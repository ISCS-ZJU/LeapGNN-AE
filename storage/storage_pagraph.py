import os
import sys

import numpy as np
import torch
from dgl import DGLGraph
from dgl.frame import Frame, FrameRef
import dgl.utils
import data
import torch.distributed as dist

import logging
# logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(asctime)s %(filename)s %(lineno)d : %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')


def Print(*content, debug=True):
    if debug:
        print(*content)


class PaGraphGraphCacheServer:
    """
    Manage graph features
    Automatically fetch the feature tensor from CPU or GPU
    """

    def __init__(self, graph, node_num, nid_map, gpuid):
        """
        Paramters:
          graph:   should be created from `dgl.contrib.graph_store`
          node_num: should be sub graph node num
          nid_map: torch tensor. map from local node id to full graph id.
                   used in fetch features from remote
        """
        self.graph = graph  # CPU full graph topo
        self.gpuid = gpuid
        self.node_num = node_num  # part_graph total node number
        self.nid_map = nid_map.clone().detach().cuda(self.gpuid)  # part-graph lnid2onid
        self.nid_map.requires_grad_(False)

        # masks for manage the feature locations: default in CPU (在开始cache前所有点都在cpu，所有都是False)
        self.gpu_flag = torch.zeros(self.node_num).bool().cuda(
            self.gpuid)  # part-graph lnid
        self.gpu_flag.requires_grad_(False)

        self.cached_num = 0  # 已经被缓存的点数
        self.capability = node_num  # 缓存容量

        # gpu tensor cache
        self.full_cached = False
        self.dims = {}          # {'field name': dims of the tensor data of a node}
        self.total_dim = 0
        # {'field name': tensor data for cached nodes in gpu} # 真正缓存的地方
        self.gpu_fix_cache = dict()
        with torch.cuda.device(self.gpuid):
            # 表示part-graph lnid -> cache id （好处是根据part-graph lnid可以很快获取cache row id）
            self.localid2cacheid = torch.cuda.LongTensor(node_num).fill_(0)
            self.localid2cacheid.requires_grad_(False)

        # logs
        self.log = False
        self.try_num = 0
        self.miss_num = 0

    def init_field(self, embed_names):
        with torch.cuda.device(self.gpuid):
            nid = torch.cuda.LongTensor([0])
        feats = self.get_feat_from_server(nid, embed_names)
        self.total_dim = 0
        for name in embed_names:
            self.dims[name] = feats[name].size(1)
            self.total_dim += feats[name].size(1)
        logging.info('total dims: {}'.format(self.total_dim))

    def auto_cache(self, dgl_g, embed_names):
        """
        Automatically cache the node features
        Params:
          g: DGLGraph for local graphs
          embed_names: field name list, e.g. ['features', 'norm']
        """
        # Step1: get available GPU memory
        peak_allocated_mem = torch.cuda.max_memory_allocated(device=self.gpuid)
        peak_cached_mem = torch.cuda.max_memory_reserved(device=self.gpuid)
        total_mem = torch.cuda.get_device_properties(self.gpuid).total_memory
        available = total_mem - peak_allocated_mem - peak_cached_mem \
            - 1024 * 1024 * 1024  # in bytes
        # Stpe2: get capability
        # assume float32 = 4 bytes
        self.capability = int(available / (self.total_dim * 4))
        #self.capability = int(6 * 1024 * 1024 * 1024 / (self.total_dim * 4))
        # self.capability = int(part_node_num * 0.2)
        logging.info('Cache Memory: {:.2f}G. Capability: {}'
              .format(available / 1024 / 1024 / 1024, self.capability))
        # Step3: cache
        if self.capability >= self.node_num:
            # fully cache
            logging.info('cache the full graph...')
            full_nids = torch.arange(self.node_num).cuda(
                self.gpuid)  # part-graph 中的lnid是0开始的
            data_frame = self.get_feat_from_server(full_nids, embed_names)
            # 最终缓存里的node id是full-graph的onid
            self.cache_fix_data(full_nids, data_frame, is_full=True)
        else:
            # choose top-cap out-degree nodes to cache
            logging.info('cache the part of graph... caching percentage: {:.4f}'
                  .format(self.capability / self.node_num))
            out_degrees = dgl_g.out_degrees()  # 对当前part-graph的node degree 排序
            sort_nid = torch.argsort(out_degrees, descending=True)
            cache_nid = sort_nid[:self.capability]  # part-graph level lnid
            data_frame = self.get_feat_from_server(cache_nid, embed_names)
            self.cache_fix_data(cache_nid, data_frame, is_full=False)

    def get_feat_from_server(self, nids, embed_names, to_gpu=False):
        """
        Fetch features of `nids` from remote server in shared CPU
        Params
          g: created from `dgl.contrib.graph_store.create_graph_from_store`
          nids: required node ids in local graph, should be in gpu
          embed_names: field name list, e.g. ['features', 'norm']
        Return:
          feature tensors of these nids (in CPU)
        """
        nids_in_full = self.nid_map[nids]  # part-graph lnid -> full-graph onid
        #cpu_frame = self.graph._node_frame[dgl.utils.toindex(nids_in_full.cpu())]
        #data_frame = {}
        # for name in embed_names:
        #  if to_gpu:
        #    data_frame[name] = cpu_frame[name].cuda(self.gpuid)
        #  else:
        #    data_frame[name] = cpu_frame[name]
        # return data_frame
        nids = nids_in_full.cpu()
        if to_gpu:
            frame = {name: self.graph._node_frame._frame[name].data[nids].cuda(self.gpuid, non_blocking=True)
                     for name in embed_names}
        else:
            frame = {
                name: self.graph._node_frame._frame[name].data[nids] for name in embed_names}
        return frame

    def cache_fix_data(self, nids, data, is_full=False):
        """
        User should make sure tensor data under every field name should
        have same num (axis 0)
        Params:
          nids: node ids to be cached in local graph.
                should be equal to data rows. should be in gpu
          data: dict: {'field name': tensor data}
        """
        rows = nids.size(0)
        self.localid2cacheid[nids] = torch.arange(rows).cuda(
            self.gpuid)  # part-graph lnid -> cache row id
        self.cached_num = rows
        for name in data:
            data_rows = data[name].size(0)
            assert (rows == data_rows)
            self.dims[name] = data[name].size(1)
            self.gpu_fix_cache[name] = data[name].cuda(self.gpuid)
        # setup flags
        self.gpu_flag[nids] = True
        self.full_cached = is_full

    def fetch_data(self, nodeflow):
        """
        copy feature from local GPU memory or
        remote CPU memory, which depends on feature
        current location.
        --Note: Should be paralleled
        Params:
          nodeflow: DGL nodeflow. all nids in nodeflow should
                    under sub-graph space
        """
        if self.full_cached:
            self.fetch_from_cache(nodeflow)
            return
        with torch.autograd.profiler.record_function('cache-idxload'):
            # 把sub-graph的lnid都加载到gpu,这里的node_mapping是从nf-level -> part-graph lnid
            nf_nids = nodeflow._node_mapping.tousertensor().cuda(self.gpuid)
            offsets = nodeflow._layer_offsets
            logging.debug(f'fetch_data batch onid, layer_offset: {nf_nids}, {offsets}')
        logging.debug(f'nf.nlayers: {nodeflow.num_layers}')
        for i in range(nodeflow.num_layers):
            # with torch.autograd.profiler.record_function('cache-idx-load'):
            #tnid = nodeflow.layer_parent_nid(i).cuda(self.gpuid)
            tnid = nf_nids[offsets[i]:offsets[i+1]]
            # get nids -- overhead ~0.1s
            with torch.autograd.profiler.record_function('cache-index'):
                gpu_mask = self.gpu_flag[tnid]
                # lnid cached in gpu (part-graph level)
                nids_in_gpu = tnid[gpu_mask]
                cpu_mask = ~gpu_mask
                # lnid cached in cpu (part-graph level)
                nids_in_cpu = tnid[cpu_mask]
            # create frame
            with torch.autograd.profiler.record_function('cache-allocate'):
                with torch.cuda.device(self.gpuid):
                    frame = {name: torch.cuda.FloatTensor(tnid.size(0), self.dims[name])
                             for name in self.dims}  # 分配存放返回当前Layer特征的空间，size是(#onid, feature-dim)
            # for gpu cached tensors: ##NOTE: Make sure it is in-place update!
            with torch.autograd.profiler.record_function('cache-gpu feat read'):
                if nids_in_gpu.size(0) != 0:
                    cacheid = self.localid2cacheid[nids_in_gpu]
                    for name in self.dims:
                        frame[name][gpu_mask] = self.gpu_fix_cache[name][cacheid]
            # for cpu cached tensors: ##NOTE: Make sure it is in-place update!
            with torch.autograd.profiler.record_function('cache-cpu feat read'):
                if nids_in_cpu.size(0) != 0:
                    cpu_data_frame = self.get_feat_from_server(
                        nids_in_cpu, list(self.dims), to_gpu=True)
                    for name in self.dims:
                        logging.debug(f'fetch features from cpu for frame["features"].size(): {frame[name].size()}, cpu_mask: {cpu_mask}, cpu_data_frame.shape: {cpu_data_frame[name].size()}')
                        frame[name][cpu_mask] = cpu_data_frame[name]
            with torch.autograd.profiler.record_function('cache-asign'):
                logging.debug(f'Final nodeflow._node_frames:{i}, frame["features"].size(): {frame["features"].size()}\n')
                nodeflow._node_frames[i] = FrameRef(Frame(frame))
            if self.log:
                self.log_miss_rate(nids_in_cpu.size(0), tnid.size(0))

    def fetch_from_cache(self, nodeflow):
        for i in range(nodeflow.num_layers):
            #nid = dgl.utils.toindex(nodeflow.layer_parent_nid(i))
            with torch.autograd.profiler.record_function('cache-idxload'):
                tnid = nodeflow.layer_parent_nid(i).cuda(self.gpuid)
            with torch.autograd.profiler.record_function('cache-gpu'):
                frame = {}
                for name in self.gpu_fix_cache:
                    frame[name] = self.gpu_fix_cache[name][tnid]
            nodeflow._node_frames[i] = FrameRef(Frame(frame))

    def log_miss_rate(self, miss_num, total_num):
        self.try_num += total_num
        self.miss_num += miss_num

    def get_miss_rate(self):
        miss_rate = float(self.miss_num) / self.try_num
        self.miss_num = 0
        self.try_num = 0
        return miss_rate
