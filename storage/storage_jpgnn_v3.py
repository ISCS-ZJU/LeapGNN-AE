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




class NewJPGNNGraphCacheServer:
    """
    Manage graph features
    Automatically fetch the feature tensor from CPU or local GPU
    """

    def __init__(self, graph, node_num, gpuid, world_size):
        """
        Paramters:
          graph:   should be created from `dgl.contrib.graph_store`
          node_num: should be sub graph node num
        """
        self.graph = graph  # CPU full graph topo
        self.gpuid = gpuid
        self.world_size = world_size
        self.node_num = node_num

        # masks for manage the feature locations: default in CPU (在开始cache前所有点都在cpu，所有都是False)
        self.gpu_flag = torch.zeros(self.node_num).bool().cuda(self.gpuid)
        self.gpu_flag.requires_grad_(False)

        self.cached_num = 0  # 已经被缓存的点数
        self.capability = node_num  # 缓存容量初始化

        # gpu tensor cache
        self.full_cached = False
        self.dims = {}          # {'field name': dims of the tensor data of a node}
        self.total_dim = 0
        # {'field name': tensor data for cached nodes in gpu} # 真正缓存的地方
        self.gpu_fix_cache = dict()
        with torch.cuda.device(self.gpuid):
            self.localid2cacheid = torch.cuda.LongTensor(node_num).fill_(0)
            self.localid2cacheid.requires_grad_(False)

        # logs
        self.log = True
        self.try_num = 0
        self.miss_num = 0
        self.f = open('jpgnn_cpu_degree_hit_rate.txt', 'a+')

        # 
        self.tol_node_mask = torch.zeros(self.node_num).bool().cuda(self.gpuid) # 全图nid数组
        self.tol_node_mask.requires_grad_(False)
        self.node_mask = [] # ngpu个全图nid数组，其中要从cpu取的位置是True，其他是False
        for j in range(self.world_size):
            self.node_mask.append(torch.zeros(self.node_num).bool().cuda(self.gpuid))
            self.node_mask[j].requires_grad_(False)
        if self.log:
            self.tol_try_mask = torch.zeros(self.node_num).bool().cuda(self.gpuid)
            self.tol_try_mask.requires_grad_(False)


    def init_field(self, embed_names):
        with torch.cuda.device(self.gpuid):
            nid = torch.cuda.LongTensor([0])
        feats = self.get_feat_from_server(nid, embed_names)
        self.total_dim = 0
        for name in embed_names:
            self.dims[name] = feats[name].size(1)
            self.total_dim += feats[name].size(1)
        logging.info('total dims: {}'.format(self.total_dim))

    def auto_cache(self, dataset, methodname, partitions, rank, embed_names):
        """
        Automatically cache the node features
        Params:
          g: DGLGraph for local graphs
          embed_names: field name list, e.g. ['features', 'norm']
        """
        # 加载分割到的这部分的图id
        sorted_part_nids = data.get_partition_results(
            dataset, methodname, partitions, rank)
        part_node_num = sorted_part_nids.shape[0]
        sorted_part_nids = torch.from_numpy(sorted_part_nids)
        logging.info(f'rank {rank} got a part of graph with {part_node_num} nodes.')

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
        self.capability = int(part_node_num * 0.2) # 缓存当前分图的20%的数据
        logging.info('Cache Memory: {:.2f}G. Capability: {}'
              .format(available / 1024 / 1024 / 1024, self.capability))
        # Step3: cache
        if self.capability >= part_node_num:
            # fully cache
            logging.info('cache the part graph... caching percentage: 100%')
            data_frame = self.get_feat_from_server(
                sorted_part_nids, embed_names)
            # 最终缓存里的node id是full-graph的onid
            self.cache_fix_data(sorted_part_nids, data_frame, is_full=True)
        else:
            # choose top-cap out-degree nodes to cache
            logging.info('cache the part of graph... caching percentage: {:.4f}'
                  .format(self.capability / part_node_num))
            cache_nid = sorted_part_nids[:self.capability]
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
        nids = nids.cpu()
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
        self.localid2cacheid[nids] = torch.arange(rows).cuda(self.gpuid)
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
            for j in range(self.world_size):
                self.fetch_from_cache(nodeflow[j])
            return
        with torch.autograd.profiler.record_function('get nf_nids'):
            # 把sub-graph的lnid都加载到gpu,这里的node_mapping是从nf-level -> part-graph lnid
            nf_nids = [] # ngpu个子树的nid
            offsets = [] # ngpu个子树不同层的划分界点
            for j in range(self.world_size):
                nf_nids.append(nodeflow[j]._node_mapping.tousertensor().cuda(self.gpuid))
                offsets.append(nodeflow[j]._layer_offsets)
            logging.debug(f'fetch_data batch onid, layer_offset: {nf_nids}, {offsets}')
            logging.debug(f'nf.nlayers: {nodeflow[j].num_layers}')
        for i in range(nodeflow[0].num_layers):
            tnid = [] # ngpu个树的第i层的nid
            for j in range(self.world_size):
                tnid.append(nf_nids[j][offsets[j][i]:offsets[j][i+1]])
            
            with torch.autograd.profiler.record_function('fetch feat overhead'):

                self.tol_node_mask[:] = False

                gpu_mask = [] # ngpu个树第i层的node是否在gpu中
                # lnid cached in gpu (part-graph level)
                nids_in_gpu = [] # ngpu个树第i层的node在gpu中的nid
                cpu_mask = [] # ngpu个树第i层的node是否在cpu中
                # lnid cached in cpu (part-graph level)                
                if self.log:
                    self.tol_try_mask[:] = False
                
                for j in range(self.world_size):
                    gpu_mask.append(self.gpu_flag[tnid[j]])
                    cpu_mask.append(~gpu_mask[j])
                    nids_in_gpu.append(tnid[j][gpu_mask[j]])
                    self.node_mask[j][:] = False
                    self.node_mask[j][tnid[j][cpu_mask[j]]] = True
                    self.tol_node_mask[tnid[j][cpu_mask[j]]] = True # ngpu个子树在i层的所有要从cpu取的id都置True
                    self.tol_try_mask[tnid[j]] = True
                nids_in_cpu = torch.where(self.tol_node_mask==True)[0] # ngpu个子树在第i层的所有要从cpu取的nid（已去重）
                
            # create frame
            with torch.autograd.profiler.record_function('fetch feat overhead'):
                with torch.cuda.device(self.gpuid):
                    frame = [] # ngpu个树第i层的nid构成的frame
                    for j in range(self.world_size):
                        frame.append({name: torch.cuda.FloatTensor(tnid[j].size(0), self.dims[name])
                                for name in self.dims})  # 分配存放返回当前Layer特征的空间，size是(#onid, feature-dim)
            # for gpu cached tensors: ##NOTE: Make sure it is in-place update!
            with torch.autograd.profiler.record_function('fetch feat from local gpu'):
                for j in range(self.world_size):
                    if nids_in_gpu[j].size(0) != 0:
                        cacheid = self.localid2cacheid[nids_in_gpu[j]]
                        for name in self.dims:
                            frame[j][name][gpu_mask[j]] = self.gpu_fix_cache[name][cacheid]
            # for cpu cached tensors: ##NOTE: Make sure it is in-place update!
            with torch.autograd.profiler.record_function('fetch feat from cpu-pure'):
                if nids_in_cpu.size(0) != 0:
                    cpu_data_frame = self.get_feat_from_server(
                        nids_in_cpu, list(self.dims), to_gpu=True) # ngpu个树的第i层缺nid都一次从cpu读取
            with torch.autograd.profiler.record_function('fetch feat from cpu-construct frames'):
                for j in range(self.world_size):
                    for name in self.dims:
                        logging.debug(f'fetch features from cpu for frame["features"].size(): {frame[j][name].size()}, cpu_mask: {cpu_mask[j]}, cpu_data_frame.shape: {cpu_data_frame[name].size()}')
                        frame[j][name][cpu_mask[j]] = cpu_data_frame[name][self.node_mask[j][self.tol_node_mask]]
            with torch.autograd.profiler.record_function('cache-asign'):
                for j in range(self.world_size):
                    logging.debug(f'Final nodeflow._node_frames:{i}, frame["features"].size(): {frame[j]["features"].size()}\n')
                    nodeflow[j]._node_frames[i] = FrameRef(Frame(frame[j]))
            if self.log:
                self.log_miss_rate(nids_in_cpu.size(0), torch.where(self.tol_try_mask==True)[0].size(0))

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
        print(f'self.miss_num, self.try_num: {self.miss_num}, {self.try_num}, {self.miss_num/self.try_num}', file=self.f)
        self.miss_num = 0
        self.try_num = 0
        return miss_rate