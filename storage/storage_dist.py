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


from rpc_client import distcache_pb2
from rpc_client import distcache_pb2_grpc
import grpc
import io

class DistCacheClient:
    def __init__(self, grpc_port, gpu, log):
        self.grpc_port = grpc_port
        # 与cache server建立连接
        self.channel = grpc.insecure_channel(self.grpc_port, options=[
                                    ('grpc.enable_retries', 1),
                                    ('grpc.keepalive_timeout_ms', 100000),
                                    ('grpc.max_receive_message_length',
                                        2000 * 1024 * 1024),  # max grpc size 2GB
        ])
        # client can use this stub to request to golang cache server
        self.stub = distcache_pb2_grpc.OperatorStub(self.channel)

        # 与cache server建立stream features的连接
        self.channel_features = grpc.insecure_channel(self.grpc_port, options=[
                                    ('grpc.enable_retries', 1),
                                    ('grpc.keepalive_timeout_ms', 100000),
                                    ('grpc.max_receive_message_length',
                                        2000 * 1024 * 1024),  # max grpc size 2GB
        ])
        # client can use this stub to request stream features to golang cache server
        self.stub_features = distcache_pb2_grpc.OperatorFeaturesStub(self.channel_features)

        
        self.gpuid = gpu
        self.feat_dim = self.get_feat_dim()
        self.dims = {'features':self.feat_dim}
        self.log = log
        self.server_log = self.get_statistic()
        assert self.log==self.server_log, 'client端和server端的log flag不一致'

        self.try_num = 0
        self.miss_num = 0
    
    def fetch_data(self, nodeflow):
        with torch.autograd.profiler.record_function('get nf_nids'):
            # 把sub-graph的lnid都加载到gpu,这里的node_mapping是从nf-level -> part-graph lnid
            nf_nids = nodeflow._node_mapping.tousertensor()
            offsets = nodeflow._layer_offsets
            logging.debug(f'fetch_data batch onid, layer_offset: {nf_nids}, {offsets}')
        logging.debug(f'nf.nlayers: {nodeflow.num_layers}')
        for i in range(nodeflow.num_layers):
            tnid = nf_nids[offsets[i]:offsets[i+1]]
            # create frame
            with torch.autograd.profiler.record_function('create frames'):
                # with torch.cuda.device(self.gpuid):
                #     frame = {name: torch.cuda.FloatTensor(tnid.size(0), self.dims[name])
                #              for name in self.dims}  # 分配存放返回当前Layer特征的空间，size是(#onid, feature-dim)
                frame = {name: torch.empty(tnid.size(0), self.dims[name]) for name in self.dims}
                tnid = tnid.tolist()
            # fetch features from cache server
            with torch.autograd.profiler.record_function('fetch feat from cache server'):
                # collect features to cpu memory
            #     features = self.get_feats_from_server(tnid)
            # with torch.autograd.profiler.record_function('convert byte features to float tensor'):
            #     for name in self.dims:
            #         frame[name].data = torch.frombuffer(features, dtype=torch.float32).reshape(len(tnid), self.feat_dim)
                row_st_idx, row_ed_idx = 0, 0
                for sub_features in self.get_stream_feats_from_server(tnid):
                    for name in self.dims:
                        sub_tensor = torch.frombuffer(sub_features, dtype=torch.float32).reshape(-1, self.feat_dim)
                        row_ed_idx += sub_tensor.shape[0]
                        # frame[name].data = torch.cat((frame[name].data, sub_tensor), dim=0)
                        frame[name][row_st_idx:row_ed_idx, :] = sub_tensor
                        row_st_idx = row_ed_idx
            with torch.autograd.profiler.record_function('move feats from CPU to GPU'):
                # move features from cpu memory to gpu memory
                for name in self.dims:
                    frame[name].data = frame[name].data.cuda(self.gpuid)
            # attach features to nodeflow
            with torch.autograd.profiler.record_function('asign frame to nodeflow'):
                logging.debug(f'Final nodeflow._node_frames:{i}, frame["features"].size(): {frame["features"].size()}\n')
                nodeflow._node_frames[i] = FrameRef(Frame(frame))
    

    def fetch_multiple_nfs(self, nfs):
        world_size = len(nfs)
        with torch.autograd.profiler.record_function('get nf_nids'):
            nf_nids = [] # ngpu个子树的nid
            offsets = [] # ngpu个子树不同层的划分界点
            for j in range(world_size):
                nf_nids.append(nfs[j]._node_mapping.tousertensor().cuda(self.gpuid))
                offsets.append(nfs[j]._layer_offsets)
        
        for i in range(nfs[0].num_layers):
            tnid = [] # ngpu个树的第i层的nid
            tnids_flat = []
            for j in range(world_size):
                tnid.append(nf_nids[j][offsets[j][i]:offsets[j][i+1]])
                tnid[j] = tnid[j].tolist()
                tnids_flat.extend(tnid[j])
            
            # create frames
            n_rows = []
            with torch.autograd.profiler.record_function('fetch feat overhead'):
                with torch.cuda.device(self.gpuid):
                    frames = [] # ngpu个树第i层的nid构成的frame
                    for j in range(world_size):
                        frames.append({name: torch.empty(len(tnid[j]), self.dims[name]) for name in self.dims})
                        n_rows.extend([len(tnid[j]) for name in self.dims])
            # fetch features from cache server
            frame_id, row_st_idx, row_ed_idx = 0, 0, 0
            with torch.autograd.profiler.record_function('fetch feat from cache server'):
                for sub_features in self.get_stream_feats_from_server(tnids_flat):
                    for name in self.dims:
                        sub_tensor = torch.frombuffer(sub_features, dtype=torch.float32).reshape(-1, self.feat_dim)
                        n_sub_tensor_row = sub_tensor.shape[0]

                    with torch.autograd.profiler.record_function('fetch feat from cache server - python while'):
                        # sub_tensor_st_idx = 0
                        while n_sub_tensor_row: # 直到将要填充的sub_tensor为空
                            cur_frame_remains_rows = n_rows[frame_id] - row_ed_idx
                            fill_rows = min(cur_frame_remains_rows, n_sub_tensor_row)
                            row_ed_idx += fill_rows
                            frames[frame_id][name][row_st_idx:row_ed_idx, :] = sub_tensor[:fill_rows, :]
                            # frames[frame_id][name][row_st_idx:row_ed_idx, :] = sub_tensor[sub_tensor_st_idx: sub_tensor_st_idx+fill_rows, :]
                            sub_tensor = sub_tensor[fill_rows:, :]
                            # sub_tensor_st_idx += fill_rows
                            n_sub_tensor_row -= fill_rows
                            if n_sub_tensor_row > 0 or row_ed_idx == n_rows[frame_id]:
                                # 为填充下一个frame准备
                                frame_id += 1
                                row_st_idx = row_ed_idx = 0
                            else:
                                row_st_idx = row_ed_idx



                        # # 如果填充得下当前的frame剩余的空间
                        # if (row_st_idx + n_sub_tensor_row) <= n_rows[frame_id] - row_ed_idx:
                        #     row_ed_idx += n_sub_tensor_row
                        #     frames[frame_id][name][row_st_idx:row_ed_idx, :] = sub_tensor
                        #     # 如果正好填充满了当前的frame
                        #     if row_ed_idx == n_rows[frame_id]:
                        #         row_st_idx, row_ed_idx = 0, 0
                        #         frame_id += 1
                        #     else:
                        #         row_st_idx = row_ed_idx
                        # # 如果当前frame填充不下，可能往下填充一个或多个
                        # else:
                        #     sub_tensor_st_idx = 0
                        #     while n_sub_tensor_row: # 直到将要填充的sub_tensor为空
                        #         cur_frame_remains_rows = n_rows[frame_id] - row_ed_idx
                        #         fill_rows = min(cur_frame_remains_rows, n_sub_tensor_row)
                        #         row_ed_idx += fill_rows
                        #         frames[frame_id][name][row_st_idx:row_ed_idx, :] = sub_tensor[sub_tensor_st_idx:sub_tensor_st_idx + fill_rows, :]
                        #         sub_tensor_st_idx += fill_rows
                        #         n_sub_tensor_row -= fill_rows
                        #         if n_sub_tensor_row > 0:
                        #             # 为填充下一个frame准备
                        #             frame_id += 1
                        #             row_st_idx = row_ed_idx = 0
                        #         else:
                        #             row_st_idx = row_ed_idx
            with torch.autograd.profiler.record_function('move feats from CPU to GPU'):
                # move multiple features from cpu memory to gpu memory
                for j in range(world_size):
                    for name in self.dims:
                        frames[j][name].data = frames[j][name].data.cuda(self.gpuid)
            # attach features to nodeflow
            with torch.autograd.profiler.record_function('asign frame to nodeflow'):
                for j in range(world_size):
                    logging.debug(f'Final nfs._node_frames:{i}, frames[j]["features"].size(): {frames[j]["features"].size()}\n')
                    nfs[j]._node_frames[i] = FrameRef(Frame(frames[j]))
            

    def get_feat_dim(self):
        response = self.stub.DCSubmit(distcache_pb2.DCRequest(
        type=distcache_pb2.get_feature_dim), timeout=1000) # response is DCReply type response
        return response.featdim
    
    def get_statistic(self):
        response = self.stub.DCSubmit(distcache_pb2.DCRequest(
        type=distcache_pb2.get_statistic), timeout=1000) # response is DCReply type response
        return response.statistic
    
    def get_feats_from_server(self, nids):
        response = self.stub.DCSubmit(distcache_pb2.DCRequest(
        type=distcache_pb2.get_features_by_client, ids=nids), timeout=100000) # TODO: large nids list occurs high overhead
        return response.features
    
    def get_stream_feats_from_server(self, nids):
        stream = self.stub_features.DCSubmitFeatures(distcache_pb2.DCRequest(
        type=distcache_pb2.get_stream_features_by_client, ids=nids), timeout=100000)
        for response in stream:
            yield response.features
    
    def get_cache_hit_info(self):
        response = self.stub.DCSubmit(distcache_pb2.DCRequest(
        type=distcache_pb2.get_cache_info), timeout=1000)
        return response.requestnum, response.localhitnum, response.local_feats_gather_time, response.remote_feats_gather_time
    
    def get_miss_rate(self):
        requestnum, localhitnum, local_feats_gather_time, remote_feats_gather_time = self.get_cache_hit_info()
        self.try_num, self.miss_num = requestnum, requestnum-localhitnum
        miss_rate = float(self.miss_num) / self.try_num
        ret = (self.miss_num, self.try_num, miss_rate)
        self.miss_num = 0
        self.try_num = 0
        return ret
    
    def get_total_local_remote_feats_gather_time(self):
        """ 输出本地cache读取feats和远程cache读取feats的总时间 """
        requestnum, localhitnum, local_feats_gather_time, remote_feats_gather_time = self.get_cache_hit_info()
        self.local_feats_gather_time = local_feats_gather_time
        self.remote_feats_gather_time = remote_feats_gather_time
        return self.local_feats_gather_time, self.remote_feats_gather_time
    
    def get_cache_partid(self):
        response = self.stub.DCSubmit(distcache_pb2.DCRequest(
        type=distcache_pb2.get_cache_info), timeout=1000)
        return response.partidx






