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

class DistCacheClient:
    def __init__(self, grpc_port, gpu, log):
        self.grpc_port = grpc_port
        # 与cache server建立连接
        self.channel = grpc.insecure_channel(self.grpc_port, options=[
                                    ('grpc.enable_retries', 1),
                                    ('grpc.keepalive_timeout_ms', 100000),
                                    ('grpc.max_receive_message_length',
                                        100 * 1024 * 1024),  # max grpc size 20MB
        ])
        # client can use this stub to request to golang cache server
        self.stub = distcache_pb2_grpc.OperatorStub(self.channel)

        
        self.gpuid = gpu
        self.feat_dim = self.get_feat_dim()
        self.dims = {'features':self.feat_dim}
        self.log = log

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
            with torch.autograd.profiler.record_function('fetch feat overhead'):
                with torch.cuda.device(self.gpuid):
                    frame = {name: torch.cuda.FloatTensor(tnid.size(0), self.dims[name])
                             for name in self.dims}  # 分配存放返回当前Layer特征的空间，size是(#onid, feature-dim)
            # fetch features from cache server
            with torch.autograd.profiler.record_function('fetch feat from cache server'):
                # collect features to cpu memory
                features = list(self.get_feats_from_server(tnid))
                cpu_features = np.array(features, dtype=np.float32).reshape(len(tnid), self.feat_dim)
                # move features to gpu
                for name in self.dims:
                    frame[name].data = torch.FloatTensor(cpu_features).cuda(self.gpuid)
            # attach features to nodeflow
            with torch.autograd.profiler.record_function('cache-asign'):
                logging.debug(f'Final nodeflow._node_frames:{i}, frame["features"].size(): {frame["features"].size()}\n')
                nodeflow._node_frames[i] = FrameRef(Frame(frame))
            if self.log:
                localhitnum, requestnum = self.get_cache_hit_info()
                self.log_miss_rate(localhitnum, requestnum)
    
    def get_feat_dim(self):
        response = self.stub.DCSubmit(distcache_pb2.DCRequest(
        type=distcache_pb2.get_feature_dim), timeout=1000) # response is DCReply type response
        return response.featdim
    
    def get_feats_from_server(self, nids):
        response = self.stub.DCSubmit(distcache_pb2.DCRequest(
        type=distcache_pb2.get_features_by_client, ids=nids), timeout=100000)
        return response.features
    
    def get_cache_hit_info(self):
        response = self.stub.DCSubmit(distcache_pb2.DCRequest(
        type=distcache_pb2.get_cache_info), timeout=1000)
        return response.requestnum, response.localhitnum
    
    def log_miss_rate(self, miss_num, total_num):
        self.try_num += total_num
        self.miss_num += miss_num
    
    def get_miss_rate(self):
        miss_rate = float(self.miss_num) / self.try_num
        print(f'self.miss_num, self.try_num: {self.miss_num}, {self.try_num}, {self.miss_num/self.try_num}', file=self.f)
        self.miss_num = 0
        self.try_num = 0
        return miss_rate
    





