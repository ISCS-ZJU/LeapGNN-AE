import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__))) # repgnn

import numpy as np
import torch
from dgl import DGLGraph
from dgl.frame import Frame, FrameRef
import torch.distributed as dist

import logging
from data import *

class LocalCacheClient:
    def __init__(self, gpu, log, nodes, datasetpath):
        self.datasetpath = datasetpath
        self.gpuid = gpu
        self.log = log
        self.nodes = nodes # total number of workers to simulate
        

        # partitioned graph data
        partgnid_npy_filepath = os.path.join(datasetpath, f'dist_True/{self.nodes}_metis')
        if not os.path.exists(partgnid_npy_filepath):
            try:
                os.system(f"python3 prepartition/metis.py --partition {self.nodes} --dataset {self.datasetpath}")
                print(f"EXECUTE python3 prepartition/metis.py --partition {self.nodes} --dataset {self.datasetpath} DONE")
            except Exception as e:
                raise Exception(f"Failed to execute metis.py, error: {e}")
        
        # load all feats into memory
        _, self.feats = get_graph_data(self.datasetpath)

        self.feat_dim = self.get_feat_dim()
        self.dims = {'features':self.feat_dim}

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
                frame = {name: torch.empty(tnid.size(0), self.dims[name]) for name in self.dims}
                tnid = tnid.tolist()
            # fetch features from local storage
            with torch.autograd.profiler.record_function('fetch feat from local storage'):
            # (non-stream method)
                features = self.get_feats_from_local(tnid)
            with torch.autograd.profiler.record_function('convert byte features to float tensor'):
                for name in self.dims:
                    frame[name].data = torch.frombuffer(features, dtype=torch.float32).reshape(len(tnid), self.feat_dim)
            with torch.autograd.profiler.record_function('move feats from CPU to GPU'):
                # move features from cpu memory to gpu memory
                for name in self.dims:
                    frame[name].data = frame[name].data.cuda(self.gpuid)
            # attach features to nodeflow
            with torch.autograd.profiler.record_function('asign frame to nodeflow'):
                logging.debug(f'Final nodeflow._node_frames:{i}, frame["features"].size(): {frame["features"].size()}\n')
                nodeflow._node_frames[i] = FrameRef(Frame(frame))
    
    def get_feat_dim(self):
        return self.feats[0].shape[0]
    
    def get_feats_from_local(self, nids):
        return self.feats[nids]






