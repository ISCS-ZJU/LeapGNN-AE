import torch
import time
import dgl
from utils import timer

class GraphGPUCache(object):
    def __init__(self, gpb, gnid2onid, local_nfeat, rank, cache_size):
        self.rank = rank
        self.cache_size = cache_size # number of max cache nodes
        self.device = torch.device(f'cuda:{rank}')
        self.cached_feats_n = 0
        # self.cached_feats_mask = [False]*len(gnid2onid) # indicate whether a node feat is cached on GPU
        self.to_cache_n = min(self.cache_size, local_nfeat['_N/feat'].size(0))
        self.feat_dim = local_nfeat['_N/feat'].size(1)
        self.local_partid2nids = gpb.partid2nids(rank) # global node id belongs to current part
        # # construct original id to global id
        # self.onid2gnid = {oid:gid: for gid,oid in enumerate(gnid2onid)}

        # gpu cache
        self.local_cache = {} # key: node id, value: node feature
        

        # put feat to cache
        print(f'Rank {rank} cuda memory BEFORE cache feat: {self.get_cuda_memory_allocated()} MB.')
        
        with timer.Timer(device=f"cuda:{rank}") as t:
            for i in range(self.to_cache_n):
                gnid = self.local_partid2nids[i]
                onid = gnid2onid[gnid]
                self.local_cache[onid] = local_nfeat['_N/feat'][i].cuda(self.device)
                self.cached_feats_n += 1
                # self.cached_feats_mask[gnid] = True # let global node id True
        # if rank==0:
        #     print(self.local_cache)
        # assert self.cached_feats_mask.sum().item() == self.cached_feats_n
        print(f'-> Rank {rank} complete cache {self.cached_feats_n} feats in total within {t.elapsed_secs} seconds.')
        print(f'Rank {rank} cuda memory AFTER cache feat: {self.get_cuda_memory_allocated()} MB.')
        

    def fetch_feat_from_cpu(self, mfgs, key, key_lst):
        return mfgs[0].srcdata['feat'][key_lst.index(key)]
    
    def fetch_feats_from_remotegpu(self, keys):
        pass

    def construct_input_feats(self, mfgs):
        keys = mfgs[0].srcdata[dgl.NID]
        key_lst = keys.tolist()
        ret = torch.zeros((len(keys), self.feat_dim)).cuda(self.device) # to be filled with values
        # print(f'{self.rank} ori:', ret)
        # 分离出在缓存和不在缓存的点的id
        cached_keys_mask = torch.tensor([item in self.local_cache for item in key_lst])
        missed_keys_mask = ~cached_keys_mask
        print(f'-> Rank {self.rank} hits {sum(cached_keys_mask)} node features in GPU, miss {sum(missed_keys_mask)} node features.')

        # get cached features
        for ret_idx, node_id in zip(cached_keys_mask.nonzero(), keys[cached_keys_mask]):
            print(ret_idx, ret_idx.item())
            ret[ret_idx.item()] = self.local_cache[node_id.item()]
        # print(f'{self.rank} after add cached:', ret)

        for ret_idx, node_id in zip(missed_keys_mask.nonzero(), keys[missed_keys_mask]):
            ret[ret_idx.item()] = self.fetch_feat_from_cpu(mfgs, node_id, key_lst)
        # print(f'{self.rank} after add missed:', ret)
        return ret
    
    def get_cuda_memory_allocated(self):
        return torch.cuda.memory_allocated(self.device)/1024/1024


