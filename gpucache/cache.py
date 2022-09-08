import torch
import time
import dgl
from utils import timer
import torch.distributed as dist
import sys

class GraphGPUCache(object):
    def __init__(self, gpb, gnid2onid, local_nfeat, rank, cache_size):
        self.rank = rank
        self.gpb = gpb
        self.cache_size = cache_size # number of max cache nodes
        self.device = torch.device(f'cuda:{rank}')
        self.cached_feats_n = 0
        # self.cached_feats_mask = [False]*len(gnid2onid) # indicate whether a node feat is cached on GPU
        self.to_cache_n = min(self.cache_size, local_nfeat['_N/feat'].size(0))
        self.feat_dim = local_nfeat['_N/feat'].size(1)
        self.local_partid2nids = gpb.partid2nids(rank) # global node id belongs to current part
        # construct original id to global id
        self.onid2gnid = {oid:gid for gid,oid in enumerate(gnid2onid)}
        # gnid to onid
        self.gnid2onid = gnid2onid

        # gpu cache
        self.local_cache = {} # key: node id, value: node feature
        

        # put feat to cache
        print(f'* Rank {rank} cuda memory BEFORE cache feat: {self.get_cuda_memory_allocated()} MB.')
        
        with timer.Timer(device=f"cuda:{rank}") as t:
            for i in range(self.to_cache_n):
                gnid = self.local_partid2nids[i]
                onid = self.gnid2onid[gnid]
                self.local_cache[onid] = local_nfeat['_N/feat'][i].cuda(self.device)
                self.cached_feats_n += 1
                # self.cached_feats_mask[gnid] = True # let global node id True
        # if rank==0:
        #     print(self.local_cache)
        # assert self.cached_feats_mask.sum().item() == self.cached_feats_n
        print(f'-> Rank {rank} complete cache {self.cached_feats_n} feats in total within {t.elapsed_secs} seconds.')
        print(f'* Rank {rank} cuda memory AFTER cache feat: {self.get_cuda_memory_allocated()} MB.')
        

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
            # print(ret_idx, ret_idx.item())
            ret[ret_idx.item()] = self.local_cache[node_id.item()]
        # print(f'{self.rank} after add cached:', ret)

        for ret_idx, node_id in zip(missed_keys_mask.nonzero(), keys[missed_keys_mask]):
            print('*')
            ret[ret_idx.item()] = self.fetch_feat_from_cpu(mfgs, node_id, key_lst)
        # print(f'{self.rank} after add missed:', ret)
        return ret
    

    def construct_input_feats_include_remote(self, mfgs):
        keys = mfgs[0].srcdata[dgl.NID]
        key_lst = keys.tolist()
        print(f'-1 rank {self.rank}: {key_lst}')
        onid2retidx = {onid:retidx for retidx,onid in enumerate(key_lst)}
        ret = torch.zeros((len(keys), self.feat_dim)).cuda(self.device) # to be filled with feats to return
        # print(f'{self.rank} ori:', ret)
        # 分离出在在各个partition上的id：由于gnid很容易区分出是在哪个partition，因此使用gnid来区分
        source_lst = [] # 将keys根据partition分配到每个gpu中
        keys_lst_gnid = [self.onid2gnid[k] for k in key_lst]
        keys_lst_gnid.sort() # 从小到大排序
        print(f'-1 rank {self.rank} gnid after sort: {keys_lst_gnid}')
        nodenum_per_partition = [dct['num_nodes'] for dct in self.gpb.metadata()] # record node number in each partiton
        gnid_bound_partition = [] # [ x, y] means the first part is gnid from 0 to x, ...
        tmpsum = 0
        for item in nodenum_per_partition:
            tmpsum += item
            gnid_bound_partition.append(tmpsum-1)
        print(f'0 rank {self.rank}: {gnid_bound_partition}')
        # classify onid to each partiton and store to each list in source_lst
        tmplst = []
        bound_idx = 0 # the first elem in gnid_bound_partition
        for gnid in keys_lst_gnid:
            if gnid <= gnid_bound_partition[bound_idx]:
                tmplst.append(self.gnid2onid[gnid])
            else:
                if len(tmplst)==0:
                    source_lst.append(torch.tensor([], device=self.device)) # in gpu
                else:
                    source_lst.append(torch.tensor(tmplst, device=self.device)) # in gpu
                tmplst = []
                bound_idx += 1
        if bound_idx <= len(gnid_bound_partition)-1:
            padding_num = len(gnid_bound_partition)-bound_idx
            if len(tmplst)>0:
                source_lst.append(torch.tensor(tmplst, device=self.device)) # in gpu
                padding_num -= 1
            for _ in range(padding_num):
                source_lst.append(torch.tensor([], device=self.device)) # in gpu
        
        print(f'1 rank {self.rank}: {source_lst}')
        assert len(source_lst) == dist.get_world_size()

        # start scatter
        onid_recv = [] # 第一个tensor表示从rank=0发来的onid，第二个tensor...rank=1...
        # n 次 scatter
        tmp_onid = [None]
        for src in range(dist.get_world_size()):
            dist.scatter_object_list(tmp_onid, source_lst, src = src)
            onid_recv.append(tmp_onid)
            tmp_onid = [None]
        print(f'2 rank {self.rank}: {onid_recv}')

        # 1. read local partition cached to fill ret
        self_part_idx = onid_recv[self.rank][0]
        local_hits_onid = []
        for onid in self_part_idx:
            if onid.item() in self.local_cache:
                # print(f'** Rank {self.rank} hits node id {onid}, retidx {onid2retidx[onid.item()]}')
                ret[onid2retidx[onid.item()]] = self.local_cache[onid.item()]
                local_hits_onid.append(onid.item())
        print(f'-> Rank {self.rank} local hits {len(local_hits_onid)} ids:', local_hits_onid)
        # print(f'** Rank {self.rank} after local fill {ret}')
        # 2. read remote request from local cache and return a dict{'remote_hit':{id:feat, ...}, 'remote_miss':[xx]}
        source_lst = []
        for src,request_onid in enumerate(onid_recv):
            response_dict = {'remote_hit':{}, 'remote_miss':[]}
            if src != self.rank:
                for onid in request_onid[0]:
                    if onid.item() in self.local_cache:
                        response_dict['remote_hit'][onid.item()] = self.local_cache[onid.item()]
                    else:
                        response_dict['remote_miss'].append(onid.item())
                source_lst.append(response_dict)
            else:
                source_lst.append(response_dict)
        # 3. scatter source_lst to send feats to remote gpus
        recv_feats_dict = []
        tmp_recv_dict = [None]
        for src in range(dist.get_world_size()):
            dist.scatter_object_list(tmp_recv_dict, source_lst, src = src)
            recv_feats_dict.append(tmp_recv_dict)
            tmp_recv_dict = [None]
        # 此时，recv_feats_dict中包含了来自各个GPU的feat和未命中的点的id
        remote_hits_onid = local_hits_onid # initialize with local_hits_onid
        print(f'-> Rank {self.rank} got recv_feats_dict', recv_feats_dict)
        for recv_dict in recv_feats_dict:
            for onid, feats in recv_dict[0]['remote_hit'].items():
                # print(f'** Rank {self.rank} get remote hits node id {onid}, retidx {onid2retidx[onid]}')
                ret[onid2retidx[onid]] = feats
                remote_hits_onid.append(onid)
        missed_local_remote_onid = set(key_lst) - set(remote_hits_onid)
        print(f'-> Rank {self.rank} remote hits {len(remote_hits_onid)-len(local_hits_onid)} ids:', remote_hits_onid)
        print(f'-> Rank {self.rank} all cache missed {len(missed_local_remote_onid)} ids:', missed_local_remote_onid)
        # print(f'** Rank {self.rank} after remote fill {ret}')
        # 4. read from cpu for both local and remote missed onid
        for onid in missed_local_remote_onid:
            ret[onid2retidx[onid]] = self.fetch_feat_from_cpu(mfgs, onid, key_lst)
        # print(f'** Rank {self.rank} after cpu fetch fill {ret}')
        
        dist.barrier()
        return ret
    

    def get_cuda_memory_allocated(self):
        return torch.cuda.memory_allocated(self.device)/1024/1024


