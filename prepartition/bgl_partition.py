import sys
sys.path.append('./')
sys.path.append('/home/qhy/gnn/repgnn')
import data
import enum
import numpy as np
import pymetis
import argparse
import scipy.sparse as spsp
import os
from queue import Queue

import logging

class disjoint_set():
    def __init__(self,size,blocks):
        self.block_remap = np.arange(size,dtype=np.int64)
        self.merged_block_size = np.zeros(size,dtype=np.int64)
        for block in blocks:
            self.merged_block_size[block.id] = block.node_num
    
    def find(self,x):
        # 并查集查找
        root = self.block_remap[x]
        while root < 0:
            root = self.block_remap[-root - 1]
        # 路径压缩 
        while x != root:
            lead = self.block_remap[x]
            self.block_remap[x] = -root - 1
            x = -lead - 1
        return root

    def union(self,x,y):
        root1 = self.find(x)
        root2 = self.find(y)
        if root1 != root2:
            self.merged_block_size[root2] += self.merged_block_size[root1]
            self.block_remap[root1] = -root2 -1
            return True
        return False
    
    def get_size(self,x):
        return self.merged_block_size[self.find(x)]
    
    def update_all(self):
        while (self.block_remap < 0).any():
            self.block_remap[self.block_remap < 0] = self.block_remap[-self.block_remap[self.block_remap < 0] - 1]



# def find(disjoint_set,x):
#     # 并查集查找
#     root = disjoint_set[x]
#     while root < 0:
#         root = disjoint_set[-root - 1]
#     # 路径压缩 
#     while x != root:
#         lead = disjoint_set[x]
#         disjoint_set[x] = -root - 1
#         x = -lead - 1
#     return root

class Part():
    def __init__(self,id):
        self.id = id
        self.blocks = []  #需要吗？
        self.node_num = 0
        self.train_node_num = 0
    
    def add_block(self,block):
        self.blocks.append(block)
        self.node_num += block.node_num
        self.train_node_num += block.train_node_num
        

class Block():
    def __init__(self,id,max_node_num):
        self.id = id
        self.nodes = []  #需要吗？
        self.node_num = 0
        self.train_node_num = 0
        self.max_node_num = max_node_num
    
    def add(self,node,train_type):
        self.node_num += 1
        if train_type:
            self.train_node_num += 1
        self.nodes.append(node)

    def get_train_node_num(self,train_mask):
        self.train_node_num = np.sum(train_mask[self.nodes])

    def full(self):
        return self.node_num >= self.max_node_num 
    
    def merge(self,block):
        self.nodes.extend(block.nodes)
        self.node_num += block.node_num
        self.train_node_num += block.train_node_num


class BlockGenerator():
    def __init__(self,adj,node_num,train_mask,block_size_scale=0.01,hop=2,assign_init_scale = 0.3):
        self.adj = adj  #邻接矩阵，压缩过的，csr？
        self.blocks = []
        self.node_num = node_num
        self.block_map = np.zeros(node_num,dtype=np.int64)
        self.block_map.fill(-1)
        self.seed_node = np.arange(node_num)
        self.queue = Queue()
        # self.expected_block_num = int(node_num/threshold) 
        self.expected_block_num = 1

        self.cur_block_id = 0
        self.train_mask = train_mask
        self.train_node_num = np.sum(train_mask)
        np.random.seed(0)
        np.random.shuffle(self.seed_node)
        self.visit_num = 0
        self.hop = hop
        self.assign_init_scale = assign_init_scale
        self.threshold = int(node_num*block_size_scale)  #停止继续搜索的阈值
        # self.merged_block_threshold=int(node_num*merged_block_size_scale)

    def _get_seed_node(self,num=None):
        #随机选取一个未访问过的点
        seed_nodes = np.where(self.block_map == -1)[0]
        seed_node = np.random.choice(seed_nodes,num,replace=False)
        return seed_node
    
    def _add_neighbor(self,node,block):
        #把node的邻居添加到对应的块中
        #如果不控制超出阈值直接停止可以一次性添加neighbor
        neighbors = self.adj.getrow(node).indices
        for neighbor in neighbors:
            if self.block_map[neighbor] < 0:
                self.block_map[neighbor] = self.block_map[node]
                self.visit_num += 1
                block.add(neighbor,self.train_mask[neighbor])
                if self.visit_num >= self.node_num or block.full():
                    #块内点的数量超出阈值
                    return
                self.queue.put(neighbor)
        return
    
    def _init_new_block(self,seed_node):
        self.block_map[seed_node] = self.cur_block_id
        self.visit_num += 1
        self.queue.put(seed_node)
        self.blocks.append(Block(self.cur_block_id,self.threshold))
        self.blocks[self.cur_block_id].add(seed_node,self.train_mask[seed_node])
        self.cur_block_id += 1
    
    def _init_bfs(self):
        seed_nodes = self._get_seed_node(self.expected_block_num)
        for seed_node in seed_nodes:
            assert self.block_map[seed_node] < 0
            self._init_new_block(seed_node)
        
    
    # def _gengerate_one_block(self):
    #     #生成单个块
    #     if self.queue.empty():
    #         self._init_new_block()

    #     while not self.queue.empty():
    #         node = self.queue.get()
    #         cur_id = self.block_map[node]
    #         cur_block = self.blocks[cur_id]
    #         if not cur_block.full():
    #             if self._add_neighbor(node,cur_block):
    #                 #块内点的数量达到阈值（不知道需不需要）
    #                 break


    def _generate_blocks(self):
        #每个点最多入队一次，每次出队最多访问其所有邻居一遍，O(E)
        self._init_bfs()
        while self.visit_num < self.node_num:
            if self.queue.empty():
                self._init_new_block(self._get_seed_node())
            while not self.queue.empty():
                node = self.queue.get()
                cur_id = self.block_map[node]
                cur_block = self.blocks[cur_id]
                if cur_block.node_num < self.threshold:
                    self._add_neighbor(node,cur_block)
                    #块内点的数量达到阈值或添加完成其所有邻居
    
    def _generate(self):
        self._generate_blocks()
        self._get_block_adj(add_reversed_edge=True)
        self._merge_blocks()
        self._get_block_adj()
        self.block2part = np.zeros(len(self.blocks),dtype = np.int64)
        self.block2part.fill(-1)
        # for block in self.blocks:
        #     block.get_train_node_num(self.train_mask)
                    
    def _get_block_adj(self,add_reversed_edge=False):
        block_num = len(self.blocks)
        rows = []
        cols = []
        self.block_adj = np.zeros((block_num,block_num))  #块的邻接矩阵
        for row,block in enumerate(self.blocks):
            for node in block.nodes:
                neighbors = self.adj.getrow(node).indices
                for neighbor in neighbors:
                    rows.append(row)
                    cols.append(self.block_map[neighbor])
                    # self.block_adj[row][self.block_map[neighbor]] = 1  #把一条边的两个点所属的块是相连的
                    if add_reversed_edge:
                        cols.append(row)
                        rows.append(self.block_map[neighbor])
                        # self.block_adj[self.block_map[neighbor]][row] = 1
        #to csr
        # row, col = np.nonzero(self.block_adj)
        # values = self.block_adj[row, col]
        values = np.ones(len(rows))
        self.block_adj = spsp.csr_matrix((values, (rows, cols)), shape=(block_num,block_num))

    
    def _merge_blocks(self):
        block_num = len(self.blocks)
        block_remap = np.arange(block_num,dtype=np.int64)
        block_remap = disjoint_set(block_num,self.blocks)
        # 并查集，-1表示指向0号块，-2表示指向1号块，正数表示根节点，值代表块id
        for block in self.blocks:
            if block_remap.get_size(block.id) < self.threshold:
                neighbors = self.block_adj.getrow(block.id).indices
                if len(neighbors) > 0:
                    min_size = self.node_num
                    min_id = -1
                    for neighbor in neighbors:
                        merged_block_size = block_remap.get_size(neighbor)
                        if merged_block_size >= self.threshold:
                            # if merged_block_size < min_size and merged_block_size < self.merged_block_threshold:
                            if merged_block_size < min_size:
                                min_size = merged_block_size
                                min_id = neighbor
                    if min_id >= 0:
                        block_remap.union(block.id,min_id)
                    else:
                        for neighbor in neighbors:
                            # merged_block_size = block_remap.get_size(neighbor)
                            # if merged_block_size < self.merged_block_threshold:
                            block_remap.union(block.id,neighbor)
                                # block_remap[block_remap.find(block.id)] = -neighbor - 1
                                # break
                        # block_remap[block.id] = -(np.random.choice(neighbors)) - 1
                        # merged_block_num -= 1
                        # 没有大块邻居，随机合并给一个邻居
        
        # 并查集搜索
        block_remap.update_all()
        # 开始合并
        new_blocks = []
        count = 0
        # print(len(block_remap),block_num)
        for i in range(block_num):
            new_block = Block(i,self.threshold)
            block_mask = (block_remap.block_remap == i)
            if block_mask.any():
                block_ids = np.nonzero(block_mask)[0]
                for block_id in block_ids:
                    new_block.merge(self.blocks[block_id])
                    block_remap.block_remap[block_id] = count
                new_block.id = count
                # print(new_block.node_num)
                new_blocks.append(new_block)
                count += 1
        self.block_map = block_remap.block_remap[self.block_map]
        self.blocks = new_blocks
        # print(len(self.blocks))



    def _score(self,block_id,part_num):
        first = np.zeros(part_num)
        second = np.zeros(part_num)
        third = np.zeros(part_num)
        score = np.zeros(part_num)
        # blocks = self.blocks[block_id]
        block_ids = np.zeros(1)
        block_ids[0] = block_id
        neighbors = []
        for _ in range(self.hop):
            for block_id in block_ids:
                neighbors.append(self.block_adj.getrow(block_id).indices)
            if len(neighbors) == 0:
                break
            neighbors = np.concatenate(neighbors)
            neighbors , _ = np.unique(neighbors, return_inverse=True)
            for neighbor in neighbors:
                block = self.blocks[neighbor]
                if self.block2part[neighbor] >= 0:
                    # first[self.block2part[neighbor]] += block.node_num
                    first[self.block2part[neighbor]] += 1
            block_ids = neighbors
            neighbors = []
        for part_id in range(part_num):
            second[part_id] = 1 - (self.parts[part_id].node_num/(self.node_num/part_num))
            third[part_id] = 1 - (self.parts[part_id].train_node_num/(self.train_node_num/part_num))
            if second[part_id] < 0 and third[part_id] < 0:
                score[part_id] = -0.1
            else:
                score[part_id] = first[part_id] * second[part_id] * third[part_id]
        # np.clip(second,0,None)
        # np.clip(third,0,None)
        # score = first * second * third
        # mask = second < 0 & third < 0
        # score[mask] = -0.1
        print(f'{score},first:{first},second:{second},third:{third}')
        return score

    def block_partition(self,partition_num):
        self._generate()
        self.parts = [Part(i) for i in range(partition_num)]
        block_num = len(self.blocks)
        print(block_num)
        block_num_per_part = max(int(self.assign_init_scale*block_num/partition_num),1)
        seed_block_part = np.random.choice(range(block_num),block_num_per_part * partition_num)
        # 随机分配初始块到区域内
        for i in range(partition_num):
            for j in range(block_num_per_part):
                self.block2part[seed_block_part[i*block_num_per_part + j]] = i
                self.parts[i].add_block(self.blocks[seed_block_part[i]])
        # 开始分配
        for i in range(block_num):
            if self.block2part[i] < 0:
                part = np.argmax(self._score(i,partition_num))
                self.block2part[i] = part
                self.parts[part].add_block(self.blocks[i])
                # print(f'block:{self.blocks[i].node_num}')
        membership = self.block2part[self.block_map]
        return membership


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hash')
    parser.add_argument("--dataset", type=str, default=None,
                        help="path to the dataset folder")
    parser.add_argument("--partition", type=int, default=2,
                        help="partition number") # 分布式情况下表示分布式node的个数，单机多卡情况下表示单机的gpu个数
    args = parser.parse_args()
    np.random.seed(2022)

    # load data
    # 加载全图topo（其中的点是从0开始编号的）
    adj = spsp.load_npz(os.path.join(args.dataset, 'adj.npz')) # <class 'scipy.sparse._coo.coo_matrix'>
    adj = adj.tocsr()
    print('load full graph adj, type:', type(adj))
    train_mask, val_mask, test_mask = data.get_masks(args.dataset)  # 加载mask
    train_nid = np.nonzero(train_mask)[0].astype(np.int64)  # 得到node id
    node_num = train_mask.shape[0]
    print(node_num)
    # [array([0, 2, 3, 4, 5, 7, 8]), array([1, 2, 3, 4, 5, 6, 9]), ..., array([1, 5])]每项表示每i个点的邻居点id
    # print(adj.row, len(adj.row), type(adj.row), np.max(adj.row))

    gengerater = BlockGenerator(adj,node_num,train_mask)
    membership = gengerater.block_partition(args.partition)
    print(membership)
    

    # 在每个partition中，按照in degree大小排序，然后写入{partition}metis文件夹中，每个文件是一个npy，存储降序的node id
    part_id = [[] for _ in range(args.partition)]
    for nid, pid in enumerate(membership):
        part_id[pid].append(nid)
    # print(part_id)
    for lst in part_id:
        lst.sort(key=lambda x: adj.getrow(x).indices.shape[0], reverse=True) # 每个part内的点按照出度从大到小排序
    # print(part_id)

    # save
    save_folder = os.path.join(args.dataset, f"dist_True/{args.partition}_metis")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        logging.info(f"mkdir {save_folder} done")
    for i, partnid in enumerate(part_id):
        # print(partnid)
        np.save(os.path.join(save_folder, f'{i}.npy'), partnid) # 每个part被分配到的graph node id
        logging.info(f"write partition nid {save_folder}/{i}.npy file  done")


# nodes_part_0 = np.argwhere(np.array(membership) == 0).ravel() # [3, 5, 6]
# nodes_part_1 = np.argwhere(np.array(membership) == 1).ravel() # [0, 1, 2, 4]
