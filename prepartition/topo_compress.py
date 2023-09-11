import scipy.sparse as spsp
from scipy.sparse import csc_array, coo_array
import os
import numpy as np
import psutil
import copy
import sys

def get_memory_usage(prefix):
    print(f'prefix: {prefix}')
    # 获取内存使用情况
    memory = psutil.virtual_memory()
    # 已用内存
    used_memory = memory.used
    # 剩余内存
    available_memory = memory.available

    # 打印结果
    print(f"    已用内存：{used_memory / (1024 ** 2)} MB")
    print(f"    剩余内存：{available_memory / (1024 ** 2)} MB")

def get_del_mask(matrix,train_mask,num_hop=3,type='csc'):
    world_size = 4
    rank = 0
    iters = 10
    fg_train_nid = np.nonzero(train_mask)[0].astype(np.int64)
    ntrain_per_gpu = int(fg_train_nid.shape[0] / world_size)
    nodes = set()
    for epoch in range(5):
        np.random.seed(epoch)
        np.random.shuffle(fg_train_nid)
        for rank in range(0,4):
            if rank == 0:
                training_nodes = (fg_train_nid[0 * ntrain_per_gpu: (1)*ntrain_per_gpu])[0:2048*iters]
            else:
                training_node = (fg_train_nid[rank * ntrain_per_gpu: (rank+1)*ntrain_per_gpu])[0:2048*iters]
                training_nodes = np.concatenate((training_nodes,training_node))
        training_nodes = training_nodes.tolist()
        print(f'training node{epoch},{len(training_nodes)}:{np.array(training_nodes)}')
        nodes.update(training_nodes)    
    print(f'batch:{len(nodes)}')
    final_nodes = copy.deepcopy(nodes)
    get_memory_usage('start running ')
    for hop in range(num_hop):
        get_memory_usage(f'hop{hop} start ')
        new_nodes = set()
        for node in nodes:
            if type == 'csc':
                new_nodes.update(matrix.getcol(node).indices.tolist())
            elif type == 'csr':
                new_nodes.update(matrix.getrow(node).indices.tolist())
        nodes = new_nodes
        final_nodes = final_nodes | new_nodes
        print(f'hop{hop}:{len(final_nodes)}')
        get_memory_usage(f'hop{hop} end ')
    del nodes
    print(f'final:{len(final_nodes)}')
    print(f'tol:{len(train_mask)}')
    del_mask = np.full_like(train_mask,True,dtype=bool)
    for i in final_nodes:
        del_mask[i] = False
    del final_nodes
    np.save(del_path,del_mask)
    # del_mask = np.load(del_path)
    return del_mask


def compress(adj,del_mask,type='csc'):
    """
    高效思路:indptr做差分拿到数组diff,对diff做掩码将删掉的部分置0,然后做前缀和恢复成新的indptr
    indices可以直接利用原indptr和掩码进行选择
    """
    new_indptr = np.diff(adj.indptr)
    new_indptr[del_mask] = 0
    
    new_indptr = np.cumsum(new_indptr)
    new_indptr = np.insert(new_indptr,0,0)
    if type == 'csc':
        print('csc type')
        new_indices = adj[:,~del_mask].indices
        new_data = adj[:,~del_mask].data
        return spsp.csc_matrix((new_data, new_indices, new_indptr),shape = (del_mask.shape[0],del_mask.shape[0]))
    elif type == 'csr':
        print('csr type')
        new_indices = adj[~del_mask,:].indices
        new_data = adj[~del_mask,:].data
        return spsp.csr_matrix((new_data, new_indices, new_indptr),shape = (del_mask.shape[0],del_mask.shape[0]))

if __name__ == '__main__':
    '''
    num_hop参数为保存邻居的跳数,num_hop=1时保存一跳邻居
    由于训练程序中实际layer数为输入采样长度+2(构建sampler时+1,dgl内部+1)(如2-2将产生4层,即三跳邻居),本处num_hop数应为训练程序中采样长度+2
    adj_type csr对应neighbor_type的out csc对应in
    '''
    adj_type='csr'
    num_hop = 3
    dataset = '/home/qhy/gnn/repgnn/dist/repgnn_data/ogbn_papers100M0'
    
    adj_path = os.path.join(dataset,'adj_bak.npz')
    save_path = os.path.join(dataset,f'adj_compressed_{adj_type}.npz')
    mask_path = os.path.join(dataset,'train.npy')
    del_path = os.path.join(dataset,'del_mask.npy')
    # part_path = os.path.join(dataset,'dist_True/4_pagraph/0.npy')

    matrix = spsp.load_npz(adj_path)
    if adj_type == 'csc':
        matrix = matrix.tocsc()
    elif adj_type == 'csr':
        matrix = matrix.tocsr()
    train_mask = np.load(mask_path)
    # part_nodes = np.load(part_path)
    # part_mask = np.full_like(train_mask,False,dtype=bool)
    # part_mask[part_nodes] = True
    # train_mask = train_mask & part_mask
    # training_node = np.where(train_mask == True)[0].tolist()
    del_mask = get_del_mask(matrix,train_mask,num_hop,type=adj_type)
    
    get_memory_usage(f'del start')
    matrix = compress(matrix,del_mask,type=adj_type)
    get_memory_usage(f'del end')
    matrix = matrix.tocoo()
    spsp.save_npz(save_path, matrix)
    print('save end')




