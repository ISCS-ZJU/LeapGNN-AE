import sys, os
sys.path.append(f'{os.path.dirname(__file__)}/../../')
from rpc_client import distcache_pb2_grpc
from rpc_client import distcache_pb2
import grpc
import random
import torch.backends.cudnn as cudnn
import argparse
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import dgl
import numpy as np
import data

import multiprocessing as mpg

from dgl import DGLGraph
import storage
from model import gcn, graphsage, gat
import logging
import time

from storage.storage_dist import DistCacheClient

import logging

import scipy.sparse as sp
from scipy.sparse.linalg import norm as sparse_norm


def main(ngpus_per_node):
    #################### 固定随机种子，增强实验可复现性；参数正确性检查 ####################
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = False
    n_gnn_trainers = args.world_size*ngpus_per_node # total GPU trainers
    logging.info(f'Total number of trainers: {n_gnn_trainers}')
    assert n_gnn_trainers > 1, 'This version only support distributed GNN training with multiple nodes'
    #################### 产生n_gnn_trainers个进程，模拟分布式训练 ####################
    barrier = mpg.Barrier(n_gnn_trainers) # 用于多进程同步
    # mpg.spawn(run, nprocs=n_gnn_trainers, args=(barrier, n_gnn_trainers, args))
    for i in range(n_gnn_trainers):
        p = mpg.Process(target=run, args=(i, barrier, n_gnn_trainers, args))
        p.start()

def get_batches(train_ind, batch_size=1, shuffle=True):
    """
    Inputs:
        train_ind: np.array
    """
    nums = train_ind.shape[0]
    if shuffle:
        np.random.shuffle(train_ind)
    i = 0
    while i < nums:
        cur_ind = train_ind[i:i + batch_size]
        yield cur_ind
        i += batch_size

def _one_layer_sampling(adj, probs, v_indices, output_size, map_arr):
    # NOTE: FastGCN described in paper samples neighboors without reference
    # to the v_indices. But in its tensorflow implementation, it has used
    # the v_indice to filter out the disconnected nodes. So the same thing
    # has been done here.
    v_indices = map_arr[v_indices]
    support = adj[v_indices, :]
    neis = np.nonzero(np.sum(support, axis=0))[1]
    p1 = probs[neis]
    p1 = p1 / np.sum(p1)
    sampled = np.random.choice(np.array(np.arange(np.size(neis))),
                                output_size, True, p1)

    u_sampled = neis[sampled]
    support = support[:, u_sampled]
    sampled_p1 = p1[sampled]

    support = support.dot(sp.diags(1.0 / (sampled_p1 * output_size)))
    return u_sampled, support

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def nontuple_preprocess_adj(adj):
    adj_normalized = normalize_adj(sp.eye(adj.shape[0]) + adj)
    # adj_normalized = sp.eye(adj.shape[0]) + normalize_adj(adj)
    return adj_normalized.tocsr()

def run(gpu, barrier, n_gnn_trainers, args):
    if gpu != 0:
        return
    #################### 参数正确性检查，打印训练参数 ####################
    sampling = args.sampling.split('-')
    assert len(set(sampling)) == 1, 'Only Support the same number of neighbors for each layer'
    if gpu == 0:
        logging.info(f'Client Args: {args}')
    args.rank = gpu  # 模拟第n个gnn trainer

    #################### rank=0的进程负责metis切分图，并保存每个trainer分到的图数据id ####################
    partition_name = 'metis' if ('papers' not in args.dataset and 'it' not in args.dataset) else 'pagraph'
    if args.rank == 0:
        # 检查metis是否切分完全，没有的话执行切分
        part_results_path = f'{args.dataset}/dist_True/{n_gnn_trainers}_{partition_name}'
        if not os.path.exists(part_results_path):
            try:
                os.system(f'python3 prepartition/{partition_name}.py --partition {n_gnn_trainers} --dataset {args.dataset}')
            except Exception as e:
                logging.error(repr(e))
                sys.exit(-1)
        logging.info(f'{partition_name}分图已经完成')
    # barrier.wait()

    #################### 各trainer加载分图结果 ####################
    part_nid_dict = {} # key: parid, value: graph nid
    n_total_graph_nodes = 0
    for pid in range(n_gnn_trainers):
        sorted_part_nid = data.get_partition_results(os.path.join(args.dataset,'dist_True'), partition_name, n_gnn_trainers, pid)
        part_nid_dict[pid] = sorted_part_nid # ndarray of graph nodes' id for trainer=pid
        n_total_graph_nodes += sorted_part_nid.size
    # 建立graph node id 到 part id的映射
    nid2pid_dict = np.empty(n_total_graph_nodes)
    for pid, nids in part_nid_dict.items():
        for nid in nids:
            nid2pid_dict[nid] = pid
    # logging.info(type(part_nid_dict[0]), part_nid_dict[0].shape)
    
    #################### 读取全图中的训练点id、计算每个gpu需要训练的nid数量、根据全图topo构建dglgraph，用于后续sampling ####################
    fg_adj = data.get_struct(args.dataset)
    fg_labels = data.get_labels(args.dataset)
    fg_train_mask, fg_val_mask, fg_test_mask = data.get_masks(args.dataset)
    fg_train_nid = np.nonzero(fg_train_mask)[0].astype(np.int64) # numpy arryay of the whole graph's training node
    ntrain_per_gpu = int(fg_train_nid.shape[0] / args.world_size) # # of training nodes per gpu
    logging.info(f'fg_train_nid: {fg_train_nid.shape[0]} ntrain_per_GPU: {ntrain_per_gpu}')
    test_nid = np.nonzero(fg_test_mask)[0].astype(np.int64)
    fg_labels = torch.from_numpy(fg_labels).type(torch.LongTensor)  # in cpu
    # construct this partition graph for sampling
    # TODO: 图的topo之后也要分布式存储
    fg = DGLGraph(fg_adj, readonly=True)

    # 构造 adj, probs 用于 layerwise sampling
    adj_csr = fg_adj.tocsr()
    adj_train = adj_csr[fg_train_nid, :][:, fg_train_nid]
    # 由于 fg_train_nid 不连续，因此构建一个映射功能的 numpy
    max_value = np.max(fg_train_nid)
    map_arr = np.full(max_value + 1, -1, dtype=np.int64)
    for idx, num in enumerate(fg_train_nid):
        # 将原始数组元素作为新数组的索引
        map_arr[num] = idx
    norm_adj_train = nontuple_preprocess_adj(adj_train)
    col_norm = sparse_norm(norm_adj_train, axis=0)
    probs = col_norm / np.sum(col_norm)
    ls_lst = eval(args.ls_lst)


    # #################### 创建本地模拟的GNN模型####################
    if args.model_name == 'gcn':
        model = gcn.GCNSampling(args.featdim, args.hidden_size, args.n_classes, len(
            sampling), F.relu, args.dropout)
    elif args.model_name == 'graphsage':
        model = graphsage.GraphSageSampling(args.featdim, args.hidden_size, args.n_classes, len(
            sampling), F.relu, args.dropout)
    elif args.model_name == 'gat':
        model = gat.GATSampling(args.featdim, args.hidden_size, args.n_classes, len(
            sampling), F.relu, [2 for _ in range(len(sampling) + 1)] ,args.dropout, args.dropout)

    #################### GNN训练 ####################
    batches_n_nodes = [] # 存放每个batch生成的子树的总点数
    n_remote_hit_nodes = [0 for _ in range(n_gnn_trainers)] # 存放在远程trainer中命中的点数
    same_target_machine_per_nf = [] # 存放每个nf中，和target node相同machine的百分比
    for epoch in range(args.epoch):
        ########## 获取当前gpu在当前epoch分配到的training node id ##########
        np.random.seed(epoch)
        np.random.shuffle(fg_train_nid)
        train_lnid = fg_train_nid[args.rank * ntrain_per_gpu: (args.rank+1)*ntrain_per_gpu]
        iter = 0
        # 下面的代码修改于 FastGCN_pytorch 实现
        for batch_inds in get_batches(train_lnid, args.batch_size):
            nf_nids = np.array([], dtype=np.int64) # 记录每个 micrograph 包含的点的id
            cur_out_nodes = batch_inds
            # 按 args.ls_list 逐层进行采
            for layer_index in range(len(ls_lst)-2, -1, -1):
                cur_sampled, _ = _one_layer_sampling(norm_adj_train, probs, cur_out_nodes, ls_lst[layer_index], map_arr)
                cur_out_nodes = cur_sampled
                # print(cur_out_nodes, nf_nids, type(cur_out_nodes), type(nf_nids))
                nf_nids = np.concatenate((cur_out_nodes, nf_nids))
            logging.info(f'nf_nids: {nf_nids}')
            batches_n_nodes.append(torch.numel(torch.from_numpy(nf_nids)))
            logging.info(f'batches_n_nodes: {batches_n_nodes}')
            # 统计命中在其他trainer上的点数
            # print(nf_nids, nf_nids.dtype)
            belongs_pid = nid2pid_dict[nf_nids]
            logging.info(f'belongs_pid: {belongs_pid}')
            unique_pid, counts = np.unique(belongs_pid, return_counts=True)
            unique_pid_counts_dict = dict(zip(unique_pid, counts))
            logging.info(f'unique_pid_counts_dict: {unique_pid_counts_dict}')
            for pid, count in unique_pid_counts_dict.items():
                pid = int(pid) # np.float64->int
                if pid != args.rank:
                    n_remote_hit_nodes[pid] += count
            logging.info(f'n_remote_hit_nodes: {n_remote_hit_nodes}')
            # nf_nids的最后bs个就是target node，判断target node和nf_nids中剩余节点属于同一Machine的比例
            target_node_machine = belongs_pid[-args.batch_size:]
            remaining_nodes = belongs_pid[:-args.batch_size]
            same_machine_nodes = np.count_nonzero(remaining_nodes==target_node_machine)
            same_target_machine_per_nf.append(same_machine_nodes / len(nf_nids))
            logging.info(f'same_target_machine_per_nf: {same_target_machine_per_nf}')
            
            # # 查看子树每层的nid        
            # offsets = nf._layer_offsets
            # for i in range(nf.num_layers):
            #     layer_nid = nf_nids[offsets[i]:offsets[i+1]] # 子树的一层中的graph node
            iter += 1
            logging.info(f'iter = {iter}')
            st = time.time()
            if iter == 10:
                break
        
        logging.info(f'=> cur_epoch {epoch} finished on rank {args.rank}')
        logging.info(f"{'=='*10} | rank={args.rank},epoch={epoch} 采样node信息输出 | {'=='*10}")
        logging.info(f'rank={args.rank}, number of training batches: {len(batches_n_nodes)}')
        logging.info(f'rank={args.rank}, the number of nodes for each tree spand by batch: {batches_n_nodes}, total nodes: {sum(batches_n_nodes)}')
        logging.info(f'rank={args.rank}, the number of nodes hits on other trainers: {n_remote_hit_nodes}')
        logging.info(f"{'=='*10} | rank={args.rank},epoch={epoch} 每个nf中和target node相同machine的占比 | {'=='*10}")
        logging.info(f"rank={args.rank}, same machine percentage with target node: {same_target_machine_per_nf}")
        logging.info(f"rank={args.rank}, same machine percentage with target node per nf: {round(sum(same_target_machine_per_nf) / len(same_target_machine_per_nf), 2)}")


def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='GNN Training')
    parser.add_argument('-d', '--dataset', default="/data/cwj/pagraph/gendemo",
                        type=str, help='training dataset name')
    parser.add_argument('-s', '--sampling', default="2-2-2",
                        type=str, help='neighborhood sampling method parameters')
    parser.add_argument('-hd', '--hidden-size', default=256,
                        type=int, help='hidden dimension size')
    parser.add_argument('-ncls', '--n-classes', default=60,
                        type=int, help='number of classes')
    parser.add_argument('-bs', '--batch-size', default=2,
                        type=int, help='training batch size')
    parser.add_argument('-dr', '--dropout', default=0.2,
                        type=float, help='dropout in training')
    parser.add_argument('-lr', '--lr', default=3e-2,
                        type=float, help='learning rate')
    parser.add_argument('-wdy', '--weight-decay', default=0,
                        type=float, help='weight decay')
    parser.add_argument('-mn', '--model-name', default='graphsage', type=str,
                        choices=['graphsage', 'gcn', 'demo'], help='GNN model name')
    parser.add_argument('-ep', '--epoch', default=3,
                        type=int, help='total trianing epoch')
    parser.add_argument('-wkr', '--num-worker', default=1,
                        type=int, help='sampling worker')
    parser.add_argument('-cs', '--cache-size', default=0,
                        type=int, help='cache size in each gpu (GB)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--log', dest='log', action='store_true',
                    help='adding this flag means log hit rate information')                    
    # simulation related
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--ngpus-per-node', default=1, type=int,
                        help='number of GPUs on each training node')
    parser.add_argument('--featdim', default=128, type=int,
                        help='dimension of each feature in simulation')
    # layerwise sampling
    parser.add_argument('--ls-lst', default='[128, 128, 1]', type=str, 
                        help='layer sizes list for layerwise sampling')
    parser.add_argument('--layerwise-sampling', action='store_true',
                        help='using layerwise sampling')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args_func(None)
    assert args.batch_size == 1, '为了验证observation 1，batch size需要设置为1'
    ngpus_per_node = args.ngpus_per_node
    modelname = args.model_name
    
    # 写日志
    log_dir = os.path.dirname(os.path.abspath(__file__))+'/logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    datasetname = args.dataset.strip('/').split('/')[-1]
    
    # log_filename = os.path.join(log_dir, f'default{modelname}_{datasetname}_trainer{args.world_size}_bs{args.batch_size}_sl{args.sampling}_ep{args.epoch}.log')
    sampling_lst = args.sampling.split('-')
    if len(args.sampling.split('-')) > 10:
        fanout = sampling_lst[0]
        sampling_len = len(sampling_lst)
        log_filename = os.path.join(log_dir, f'default_{modelname}_{datasetname}_trainer{args.world_size}_bs{args.batch_size}_sl{fanout}x{sampling_len}_ep{args.epoch}.log')
    else:
        log_filename = os.path.join(log_dir, f'default_{modelname}_{datasetname}_trainer{args.world_size}_bs{args.batch_size}_sl{args.sampling}_ep{args.epoch}.log')
    
    if os.path.exists(log_filename):
        if_delete = input(f'{log_filename} has exists, whether to delete? [y/n] ')
        if if_delete=='y' or if_delete=='Y':
            os.remove(log_filename) # 删除已有日志，重新运行
        else:
            print('已经运行过，无需重跑，直接退出程序')
            sys.exit(-1) # 退出程序
    logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='a+', format='%(levelname)s %(asctime)s %(filename)s %(lineno)d : %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
    logging.info(f"ngpus_per_trainer: {ngpus_per_node}")

    # 开始执行
    main(ngpus_per_node)