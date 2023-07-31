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
from model import gcn, gat, graphsage
from model.deep import deepergcn
import logging
import time

from storage.storage_dist import DistCacheClient

import logging


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
    partition_name = 'metis' if 'papers' not in args.dataset else 'pagraph'
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

    # #################### 创建本地模拟的GNN模型####################
    # model = gcn.GCNSampling(args.featdim, args.hidden_size, args.n_classes, len(sampling), F.relu, args.dropout)
    if args.model_name == 'gcn':
        model = gcn.GCNSampling(args.featdim, args.hidden_size, args.n_classes, len(
            sampling), F.relu, args.dropout)
    elif args.model_name == 'graphsage':
        model = graphsage.GraphSageSampling(args.featdim, args.hidden_size, args.n_classes, len(
            sampling), F.relu, args.dropout)
    elif args.model_name == 'gat':
        model = gat.GATSampling(args.featdim, args.hidden_size, args.n_classes, len(
            sampling), F.relu, [2 for _ in range(len(sampling) + 1)] ,args.dropout, args.dropout)
    elif args.model_name == 'deepergcn':
        args.n_layers = len(sampling)
        args.in_feats = args.featdim
        args.gpu = 0
        model = deepergcn.DeeperGCN(args)
    n_model_param = sum([p.numel() for p in model.parameters()])
    print(f'n_model_param = {n_model_param}')

    #################### GNN训练 ####################
    batches_n_nodes = [] # 存放每个batch生成的子树的总点数
    n_remote_hit_nodes = [0 for _ in range(n_gnn_trainers)] # 存放在远程trainer中命中的点数
    for epoch in range(args.epoch):
        ########## 获取当前gpu在当前epoch分配到的training node id ##########
        np.random.seed(epoch)
        np.random.shuffle(fg_train_nid)
        train_lnid = fg_train_nid[args.rank * ntrain_per_gpu: (args.rank+1)*ntrain_per_gpu]
            
        ########## 根据分配到的Training node id, 构造图节点batch采样器 ###########
        sampler = dgl.contrib.sampling.NeighborSampler(fg, args.batch_size, expand_factor=int(sampling[0]), num_hops=len(sampling)+1, neighbor_type='in', shuffle=True, num_workers=args.num_worker, seed_nodes=train_lnid, prefetch=True, add_self_loop=True)

        ########## 利用每个batch的训练点采样生成的子树nf，进行GNN训练 ###########
        # model.train()
        iter = 0
        for nf in sampler:
            # 对于deepergcn提早结束
            if len(sampling) > 10 and iter>=3:
                break
            # 统计一个batch生成的一个nf中，点的总个数（每层已经做过去重）、这些点在每个trainer分图结果中命中的点数
            nf_nids = nf._node_mapping.tousertensor() # 一个batch对应的子树的所有graph node （层内点id去重）
            batches_n_nodes.append(torch.numel(nf_nids))
            # 统计命中在其他trainer上的点数
            belongs_pid = nid2pid_dict[nf_nids]
            unique_pid, counts = np.unique(belongs_pid, return_counts=True)
            unique_pid_counts_dict = dict(zip(unique_pid, counts))
            for pid, count in unique_pid_counts_dict.items():
                pid = int(pid) # np.float64->int
                if pid != args.rank:
                    n_remote_hit_nodes[pid] += count
            # # 查看子树每层的nid        
            # offsets = nf._layer_offsets
            # for i in range(nf.num_layers):
            #     layer_nid = nf_nids[offsets[i]:offsets[i+1]] # 子树的一层中的graph node
            iter += 1
            st = time.time()
        logging.info(f'=> cur_epoch {epoch} finished on rank {args.rank}')
        logging.info(f"{'=='*10} | rank={args.rank},epoch={epoch} 采样node信息输出 | {'=='*10}")
        logging.info(f'rank={args.rank}, number of training batches: {len(batches_n_nodes)}')
        logging.info(f'rank={args.rank}, the number of nodes for each tree spand by batch: {batches_n_nodes}, total nodes: {sum(batches_n_nodes)}')
        logging.info(f'rank={args.rank}, the number of nodes hits on other trainers: {n_remote_hit_nodes}')


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
                        choices=['graphsage', 'gcn', 'gat', 'deepergcn', 'demo'], help='GNN model name')
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
    
    # args for deepergcn
    parser.add_argument('--mlp_layers', type=int, default=1,
                            help='the number of layers of mlp in conv')
    parser.add_argument('--block', default='res+', type=str,
                            help='deepergcn layer: graph backbone block type {res+, res, dense, plain}')
    parser.add_argument('--conv', type=str, default='gen',
                            help='the type of deepergcn GCN layers')
    parser.add_argument('--gcn_aggr', type=str, default='max',
                            help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, softmax_sum, power, power_sum]')
    parser.add_argument('--norm', type=str, default='batch',
                            help='the type of normalization layer')
    parser.add_argument('--t', type=float, default=1.0,
                            help='the temperature of SoftMax')
    parser.add_argument('--p', type=float, default=1.0,
                            help='the power of PowerMean')
    parser.add_argument('--y', type=float, default=0.0,
                            help='the power of degrees')
    parser.add_argument('--learn_t', action='store_true')
    parser.add_argument('--learn_p', action='store_true')
    parser.add_argument('--learn_y', action='store_true')
    parser.add_argument('--msg_norm', action='store_true')
    parser.add_argument('--learn_msg_scale', action='store_true')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args_func(None)
    ngpus_per_node = args.ngpus_per_node
    model_name = args.model_name
    
    # 写日志
    log_dir = os.path.dirname(os.path.abspath(__file__))+'/logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    datasetname = args.dataset.strip('/').split('/')[-1]
    sampling_lst = args.sampling.split('-')
    if len(args.sampling.split('-')) > 10:
        fanout = sampling_lst[0]
        sampling_len = len(sampling_lst)
        log_filename = os.path.join(log_dir, f'default_{model_name}_{datasetname}_trainer{args.world_size}_bs{args.batch_size}_sl{fanout}x{sampling_len}_ep{args.epoch}.log')
    else:
        log_filename = os.path.join(log_dir, f'default_{model_name}_{datasetname}_trainer{args.world_size}_bs{args.batch_size}_sl{args.sampling}_ep{args.epoch}.log')

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
