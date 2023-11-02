import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import random
import torch.backends.cudnn as cudnn
import argparse
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import dgl
import numpy as np
import data

from dgl import DGLGraph
from utils.help import Print
import storage
from model import gcn, graphsage, gat, deep
import logging
import time

from storage.storage_local import LocalCacheClient

from common.log import setup_primary_logging, setup_worker_logging

import pickle
import threading
sem = threading.Semaphore(1000) #设置线程数限制防止崩溃 

def main(ngpus_per_node):
    #################### 固定随机种子，增强实验可复现性；参数正确性检查 ####################
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = False

    #################### 单进程计算 ####################
    # run(0,ngpus_per_node, args, log_queue)
    mp.spawn(run, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args, log_queue))


def get_balanced_parted_train_nids(parts_dir, ntrain_per_gpu, fg_train_nid, args):
    parts_train_set = []
    for cur_rank in range(args.distnodes):
        part_npy = os.path.join(parts_dir, f'{cur_rank}.npy')
        part_nid = np.load(part_npy)
        part_train_nid = np.intersect1d(part_nid, fg_train_nid)
        parts_train_set.append(part_train_nid)
    # balance number of training nodes of each parts graph
    ret = np.concatenate(parts_train_set)
    # delete remaining elements
    ret = ret[:ntrain_per_gpu*args.distnodes]
    ret = ret.reshape(args.distnodes, -1)
    return ret



def run(gpu, ngpus_per_node, args, log_queue):
    #################### 配置logger ####################
    # args.gpu = gpu  # 表示使用本地节点的gpu id
    setup_worker_logging(args.rank, log_queue)

    #################### 参数正确性检查，打印训练参数 ####################
    sampling = args.sampling.split('-')
    assert len(set(sampling)) == 1, 'Only Support the same number of neighbors for each layer'
    logging.info(f'Client Args: {args}')
    logging.info(f'Using 1 GPU in total to simulate distributed {args.distnodes} GPUs training...')
    
    #################### 读取全图中的训练点id、计算每个gpu需要训练的nid数量、根据全图topo构建dglgraph，用于后续sampling ####################
    fg_adj = data.get_struct(args.dataset)
    fg_labels = data.get_labels(args.dataset)
    fg_train_mask, fg_val_mask, fg_test_mask = data.get_masks(args.dataset)
    fg_train_nid = np.nonzero(fg_train_mask)[0].astype(np.int64) # numpy arryay of the whole graph's training node
    ntrain_per_gpu = int(fg_train_nid.shape[0] / args.distnodes) # # of training nodes per gpu
    print('fg_train_nid:',fg_train_nid.shape[0], 'ntrain_per_GPU:', ntrain_per_gpu)
    test_nid = np.nonzero(fg_test_mask)[0].astype(np.int64)
    fg_labels = torch.from_numpy(fg_labels).type(torch.LongTensor)  # in cpu
    # construct this partition graph for sampling
    # TODO: 图的topo之后也要分布式存储
    fg = DGLGraph(fg_adj, readonly=True)

    #################### 创建用于从分布式缓存中获取features数据的客户端对象 ####################
    cache_client = LocalCacheClient(args.gpu, args.log, args.distnodes, args.dataset)
    featdim = cache_client.feat_dim
    print(f'Got feature dim from server: {featdim}')

    #################### 创建分布式训练GNN模型、优化器 ####################
    print(f'dataset:', args.dataset)
    if 'ogbn_arxiv' in args.dataset:
        args.n_classes = 40
    elif 'ogbn_products' in args.dataset:
        args.n_classes = 47
    elif 'citeseer' in args.dataset:
        args.n_classes = 6
    elif 'pubmed' in args.dataset:
        args.n_classes = 3
    elif 'reddit' in args.dataset:
        args.n_classes = 41
    elif 'cora_full' in args.dataset:
        args.n_classes = 70
    else:
        raise Exception("ERRO: Unsupported dataset.")
    if args.model_name == 'gcn':
        model = gcn.GCNSampling(featdim, args.hidden_size, args.n_classes, len(
            sampling), F.relu, args.dropout)
    elif args.model_name == 'graphsage':
        model = graphsage.GraphSageSampling(featdim, args.hidden_size, args.n_classes, len(
            sampling), F.relu, args.dropout)
    elif args.model_name == 'gat':
        model = gat.GATSampling(featdim, args.hidden_size, args.n_classes, len(
            sampling), F.relu, [2 for _ in range(len(sampling) + 1)] ,args.dropout, args.dropout)
    elif args.model_name == 'deepergcn':
        args.n_layers = len(sampling)
        args.in_feats = featdim
        model = deep.DeeperGCN(args)
    model.cuda(args.gpu)

    # count number of model params
    print('Total number of model params:', sum([p.numel() for p in model.parameters()]))

    #################### GNN训练 ####################
    avg_amplify = 0
    for epoch in range(args.epoch):
        print(f'epoch = {epoch}')
        ########## 模拟每个 rank 的 sampling 并把结果用 pickle 保存到本地文件系统 ##########
        nf_list = [[] for _ in range(args.distnodes)]
        for cur_rank in range(args.distnodes):
            print(f'cur_rank = {cur_rank}')
            ##### 获取当前gpu在当前epoch分配到的training node id #####
            # global shuffling
            if cur_rank == 0:
                np.random.seed(epoch)
                np.random.shuffle(fg_train_nid)
            train_lnid = fg_train_nid[cur_rank * ntrain_per_gpu: (cur_rank+1)*ntrain_per_gpu]
            sampler = dgl.contrib.sampling.NeighborSampler(fg, args.batch_size, expand_factor=int(sampling[0]), num_hops=len(sampling)+1, neighbor_type='in', shuffle=False, num_workers=args.num_worker, seed_nodes=train_lnid, prefetch=True, add_self_loop=True) # shuffle=False because train_lnid has been shuffled
            for nfidx, nf in enumerate(sampler):
                nf_list[cur_rank].append(len(nf._node_mapping.tousertensor())) # 直接放入nf的大小
                num_iters = len(nf_list[0])
            print(f'num of iters per epoch per machine: {nf_list}')
        # 统计当前 epoch 的每个节点上的nf大小均值，和batch size相除得到扩增倍数
        avg_amplify += sum([sum([nflen/args.batch_size for nflen in lst]) for lst in nf_list])

    print(f'avg_amplify = {avg_amplify / args.epoch / args.distnodes / num_iters }')
        

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
                        choices=['deepergcn', 'gat', 'graphsage', 'gcn', 'demo'], help='GNN model name')
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
    parser.add_argument('--eval', action='store_true', help='whether to evaluate the GNN model')

    # distributed related
    parser.add_argument('--distnodes', default=2, type=int,
                        help='number of simulation nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--local', action='store_true')
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
    # Set multiprocessing type to spawn
    torch.multiprocessing.set_start_method("spawn", force=True)
    
    args = parse_args_func(None)
    model_name = args.model_name
    datasetname = args.dataset.strip('/').split('/')[-1]

    # 写日志
    log_dir = os.path.dirname(os.path.abspath(__file__))+'/logs/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    sampling_lst = args.sampling.split('-')
    if len(args.sampling.split('-')) > 10:
        fanout = sampling_lst[0]
        sampling_len = len(sampling_lst)
        log_filename = os.path.join(log_dir, f'default_{model_name}_{datasetname}_trainer{args.distnodes}_bs{args.batch_size}_sl{fanout}x{sampling_len}_local{args.local}.log')
    else:
        log_filename = os.path.join(log_dir, f'default_{model_name}_{datasetname}_trainer{args.distnodes}_bs{args.batch_size}_sl{args.sampling}_local{args.local}.log')
    if os.path.exists(log_filename):
        # if_delete = input(f'{log_filename} has exists, whether to delete? [y/n] ')
        if_delete = 'y'
        if if_delete=='y' or if_delete=='Y':
            os.remove(log_filename) # 删除已有日志，重新运行
        else:
            print('已经运行过，无需重跑，直接退出程序')
            sys.exit(-1) # 退出程序
    
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    ngpus_per_node = 1
    
    # logging for multiprocessing
    log_queue = setup_primary_logging(log_filename, "error.log")

    # main function
    main(ngpus_per_node)
