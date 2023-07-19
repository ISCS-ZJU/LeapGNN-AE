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
from model_inter import gcn, graphsage, gat
import logging
import time

from storage.storage_dist import DistCacheClient

import logging

from dgl.frame import Frame, FrameRef
import random

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
    #################### 参数正确性检查，打印训练参数 ####################
    sampling = args.sampling.split('-')
    assert len(set(sampling)) == 1, 'Only Support the same number of neighbors for each layer'
    if gpu == 0:
        logging.info(f'Client Args: {args}')
    args.rank = gpu  # 模拟第n个gnn trainer

    #################### rank=0的进程负责metis切分图，并保存每个trainer分到的图数据id ####################
    if args.rank == 0:
        # 检查metis是否切分完全，没有的话执行切分
        part_results_path = f'{args.dataset}/dist_True/{n_gnn_trainers}_metis'
        if not os.path.exists(part_results_path):
            try:
                os.system(f'python3 prepartition/metis.py --partition {n_gnn_trainers} --dataset {args.dataset}')
            except Exception as e:
                logging.error(repr(e))
                sys.exit(-1)
        logging.info('metis分图已经完成')
    barrier.wait()

    #################### 各trainer加载分图结果 ####################
    part_nid_dict = {} # key: parid, value: graph nid
    n_total_graph_nodes = 0
    for pid in range(n_gnn_trainers):
        sorted_part_nid = data.get_partition_results(os.path.join(args.dataset,'dist_True'), "metis", n_gnn_trainers, pid)
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
    if args.model_name == 'gcn':
        model = gcn.GCNSampling(args.featdim, args.hidden_size, args.n_classes, len(
            sampling), F.relu, args.dropout)
    elif args.model_name == 'graphsage':
        model = graphsage.GraphSageSampling(args.featdim, args.hidden_size, args.n_classes, len(
            sampling), F.relu, args.dropout)
    elif args.model_name == 'gat':
        model = gat.GATSampling(args.featdim, args.hidden_size, args.n_classes, len(
            sampling), F.relu, [2 for _ in range(len(sampling) + 1)] ,args.dropout, args.dropout)
    # model = gcn.GCNSampling(args.featdim, args.hidden_size, args.n_classes, len(sampling), F.relu, args.dropout)
    n_model_param = sum([p.numel() for p in model.parameters()])

    #################### GNN训练 ####################
    batches_n_nodes = [] # 存放每个batch生成的子树的总点数
    n_remote_hit_nodes = [0 for _ in range(n_gnn_trainers)] # 存放在远程trainer中命中的点数
    block_trans_model_opti = [0 for _ in range(len(sampling)+1)] # 存放每block GNN的模型和优化器迁移的参数量
    block_trans_aggr = [0 for _ in range(len(sampling)+1)] # 存放每block GNN的中间数据aggr的移动参数量
    nf_trans_comb = [] # 存放每个nf执行完时GNN的中间数据comb的移动参数量
    nf_trans_actv = [] # 存放每个nf执行完时GNN的中间数据actv的移动参数量
    for epoch in range(args.epoch):
        ########## 获取当前gpu在当前epoch分配到的training node id ##########
        np.random.seed(epoch)
        np.random.shuffle(fg_train_nid)
        train_lnid = fg_train_nid[args.rank * ntrain_per_gpu: (args.rank+1)*ntrain_per_gpu]
            
        ########## 根据分配到的Training node id, 构造图节点batch采样器 ###########
        sampler = dgl.contrib.sampling.NeighborSampler(fg, args.batch_size, expand_factor=int(sampling[0]), num_hops=len(sampling)+1, neighbor_type='in', shuffle=True, num_workers=args.num_worker, seed_nodes=train_lnid, prefetch=True, add_self_loop=True)

        ########## 利用每个batch的训练点采样生成的子树nf，进行GNN训练 ###########
        model.train()
        iter = 0
        for nf in sampler:
            # 统计一个batch生成的一个nf中，点的总个数（每层已经做过去重）、这些点在每个trainer分图结果中命中的点数
            nf_nids = nf._node_mapping.tousertensor() # 一个batch对应的子树的所有graph node （层内点id去重）
            # print('nf_nids:', nf_nids)
            batches_n_nodes.append(torch.numel(nf_nids))
            # 统计命中在其他trainer上的点数
            belongs_pid = nid2pid_dict[nf_nids]
            unique_pid, counts = np.unique(belongs_pid, return_counts=True)
            unique_pid_counts_dict = dict(zip(unique_pid, counts))
            for pid, count in unique_pid_counts_dict.items():
                pid = int(pid) # np.float64->int
                if pid != args.rank:
                    n_remote_hit_nodes[pid] += count
            
            ########## 计算naive方式下数据迁移量 ##########
            # 查看子树每层的nid        
            offsets = nf._layer_offsets

            # 填充假数据，从而可以获取每个block执行完后的activation的shape
            activation_shape = []
            for i in range(nf.num_layers):
                layer_nid = nf_nids[offsets[i]:offsets[i+1]]
                nf._node_frames[i] = FrameRef(Frame({'features': torch.FloatTensor(layer_nid.size(0), args.featdim)}))
            # model(nf)

            block_num = nf.num_layers - 1
            # 获取每个block输入的维度用于后续计算aggr迁移量
            block_input_dim = [] # 只有第一个block输入是featdim，其他都是hiden_size
            for j in range(block_num):
                if j==0:
                    block_input_dim.append(args.featdim)
                else:
                    block_input_dim.append(args.hidden_size)
            # 获取每个nf前传时combine需要传输的数据量
            fwdoutput, nf_total_comb_size, nf_total_actv_size = model(nf)
            
            for i in range(block_num):
                block_input_nid = nf_nids[offsets[i]:offsets[i+1]] # 子树的一层中的graph node
                # print('block_input_nid:', block_input_nid)
                # 计算该block的nid跨越的机器数
                layer_belongs_pid = nid2pid_dict[block_input_nid]
                layer_unique_pid, layer_counts = np.unique(layer_belongs_pid, return_counts=True)
                layer_accross_machines = len(layer_unique_pid)
                # 计算完成该block GNN计算需要的模型参数和优化器的迁移数据量
                # logging.info(f'i = {i}, len(block_trans_model_opti)={len(block_trans_model_opti)}')
                block_trans_model_opti[i] += n_model_param * 3 * (layer_accross_machines - 1) # adam优化器的参数量是模型参数量2倍
                ## 计算完成该block GNN计算需要的中间数据的迁移数据量
                # 迁移节点顺序
                trans_machine_sequence = [(args.rank+i)%args.world_size for i in range(layer_accross_machines)]
                
                # 模拟该block计算时的迁移过程，计算aggr的迁移数量
                  # 1. 确定当前machine上gnids的父节点有几个，乘以feat.dim（由于父节点数量难以确定，取[当前block输入层的本机gnids数/fanout, 当前block输出gnids数]
                  # 2. 累加之前的aggr值，即得到当前m迁移到下一个m时要传输的aggr数据量(由于累加会涉及父节点合并的问题，但不确定有多少，因此通过不累加来估计结果)
                
                block_aggr_trans_ndata = 0
                for mid in trans_machine_sequence[:-1]:
                    cur_machine_nids = np.count_nonzero([layer_belongs_pid==mid])
                    rand_parent_nids = random.randint(cur_machine_nids//int(sampling[0]), len(nf_nids[offsets[i+1]:offsets[i+2]]))
                    block_aggr_trans_ndata += rand_parent_nids * block_input_dim[i]
                block_trans_aggr[i] += block_aggr_trans_ndata
            # 计算当前nf的combine迁移数量
            nf_trans_comb.append(nf_total_comb_size)

            # 确定激活中间结果的大小
            nf_trans_actv.append(nf_total_actv_size)

            iter += 1
            st = time.time()
        logging.info(f'=> cur_epoch {epoch} finished on rank {args.rank}')
        logging.info(f"{'=='*10} | rank={args.rank},epoch={epoch} 采样node信息输出 | {'=='*10}")
        logging.info(f'rank={args.rank}, number of training batches: {len(batches_n_nodes)}')
        logging.info(f'rank={args.rank}, the number of nodes for each tree spand by batch: {batches_n_nodes}, total nodes: {sum(batches_n_nodes)}')
        logging.info(f'rank={args.rank}, the number of nodes hits on other trainers: {n_remote_hit_nodes}')
        
    logging.info(f"{'=='*10} | rank={args.rank}, total_epoch={args.epoch} naive模型迁移方式的数据传输量 | {'=='*10}")
    logging.info(f'rank={args.rank}, each block model&opt size transfer: {block_trans_model_opti}, total model&opt size transfer: {sum(block_trans_model_opti)}')
    logging.info(f'rank={args.rank}, each block aggr size transfer: {block_trans_aggr}, total aggr size transfer: {sum(block_trans_aggr)}')
    logging.info(f'rank={args.rank}, each nf comb size transfer: {nf_trans_comb}, total comb size transfer: {sum(nf_trans_comb)}')
    logging.info(f'rank={args.rank}, each nf actv size transfer: {nf_trans_actv}, total actv size transfer: {sum(nf_trans_actv)}')
    logging.info(f'rank={args.rank}, total data transfer: {sum(block_trans_model_opti) + sum(block_trans_aggr) + sum(nf_trans_comb) + sum(nf_trans_actv)}')
    



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

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args_func(None)
    ngpus_per_node = args.ngpus_per_node
    modelname = args.model_name
    
    # 写日志
    log_dir = os.path.dirname(os.path.abspath(__file__))+'/logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    datasetname = args.dataset.strip('/').split('/')[-1]
    log_filename = os.path.join(log_dir, f'default{modelname}_{datasetname}_trainer{args.world_size}_bs{args.batch_size}_sl{args.sampling}_ep{args.epoch}.log')
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
