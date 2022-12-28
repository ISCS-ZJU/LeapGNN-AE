import sys, os
sys.path.append(f'{os.path.dirname(__file__)}/../')
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

from dgl import DGLGraph
from utils.help import Print
import storage
from model import gcn
import logging
import time

from storage.storage_dist import DistCacheClient

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO, filename="./dgl_cpu_dist_1114_bs8000.txt", filemode='a+',
                    format='%(levelname)s %(asctime)s %(filename)s %(lineno)d : %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
# torch.set_printoptions(threshold=np.inf)




def main(ngpus_per_node):
    #################### 固定随机种子，增强实验可复现性；参数正确性检查 ####################
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = False
    n_gnn_trainers = args.world_size*ngpus_per_node # total GPU trainers
    assert n_gnn_trainers > 1, 'This version only support distributed GNN training with multiple nodes'
    #################### 产生n_gnn_trainers个进程，模拟分布式训练 ####################
    barrier = mp.Barrier(n_gnn_trainers) # 用于多进程同步
    mp.spawn(run, nprocs=n_gnn_trainers, args=(barrier, n_gnn_trainers, args))



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
                print(repr(e))
                sys.exit(-1)
        print('metis分图已经完成')
    barrier.wait()

    #################### 各trainer加载分图结果 ####################
    part_nid_dict = {}
    for pid in range(n_gnn_trainers):
        sorted_part_nid = data.get_partition_results(os.path.join(args.dataset,'dist_True'), "metis", n_gnn_trainers, pid)
        part_nid_dict[pid] = sorted_part_nid # ndarray of graph nodes' id for trainer=pid
    print(type(part_nid_dict), part_nid_dict.shape)
    
    #################### 读取全图中的训练点id、计算每个gpu需要训练的nid数量、根据全图topo构建dglgraph，用于后续sampling ####################
    fg_adj = data.get_struct(args.dataset)
    fg_labels = data.get_labels(args.dataset)
    fg_train_mask, fg_val_mask, fg_test_mask = data.get_masks(args.dataset)
    fg_train_nid = np.nonzero(fg_train_mask)[0].astype(np.int64) # numpy arryay of the whole graph's training node
    ntrain_per_gpu = int(fg_train_nid.shape[0] / args.world_size) # # of training nodes per gpu
    print('fg_train_nid:',fg_train_nid.shape[0], 'ntrain_per_GPU:', ntrain_per_gpu)
    test_nid = np.nonzero(fg_test_mask)[0].astype(np.int64)
    fg_labels = torch.from_numpy(fg_labels).type(torch.LongTensor)  # in cpu
    # construct this partition graph for sampling
    # TODO: 图的topo之后也要分布式存储
    fg = DGLGraph(fg_adj, readonly=True)

    #################### 创建本地模拟的GNN模型####################
    model = gcn.GCNSampling(args.featdim, args.hidden_size, args.n_classes, len(sampling), F.relu, args.dropout)

    #################### GNN训练 ####################
    batches_n_nodes = [] # 存放每个batch生成的子树的总点数
    n_remote_hit_nodes = [[] for _ in range(n_gnn_trainers)] # 存放在远程trainer中命中的点数
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
            logging.debug(f'iter: {iter}')
            # 统计一个batch生成的一个nf中，点的总个数（每层已经做过去重）、这些点在每个trainer分图结果中命中的点数
            nf_nids = nf._node_mapping.tousertensor()
            offsets = nf._layer_offsets
            for i in range(nf.num_layers):
                layer_nid = nf_nids[offsets[i]:offsets[i+1]]
                print(args.rank, layer_nid)
                sys.exit(-1)
            batch_nid = nf.layer_parent_nid(-1)
            iter += 1
            st = time.time()
        logging.info(f'rank: {args.rank}, iter_num: {iter}')
        print(f'=> cur_epoch {epoch} finished on rank {args.rank}')


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
    logging.info(f"ngpus_per_node: {ngpus_per_node}")

    main(ngpus_per_node)
