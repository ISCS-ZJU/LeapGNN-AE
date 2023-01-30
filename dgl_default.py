from rpc_client import distcache_pb2_grpc
from rpc_client import distcache_pb2
import grpc
import random
import torch.backends.cudnn as cudnn
import argparse
import os
import sys
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import dgl
import numpy as np
import data

from dgl import DGLGraph
from utils.help import Print
import storage
from model import gcn, graphsage
import logging
import time

from storage.storage_dist import DistCacheClient

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO, filename=f"./dgl_default.txt", filemode='a+',
                    format='%(levelname)s %(asctime)s %(filename)s %(lineno)d : %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
# torch.set_printoptions(threshold=np.inf)




def main(ngpus_per_node):
    #################### 固定随机种子，增强实验可复现性；参数正确性检查 ####################
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = False
    assert args.world_size > 1, 'This version only support distributed GNN training with multiple nodes'
    args.distributed = args.world_size > 1 # using DDP for multi-node training

    #################### 为每个GPU卡产生一个训练进程 ####################
    if args.distributed:
        # total # of DDP training process
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(run, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # run(0, ngpus_per_node, args)
        sys.exit(-1)


def run(gpu, ngpus_per_node, args):
    #################### 参数正确性检查，打印训练参数 ####################
    sampling = args.sampling.split('-')
    assert len(set(sampling)) == 1, 'Only Support the same number of neighbors for each layer'
    if gpu == 0:
        logging.info(f'Client Args: {args}')
    
    #################### 构建GNN分布式训练环境 ####################
    args.gpu = gpu  # 表示使用本地节点的gpu id
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu  # 传入的rank表示节点个数
    dist_init_method = args.dist_url
    if torch.cuda.device_count() < 1:
        device = torch.device('cpu')
        torch.distributed.init_process_group(
            backend='gloo', init_method=dist_init_method, world_size=args.world_size, rank=args.rank)
        logging.info(f'Using CPU for training...')
    else:
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:' + str(args.rank))
        torch.distributed.init_process_group(
            backend='gloo', init_method=dist_init_method, world_size=args.world_size, rank=args.rank)
        logging.info(f'Using {args.world_size} distributed GPUs in total for training...')
    
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
    torch.distributed.barrier()

    #################### 创建用于从分布式缓存中获取features数据的客户端对象 ####################
    cache_client = DistCacheClient(args.grpc_port, args.gpu, args.log)
    featdim = cache_client.get_feat_dim()
    print(f'Got feature dim from server: {featdim}')

    #################### 创建分布式训练GNN模型、优化器 ####################
    print(f'dataset:', args.dataset)
    if 'ogbn_arxiv' in args.dataset:
        args.n_classes = 40
    elif 'ogbn_products0' in args.dataset:
        args.n_classes = 47
    else:
        raise Exception("ERRO: Unsupported dataset.")
    if args.model_name == 'gcn':
        model = gcn.GCNSampling(featdim, args.hidden_size, args.n_classes, len(
            sampling), F.relu, args.dropout)
    elif args.model_name == 'graphsage':
        model = graphsage.GraphSageSampling(featdim, args.hidden_size, args.n_classes, len(
            sampling), F.relu, args.dropout)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    #################### GNN训练 ####################
    with torch.autograd.profiler.profile(enabled=(args.gpu == 0), use_cuda=True) as prof:
        with torch.autograd.profiler.record_function('total epochs time'):
            for epoch in range(args.epoch):
                with torch.autograd.profiler.record_function('train data prepare'):
                    ########## 获取当前gpu在当前epoch分配到的training node id ##########
                    np.random.seed(epoch)
                    np.random.shuffle(fg_train_nid)
                    train_lnid = fg_train_nid[args.rank * ntrain_per_gpu: (args.rank+1)*ntrain_per_gpu]
                    
                ########## 根据分配到的Training node id, 构造图节点batch采样器 ###########
                sampler = dgl.contrib.sampling.NeighborSampler(fg, args.batch_size, expand_factor=int(sampling[0]), num_hops=len(sampling)+1, neighbor_type='in', shuffle=True, num_workers=args.num_worker, seed_nodes=train_lnid, prefetch=True, add_self_loop=True)

                ########## 利用每个batch的训练点采样生成的子树nf，进行GNN训练 ###########
                model.train()
                iter = 0
                wait_sampler = []
                st = time.time()
                # each_sub_iter_nsize = [] #  记录每次前传计算的 sub_batch的树的点树
                for nf in sampler:
                    wait_sampler.append(time.time()-st)
                    logging.debug(f'iter: {iter}')
                    with torch.autograd.profiler.record_function('fetch feat'):
                        # 将nf._node_frame中填充每层神经元的node Frame (一个frame是一个字典，存储feat)
                        cache_client.fetch_data(nf)
                    batch_nid = nf.layer_parent_nid(-1)
                    with torch.autograd.profiler.record_function('fetch label'):
                        labels = fg_labels[batch_nid].cuda(
                            args.gpu, non_blocking=True)
                    with torch.autograd.profiler.record_function('gpu-compute with optimizer.step'):
                        # each_sub_iter_nsize.append(nf._node_mapping.tousertensor().size(0))
                        with torch.autograd.profiler.record_function('DDP forward'):
                            pred = model(nf)
                        with torch.autograd.profiler.record_function('DDP calculate loss'):
                            loss = loss_fn(pred, labels)
                        with torch.autograd.profiler.record_function('DDP optimizer.zero()'):
                            optimizer.zero_grad()
                        with torch.autograd.profiler.record_function('DDP backward'):
                            loss.backward()
                        with torch.autograd.profiler.record_function('DDP optimizer.step()'):
                            optimizer.step()
                    iter += 1
                    st = time.time()
                logging.info(f'rank: {args.rank}, iter_num: {iter}')
                
                ########## 当一个epoch结束，打印从当前client发送到本地cache server的请求中，本地命中数/本地总请求数 ###########
                if cache_client.log:
                    miss_rate = cache_client.get_miss_rate()
                    print('Epoch miss rate for epoch {} on rank {}: {:.4f}'.format(
                        epoch, args.rank, miss_rate))
                    time_local, time_remote = cache_client.get_total_local_remote_feats_gather_time() 
                    print(f'Up to now, total_local_feats_gather_time = {time_local*0.001} s, total_remote_feats_gather_time = {time_remote*0.001} s')
                    # print(f'Sub_iter nsize mean, max, min: {int(sum(each_sub_iter_nsize) / len(each_sub_iter_nsize))}, {max(each_sub_iter_nsize)}, {min(each_sub_iter_nsize)}')
                print(f'=> cur_epoch {epoch} finished on rank {args.rank}')
    
    logging.info(prof.key_averages().table(sort_by='cuda_time_total'))
    logging.info(
        f'wait sampler total time: {sum(wait_sampler)}, total iters: {len(wait_sampler)}, avg iter time:{sum(wait_sampler)/len(wait_sampler)}')


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
    # distributed related
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--grpc-port', default="10.5.30.43:18110", type=str,
                        help='grpc port to connect with cache servers.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args_func(None)
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    logging.info(f"ngpus_per_node: {ngpus_per_node}")

    main(ngpus_per_node)
