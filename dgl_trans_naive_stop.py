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
import math

from dgl import DGLGraph
from utils.help import Print
import storage
from model import gcn, graphsage, gat, deep
import logging
import time

from storage.storage_naive import NaiveCacheClient

from common.log import setup_primary_logging, setup_worker_logging
import psutil,gc,tracemalloc

trans_num={
'gat_in_2004_8000_16_sl10-10' : 2781317624,
'gat_in_2004_8000_32_sl10-10' : 3012517288,
'gat_ogbn_arxiv0_8000_16_sl10-10' : 76163248,
'gat_ogbn_arxiv0_8000_32_sl10-10' : 120938368,
'gat_ogbn_products0_8000_16_sl10-10' : 7665969576,
'gat_ogbn_products0_8000_32_sl10-10' : 11645865104,
'gat_uk_2007_8000_16_sl10-10' : 2581920280,
'gat_uk_2007_8000_32_sl10-10' : 2942595528,
'gcn_in_2004_8000_16_sl10-10' : 2386754004,
'gcn_in_2004_8000_32_sl10-10' : 2639711836,
'gcn_ogbn_arxiv0_8000_16_sl10-10' : 55583104,
'gcn_ogbn_arxiv0_8000_32_sl10-10' : 79518976,
'gcn_ogbn_products0_8000_16_sl10-10' : 5581949790,
'gcn_ogbn_products0_8000_32_sl10-10' : 7008621054,
'gcn_uk_2007_8000_16_sl10-10' : 2487498684,
'gcn_uk_2007_8000_32_sl10-10' : 2389154140,
'graphsage_in_2004_8000_16_sl10-10' : 2343359776,
'graphsage_in_2004_8000_32_sl10-10' : 2677272488,
'graphsage_ogbn_arxiv0_8000_16_sl10-10' : 53277152,
'graphsage_ogbn_arxiv0_8000_32_sl10-10' : 72044416,
'graphsage_ogbn_products0_8000_16_sl10-10' : 5486376784,
'graphsage_ogbn_products0_8000_32_sl10-10' : 7291382568,
'graphsage_uk_2007_8000_16_sl10-10' : 2344841736,
'graphsage_uk_2007_8000_32_sl10-10' : 2517359080,
'gat_it_1024_16_sl10-10' : 81464723584,
'gat_it_1024_32_sl10-10' : 96123999568,
'gcn_it_1024_16_sl10-10' : 73667161620,
'gcn_it_1024_32_sl10-10' : 80484347700,
'graphsage_it_1024_16_sl10-10' : 75380668752,
'gat_in_2004_8000_128_sl10-10' : 5414342256,
'gat_ogbn_arxiv0_8000_128_sl10-10' : 333347264,
'gat_ogbn_products0_8000_128_sl10-10' : 35028645104,
'gat_uk_2007_8000_128_sl10-10' : 5865055400,
'gcn_in_2004_8000_128_sl10-10' : 3850048644,
'gcn_ogbn_arxiv0_8000_128_sl10-10' : 176132528,
'gcn_ogbn_products0_8000_128_sl10-10' : 17329167144,
'gcn_uk_2007_8000_128_sl10-10' : 4249001012,
'graphsage_in_2004_8000_128_sl10-10' : 3919758984,
'graphsage_ogbn_arxiv0_8000_128_sl10-10' : 209975392,
'graphsage_ogbn_products0_8000_128_sl10-10' : 17875151976,
'graphsage_uk_2007_8000_128_sl10-10' : 4132324080,
'gcn_ogbn_arxiv0_80000000_128_sl10000000-10000000' : 146376332,
'gcn_ogbn_arxiv0_80000000_16_sl10000000-10000000' : 23033244,
'gcn_ogbn_arxiv0_80000000_128_sl10000000' : 52445192,
'gcn_ogbn_arxiv0_80000000_16_sl10000000' : 17593720,
'gcn_uk_2007_80000000_128_sl10000000-10000000' : 2103382628,
'gcn_uk_2007_80000000_16_sl10000000-10000000' : 901939612,
'gcn_uk_2007_80000000_128_sl10000000' : 821232232,
'gcn_uk_2007_80000000_16_sl10000000' : 365788728,
'gcn_in_2004_80000000_128_sl10000000-10000000' : 2121678188,
'gcn_in_2004_80000000_16_sl10000000-10000000' : 1259766692,
'gcn_in_2004_80000000_128_sl10000000' : 909969440,
'gcn_in_2004_80000000_16_sl10000000' : 531851848,
'deepergcn_in_2004_512_16_sl2-2-2-2-2-2' : 9841759948,
'deepergcn_ogbn_arxiv0_512_16_sl2-2-2-2-2-2-2-2-2' : 733387536,
'deepergcn_ogbn_arxiv0_512_16_sl2-2-2-2-2-2' : 233527192,
'deepergcn_ogbn_arxiv0_512_16_sl2-2-2-2-2' : 150369632,
'deepergcn_ogbn_products0_512_16_sl2-2-2-2-2-2' : 70849854084,
'deepergcn_uk_2007_512_16_sl2-2-2-2-2-2' : 12643117600,
'film_in_2004_512_16_sl2-2-2-2-2-2-2-2-2' : 30839134312,
'film_in_2004_512_16_sl2-2-2-2-2-2' : 10528277308,
'film_ogbn_arxiv0_512_16_sl2-2-2-2-2-2-2-2-2' : 820020560,
'film_ogbn_arxiv0_512_16_sl2-2-2-2-2-2' : 274081688,
'film_ogbn_products0_512_16_sl2-2-2-2-2-2-2-2-2' : 616100693564,
'film_ogbn_products0_512_16_sl2-2-2-2-2-2' : 68937897772,
'film_uk_2007_512_16_sl2-2-2-2-2-2-2-2-2' : 52381764272,
'film_uk_2007_512_16_sl2-2-2-2-2-2' : 13185378184,
}



def send_recv(val,new_val,rank,world_size):
    # new_val = torch.zeros_like(val)
    if rank % 2:
        torch.distributed.send(val, dst = (rank-1+world_size)%world_size)
        torch.distributed.recv(new_val)
    else:
        torch.distributed.recv(new_val)
        torch.distributed.send(val, dst = (rank-1+world_size)%world_size)
    return new_val


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
                 args=(ngpus_per_node, args, log_queue))
    else:
        # run(0, ngpus_per_node, args)
        sys.exit(-1)


def run(gpu, ngpus_per_node, args, log_queue):
    #################### 配置logger ####################
    tracemalloc.start()
    args.gpu = gpu  # 表示使用本地节点的gpu id
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu  # 传入的rank表示节点个数
    setup_worker_logging(args.rank, log_queue)

    #################### 参数正确性检查，打印训练参数 ####################
    sampling = args.sampling.split('-')
    assert len(set(sampling)) == 1, 'Only Support the same number of neighbors for each layer'
    if gpu == 0:
        logging.info(f'Client Args: {args}')
    
    #################### 构建GNN分布式训练环境 ####################
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
    cache_client = NaiveCacheClient(args.grpc_port, args.gpu, args.log, args.rank, args.dataset)
    cache_client.ConstructNid2Pid(args.dataset, args.world_size, 'metis', len(fg_train_mask))
    cache_client.Reset()
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
    elif 'ogbn_papers100M' in args.dataset:
        args.n_classes = 172
    elif 'twitter' in args.dataset:
        args.n_classes = 172
    elif 'uk_2007' in args.dataset:
        args.n_classes = 60
    elif 'in_2004' in args.dataset:
        args.n_classes = 60
    elif 'it' in args.dataset:
        args.n_classes = 60
    else:
        raise Exception("ERRO: Unsupported dataset.")
    max_acc = 0
    exp_iter_num = math.ceil(ntrain_per_gpu/args.batch_size)
    # count number of model params
    

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
                sampler = dgl.contrib.sampling.NeighborSampler(fg, args.batch_size, expand_factor=int(sampling[0]), num_hops=len(sampling)+1, neighbor_type='in', shuffle=False, num_workers=args.num_worker, seed_nodes=train_lnid, prefetch=True, add_self_loop=True)

                ########## 利用每个batch的训练点采样生成的子树nf，进行GNN训练 ###########
                iter = 0
                wait_sampler = []
                st = time.time()
                # each_sub_iter_nsize = [] #  记录每次前传计算的 sub_batch的树的点树
                # snapshot1 = tracemalloc.take_snapshot()
                for nf in sampler:
                    wait_sampler.append(time.time()-st)
                    # print(f'iter: {iter}')
                    with torch.autograd.profiler.record_function('fetch feat'):
                        # 将nf._node_frame中填充每层神经元的node Frame (一个frame是一个字典，存储feat)
                        cache_client.fetch_data(nf)
                    # logging.info(f'avg loss = {sum(avg_loss)/len(avg_loss)}')
                    num = 36
                    for _ in range(num):
                        with torch.autograd.profiler.record_function('init data'):
                            val = torch.empty(trans_num[f"{args.model_name}_{ args.dataset.strip('/').split('/')[-1]}_{args.batch_size}_{args.hidden_size}_sl{args.sampling}"]//exp_iter_num//num,dtype=torch.float32)
                            new_val = torch.empty_like(val)
                        with torch.autograd.profiler.record_function('model transfer'):
                            ########## 模型参数在分布式GPU间进行传输 ###########
                            new_val = send_recv(val,new_val,args.rank,args.world_size)
                    iter += 1
                    st = time.time()
                    if iter >= args.iter_stop:
                        break
                logging.info(f'rank: {args.rank}, iter_num: {iter}')
                
                ########## 当一个epoch结束，打印从当前client发送到本地cache server的请求中，本地命中数/本地总请求数 ###########
                if cache_client.log:
                    miss_num, try_num, miss_rate = cache_client.get_miss_rate()
                    logging.info(f'Epoch miss rate ( miss_num/try_num ) for epoch {epoch} on rank {args.rank}: {miss_num} / {try_num} = {miss_rate}')
                    time_local, time_remote = cache_client.get_total_local_remote_feats_gather_time() 
                    logging.info(f'Up to now, total_local_feats_gather_time = {time_local*0.001} s, total_remote_feats_gather_time = {time_remote*0.001} s')
                    # print(f'Sub_iter nsize mean, max, min: {int(sum(each_sub_iter_nsize) / len(each_sub_iter_nsize))}, {max(each_sub_iter_nsize)}, {min(each_sub_iter_nsize)}')
                # print(f'=> cur_epoch {epoch} finished on rank {args.rank}')
                logging.info(f'=> cur_epoch {epoch} finished on rank {args.rank}')


    if args.eval:
        logging.info(f'Max acc:{max_acc}')
    if not args.eval:
        logging.info(prof.key_averages().table(sort_by='cuda_time_total'))
    logging.info(
        f'wait sampler total time: {sum(wait_sampler)}, total iters: {len(wait_sampler)}, avg iter time:{sum(wait_sampler)/len(wait_sampler)}')
    torch.distributed.barrier()

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
                        choices=['deepergcn', 'gat', 'graphsage', 'gcn', 'demo', 'film'], help='GNN model name')
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
    parser.add_argument('--iter_stop', type=int, default=2, help='early stop to avoid oom')
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
    log_dir = os.path.dirname(os.path.abspath(__file__))+'/logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    sampling_lst = args.sampling.split('-')
    if len(args.sampling.split('-')) > 10:
        fanout = sampling_lst[0]
        sampling_len = len(sampling_lst)
        log_filename = os.path.join(log_dir, f'naive_{model_name}_{datasetname}_trainer{args.world_size}_bs{args.batch_size}_sl{fanout}x{sampling_len}_ep{args.epoch}_hd{args.hidden_size}.log')
    else:
        log_filename = os.path.join(log_dir, f'naive_{model_name}_{datasetname}_trainer{args.world_size}_bs{args.batch_size}_sl{args.sampling}_ep{args.epoch}_hd{args.hidden_size}.log')
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
