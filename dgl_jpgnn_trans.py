from rpc_client import distcache_pb2_grpc
from rpc_client import distcache_pb2
import grpc
import torch.backends.cudnn as cudnn
import random
import argparse
import os, sys, time
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.distributed as dist
import dgl
import numpy as np
import data

from dgl import DGLGraph
from utils.help import Print
import storage
from model import gcn
from utils.ring_all_reduce_demo import allreduce
from multiprocessing import Process, Queue
from storage.storage_dist import DistCacheClient

import warnings
warnings.filterwarnings("ignore")


import logging
# logging.basicConfig(level=logging.DEBUG) # 级别升序：DEBUG INFO WARNING ERROR CRITICAL；需要记录到文件则添加filename=path参数；
logging.basicConfig(level=logging.INFO, filename=f"./dgl_jpgnn_trans.txt", filemode='a+', 
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

def run(rank, ngpus_per_node, args):
    #################### 参数正确性检查，打印训练参数 ####################
    sampling = args.sampling.split('-')
    assert len(set(sampling)) == 1, 'Only Support the same number of neighbors for each layer'
    if rank == 0:
        logging.info(f'Client Args: {args}')
    
    #################### 构建GNN分布式训练环境 ####################
    args.gpu = rank  # 表示使用本地节点的gpu id
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + rank  # 传入的rank表示节点个数
    world_size = args.world_size
    dist_init_method = args.dist_url
    if torch.cuda.device_count() < 1:
        device = torch.device('cpu')
        torch.distributed.init_process_group(
            backend='gloo', init_method=dist_init_method, world_size=world_size, rank=args.rank)
        logging.info(f'Using CPU for training...')
    else:
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:' + str(args.gpu))
        torch.distributed.init_process_group(
            backend='gloo', init_method=dist_init_method, world_size=world_size, rank=args.rank)
        logging.info(f'Using {world_size} distributed GPUs in total for training...')
    
    #################### 读取全图中的训练点id、计算每个gpu需要训练的nid数量、根据全图topo构建dglgraph，用于后续sampling ####################
    fg_adj = data.get_struct(args.dataset)
    fg_labels = data.get_labels(args.dataset)
    fg_train_mask, fg_val_mask, fg_test_mask = data.get_masks(args.dataset)
    fg_train_nid = np.nonzero(fg_train_mask)[0].astype(np.int64) # numpy arryay of the whole graph's training node
    ntrain_per_gpu = int(fg_train_nid.shape[0] / world_size) # # of training nodes per gpu
    print('fg_train_nid:',fg_train_nid.shape[0], 'ntrain_per_GPU:', ntrain_per_gpu)
    test_nid = np.nonzero(fg_test_mask)[0].astype(np.int64)
    fg_labels = torch.from_numpy(fg_labels).type(torch.LongTensor) # in cpu
    # construct this partition graph for sampling
    # TODO: 图的topo之后也要分布式存储
    fg = DGLGraph(fg_adj, readonly=True)
    torch.distributed.barrier()

    #################### 创建用于从分布式缓存中获取features数据的客户端对象 ####################
    cache_client = DistCacheClient(args.grpc_port, args.gpu, args.log)
    featdim = cache_client.get_feat_dim()
    print(f'Got feature dim from server: {featdim}')

    #################### 创建分布式训练GNN模型、优化器 ####################
    model = gcn.GCNSampling(featdim, args.hidden_size, args.n_classes, len(
        sampling), F.relu, args.dropout)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    #################### 每个训练node id对应到part id ####################
    max_train_nid = np.max(fg_train_nid)+1
    nid2pid = np.zeros(max_train_nid, dtype=np.int64)-1
    for pid in range(world_size):
        sorted_part_nid = data.get_partition_results(os.path.join(args.dataset,'dist_True'), "metis", world_size, pid)
        necessary_nid = sorted_part_nid[sorted_part_nid<max_train_nid] # 只需要training node id即可
        nid2pid[necessary_nid] = pid

    model_idx = rank
        
    def split_fn(a):
        if len(a) <= args.batch_size:
            return np.split(a, np.array([len(a)])) # 最后会产生一个空的array, necessary
        else:
            return np.split(a, np.arange(args.batch_size, len(a), args.batch_size)) # 例如a=[0,1, ..., 9]，bs=3，那么切割的结果是[0,1,2], [3,4,5], [6,7,8], [9]
    
    #################### GNN训练 ####################
    with torch.autograd.profiler.profile(enabled=(rank == 0), use_cuda=True, with_stack=True) as prof:
        with torch.autograd.profiler.record_function('total epochs time'):
            for epoch in range(args.epoch):
                with torch.autograd.profiler.record_function('train data prepare'):
                    ########## 确定当前epoch每个gpu要训练的batch nid ###########
                    np.random.seed(epoch)
                    np.random.shuffle(fg_train_nid)
                    useful_fg_train_nid = fg_train_nid[:world_size*ntrain_per_gpu]
                    useful_fg_train_nid = useful_fg_train_nid.reshape(world_size, ntrain_per_gpu) # 每行表示一个gpu要训练的epoch train nid
                    logging.debug(f'rank: {rank} useful_fg_train_nid:{useful_fg_train_nid}')
                    # 根据args.batch_size将每行切分为多个batch，然后转置，最终结果类似：array([[array([0, 1]), array([5, 6])], [array([2, 3]), array([7, 8])], [array([4]), array([9])]], dtype=object)；行数表示batch数量
                    useful_fg_train_nid = np.apply_along_axis(split_fn, 1, useful_fg_train_nid,).T
                    logging.debug(f'rank:{rank} useful_fg_train_nid.split.T:{useful_fg_train_nid}')
                    
                    ########## 确定该gpu在当前epoch中将要训练的所有sub-batch的nid，放入sub_batch_nid中，同时构建sub_batch_offsets，以备NeighborSamplerWithDiffBatchSz中使用 ###########
                    sub_batch_nid = []
                    sub_batch_offsets = [0]
                    cur_offset = 0
                    for row in useful_fg_train_nid:
                        for batch in row:
                            cur_gpu_nid_mask = (nid2pid[batch]==rank)
                            sub_batch = batch[cur_gpu_nid_mask]
                            sub_batch_nid.extend(sub_batch)
                            cur_offset += len(sub_batch)
                            sub_batch_offsets.append(cur_offset)
                            if rank==0:
                                logging.debug(f'put sub_batch: {batch[cur_gpu_nid_mask]}')
                
                with torch.autograd.profiler.record_function('create sampler'):
                    ########## 根据分配到的sub_batch_nid和sub_batch_offsets，构造采样器 ###########
                    sampler = dgl.contrib.sampling.NeighborSamplerWithDiffBatchSz(fg, sub_batch_offsets, expand_factor=int(sampling[0]), num_hops=len(sampling)+1, neighbor_type='in', shuffle=False, num_workers=1, seed_nodes=sub_batch_nid, add_self_loop=True)
                
                ########## 利用每个sub_batch的训练点采样生成的子树nf，进行GNN训练 ###########
                model.train()
                n_sub_batches = len(sub_batch_nid)
                logging.info(f'n_sub_batches:{n_sub_batches}')
                
                sub_iter = 0
                for sub_nf in sampler:
                    print(f'sub_iter:', sub_iter, sub_nf==None)
                    with torch.autograd.profiler.record_function('model transfer'):
                            ########## 模型参数在分布式GPU间进行传输 ###########
                            send_recv(model,args.gpu,args.rank,world_size)
                    if sub_nf!=None:
                        with torch.autograd.profiler.record_function('fetch feat'):
                            ########## 跨结点获取sub_nf的feature数据 ###########
                            cache_client.fetch_data(sub_nf)

                        batch_nid = sub_nf.layer_parent_nid(-1)
                        with torch.autograd.profiler.record_function('fetch label'):
                            labels = fg_labels[batch_nid].cuda(rank, non_blocking=True)
                        with torch.autograd.profiler.record_function('gpu-compute'):
                            pred = model(sub_nf)
                            loss = loss_fn(pred, labels)
                            loss.backward()
                            # for x in model.named_parameters():
                            #     logging.info(x[1].grad.size())
                            logging.info(f'rank: {rank} sub_batch backward done.')
                    
                    with torch.autograd.profiler.record_function('sync for each sub_iter'):    
                        # 同步
                        dist.barrier()
                    # 如果已经完成了一个batch的数据并行训练
                    if (sub_iter+1) % world_size == 0:
                        with torch.autograd.profiler.record_function('gpu-compute'):
                            optimizer.step()
                        # 至此，一个iteration结束
                    sub_iter += 1
                if cache_client.log:
                    miss_rate = cache_client.get_miss_rate()
                    print('Epoch miss rate for epoch {} on rank {}: {:.4f}'.format(epoch, args.rank, miss_rate))
                print(f'=> cur_epoch {epoch} finished on rank {args.rank}')
      
    if args.rank == 0:
        logging.info(prof.export_chrome_trace('tmp.json'))
        logging.info(prof.key_averages().table(sort_by='cuda_time_total'))

def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='GNN Training')
    parser.add_argument('-d', '--dataset', default="/data/cwj/pagraph/gendemo", type=str, help='training dataset name')
    parser.add_argument('-ngpu', '--num-gpu', default=1,
                        type=int, help='# of gpus to train gnn with DDP')
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
    parser.add_argument('-ckpt', '--ckpt-path', default='/dev/shm', type=str, help='ckpt path for jpgnn')
    
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--log', dest='log', action='store_true',
                    help='adding this flag means log hit rate information')  

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
    # args = parse_args_func(None)
    # mp.spawn(run, args=(list(range(args.num_gpu)), args), nprocs=args.num_gpu)


def send_recv(model,gpu,rank,world_size):
    for val in model.parameters():
        val_cpu = val.cpu()
        new_val = torch.zeros_like(val_cpu)
        if rank % 2:
            torch.distributed.send(val_cpu, dst = (rank-1+world_size)%world_size)
            torch.distributed.recv(new_val)
        else:
            torch.distributed.recv(new_val)
            torch.distributed.send(val_cpu, dst = (rank-1+world_size)%world_size)
        with torch.no_grad():
            val[:] = new_val.cuda(gpu)

        # torch.distributed.barrier()
