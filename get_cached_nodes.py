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
from model import gcn, graphsage, deep
from model import gat
# from model.dgl_gat_nodeflow.model import gat_nodeflow as gat
import logging
import time

from storage.storage_dist import DistCacheClient

from common.log import setup_primary_logging, setup_worker_logging

# import GPUtil
# from threading import Thread
# import time

# class Monitor(Thread):
#     def __init__(self, delay):
#         super(Monitor, self).__init__()
#         self.stopped = False
#         self.delay = delay # Time between calls to GPUtil
#         self.load = []
#         self.start()

#     def run(self):
#         while not self.stopped:
#             # GPUtil.showUtilization()
#             gpu = GPUtil.getGPUs()
#             self.load.append(gpu[1].load*100)
#             time.sleep(self.delay)

#     def stop(self):
#         self.stopped = True

from dgl.frame import Frame, FrameRef
def fetch_data(gpuid,feat, nodeflow):
    feat_dim = feat.shape[1]
    dims = {'features':feat_dim}
    with torch.autograd.profiler.record_function('get nf_nids'):
        # 把sub-graph的lnid都加载到gpu,这里的node_mapping是从nf-level -> part-graph lnid
        nf_nids = nodeflow._node_mapping.tousertensor()
        offsets = nodeflow._layer_offsets
        logging.debug(f'fetch_data batch onid, layer_offset: {nf_nids}, {offsets}')
    logging.debug(f'nf.nlayers: {nodeflow.num_layers}')
    for i in range(nodeflow.num_layers):
        tnid = nf_nids[offsets[i]:offsets[i+1]]
        # create frame
        with torch.autograd.profiler.record_function('create frames'):
            # with torch.cuda.device(self.gpuid):
            #     frame = {name: torch.cuda.FloatTensor(tnid.size(0), self.dims[name])
            #              for name in self.dims}  # 分配存放返回当前Layer特征的空间，size是(#onid, feature-dim)
            frame = {name: torch.empty(tnid.size(0), dims[name]) for name in dims}
            tnid = tnid.tolist()
        with torch.autograd.profiler.record_function('fetch feat from cache server'):
        # # fetch features from cache server
        
        # (non-stream method)
            features = feat[tnid]
        with torch.autograd.profiler.record_function('convert byte features to float tensor'):
            for name in dims:
                frame[name].data = torch.frombuffer(features, dtype=torch.float32).reshape(len(tnid), feat_dim)
        
        with torch.autograd.profiler.record_function('move feats from CPU to GPU'):
            # move features from cpu memory to gpu memory
            for name in dims:
                frame[name].data = frame[name].data.cuda(gpuid)
        # attach features to nodeflow
        with torch.autograd.profiler.record_function('asign frame to nodeflow'):
            logging.debug(f'Final nodeflow._node_frames:{i}, frame["features"].size(): {frame["features"].size()}\n')
            nodeflow._node_frames[i] = FrameRef(Frame(frame))

def send_recv(val,new_val,rank,world_size):
    # new_val = torch.zeros_like(val)
    if rank % 2:
        torch.distributed.send(val, dst = (rank-1+world_size)%world_size)
        torch.distributed.recv(new_val)
    else:
        torch.distributed.recv(new_val)
        torch.distributed.send(val, dst = (rank-1+world_size)%world_size)
    return new_val
        

def get_comm_time():
    t = 7.805 / 5
    t = t / 176132528
    # num = 10000
    # val = torch.empty(num,dtype=torch.float32)
    # new_val = torch.empty_like(val)
    # st = time.time()
    # with torch.autograd.profiler.record_function('comm time'):
    #     ########## 模型参数在分布式GPU间进行传输 ###########
    #     new_val = send_recv(val,new_val,args.rank,args.world_size)
    logging.info(f'comm time: {t}')
    return t

def get_compute_time(nf,model,gpu,feat):
    fetch_data(gpu,feat,nf)
    enum = nf._block_offsets[-1]
    vnum = nf._layer_offsets[-1]
    dim = feat.shape[-1]
    st = time.time()
    with torch.autograd.profiler.record_function('compute once time'):
        pred = model(nf)
    dt = time.time()-st
    dt = dt / dim
    dt = dt / (vnum + enum)
    logging.info(f'compute time: {dt}, {feat.shape}, {vnum}')
    return dt
    
import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, item, priority):
        heapq.heappush(self.heap, (priority, item))

    def pop(self):
        _, item = heapq.heappop(self.heap)
        return item
    

def get_tru(u,t_comm,t_compute,dim,fg,vreps,i,hd):
    tru = 0
    sampler = dgl.contrib.sampling.NeighborSampler(fg, 1, expand_factor=int(1000000), num_hops=i, neighbor_type='in', shuffle=False, num_workers=4, seed_nodes=[u], prefetch=True, add_self_loop=True)
    for sub in sampler:
        nodes = sub._node_mapping.tousertensor()
        off = sub._layer_offsets
        for hop in range(1,i+1):
            nei = set(nodes[off[hop]:off[hop+1]]) - vreps[hop]
            tru = tru + t_compute * (len(nei) + sub._block_offsets[hop] - sub._block_offsets[hop - 1]) * hd
    return tru * 0.8, nodes , off

def get_cached_nodes(nf,t_comm,t_compute,dim,fg,hd):
    nf_nids = nf._node_mapping.tousertensor()
    offsets = nf._layer_offsets
    vreps = [set() for _ in range(nf.num_layers)]
    r = []
    for i in range(1,nf.num_layers+1):
        ri = set()
        heap = PriorityQueue()
        # src, dst, _ = nf.block_edges(i, remap_local=True)
        # col = nf.map_from_parent_nid(i,nf.map_to_parent_nid(col),remap_local=True)
        # row = nf.map_from_parent_nid(i,nf.map_to_parent_nid(row),remap_local=True)
        tnid = nf_nids[offsets[i]:offsets[i+1]]
        uid = nf_nids[offsets[i-1]:offsets[i]]
        logging.info(f'num: {len(tnid)}')
        for u in tnid:
            tru,_,_ = get_tru(u,t_comm,t_compute,dim,fg,vreps,i,hd)
            # nodes = set(u)
            # for j in range(i):
            #     src, dst, eid = nf.block_edges(i - j, remap_local=True)
            #     hops.append(set(src[dst == nodes]))
            #     tru = tru + t_compute * (len(hops[j] - vreps[i - j]) + eid.shpe[0])
            #     nodes = hops[j] - vreps[i - j]
            trc = t_comm * hd
            # logging.info(f'time: {trc}, {tru}')
            heap.push(u,tru)
            # src[dst == u]
        logging.info(f'heap: {len(heap.heap)}')
        while len(heap.heap) > 0:
            u = heap.pop()
            trc = t_comm * hd
            tru, nodes, off = get_tru(u,t_comm,t_compute,dim,fg,vreps,i)
            if tru < trc:
                ri.add(u)
                for hop in range(i):
                    vreps[hop] = vreps[hop] | nodes[off[hop]:off[hop+1]]
        r.append(ri)
        logging.info(f'ri: {ri}')
                    
        





from dgl.frame import Frame, FrameRef
def fetch_data(gpuid,feat, nodeflow):
    feat_dim = feat.shape[1]
    dims = {'features':feat_dim}
    with torch.autograd.profiler.record_function('get nf_nids'):
        # 把sub-graph的lnid都加载到gpu,这里的node_mapping是从nf-level -> part-graph lnid
        nf_nids = nodeflow._node_mapping.tousertensor()
        offsets = nodeflow._layer_offsets
        logging.debug(f'fetch_data batch onid, layer_offset: {nf_nids}, {offsets}')
    logging.debug(f'nf.nlayers: {nodeflow.num_layers}')
    for i in range(nodeflow.num_layers):
        tnid = nf_nids[offsets[i]:offsets[i+1]]
        # create frame
        with torch.autograd.profiler.record_function('create frames'):
            # with torch.cuda.device(self.gpuid):
            #     frame = {name: torch.cuda.FloatTensor(tnid.size(0), self.dims[name])
            #              for name in self.dims}  # 分配存放返回当前Layer特征的空间，size是(#onid, feature-dim)
            frame = {name: torch.empty(tnid.size(0), dims[name]) for name in dims}
            tnid = tnid.tolist()
        with torch.autograd.profiler.record_function('fetch feat from cache server'):
        # # fetch features from cache server
        
        # (non-stream method)
            features = feat[tnid]
        with torch.autograd.profiler.record_function('convert byte features to float tensor'):
            for name in dims:
                frame[name].data = torch.frombuffer(features, dtype=torch.float32).reshape(len(tnid), feat_dim)
        
        with torch.autograd.profiler.record_function('move feats from CPU to GPU'):
            # move features from cpu memory to gpu memory
            for name in dims:
                frame[name].data = frame[name].data.cuda(gpuid)
        # attach features to nodeflow
        with torch.autograd.profiler.record_function('asign frame to nodeflow'):
            logging.debug(f'Final nodeflow._node_frames:{i}, frame["features"].size(): {frame["features"].size()}\n')
            nodeflow._node_frames[i] = FrameRef(Frame(frame))

def main(ngpus_per_node):
    #################### 固定随机种子，增强实验可复现性；参数正确性检查 ####################
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = False
    # assert args.world_size > 1, 'This version only support distributed GNN training with multiple nodes'
    args.distributed = args.world_size > 1 # using DDP for multi-node training

    #################### 为每个GPU卡产生一个训练进程 ####################
    if args.distributed:
        # total # of DDP training process
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(run, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args, log_queue))
    else:
        run(0, ngpus_per_node, args)
        # sys.exit(-1)


def run(gpu, ngpus_per_node, args, log_queue):
    #################### 配置logger ####################
    args.gpu = gpu  # 表示使用本地节点的gpu id
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu  # 传入的rank表示节点个数
    setup_worker_logging(args.rank, log_queue)

    #################### 参数正确性检查，打印训练参数 ####################
    sampling = args.sampling.split('-')
    sampling_intlst = list(map(int, sampling))
    assert len(set(sampling)) == 1, 'Only Support the same number of neighbors for each layer'
    if gpu == 0:
        logging.info(f'Client Args: {args}')
    
    #################### 构建GNN分布式训练环境 ####################
    dist_init_method = args.dist_url
    if torch.cuda.device_count() < 1:
        device = torch.device('cpu')
        # torch.distributed.init_process_group(
        #     backend='gloo', init_method=dist_init_method, world_size=args.world_size, rank=args.rank)
        logging.info(f'Using CPU for training...')
    else:
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:' + str(args.rank))
        # torch.distributed.init_process_group(
        #     backend='gloo', init_method=dist_init_method, world_size=args.world_size, rank=args.rank)
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
    feat = np.load(os.path.join(args.dataset, 'feat.npy'))
    # construct this partition graph for sampling
    # TODO: 图的topo之后也要分布式存储
    fg = DGLGraph(fg_adj, readonly=True)
    # torch.distributed.barrier()

    #################### 创建用于从分布式缓存中获取features数据的客户端对象 ####################
    # cache_client = DistCacheClient(args.grpc_port, args.gpu, args.log)
    # cache_client.Reset()
    # cache_client.ConstructNid2Pid(args.dataset, args.world_size, 'metis', len(fg_train_mask))
    featdim = feat.shape[1]
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
    elif 'in' in args.dataset:
        args.n_classes = 60
    elif 'uk' in args.dataset:
        args.n_classes = 60
    else:
        raise Exception("ERRO: Unsupported dataset.")
    if args.model_name == 'gcn':
        model = gcn.GCNSampling(featdim, args.hidden_size, args.n_classes, len(
            sampling), F.relu, args.dropout)
    elif args.model_name == 'graphsage':
        model = graphsage.GraphSageSampling(featdim, args.hidden_size, args.n_classes, len(
            sampling), F.relu, args.dropout)
    elif args.model_name == 'gat':
        model = gat.GATSampling(featdim, args.hidden_size, args.n_classes, len(sampling), F.relu, [2 for _ in range(len(sampling) + 1)] ,args.dropout, args.dropout)
        # sampling_intlst = sampling_intlst + [sampling_intlst[-1]]
        # model = gat.GATNodeFlow(len(sampling_intlst)-1, featdim, [args.hidden_size], args.n_classes, sampling_intlst, args.dropout, args.dropout, True)
    elif args.model_name == 'deepergcn':
        args.n_layers = len(sampling)
        args.in_feats = featdim
        model = deep.DeeperGCN(args)
    elif args.model_name == 'film':
        model = deep.GNNFiLM(featdim, args.hidden_size, args.n_classes, len(sampling) + 1, args.dropout)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,eps=1e-5)
    model.cuda(args.gpu)
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model, device_ids=[args.gpu])
    max_acc = 0

    # count number of model params
    print('Total number of model params:', sum([p.numel() for p in model.parameters()]))
    # monitor = Monitor(0.1)

    #################### GNN训练 ####################
    with torch.autograd.profiler.profile(enabled=(args.gpu == 0), use_cuda=True) as prof:
        with torch.autograd.profiler.record_function('total epochs time'):
            for epoch in range(args.epoch):
                with torch.autograd.profiler.record_function('train data prepare'):
                    ########## 获取当前gpu在当前epoch分配到的training node id ##########
                    np.random.seed(epoch)
                    np.random.shuffle(fg_train_nid)
                    train_lnid = fg_train_nid
                    
                ########## 根据分配到的Training node id, 构造图节点batch采样器 ###########
                sampler = dgl.contrib.sampling.NeighborSampler(fg, args.batch_size, expand_factor=int(sampling[0]), num_hops=len(sampling)+1, neighbor_type='in', shuffle=False, num_workers=args.num_worker, seed_nodes=train_lnid, prefetch=True, add_self_loop=True)

                ########## 利用每个batch的训练点采样生成的子树nf，进行GNN训练 ###########
                model.train()
                iter = 0
                wait_sampler = []
                st = time.time()
                avg_loss = []
                # each_sub_iter_nsize = [] #  记录每次前传计算的 sub_batch的树的点树
                for nf in sampler:
                    wait_sampler.append(time.time()-st)
                    # print(f'iter: {iter}')
                    t_comm = get_comm_time()
                    t_compute = get_compute_time(nf,model,gpu,feat)
                    get_cached_nodes(nf,t_comm,t_compute,feat.shape[-1],fg,args.hidden_size)
                        # torch.distributed.barrier()
                    # logging.info(f'avg loss = {sum(avg_loss)/len(avg_loss)}')
                        
                    iter += 1
                    st = time.time()
                logging.info(f'rank: {args.rank}, iter_num: {iter}')
                
                ########## 当一个epoch结束，打印从当前client发送到本地cache server的请求中，本地命中数/本地总请求数 ###########
                # print(f'=> cur_epoch {epoch} finished on rank {args.rank}')
                logging.info(f'=> cur_epoch {epoch} finished on rank {args.rank}')

                if args.eval:
                    num_acc = 0  
                    for nf in dgl.contrib.sampling.NeighborSampler(fg,len(test_nid),
                                                                expand_factor=int(sampling[0]),
                                                                neighbor_type='in',
                                                                num_workers=args.num_worker,
                                                                num_hops=len(sampling)+1,
                                                                seed_nodes=test_nid,
                                                                prefetch=True,
                                                                add_self_loop=True):
                        model.eval()
                        with torch.no_grad():
                            fetch_data(gpu,feat,nf)
                            pred = model(nf)
                            batch_nids = nf.layer_parent_nid(-1)
                            batch_labels = fg_labels[batch_nids].cuda(args.gpu)
                            num_acc += (pred.argmax(dim=1) == batch_labels).sum().cpu().item()
                    max_acc = max(num_acc / len(test_nid),max_acc)
                    logging.info(f'Epoch: {epoch}, Test Accuracy {num_acc / len(test_nid)}')
    # monitor.stop()

    if args.eval:
        logging.info(f'Max acc:{max_acc}')
    if not args.eval:
        logging.info(prof.key_averages().table(sort_by='cuda_time_total'))
    logging.info(
        f'wait sampler total time: {sum(wait_sampler)}, total iters: {len(wait_sampler)}, avg iter time:{sum(wait_sampler)/len(wait_sampler)}')
    # logging.info(f'gpu util:{monitor.load}')
    # torch.distributed.barrier()

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
                        choices=['deepergcn', 'gat', 'graphsage', 'gcn', 'film', 'demo'], help='GNN model name')
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
        log_filename = os.path.join(log_dir, f'get_cache_{model_name}_{datasetname}_trainer{args.world_size}_bs{args.batch_size}_sl{fanout}x{sampling_len}_ep{args.epoch}_hd{args.hidden_size}_localFalse.log')
    else:
        log_filename = os.path.join(log_dir, f'get_cache_{model_name}_{datasetname}_trainer{args.world_size}_bs{args.batch_size}_sl{args.sampling}_ep{args.epoch}_hd{args.hidden_size}_localFalse.log')
    if os.path.exists(log_filename):
        # if_delete = input(f'{log_filename} has exists, whether to delete? [y/n] ')
        if_delete = 'y'
        if if_delete=='y' or if_delete=='Y':
            os.remove(log_filename) # 删除已有日志，重新运行
        else:
            print('已经运行过，无需重跑，直接退出程序')
            sys.exit(-1) # 退出程序
        # while os.path.exists(log_filename):
        #     base, extension = os.path.splitext(log_filename)
        #     log_filename = f"{base}_1{extension}"
        #     print(f"new log_filename: {log_filename}")
    
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    ngpus_per_node = 1
    
    # logging for multiprocessing
    log_queue = setup_primary_logging(log_filename, "error.log")

    # main function
    main(ngpus_per_node)