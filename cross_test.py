import argparse
import os, sys, time
import torch
import torch.multiprocessing as m
import torch.nn.functional as F
import torch.distributed as dist
import dgl
import numpy as np
import data
import random

from dgl import DGLGraph
from utils.help import Print
import storage
from model import gcn
from utils.ring_all_reduce_demo import allreduce
import multiprocessing

import warnings
warnings.filterwarnings("ignore")


import logging
# logging.basicConfig(level=logging.DEBUG) # 级别升序：DEBUG INFO WARNING ERROR CRITICAL；需要记录到文件则添加filename=path参数；
logging.basicConfig(level=logging.INFO, filename="./2023.1.12", filemode='a+', format='%(levelname)s %(asctime)s %(filename)s %(lineno)d : %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
# torch.set_printoptions(threshold=np.inf)


######################################################################
# Defining Training Procedure
# ---------------------------

def run(rank, devices_lst, hit_q, max_q, args):
    # print config parameters
    if rank == 0:
        logging.info(f'Client Args: {args}')
    total_epochs = args.epoch
    world_size = len(devices_lst)
    # Initialize distributed training context.
    dev_id = devices_lst[rank]
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12365')
    if torch.cuda.device_count() < 1:
        device = torch.device('cpu')
        torch.distributed.init_process_group(
            backend='gloo', init_method=dist_init_method, world_size=world_size, rank=rank)
    else:
        # torch.cuda.set_device(dev_id)
        device = torch.device('cuda:' + str(dev_id))
        torch.distributed.init_process_group(
            backend='gloo', init_method=dist_init_method, world_size=world_size, rank=rank)

    # connect to cpu graph server
    dataset_name = os.path.basename(args.dataset)
    cpu_g = dgl.contrib.graph_store.create_graph_from_store(
        dataset_name, "shared_mem", port=8004)

    # rank = 0 partition graph
    sampling = args.sampling.split('-')
    assert len(
        set(sampling)) == 1, 'Only Support the same number of neighbors for each layer'

    # get train nid (consider DDP) as corresponding labels
    fg_adj = data.get_struct(args.dataset)
    logging.debug(f'full graph:{fg_adj}')
    fg_labels = data.get_labels(args.dataset)
    fg_train_mask, fg_val_mask, fg_test_mask = data.get_masks(args.dataset)
    fg_train_nid = np.nonzero(fg_train_mask)[0].astype(np.int64)
    ntrain_per_node = int(fg_train_nid.shape[0] / world_size) - 1
    test_nid = np.nonzero(fg_test_mask)[0].astype(np.int64)
    
    fg_labels = torch.from_numpy(fg_labels).type(torch.LongTensor) # in cpu

    # metis切分图，然后根据切分的结果，每个GPU缓存对应各部分图的热点feat（有不知道缓存量大小，第一个iter结束的时候再缓存）
    # if rank == 0:
    #     st = time.time()
    #     os.system(
    #         f"python3 prepartition/metis.py --partition {world_size} --dataset {args.dataset}")
    #     logging.info(f'It takes {time.time()-st}s on metis algorithm.')
    # torch.distributed.barrier()
    # print(123)
    
    # 1. 每个gpu加载分图的结果，之后用于对train_lnid根据所在GPU进行切分
    max_train_nid = np.max(fg_train_nid)+1
    nid2pid = np.zeros(max_train_nid, dtype=np.int64)-1
    for pid in range(world_size):
        sorted_part_nid = data.get_partition_results(os.path.join(args.dataset,"dist_True"), "metis", world_size, pid)
        necessary_nid = sorted_part_nid[sorted_part_nid<max_train_nid]
        nid2pid[necessary_nid] = pid
    

    # construct this partition graph for sampling
    fg = DGLGraph(fg_adj, readonly=True)

    # 建立当前GPU的cache-第一个参数是cpu-full-graph（因为读取feat要从原图读）, 第二个参数用于形成bool数组，判断某个train_nid是否在缓存中
    # cacher = storage.NewJPGNNGraphCacheServer(cpu_g, fg_adj.shape[0], rank, world_size)
    # cacher.init_field(['features'])

    # build DDP model and helpers
    torch.manual_seed(2022)
    # model = gcn.GCNSampling(cacher.dims['features'], args.hidden_size, args.n_classes, len(
    #     sampling), F.relu, args.dropout)
    # loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # model.cuda(rank)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank]) # 每个sub_batch都会进行梯度同步（不过并不执行optimizer更新）
    # ctx = torch.device(rank)

    model_idx = rank
    
    # remove old ckpt before starting
    # model_ckpt_path = os.path.join(args.ckpt_path, f'model_{rank}.pt')
    # grad_ckpt_path = os.path.join(args.ckpt_path, f'model_{rank}_grad.pt')
    # if os.path.exists(model_ckpt_path):
    #     os.remove(model_ckpt_path)
    # torch.save(model.state_dict(), model_ckpt_path) # 保存最初始的模型参数
    # if os.path.exists(grad_ckpt_path):
    #     os.remove(grad_ckpt_path)
        
    def split_fn(a):
        return np.split(a, np.arange(args.batch_size, len(a), args.batch_size)) # 例如a=[0,9]，bs=3，那么切割的结果是[0,1,2], [3,4,5], [6,7,8], [9]

    # start training
    with torch.autograd.profiler.profile(enabled=(rank == 0), use_cuda=True) as prof:
        # with torch.autograd.profiler.record_function('total epochs time'):
        for epoch in range(args.epoch):
                # with torch.autograd.profiler.record_function('train data prepare'):
                    # 切分训练数据
            np.random.seed(epoch)
            np.random.shuffle(fg_train_nid)
            if rank==0:
                logging.info(f'=> Shuffled epoch training nid for epoch {epoch}: {fg_train_nid}')
            useful_fg_train_nid = fg_train_nid[:world_size*ntrain_per_node]
            useful_fg_train_nid = useful_fg_train_nid.reshape(world_size, ntrain_per_node) # 每行表示一个gpu要训练的epoch train nid
            logging.debug(f'rank: {rank} useful_fg_train_nid:{useful_fg_train_nid}')
            # 根据args.batch_size将每行切分为多个batch，然后转置，最终结果类似：array([[array([0, 1]), array([5, 6])], [array([2, 3]), array([7, 8])], [array([4]), array([9])]], dtype=object)；行数表示batch数量
            useful_fg_train_nid = np.apply_along_axis(split_fn, 1, useful_fg_train_nid,).T
            logging.debug(f'rank:{rank} useful_fg_train_nid.split.T:{useful_fg_train_nid}')
            
            # 遍历二维数组中每行的batch nid，收集其中属于当前GPU的等数量的sub-batch nid
            sub_batch_offsets = []
            sub_batch_nid = [] # k个连续的sub_batch nparray为一组，表示一次iteration中所有的worker中属于当前GPU的nid
            sub_batch_offsets.append(0)
            now_start = 0
            for row in useful_fg_train_nid:
                for batch in row:
                    cur_gpu_nid_mask = (nid2pid[batch]==rank)
                    #cur_gpu_nid_mask = np.random.random(len(nid2pid[batch])) < 0.5 #for test
                    sub_batch = batch[cur_gpu_nid_mask]
                    sub_batch_nid.extend(sub_batch) 
                    now_start += len(sub_batch)
                    sub_batch_offsets.append(now_start)# 即使是空也会占一个位置
                    if rank==0:
                        logging.debug(f'put sub_batch: {batch[cur_gpu_nid_mask]}')
            #if epoch == 0:
            sampler = dgl.contrib.sampling.NeighborSamplerWithDiffBatchSz(fg, sub_batch_offsets, expand_factor=int(sampling[0]), num_hops=len(sampling)+1, neighbor_type='in', shuffle=False, num_workers=1, seed_nodes=sub_batch_nid, add_self_loop=True)
            count = 0
            for nf in sampler:
                nf_nids = nf._node_mapping.tousertensor()
                offsets = nf._layer_offsets
                tol_hit = []
                hit_num = 0
                hit_max = 0
                for j in range(0,args.num_gpu):
                    tol_hit.append(0)
                for i in range(nf.num_layers):
                    tnid = nf_nids[offsets[i]:offsets[i+1]]
                    layer_hit = []
                    for j in range(0,args.num_gpu):
                        layer_hit.append(0)
                    for j in tnid:
                        layer_hit[nid2pid[j]] += 1
                    # layer_hit[rank] = -1
                    # logging.info(f'layer: rank: {rank} : {layer_hit}')
                    for j in range(0,args.num_gpu):
                        tol_hit[j] += layer_hit[j]
                    # dist.barrier()
                    # if rank == 0:
                    #     logging.info(f'')
                for i in range(0,args.num_gpu):
                    if i != rank:
                        hit_num += tol_hit[i]
                        if tol_hit[i] > hit_max:
                            hit_max = tol_hit[i]
                hit_q.put(hit_num)
                max_q.put(hit_max)
                for i in range(0,args.num_gpu):
                    if i == rank:
                        logging.info(f'tol: rank: {rank} : {tol_hit}')
                    dist.barrier()
                
                if rank == 0:
                    hit_num = 0
                    m = 0
                    for i in range(0,args.num_gpu):
                        hit_num += hit_q.get()
                        m = max_q.get()
                        if m > hit_max:
                            hit_max = m
                    # logging.info(f'ave:{hit_num/(args.num_gpu*(args.num_gpu-1))}  max:{hit_max}  max/ave:{hit_max/(hit_num/(args.num_gpu*(args.num_gpu-1)))}')
                print(f'=> cur_epoch {epoch} finished on rank {rank}')
                dist.barrier()
                # fetch_done.put(1)
                # nf_gen_proc.join() # 一个epoch结束
    # if rank == 0:
        # logging.info(prof.key_averages().table(sort_by='cuda_time_total'))


def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='GNN Training')
    parser.add_argument('-d', '--dataset', default="/data/cwj/repgnn/ogbn_arxiv128", type=str, help='training dataset name')
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
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args_func(None)
    mp = multiprocessing.get_context('spawn')
    hit_q = mp.Queue(args.num_gpu)
    max_q = mp.Queue(args.num_gpu)
    # run(0,list(range(args.num_gpu)), hit_q, max_q, args)
    m.spawn(run, args=(list(range(args.num_gpu)), hit_q, max_q, args), nprocs=args.num_gpu)
