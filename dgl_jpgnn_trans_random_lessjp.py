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
from model import gat, gcn, graphsage
from utils.ring_all_reduce_demo import allreduce
from multiprocessing import Process, Queue
from storage.storage_dist import DistCacheClient

import warnings
warnings.filterwarnings("ignore")


import logging
from common.log import setup_primary_logging, setup_worker_logging

import threading
sem = threading.Semaphore(1000) #设置线程数限制防止崩溃 

"""
lessjp 功能实现思路:
1. 前k个epoch 构造矩阵，n*n, 每个格子填写 miss_rate；
2. 用启发式方法调整每个 iteration 的计算矩阵图；保证每个iteration、每个模型的移动的次数都是相同的；
   每次迭代后，根据生成的计算矩阵图，构建模型移动路径链，按照这个链进行send_recv；
3. 一次迭代后，总计k个epoch 的运行时间，直到平均epoch运行时间小于等于目标时间或者移动次数为0，算法迭代停止；
"""

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

def reverse_columns(arr, k):
    """
    把第k列的数据轮转到第0列
    """
    num_cols = arr.shape[1]
    reversed_arr = np.concatenate((arr[:, k:num_cols], arr[:, 0:k]), axis=1)
    return reversed_arr

def substract_columns(arr):
    """
    根据 offsets arr 计算出每个 sub_batch 包含的点的数量
    """
    prev_cols = arr[ :, :-1]
    next_cols = arr[ :, 1:]
    return next_cols - prev_cols


def get_model_trace(jp_times, world_size, sub_batch_offsets):
    # 从各个节点 gather 每个 micro-batch 的样本数量
    sub_batches = [[None for _ in range(len(sub_batch_offsets))] for _ in range(world_size)] # world_size * len_of_sub_batches_of_each_rank
    # print('sub_batches', sub_batches)
    dist.all_gather_object(sub_batches, sub_batch_offsets)
    # convert into 2-d numpy array
    sub_batches = np.array(sub_batches)
    sub_batches_size = substract_columns(sub_batches)
    assert sub_batches_size.shape[1] % world_size == 0 , f"error: sub_batches_size is not correct, shape: {sub_batches_size.shape}"
    sub_batches_num = sub_batches_size.shape[1] // world_size
    # 构造在 lessjp 前的 model_trace
    model_trace = np.array([[(j-i+world_size) % world_size for i in range(world_size)]*sub_batches_num for j in range(world_size)]) # 因为模型左移，所以之前的 (i+j)%world_size 错误
    
    if jp_times > 0:
        for lesstimes in range(world_size - jp_times):
            # 每 group_size 个列里，去掉 1 列 （表示减少了跳的次数）
            group_size = world_size - lesstimes
            col_min = np.min(sub_batches_size, axis=0)
            reshaped_col_min = col_min.reshape(-1, group_size)
            min_indices = np.argmin(reshaped_col_min, axis=1).flatten()
            min_indices += np.arange(len(min_indices))*group_size # 加上其所在下标 *group_size
            # 找出每个列的最小值，从 model_trace 去掉最小值中最小的所在的列
            keep_cols = np.ones(model_trace.shape[1], dtype=bool) # 创建 bool 数组标记要保留的列
            keep_cols[min_indices] = False
            model_trace = model_trace[:, keep_cols]
            sub_batches_size = sub_batches_size[:, keep_cols] # 删除掉部分列
    else:
        # jp_times == 0 meas no jp
        model_trace = np.array([[j]*sub_batches_num for j in range(world_size)])
    return model_trace



def get_sub_batchs(epoch, fg_train_nid, world_size, ntrain_per_gpu, cache_client, nid2pid, gpuid, rank, split_fn, jp_times, miss_rate_lst):
    with torch.autograd.profiler.record_function('train data prepare'):
        np.random.seed(epoch)
        np.random.shuffle(fg_train_nid)
        useful_fg_train_nid = fg_train_nid[:world_size*ntrain_per_gpu]
        useful_fg_train_nid = useful_fg_train_nid.reshape(world_size, ntrain_per_gpu) # 每行表示一个gpu要训练的epoch train nid
        # logging.debug(f'rank: {rank} useful_fg_train_nid:{useful_fg_train_nid}')

        # 根据args.batch_size将每行切分为多个batch，然后转置，最终结果类似：array([[array([0, 1]), array([5, 6])], [array([2, 3]), array([7, 8])], [array([4]), array([9])]], dtype=object)；行数表示batch数量
        useful_fg_train_nid = np.apply_along_axis(split_fn, 1, useful_fg_train_nid,).T
        # logging.debug(f'rank:{rank} useful_fg_train_nid.split.T:{useful_fg_train_nid}')
        
        ########## 确定该gpu在当前epoch中将要训练的所有sub-batch的nid，放入sub_batch_nid中，同时构建sub_batch_offsets，以备NeighborSamplerWithDiffBatchSz中使用 ###########
        cache_partidx = cache_client.get_cache_partid()
        assert cache_partidx == rank, 'rank设置需要与partidx相同，否则影响命中率'
        
        sub_batch_nid = []
        sub_batch_offsets = [0]
        cur_offset = 0
        # 为确保每个 model 学习对应 mini-batch 的训练数据，需要根据交换列的顺序
        reversed_useful_fg_train_nid = reverse_columns(useful_fg_train_nid, rank)
        for row in reversed_useful_fg_train_nid:
            for batch in row:
                cur_gpu_nid_mask = (nid2pid[batch]==cache_partidx)
                sub_batch = batch[cur_gpu_nid_mask]
                sub_batch_nid.extend(sub_batch)
                cur_offset += len(sub_batch)
                sub_batch_offsets.append(cur_offset)
                if gpuid==0:
                    logging.debug(f'put sub_batch: {batch[cur_gpu_nid_mask]}')
        ##### 计算 jp_times 时的模型迁移路径 #####
        model_trace = get_model_trace(jp_times, world_size, sub_batch_offsets)
        # logging.debug(f'model_trace: {model_trace}')
        
        ##### 根据 model_trace, 当前rank 对每个 iteration 中的 sub_batches 重新划分 #####
        # 将 model_trace 转置， 找出每行中值为当前 rank 所在的列的 indices 列表，表示将移动到本节点的模型 id 顺序
        model_trace = model_trace.T
        find_col_indices = np.argwhere(model_trace == rank)[:, 1]
        # 每 jp_times 为一组，表示一个 iteration 中每个 sub_batch 来源的 batch id (针对没有反转过的 useful_fg_train_nid)
        # 据此，重新构造 sub_batch_nid 和 sub_batch_offsets
        new_sub_batch_nids = []
        new_sub_batch_offsets = [0]
        new_cur_offsets = 0
        for rowid, row in enumerate(useful_fg_train_nid): # one interation for one row
            # 原本在当前节点要训练的各个 sub_batch 
            total_nodes = 0
            ori_sub_batch = []
            for batch in row:
                cur_gpu_nid_mask = (nid2pid[batch]==cache_partidx)
                sub_batch = batch[cur_gpu_nid_mask]
                ori_sub_batch.append(sub_batch)
                total_nodes += len(sub_batch)
            ori_sub_batch = np.array(ori_sub_batch) # convert from list into numpy array
            # lessjp后被选中的 sub_batch 存储为 keep_sub_batches
            batchids = find_col_indices[rowid*jp_times:(rowid+1)*jp_times]
            keep_bool = np.zeros(world_size, dtype=bool)
            keep_bool[batchids] = True
            keep_sub_batches = ori_sub_batch[keep_bool]
            # logging.info(f'keep_sub_batches: {keep_sub_batches}')
            # 将未选中的 sub_batch 拆开分配到 keep_sub_batches
            unkeep_nids = []
            unkeep_batches = ori_sub_batch[~keep_bool]
            for unkeep_batch in unkeep_batches:
                unkeep_nids.extend(unkeep_batch)
            avg_len = total_nodes // len(batchids) + 1
            # logging.debug(f'avg_len: {avg_len}, unkeep_nids_len: {len(unkeep_nids)}')
            for sub_batch in keep_sub_batches:
                len_sub_batch = len(sub_batch)
                if len_sub_batch < avg_len and len(unkeep_nids) > 0:
                    sub_batch = np.concatenate((sub_batch, unkeep_nids[ : avg_len - len_sub_batch]))
                    unkeep_nids = unkeep_nids[avg_len - len_sub_batch:]
                # 添加到最终的结果中
                new_sub_batch_nids.extend(sub_batch)
                new_cur_offsets += len(sub_batch)
                new_sub_batch_offsets.append(new_cur_offsets)
            assert len(unkeep_nids)==0, 'unkeep_nids was not used up'
        # logging.debug(f'iteration {rowid} done, len nids {len(new_sub_batch_nids)}, unkeep_nids_len: {len(unkeep_nids)}')
        return new_sub_batch_nids, new_sub_batch_offsets, model_trace.T



def run(gpuid, ngpus_per_node, args, log_queue):
    #################### 配置logger ####################
    args.gpu = gpuid  # 表示使用本地节点的gpu id
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpuid  # 传入的rank表示节点个数
    setup_worker_logging(args.rank, log_queue)

    #################### 参数正确性检查，打印训练参数 ####################
    sampling = args.sampling.split('-')
    assert len(set(sampling)) == 1, 'Only Support the same number of neighbors for each layer'
    if gpuid == 0:
        logging.info(f'Client Args: {args}')
    
    #################### 构建GNN分布式训练环境 ####################
    args.gpu = gpuid  # 表示使用本地节点的gpu id
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpuid  # 传入的rank表示节点个数
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
    logging.info(f'fg_train_nid: {fg_train_nid.shape[0]}, ntrain_per_GPU: {ntrain_per_gpu}')
    test_nid = np.nonzero(fg_test_mask)[0].astype(np.int64)
    fg_labels = torch.from_numpy(fg_labels).type(torch.LongTensor) # in cpu
    # construct this partition graph for sampling
    # TODO: 图的topo之后也要分布式存储
    fg = DGLGraph(fg_adj, readonly=True)
    torch.distributed.barrier()

    #################### 创建用于从分布式缓存中获取features数据的客户端对象 ####################
    cache_client = DistCacheClient(args.grpc_port, args.gpu, args.log)
    cache_client.Reset()
    cache_client.ConstructNid2Pid(args.dataset, args.world_size, 'metis', len(fg_train_mask))
    featdim = cache_client.feat_dim
    print(f'Got feature dim from server: {featdim}')

    #################### 创建分布式训练GNN模型、优化器 ####################
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
        model = gat.GATSampling(featdim, args.hidden_size, args.n_classes, len(
            sampling), F.relu, [2 for _ in range(len(sampling) + 1)] ,args.dropout, args.dropout)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,eps=1e-5)
    model.cuda(gpuid)
    logging.info(f'create model')
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpuid])
    logging.info(f'ddp model set up')
    max_acc = 0

    #################### 每个训练node id对应到part id ####################
    max_train_nid = np.max(fg_train_nid)+1
    nid2pid = np.zeros(max_train_nid, dtype=np.int64)-1
    partition_name = 'metis' if 'papers' not in args.dataset else 'pagraph'
    for pid in range(world_size):
        sorted_part_nid = data.get_partition_results(os.path.join(args.dataset,'dist_True'), partition_name, world_size, pid)
        necessary_nid = sorted_part_nid[sorted_part_nid<max_train_nid] # 只需要training node id即可
        nid2pid[necessary_nid] = pid

    model_idx = args.rank
        
    def split_fn(a):
        if len(a) <= args.batch_size:
            return np.split(a, np.array([len(a)])) # 最后会产生一个空的array, necessary
        else:
            return np.split(a, np.arange(args.batch_size, len(a), args.batch_size)) # 例如a=[0,1, ..., 9]，bs=3，那么切割的结果是[0,1,2], [3,4,5], [6,7,8], [9]
    
    #################### GNN训练 ####################\
    # 选择client的fetch函数
    if args.deduplicate:
        fetch_func = cache_client.fetch_multiple_nfs_elimredun
        print('-> 将调用去冗余的 fetch_multiple_nfs_elimredun 函数')
        logging.info('-> 调用去冗余的 fetch_multiple_nfs_elimredun 函数')
    else:
        fetch_func = cache_client.fetch_multiple_nfs_v2
        print('-> 将调用不去冗余的 fetch_multiple_nfs_v2 函数')
        logging.info('-> 调用不去冗余的 fetch_multiple_nfs_v2 函数')

    """
    additional varibales for less jump
    """
    jp_times = args.world_size # 跳跃次数
    moniter_epoch_time = [] # moniter the avg. epoch time
    all_epoch_time = [] # moniter each epoch time for exp details
    adjust_jp_times = True # 是否继续缩小 jp_times
    miss_rate_lst = [None for _ in range(args.world_size)] # each rank's miss rate, (rank0, 1, 2, ...)
    machine2model = [i for i in range(args.world_size)] # 记录随着模型的迁移，每个机器(按顺序)上对应的 model_id, 初始状态下机器id=model_id
    
    last_avg_epoch_time = float('inf')
    with torch.autograd.profiler.profile(enabled=(gpuid == 0), use_cuda=True, with_stack=True) as prof:
        with torch.autograd.profiler.record_function('total epochs time'):
            epoch_st = time.time()
            for epoch in range(args.epoch):
                epoch_time_step = 0
                colide_table = None
                ########## 计算监视下的一个epoch的平均训练时间 ##########
                if len(moniter_epoch_time) >= args.moniter_epochs and adjust_jp_times==True:
                    avg_epoch_time = sum(moniter_epoch_time) / len(moniter_epoch_time)
                    # if avg_epoch_time < last_avg_epoch_time:
                    if jp_times > args.world_size - 1:
                        logging.info(f"avg_epoch_time of {moniter_epoch_time} is {avg_epoch_time} which is smaller than last avg epoch time {last_avg_epoch_time}")
                        jp_times -= 1
                        logging.info(f"new jp_times is {jp_times}")
                        if jp_times == 0:
                            jp_times += 1 # 并不算完全退化，保留第一次的跳跃权利
                            adjust_jp_times = False # 保持一跳的方式
                        moniter_epoch_time = []
                        last_avg_epoch_time = avg_epoch_time # update last_avg_epoch_time
                    else:
                        # adjust_jp_times = False
                        # jp_times += 1 # 恢复性能更好的jp_times
                        # jp_times 后续都保持3
                        # 生成当前 epoch 需要的time_step列表
                        colide_table = generate_collision_table_dict(4, len(sub_batch_offsets))
                    logging.info(f"cur_jp_times = {jp_times}")

                
                # # sync jp_times
                # jp_times_tensor = torch.tensor(jp_times)
                # dist.all_reduce(jp_times_tensor, op=torch.distributed.ReduceOp.SUM)
                # jp_times = jp_times_tensor.item() // args.world_size

                if jp_times > 0:
                    ########## 确定当前epoch每个gpu要训练的batch nid ###########
                    sub_batch_nid, sub_batch_offsets, model_trace = get_sub_batchs(epoch, fg_train_nid, world_size, ntrain_per_gpu, cache_client, nid2pid, gpuid, args.rank, split_fn, jp_times, miss_rate_lst)
                    logging.info(f'get_sub_batchs done')
                        
                    with torch.autograd.profiler.record_function('create sampler'):
                        ########## 根据分配到的sub_batch_nid和sub_batch_offsets，构造采样器 ###########
                        with sem:
                            # logging.debug(f'seed_nodes len: {len(sub_batch_nid), type(sub_batch_nid), sub_batch_nid[-10:]}, sub_batch_offsets len: {len(sub_batch_offsets), type(sub_batch_offsets), sub_batch_offsets[-10:]}')
                            sampler = dgl.contrib.sampling.NeighborSamplerWithDiffBatchSz(fg, sub_batch_offsets, expand_factor=int(sampling[0]), num_hops=len(sampling)+1, neighbor_type='in', shuffle=False, num_workers=args.num_worker, seed_nodes=sub_batch_nid, prefetch=True, add_self_loop=True)
                            # logging.debug(f'init sampler done')
                else:
                    # 退化为 default 模式
                    np.random.seed(epoch)
                    np.random.shuffle(fg_train_nid)
                    train_lnid = fg_train_nid[args.rank * ntrain_per_gpu: (args.rank+1)*ntrain_per_gpu]
                    sampler = dgl.contrib.sampling.NeighborSampler(fg, args.batch_size, expand_factor=int(sampling[0]), num_hops=len(sampling)+1, neighbor_type='in', shuffle=True, num_workers=args.num_worker, seed_nodes=train_lnid, prefetch=True, add_self_loop=True)

                
                ########## 利用每个sub_batch的训练点采样生成的子树nf，进行GNN训练 ###########
                model.train()
                n_sub_batches = len(sub_batch_offsets)-1 # trick: 当 jp_times == 0 时， n_sub_batches 与 jp_times == 1 时相同
                logging.info(f'n_sub_batches:{n_sub_batches}')
                
                sub_iter = 0
                wait_sampler = []
                

                sampler_iterator = iter(sampler)
                jp_cnt = 0
                for sub_nf_id in range(len(sub_batch_offsets)-1):
                    if jp_times > 0:
                        with torch.autograd.profiler.record_function('model transfer'):
                            ##### 在一个 sub_batch 训练开始前，迁移模型 #####
                            send_recv_model_trace_trace(model_trace, model, args.gpu, args.rank, jp_cnt, machine2model, world_size)
                        jp_cnt += 1
                        ########## 获取sub_nfs，跨结点获取sub_nfs的feature数据 ###########
                        st = time.time()
                        if sub_nf_id % jp_times == 0:
                            # 一次获取world_size个nf，进行预取
                            sub_nfs_lst = [] # 存放提前预取的包含features的nf
                            for j in range(jp_times):
                                try:
                                    sub_nfs_lst.append(next(sampler_iterator)) # 获取子图topo
                                except StopIteration:
                                    continue # 可能不足batch size个
                            wait_sampler.append(time.time() - st)
                            with torch.autograd.profiler.record_function('fetch feat'):
                                fetch_func(sub_nfs_lst) # 获取feats存入sub_nfs_lst列中中的对象属性    
                        # 选择其中一个sub_nf参与后续计算
                        sub_nf = sub_nfs_lst[sub_nf_id%jp_times]
                    else:
                        st = time.time()
                        sub_nf = next(sampler_iterator)
                        wait_sampler.append(time.time() - st)
                        fetch_func([sub_nf])

                    if sub_nf._node_mapping.tousertensor().shape[0] > 0:
                            st = time.time()
                            batch_nid = sub_nf.layer_parent_nid(-1)
                            with torch.autograd.profiler.record_function('fetch label'):
                                labels = fg_labels[batch_nid].cuda(gpuid, non_blocking=True)
                            with torch.autograd.profiler.record_function('gpu-compute'):
                                pred = model(sub_nf)
                                loss = loss_fn(pred, labels)
                            with torch.autograd.profiler.record_function('sync before compute'):    
                            # 同步
                                dist.barrier()
                            with torch.autograd.profiler.record_function('gpu-compute'):
                                loss.backward()
                            ed = time.time()
                    # two model compute
                    if colide_table is not None:
                        # logging.info("colide")
                        with torch.autograd.profiler.record_function('gpu-compute'):
                            if colide_table[args.rank][epoch_time_step]:
                                # simulate another model compute
                                time.sleep(ed-st)
                    
                    epoch_time_step += 1
                    
                    with torch.autograd.profiler.record_function('sync for each sub_iter'):    
                        # 同步
                        dist.barrier()
                    if jp_times > 0:
                        # 如果已经完成了一个batch的数据并行训练
                        if (sub_iter+1) % jp_times == 0:
                            with torch.autograd.profiler.record_function('gpu-compute'):
                                optimizer.step() # 至此，一个iteration结束
                                optimizer.zero_grad()
                    else:
                        with torch.autograd.profiler.record_function('gpu-compute'):
                                optimizer.step() # 至此，一个iteration结束
                                optimizer.zero_grad()
                        
                    sub_iter += 1
                    st = time.time()
                if cache_client.log:
                    miss_num, try_num, miss_rate = cache_client.get_miss_rate()
                    if epoch==0:
                        # all gather miss_rates
                        dist.all_gather_object(miss_rate_lst, miss_rate)

                    logging.info(f'Epoch miss rate ( miss_num/try_num ) for epoch {epoch} on rank {args.rank}: {miss_num} / {try_num} = {miss_rate}')
                    time_local, time_remote = cache_client.get_total_local_remote_feats_gather_time() 
                    logging.info(f'Up to now, total_local_feats_gather_time = {time_local*0.001} s, total_remote_feats_gather_time = {time_remote*0.001} s')
                print(f'=> cur_epoch {epoch} finished on rank {args.rank}')
                logging.info(f'=> cur_epoch {epoch} finished on rank {args.rank}')

                if args.eval:
                    num_acc = 0  
                    with sem:
                        sampler = dgl.contrib.sampling.NeighborSampler(fg,len(test_nid),
                                                                expand_factor=int(sampling[0]),
                                                                neighbor_type='in',
                                                                num_workers=args.num_worker,
                                                                num_hops=len(sampling)+1,
                                                                seed_nodes=test_nid,
                                                                prefetch=True,
                                                                add_self_loop=True)
                    for nf in sampler:
                        model.eval()
                        with torch.no_grad():
                            cache_client.fetch_data(nf)
                            pred = model(nf)
                            batch_nids = nf.layer_parent_nid(-1)
                            batch_labels = fg_labels[batch_nids].cuda(args.gpu)
                            num_acc += (pred.argmax(dim=1) == batch_labels).sum().cpu().item()
                    max_acc = max(num_acc / len(test_nid),max_acc)
                    logging.info(f'Epoch: {epoch}, Test Accuracy {num_acc / len(test_nid)}')

                moniter_epoch_time.append(time.time() - epoch_st)
                all_epoch_time.append(time.time() - epoch_st)
                epoch_st = time.time()
                

    
    # logging.info(prof.export_chrome_trace('tmp.json'))
    if args.eval:
        logging.info(f'Max acc:{max_acc}')
    logging.info(prof.key_averages().table(sort_by='cuda_time_total'))
    logging.info(
        f'wait sampler total time: {sum(wait_sampler)}, total sub_iters: {len(wait_sampler)}, avg sub_iter time:{sum(wait_sampler)/len(wait_sampler)}')
    logging.info(f'all_epoch_time:{all_epoch_time}')

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
                        choices=['gat', 'graphsage', 'gcn', 'demo'], help='GNN model name')
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
    parser.add_argument('--eval', action='store_true', help='whether to evaluate the GNN model')

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
    # --nodedup的时候，会把该参数设置为false，否则默认就是true
    parser.add_argument('--nodedup', dest='deduplicate', action='store_false', default=True)

    # less jumps
    parser.add_argument('--default-time', default='-1', type=float, help='one epoch time of default mode')
    parser.add_argument('--moniter-epochs', default=1, type=int, help='number of epochs to moniter each epoch training time')

    parser.add_argument('--iter_stop', type=int, default=2, help='early stop to avoid oom')
    parser.add_argument('--gputil', action='store_true', help='Enable GPU utilization monitoring')
    parser.add_argument('--util-interval', type=float, default=0.1, help='Time interval to call gputil (unit: second)')
    # args for deepergcn

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
        log_filename = os.path.join(log_dir, f'jpgnn_trans_random_lessjp_dedup_{args.deduplicate}_{model_name}_{datasetname}_trainer{args.world_size}_bs{args.batch_size}_sl{fanout}x{sampling_len}_ep{args.epoch}_hd{args.hidden_size}.log')
    else:
        log_filename = os.path.join(log_dir, f'jpgnn_trans_random_lessjp_dedup_{args.deduplicate}_{model_name}_{datasetname}_trainer{args.world_size}_bs{args.batch_size}_sl{args.sampling}_ep{args.epoch}_hd{args.hidden_size}.log')
    if os.path.exists(log_filename):
        # # if_delete = input(f'{log_filename} has exists, whether to delete? [y/n] ')
        # if_delete = 'y'
        # if if_delete=='y' or if_delete=='Y':
        #     os.remove(log_filename) # 删除已有日志，重新运行
        # else:
        #     print('已经运行过，无需重跑，直接退出程序')
        #     sys.exit(-1) # 退出程序
        while os.path.exists(log_filename):
            base, extension = os.path.splitext(log_filename)
            log_filename = f"{base}_1{extension}"
            print(f"new log_filename: {log_filename}")

    # if torch.cuda.is_available():
    #     ngpus_per_node = torch.cuda.device_count()
    # else:
    #     ngpus_per_node = 1
    ngpus_per_node = 1
    logging.info(f"ngpus_per_node: {ngpus_per_node}")

    # logging for multiprocessing
    log_queue = setup_primary_logging(log_filename, "error.log")

    main(ngpus_per_node)


def send_recv_model_trace_trace(model_trace, model, gpu, rank, jp_cnt, machine2model, world_size):
    # 根据当前机器上的 modelid 和 model_trace 确定应该哪个机器发送参数
    # 计算当前机器的模型下一步的目的机器id，如果等于当前机器id（即rank），则不用迁移
    cur_model_id = machine2model[rank]
    dst_machine_id = model_trace[cur_model_id][jp_cnt]
    # logging.debug(f"Start jump model the {jp_cnt}th times, cur_model_id={cur_model_id}, next_dst_machine_id={dst_machine_id}")

    # 确定哪些机器发送数据，例如 0->2 1->3 2->0 3->1 则原始的奇偶交错发送的方法就会导致死锁
    chains_lst = []
    checked = [False for _ in range(world_size)]
    send_ranks = [] # rank to send first
    for tmprank in range(world_size):
        tmp_dst = model_trace[machine2model[tmprank]][jp_cnt]
        chains_lst.append((tmprank, tmp_dst))
    chains_lst.sort()
    for s,d in chains_lst:
        if checked[s] == False:
            send_ranks.append(s)
            checked[s] = True
            checked[d] = True # 源节点和目的结点不能同时 send，会死锁
        else:
            if checked[d] == False:
                send_ranks.append(d)
                checked[d] = True
    # logging.debug(f'chains_lst: {chains_lst} send_ranks: {send_ranks}')

    if dst_machine_id != rank:
        send_first = (rank in send_ranks)
        for val in model.parameters():
            val_cpu = val.cpu()
            new_val = torch.zeros_like(val_cpu)
            if send_first:
                torch.distributed.send(val_cpu, dst = dst_machine_id)
                torch.distributed.recv(new_val)
            else:
                torch.distributed.recv(new_val)
                torch.distributed.send(val_cpu, dst = dst_machine_id)
            with torch.no_grad():
                val[:] = new_val.cuda(gpu)
    # logging.debug(f"End jump model")
    # update machine2model
    for rowid in range(world_size):
        machineid = model_trace[rowid][jp_cnt]
        machine2model[machineid] = rowid
    # logging.debug(f'cur machine2modelid: {machine2model}')


def generate_collision_table_dict(k, times):
    # 生成 random 删除后的碰撞表格; k=num_parallel_model or =world_size;
    # times: 重复执行几次
    # return: 每个模型在每个time step是否有colide (不会重复计算，以前一个模型为准)
    ret = None
    # initial table
    table = np.zeros((k,k))
    table[0] = np.arange(k)
    for i in range(1, k):
        table[i] = np.roll(table[i-1], -1)
    # print('init table:')
    print(table)
    logging.info(f'table:{table}')
    # random delete
    delnum = 1
    # for delnum in range(1, k):
    for _ in range(times):
        new_table = np.zeros((k, k-delnum))
        for row in range(table.shape[0]):
            delete_col_idx = np.random.randint(0, len(table[row]))
            new_table[row] = np.delete(table[row], delete_col_idx)
        # table = new_table
        # print(f'delnum = {delnum}, table:')
        # print(new_table)
        # 对于每个机器来说，计算发生碰撞的iter id
        colide_table = calculate_colide(new_table)
        # print(colide_table)
        if ret is None:
            ret = colide_table
        else:
            ret = np.concatenate((ret, colide_table), axis=1)
    return ret

def calculate_colide(ar):
    row, col = ar.shape[0], ar.shape[1]
    ret_ar = np.zeros_like(ar)
    for r in range(row):
        for c in range(col):
            for tmpr in range(r+1, row):
                ret_ar[r][c] += (ar[tmpr][c] == ar[r][c])
    return ret_ar
            
