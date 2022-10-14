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

import warnings
warnings.filterwarnings("ignore")


import logging
# logging.basicConfig(level=logging.DEBUG) # 级别升序：DEBUG INFO WARNING ERROR CRITICAL；需要记录到文件则添加filename=path参数；
logging.basicConfig(level=logging.INFO, filename="./jpgnn_cpu_degree.txt", filemode='a+', format='%(levelname)s %(asctime)s %(filename)s %(lineno)d : %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
# torch.set_printoptions(threshold=np.inf)


######################################################################
# Defining Training Procedure
# ---------------------------

def run(rank, devices_lst, args):
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
        torch.cuda.set_device(dev_id)
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
    if rank == 0:
        st = time.time()
        os.system(
            f"python3 prepartition/metis.py --partition {world_size} --dataset {args.dataset}")
        logging.info(f'It takes {time.time()-st}s on metis algorithm.')
    torch.distributed.barrier()
    
    # 1. 每个gpu加载分图的结果，之后用于对train_lnid根据所在GPU进行切分
    max_train_nid = np.max(fg_train_nid)+1
    nid2pid = np.zeros(max_train_nid, dtype=np.int64)-1
    for pid in range(world_size):
        sorted_part_nid = data.get_partition_results(args.dataset, "metis", world_size, pid)
        necessary_nid = sorted_part_nid[sorted_part_nid<max_train_nid]
        nid2pid[necessary_nid] = pid
    

    # construct this partition graph for sampling
    fg = DGLGraph(fg_adj, readonly=True)

    # 建立当前GPU的cache-第一个参数是cpu-full-graph（因为读取feat要从原图读）, 第二个参数用于形成bool数组，判断某个train_nid是否在缓存中
    cacher = storage.NewJPGNNGraphCacheServer(cpu_g, fg_adj.shape[0], rank, world_size)
    cacher.init_field(['features'])

    # build DDP model and helpers
    torch.manual_seed(2022)
    model = gcn.GCNSampling(cacher.dims['features'], args.hidden_size, args.n_classes, len(
        sampling), F.relu, args.dropout)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.cuda(rank)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    ctx = torch.device(rank)

    model_idx = rank
    
    # remove old ckpt before starting
    model_ckpt_path = os.path.join(args.ckpt_path, f'model_{rank}.pt')
    grad_ckpt_path = os.path.join(args.ckpt_path, f'model_{rank}_grad.pt')
    if os.path.exists(model_ckpt_path):
        os.remove(model_ckpt_path)
    torch.save(model.state_dict(), model_ckpt_path) # 保存最初始的模型参数
    if os.path.exists(grad_ckpt_path):
        os.remove(grad_ckpt_path)
        
    def split_fn(a):
        return np.split(a, np.arange(args.batch_size, len(a), args.batch_size)) # 例如a=[0,9]，bs=3，那么切割的结果是[0,1,2], [3,4,5], [6,7,8], [9]

    # start training
    
    with torch.autograd.profiler.profile(enabled=(rank == 0), use_cuda=True) as prof:
        with torch.autograd.profiler.record_function('avg one epoch'):
            for epoch in range(args.epoch):
                with torch.autograd.profiler.record_function('train data prepare'):
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
                    sub_batch_nid = [] # k个连续的sub_batch nparray为一组，表示一次iteration中所有的worker中属于当前GPU的nid
                    for row in useful_fg_train_nid:
                        for batch in row:
                            cur_gpu_nid_mask = (nid2pid[batch]==rank)
                            sub_batch_nid.append(batch[cur_gpu_nid_mask]) # 即使是空也会占一个位置
                            if rank==0:
                                logging.debug(f'put sub_batch: {batch[cur_gpu_nid_mask]}')
                
                with torch.autograd.profiler.record_function('create Queue&Process'):
                    # 构造sampler生成器，从sub_batch_nid中取一个np.array生成一棵子树，放入queue中，等待主进程被取到后进行前传反传
                    nf_q = Queue(20)
                    fetch_done = Queue(1)
                    nf_gen_proc = Process(target=generate_nodeflows, args=(sub_batch_nid, fg, sampling, nf_q, fetch_done))
                    nf_gen_proc.daemon = True
                    nf_gen_proc.start()
                # mp.spawn(generate_nodeflows, args=(sub_batch_nid, fg, sampling, nf_q), nprocs=1)
                
                # 从nf_q中读取nf，开始模型训练
                model.train()
                n_batches = useful_fg_train_nid.shape[0]
                # n_sub_batches = n_batches * world_size
                n_sub_batches = len(sub_batch_nid)
                logging.debug(f'n_sub_batches:{n_sub_batches}')

                cur_batch_piece_id = rank
                for sub_iter in range(n_sub_batches):
                    iter = sub_iter // world_size
                    if sub_iter % world_size == 0:
                        with torch.autograd.profiler.record_function('wait sampler'):
                            try:
                                nfs = []
                                for _ in range(0,world_size):
                                    nf = nf_q.get(True)
                                    nfs.append(nf)
                            except Exception as e:
                                logging.debug(f'* {repr(e)}') # TODO: 会有Bug输出，但是似乎还是正常运行，不是很懂为什么
                        logging.debug('got sampler results.')
                        if epoch==0 and sub_iter==0:
                            cacher.fetch_data(nf) # 没有缓存的时候的fetch_data时间不要算入
                        else:
                            with torch.autograd.profiler.record_function('fetch feat'):
                                cacher.fetch_data(nf)
                    nf = nfs[sub_iter % world_size]
                    if nf!=None:
                        with torch.autograd.profiler.record_function('model transfer'):
                            # 加载其他worker写入的模型
                            load_model_ckpt_path = os.path.join(args.ckpt_path, f'model_{cur_batch_piece_id}.pt')
                            model.load_state_dict(torch.load(load_model_ckpt_path))
                            model.cuda(rank)
                        # 前传反传获取梯度
                        # if epoch==0 and sub_iter==0:
                        #     cacher.fetch_data(nf) # 没有缓存的时候的fetch_data时间不要算入
                        # else:
                        #     with torch.autograd.profiler.record_function('fetch feat'):
                        #         cacher.fetch_data(nf)

                        batch_nid = nf.layer_parent_nid(-1)
                        with torch.autograd.profiler.record_function('fetch label'):
                            labels = fg_labels[batch_nid].cuda(rank, non_blocking=True)
                        with torch.autograd.profiler.record_function('gpu-compute'):
                            pred = model(nf)
                            loss = loss_fn(pred, labels)
                            # loss = cur_train_batch_piece.size / args.batch_size # for accumulating gradient
                            loss.backward()
                            # for x in model.named_parameters():
                            #     logging.info(x[1].grad.size())
                            logging.debug(f'rank: {rank} local backward done.')
                        with torch.autograd.profiler.record_function('model transfer'):
                            new_grad_dict = {}
                            load_grad_ckpt_path = os.path.join(args.ckpt_path, f'model_{cur_batch_piece_id}_grad.pt')
                            if os.path.exists(load_grad_ckpt_path):
                                # load gradient data from previous model
                                with torch.no_grad():
                                    pre_grad_dict = torch.load(load_grad_ckpt_path)
                                    for x in model.named_parameters():
                                        x[1].grad.data += pre_grad_dict[x[0]].cuda(rank) # accumulate grad
                                        new_grad_dict[x[0]] = x[1].grad.data
                                logging.debug(f'rank: {rank} iter/sub_iter: {iter}/{sub_iter} load and accumulate grad_ckpt_path: {load_grad_ckpt_path}')
                            else:
                                new_grad_dict = {x[0]:x[1].grad.data for x in model.named_parameters()}
                                logging.debug(f'rank: {rank} iter/sub_iter: {iter}/{sub_iter} save grad_ckpt_path with new grad: {load_grad_ckpt_path}')
                            torch.save(new_grad_dict, load_grad_ckpt_path)
                    
                    with torch.autograd.profiler.record_function('sync for each sub_iter'):    
                        # 同步
                        dist.barrier()
                    # 将cur_batch_piece_id左移
                    cur_batch_piece_id = (cur_batch_piece_id-1+world_size)%world_size
                    
                    if (sub_iter+1) % world_size == 0: # 如果已经完成了一个batch的数据并行训练，那么各模型加载对应rank的模型参数、梯度，进行梯度的allreduce，然后使用优化器更新参数；覆写最新的模型参数和归零后的梯度值；
                        # 加载自己rank的模型参数
                        with torch.autograd.profiler.record_function('model transfer'):
                            model_ckpt_path = os.path.join(args.ckpt_path, f'model_{rank}.pt')
                            model.load_state_dict(torch.load(load_model_ckpt_path))
                            model.cuda(rank)
                            # 加载自己rank的梯度，然后进行allreduce同步，再更新本地模型参数
                            grad_ckpt_path = os.path.join(args.ckpt_path, f'model_{rank}_grad.pt')
                            batch_grad_dict = torch.load(grad_ckpt_path, map_location=torch.device('cpu'))
                        with torch.autograd.profiler.record_function('gradient allreduce'):
                            for param_name, param in model.named_parameters():
                                # logging.debug(f'rank: {rank} before optimizer param: {param_name} {param}')
                                # logging.debug(f'rank: {rank} before allreduce grad: {param_name} {batch_grad_dict[param_name]}')
                                recv = torch.zeros_like(batch_grad_dict[param_name])
                                allreduce(send=batch_grad_dict[param_name], recv=recv) # recv的值已经在allreduce中做了平均处理
                                param.grad = recv.cuda(rank)
                                    # logging.debug(f'rank: {rank} after allreduce grad: {param_name} {param.grad.data}')
                        with torch.autograd.profiler.record_function('gpu-compute'):
                            optimizer.step()
                            # for param_name, param in model.named_parameters():
                                # logging.debug(f'rank: {rank} after optimizer grad: {param_name} {param}')
                        
                        # 覆写新参数、梯度归零、覆写梯度
                        with torch.autograd.profiler.record_function('model transfer'):
                            torch.save(model.state_dict(), model_ckpt_path)
                            optimizer.zero_grad()
                            new_grad_dict = {x[0]:x[1].grad.data for x in model.named_parameters()}
                            torch.save(new_grad_dict, grad_ckpt_path)
                        # 至此，一个iteration结束
                    with torch.autograd.profiler.record_function('auto cache time'):
                        if epoch == 0 and sub_iter == 0:
                            cacher.auto_cache(args.dataset, "metis", world_size, rank, ['features'])
                if cacher.log:
                    miss_rate = cacher.get_miss_rate()
                    print('Epoch miss rate for epoch {} on rank {}: {:.4f}'.format(epoch, rank, miss_rate))
                print(f'=> cur_epoch {epoch} finished on rank {rank}')
                fetch_done.put(1)
                nf_gen_proc.join() # 一个epoch结束
    if rank == 0:
        logging.info(prof.key_averages().table(sort_by='cuda_time_total'))


def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='GNN Training')
    parser.add_argument('-d', '--dataset', default="/data/pagraph/ogb/set/tmp", type=str, help='training dataset name')
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
    mp.spawn(run, args=(list(range(args.num_gpu)), args), nprocs=args.num_gpu)




def generate_nodeflows(sub_batch_nid, fg, sampling, queue, fetch_done):
    sub_iter = 0
    for sub_batch in sub_batch_nid:
        # logging.info(f'=> sub_batch in queue to generate nf: {sub_batch}, sub_iter:{sub_iter}')
        sub_iter += 1
        if sub_batch.size>0:
            sampler = dgl.contrib.sampling.NeighborSampler(fg, sub_batch.size, expand_factor=int(sampling[0]), num_hops=len(sampling)+1, neighbor_type='in', shuffle=False, num_workers=1, seed_nodes=sub_batch, add_self_loop=True)
            asure = 0           
            for nf in sampler:
                asure += 1
                queue.put(nf)
                print(nf)
            assert asure<=1, 'Error when create sampler'
        else:
            queue.put(None) # 当前sub_batch为空
    # time.sleep(2) # TODO: change a way to fix this error
    while fetch_done.empty():
        time.sleep(0.1)


    