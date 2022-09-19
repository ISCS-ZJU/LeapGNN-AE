import argparse
import os, sys
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


# torch.set_printoptions(threshold=np.inf)


######################################################################
# Defining Training Procedure
# ---------------------------

def run(rank, devices_lst, args):
    # print config parameters
    if rank == 0:
        print('Client Args:', args)
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
    Print('full graph:', fg_adj)
    fg_labels = data.get_labels(args.dataset)
    fg_train_mask, fg_val_mask, fg_test_mask = data.get_masks(args.dataset)
    fg_train_nid = np.nonzero(fg_train_mask)[0].astype(np.int64)
    ntrain_per_node = int(fg_train_nid.shape[0] / world_size) - 1
    test_nid = np.nonzero(fg_test_mask)[0].astype(np.int64)

    # metis切分图，然后根据切分的结果，每个GPU缓存对应各部分图的热点feat（有不知道缓存量大小，第一个iter结束的时候再缓存）
    if rank == 0:
        os.system(
            f"python3 prepartition/metis.py --partition {world_size} --dataset {args.dataset}")
    torch.distributed.barrier()
    
    # 1. 每个gpu加载分图的结果，之后用于对train_lnid根据所在GPU进行切分
    max_train_nid = np.max(fg_train_nid)+1
    nid2pid = np.zeros(max_train_nid, dtype=np.int64)-1
    for pid in range(world_size):
        sorted_part_nid = data.get_partition_results(args.dataset, "metis", world_size, pid)
        necessary_nid = sorted_part_nid[sorted_part_nid<max_train_nid]
        nid2pid[sorted_part_nid] = pid
    

    # construct this partition graph for sampling
    fg = DGLGraph(fg_adj, readonly=True)

    # 建立当前GPU的cache-第一个参数是cpu-full-graph（因为读取feat要从原图读）, 第二个参数用于形成bool数组，判断某个train_nid是否在缓存中
    cacher = storage.JPGNNGraphCacheServer(cpu_g, fg_adj.shape[0], rank)
    cacher.init_field(['features'])

    # build DDP model and helpers
    model = gcn.GCNSampling(cacher.dims['features'], args.hidden_size, args.n_classes, len(
        sampling), F.relu, args.dropout)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.cuda(rank)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    ctx = torch.device(rank)

    model_idx = rank

    # start training
    with torch.autograd.profiler.profile(enabled=(rank == 0), use_cuda=True) as prof:
        for epoch in range(args.epoch):
            # 切分训练数据
            np.random.seed(epoch)
            np.random.shuffle(fg_train_nid)
            if rank==0:
                print('=> Shuffled epoch training nid:', fg_train_nid)
            train_lnid = fg_train_nid[rank * ntrain_per_node: (rank+1)*ntrain_per_node] # 当前rank的当前epoch要训练的点
            train_labels = fg_labels[train_lnid]
            part_labels = np.zeros(np.max(train_lnid) + 1, dtype=np.int)
            part_labels[train_lnid] = train_labels
            part_labels = torch.LongTensor(part_labels)  # to torch tensors
            Print('rank:', rank, 'local epoch training nid:', train_lnid)

            # # 构造sampler
            # sampler = dgl.contrib.sampling.NeighborSampler(fg, args.batch_size, expand_factor=int(sampling[0]), num_hops=len(sampling)+1, neighbor_type='in', shuffle=True, num_workers=args.num_worker, seed_nodes=train_lnid, prefetch=True, add_self_loop=True)

            model.train()
            iter = 0
            # padding train_lnid using replicas
            pad_num =0 if (train_lnid.size % args.batch_size==0) else args.batch_size - train_lnid.size % args.batch_size
            if pad_num:
                pad_ndarray = train_lnid[:pad_num]
                train_lnid = np.concatenate((train_lnid, pad_ndarray))
            train_lnid = train_lnid.reshape(-1, args.batch_size)
            Print('rank:', rank, 'train_lnid after padding:', train_lnid)
            for batch_nid in train_lnid:
                Print('rank:', rank, 'iter:', iter, 'batch_nid:', batch_nid)
                # 根据每个train nid分类出所属的优势GPU组别
                train_nid_per_gpu = []
                for pid in range(world_size):
                    cur_pid_mask = nid2pid[train_lnid]==pid
                    train_nid_per_gpu.append(train_lnid[cur_pid_mask])
                # train_nid_per_gpu = torch.tensor(train_nid_per_gpu) # then, isend is enabled
                Print('rank:', rank, 'train_nid_per_gpu:', train_nid_per_gpu)
                
                # scatter
                train_nid_after_scatter = []
                rcv_tmp_tensor = [None]
                for src in range(world_size):
                    dist.scatter_object_list(rcv_tmp_tensor, train_nid_per_gpu, src=src)
                    train_nid_after_scatter.append(rcv_tmp_tensor[0])
                Print('rank:', rank, 'train_nid_after_scatter:', train_nid_after_scatter)
                
                # 保留初始的model_ckpt和grad到CPU memory (tmp file:/dev/shm/model_{mid}.pt, model_{mid}_grad.pt)
                model_ckpt_path = os.path.join(args.ckpt_path, f'model_{rank}.pt')
                grad_ckpt_path = os.path.join(args.ckpt_path, f'model_{rank}_grad.pt')
                torch.save(model.state_dict(), model_ckpt_path)
                optimizer.zero_grad()
                with torch.no_grad():
                    grad_dict = {x[0]:x[1].data.grad for x in model.named_parameters()}
                    torch.save(grad_dict, grad_ckpt_path)
                
                cur_batch_piece_id = rank
                sub_iter = 0
                for _ in range(world_size):
                    sub_iter += 1
                    if rank==0:
                        print("===="*10, 'sub_iter:',sub_iter, "===="*10)
                    # 加载其他worker写入的模型和grad
                    load_model_ckpt_path = os.path.join(args.ckpt_path, f'model_{cur_batch_piece_id}.pt')
                    load_grad_ckpt_path = os.path.join(args.ckpt_path, f'model_{cur_batch_piece_id}_grad.pt')
                    model.load_state_dict(torch.load(load_model_ckpt_path))
                    # model.train()
                    with torch.no_grad():
                        grad_dict = torch.load(load_grad_ckpt_path)
                        for x in model.named_parameters():
                            x[1].data.grad = grad_dict[x[0]]
                    Print('rank:', rank, f'iter/sub_iter: {iter}/{sub_iter}', 'load_ckpt_path:', load_model_ckpt_path, load_grad_ckpt_path)
                    
                    # 构造sampler产生子树
                    cur_train_batch_piece = train_nid_after_scatter[cur_batch_piece_id]
                    if cur_train_batch_piece.size:
                        sampler = dgl.contrib.sampling.NeighborSampler(fg, cur_train_batch_piece.size, expand_factor=int(sampling[0]), num_hops=len(sampling)+1, neighbor_type='in', shuffle=True, num_workers=1, seed_nodes=cur_train_batch_piece, add_self_loop=True)
                        Print('rank:', rank, 'cur_train_batch_piece and size:', cur_train_batch_piece, cur_train_batch_piece.size)
                        
                        # 前传反传
                        asure = 0
                        for nf in sampler:
                            asure += 1
                            with torch.autograd.profiler.record_function('featch batch data'):
                                cacher.fetch_data(nf)
                                batch_nid = nf.layer_parent_nid(-1)
                                labels = part_labels[batch_nid].cuda(rank, non_blocking=True)
                            with torch.autograd.profiler.record_function('gpu-compute'):
                                pred = model(nf)
                                Print('rank:', rank, 'forward done.')
                                loss = loss_fn(pred, labels)
                                loss.backward()
                                Print('rank:', rank, 'local backward done.')
                        assert asure==1, 'Error when constructing sampler and need to check sampler again'
                    # 如果不是最后一次，直接保存累加了这次梯度的新梯度值
                    if sub_iter != world_size:
                        with torch.no_grad():
                            grad_dict = {x[0]:x[1].data.grad for x in model.named_parameters()}
                            torch.save(grad_dict, load_grad_ckpt_path)
                        Print('rank:', rank, f'iter/sub_iter: {iter}/{sub_iter}', 'save grad_ckpt_path with accumulated grad:', load_grad_ckpt_path)
                    # 如果是最后一次了，更新参数，梯度清零，覆写ckpt和梯度
                    else:
                        optimizer.step()
                        torch.save(model.state_dict(), load_model_ckpt_path)
                        optimizer.zero_grad()
                        with torch.no_grad():
                            grad_dict = {x[0]:x[1].data.grad for x in model.named_parameters()}
                            torch.save(grad_dict, load_grad_ckpt_path)
                        Print('rank:', rank, f'iter/sub_iter: {iter}/{sub_iter}', '_UPDATE_ params and save ckpt:', load_model_ckpt_path, load_grad_ckpt_path)
                    # 同步
                    dist.barrier()
                    # 将cur_batch_piece_id左移
                    cur_batch_piece_id = (cur_batch_piece_id-1+world_size)%world_size

                iter += 1
                if epoch == 0 and iter == 1:
                    cacher.auto_cache(args.dataset, "metis",
                                      world_size, rank, ['features'])
            if cacher.log:
                miss_rate = cacher.get_miss_rate()
                print('Epoch miss rate: {:.4f}'.format(miss_rate))
            print(f'cur_epoch {epoch} finished on rank {rank}', '===='*20)
    if rank == 0:
        print(prof.key_averages().table(sort_by='cuda_time_total'))


def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='GNN Training')
    parser.add_argument('-d', '--dataset', default="/data/pagraph/gendemo", type=str, choices=[
                        'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-mag'], help='training dataset name')
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
