import random
import torch.backends.cudnn as cudnn
import argparse
import os, sys
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
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO, filename="./dgl_cpu_degree_1031.txt", filemode='a+',
                    format='%(levelname)s %(asctime)s %(filename)s %(lineno)d : %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
# torch.set_printoptions(threshold=np.inf)



#############################
# WARNING: using DDP for multi-node training
# ---------------------------

def main(ngpus_per_node):
    # reproduce the same results
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = False

    assert args.world_size > 1, 'This version only support distributed GNN training with multiple nodes'
    args.distributed = args.world_size > 1 # using DDP for multi-node training

    
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size  # total # of DDP training process
        mp.spawn(run, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        sys.exit(-1)


def run(gpu, ngpus_per_node, args):
    # print config parameters
    if gpu == 0:
        logging.info(f'Client Args: {args}')
    
    args.gpu = gpu # 表示使用本地节点的gpu id
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu # 传入的rank表示节点个数
    # Initialize distributed training context.
    dist_init_method = args.dist_url
    if torch.cuda.device_count() < 1:
        device = torch.device('cpu')
        torch.distributed.init_process_group(
            backend='gloo', init_method=dist_init_method, world_size=args.world_size, rank=args.rank)
        logging.info(f'Using CPU for training...')
    else:
        torch.cuda.set_device(args.rank)
        device = torch.device('cuda:' + str(args.rank))
        torch.distributed.init_process_group(
            backend='nccl', init_method=dist_init_method, world_size=args.world_size, rank=args.rank)
        logging.info(f'Using {args.world_size} GPUs in total for training...')

    # # connect to cpu graph server
    # dataset_name = os.path.basename(args.dataset)
    # cpu_g = dgl.contrib.graph_store.create_graph_from_store(
    #     dataset_name, "shared_mem", port=8004)

    # rank = 0 partition graph
    sampling = args.sampling.split('-')
    assert len(
        set(sampling)) == 1, 'Only Support the same number of neighbors for each layer'

    # get train nid (consider DDP) as corresponding labels
    fg_adj = data.get_struct(args.dataset)
    fg_labels = data.get_labels(args.dataset)
    fg_train_mask, fg_val_mask, fg_test_mask = data.get_masks(args.dataset)
    fg_train_nid = np.nonzero(fg_train_mask)[0].astype(np.int64)
    ntrain_per_node = int(fg_train_nid.shape[0] / args.world_size)
    print('fg_train_nid:',
          fg_train_nid.shape[0], 'ntrain_per_GPU:', ntrain_per_node)
    test_nid = np.nonzero(fg_test_mask)[0].astype(np.int64)

    fg_labels = torch.from_numpy(fg_labels).type(torch.LongTensor)  # in cpu

    torch.distributed.barrier()

    # construct this partition graph for sampling
    # TODO: 图的topo之后也要分布式
    fg = DGLGraph(fg_adj, readonly=True)

    # build DDP model and helpers
    model = gcn.GCNSampling(cacher.dims['features'], args.hidden_size, args.n_classes, len(
        sampling), F.relu, args.dropout)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    ctx = torch.device(rank)

    # start training
    with torch.autograd.profiler.profile(enabled=(rank == 0), use_cuda=True) as prof:
        with torch.autograd.profiler.record_function('total epochs time'):
            for epoch in range(args.epoch):
                with torch.autograd.profiler.record_function('train data prepare'):
                    # 切分训练数据
                    np.random.seed(epoch)
                    np.random.shuffle(fg_train_nid)
                    train_lnid = fg_train_nid[rank *
                                              ntrain_per_node: (rank+1)*ntrain_per_node]
                    train_labels = fg_labels[train_lnid]
                    part_labels = np.zeros(
                        np.max(train_lnid) + 1, dtype=np.int)
                    part_labels[train_lnid] = train_labels
                    part_labels = torch.LongTensor(
                        part_labels)  # to torch tensors

                # 构造sampler
                sampler = dgl.contrib.sampling.NeighborSampler(fg, args.batch_size, expand_factor=int(sampling[0]), num_hops=len(
                    sampling)+1, neighbor_type='in', shuffle=True, num_workers=args.num_worker, seed_nodes=train_lnid, prefetch=True, add_self_loop=True)

                model.train()
                iter = 0
                wait_sampler = []
                st = time.time()
                # each_sub_iter_nsize = [] #  记录每次前传计算的 sub_batch的树的点树
                for nf in sampler:
                    wait_sampler.append(time.time()-st)
                    logging.debug(f'iter: {iter}')
                    if epoch == 0 and iter == 0:
                        cacher.fetch_data(nf)  # 没有缓存的时候的fetch_data时间不要算入
                    else:
                        with torch.autograd.profiler.record_function('fetch feat'):
                            # 将nf._node_frame中填充每层神经元的node Frame (一个frame是一个字典，存储feat)
                            cacher.fetch_data(nf)
                    batch_nid = nf.layer_parent_nid(-1)
                    with torch.autograd.profiler.record_function('fetch label'):
                        labels = part_labels[batch_nid].cuda(
                            rank, non_blocking=True)
                    with torch.autograd.profiler.record_function('gpu-compute with optimizer.step'):
                        # each_sub_iter_nsize.append(nf._node_mapping.tousertensor().size(0))
                        pred = model(nf)
                        loss = loss_fn(pred, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    iter += 1
                    with torch.autograd.profiler.record_function('auto cache time'):
                        if epoch == 0 and iter == 1:
                            cacher.auto_cache(args.dataset, "metis",
                                              world_size, rank, ['features'])
                    st = time.time()
                logging.info(f'rank: {rank}, iter_num: {iter}')
                if cacher.log:
                    miss_rate = cacher.get_miss_rate()
                    print('Epoch miss rate for epoch {} on rank {}: {:.4f}'.format(
                        epoch, rank, miss_rate))
                    # print(f'Sub_iter nsize mean, max, min: {int(sum(each_sub_iter_nsize) / len(each_sub_iter_nsize))}, {max(each_sub_iter_nsize)}, {min(each_sub_iter_nsize)}')
                print(f'=> cur_epoch {epoch} finished on rank {rank}')
    if rank == 0:
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
    # distributed related
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args_func(None)
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    logging.info(f"ngpus_per_node:", ngpus_per_node)

    main(ngpus_per_node)
