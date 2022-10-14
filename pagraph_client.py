import argparse, os
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
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='12365')
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
    cpu_g = dgl.contrib.graph_store.create_graph_from_store(dataset_name, "shared_mem", port=8004)

    # rank = 0 partition graph
    sampling = args.sampling.split('-')
    assert len(set(sampling))==1, 'Only Support the same number of neighbors for each layer'
    if rank == 0:
        # --num-hops 0 means only cache distributed node features themselves
        os.system(f"python3 prepartition/hash.py --num-hops 1 --partition {world_size} --dataset {args.dataset}")
    torch.distributed.barrier()

    # get part-graph;注意，下面注释的lnid是针对part-graph，而不是sample出来的sub-graph
    partgraph_adj, lnid2onid = data.get_sub_train_graph(args.dataset, rank, world_size) # local graph topo, lnid->onid(这里的onid似乎要求就是从0开始编号的，否则最开始数据集的mask就无法适应了)
    Print("adj, lnid2onid of part-graph", partgraph_adj, lnid2onid)
    train_lnid = data.get_sub_train_nid(args.dataset, rank, world_size) # 在partgraph中
    Print("train_lnid of part-graph", train_lnid)
    train_labels = data.get_sub_train_labels(args.dataset, rank, world_size)
    Print("train_labels of part-graph", train_labels)
    partgrap_labels = np.zeros(np.max(train_lnid) + 1, dtype=np.int) # 根据train_lnid -> train_label
    partgrap_labels[train_lnid] = train_labels
    # construct this partition graph for sampling 
    part_g = DGLGraph(partgraph_adj, readonly=True)

    # to tensor
    lnid2onid = torch.LongTensor(lnid2onid)
    partgrap_labels = torch.LongTensor(partgrap_labels)

    # 建立当前GPU的cache-第一个参数是cpu-full-graph（因为读取feat要从原图读）, 第二个点的数量是partgraph（因为是当前GPU的feat缓存对象，要用bool array表示，使用full-graph太浪费空间）
    cacher = storage.PaGraphGraphCacheServer(cpu_g, partgraph_adj.shape[0], lnid2onid, rank) 
    cacher.init_field(['features'])

    # build DDP model and helpers
    model = gcn.GCNSampling(cacher.dims['features'], args.hidden_size, args.n_classes, len(sampling), F.relu, args.dropout)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    ctx = torch.device(rank)

    # sampling on sub-graph
    sampler = dgl.contrib.sampling.NeighborSampler(part_g, args.batch_size, expand_factor=int(sampling[0]), num_hops=len(sampling)+1, neighbor_type='in', shuffle=True, num_workers=args.num_worker, seed_nodes=train_lnid, prefetch=True, add_self_loop=True)

    # start training
    with torch.autograd.profiler.profile(enabled=(rank==0), use_cuda=True) as prof:
        for epoch in range(args.epoch):
            model.train()
            iter = 0
            for nf in sampler:
                with torch.autograd.profiler.record_function('featch batch data'):
                    cacher.fetch_data(nf) # 将nf._node_frame中填充每层神经元的node Frame (一个frame是一个字典，存储feat)
                    batch_nid = nf.layer_parent_nid(-1) # part-graph lnid
                    labels = partgrap_labels[batch_nid].cuda(rank, non_blocking=True)
                with torch.autograd.profiler.record_function('gpu-compute'):
                    pred = model(nf)
                    loss = loss_fn(pred, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                iter += 1
                if epoch==0 and iter==1:
                    cacher.auto_cache(part_g, ['features']) # 这时候做的好处是根据历史第一轮的gpu memory利用情况，自适应判断可以缓存多少的feat
            if cacher.log:
                miss_rate = cacher.get_miss_rate()
                print('Epoch miss rate: {:.4f}'.format(miss_rate))
    if rank == 0:
        print(prof.key_averages().table(sort_by='cuda_time_total'))




def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='GNN Training')
    parser.add_argument('-d', '--dataset', default="/data/cwj/pagraph/gendemo", type=str, choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-mag'], help='training dataset name')
    parser.add_argument('-ngpu', '--num-gpu', default=1, type=int, help='# of gpus to train gnn with DDP')
    parser.add_argument('-s', '--sampling', default="2-2-2", type=str, help='neighborhood sampling method parameters')
    parser.add_argument('-hd', '--hidden-size', default=256, type=int, help='hidden dimension size')
    parser.add_argument('-ncls', '--n-classes', default=60, type=int, help='number of classes')
    parser.add_argument('-bs', '--batch-size', default=2, type=int, help='training batch size')
    parser.add_argument('-dr', '--dropout', default=0.2, type=float, help='dropout in training')
    parser.add_argument('-lr', '--lr', default=3e-2, type=float, help='learning rate')
    parser.add_argument('-wdy', '--weight-decay', default=0, type=float, help='weight decay')
    parser.add_argument('-mn', '--model-name', default='graphsage', type=str, choices=['graphsage', 'gcn', 'demo'], help='GNN model name')
    parser.add_argument('-ep', '--epoch', default=3, type=int, help='total trianing epoch')
    parser.add_argument('-wkr', '--num-worker', default=1, type=int, help='sampling worker')
    parser.add_argument('-cs', '--cache-size', default=0, type=int, help='cache size in each gpu (GB)')
    return parser.parse_args(argv)



if __name__ == '__main__':
    args = parse_args_func(None)
    mp.spawn(run, args=(list(range(args.num_gpu)), args), nprocs=args.num_gpu)
