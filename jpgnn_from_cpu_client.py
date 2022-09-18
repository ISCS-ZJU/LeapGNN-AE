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
    
    # get train nid (consider DDP) as corresponding labels
    fg_adj = data.get_struct(args.dataset)
    Print('full graph:', fg_adj)
    fg_labels = data.get_labels(args.dataset)
    fg_train_mask, fg_val_mask, fg_test_mask = data.get_masks(args.dataset)
    fg_train_nid = np.nonzero(fg_train_mask)[0].astype(np.int64)
    ntrain_per_node = int(fg_train_nid.shape[0] / world_size) -1
    test_nid = np.nonzero(fg_test_mask)[0].astype(np.int64)
    

    # metis切分图，然后根据切分的结果，每个GPU缓存对应各部分图的热点feat（有不知道缓存量大小，第一个iter结束的时候再缓存）
    if rank == 0:
        os.system(f"python3 prepartition/metis.py --partition {world_size} --dataset {args.dataset}")
    torch.distributed.barrier()

    # construct this partition graph for sampling 
    fg = DGLGraph(fg_adj, readonly=True)
    

    # 建立当前GPU的cache-第一个参数是cpu-full-graph（因为读取feat要从原图读）, 第二个参数用于形成bool数组，判断某个train_nid是否在缓存中
    cacher = storage.JPGNNGraphCacheServer(cpu_g, fg_adj.shape[0], rank)
    cacher.init_field(['features'])

    # build DDP model and helpers
    model = gcn.GCNSampling(cacher.dims['features'], args.hidden_size, args.n_classes, len(sampling), F.relu, args.dropout)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    ctx = torch.device(rank)

    model_idx = rank

    # start training
    with torch.autograd.profiler.profile(enabled=(rank==0), use_cuda=True) as prof:
        for epoch in range(args.epoch):
            # 切分训练数据
            np.random.seed(epoch)
            np.random.shuffle(fg_train_nid)
            train_lnid = fg_train_nid[rank*ntrain_per_node: (rank+1)*ntrain_per_node]
            train_labels = fg_labels[train_lnid]
            part_labels = np.zeros(np.max(train_lnid) + 1, dtype=np.int)
            part_labels[train_lnid] = train_labels
            part_labels = torch.LongTensor(part_labels) # to torch tensors
            Print('rank:',rank, 'local training nid:',train_lnid)
            
            # 根据每个train nid分类出所属的优势GPU组别
            train_nid_per_gpu = 

            # 构造sampler
            sampler = dgl.contrib.sampling.NeighborSampler(fg, args.batch_size, expand_factor=int(sampling[0]), num_hops=len(sampling)+1, neighbor_type='in', shuffle=True, num_workers=args.num_worker, seed_nodes=train_lnid, prefetch=True, add_self_loop=True)

            model.train()
            iter = 0
            for nf in sampler:
                Print('iter:', iter)
                with torch.autograd.profiler.record_function('featch batch data'):
                    cacher.fetch_data(nf) # 将nf._node_frame中填充每层神经元的node Frame (一个frame是一个字典，存储feat)
                    batch_nid = nf.layer_parent_nid(-1)
                    labels = part_labels[batch_nid].cuda(rank, non_blocking=True)
                with torch.autograd.profiler.record_function('gpu-compute'):
                    print(f'rank: {rank}, model locate on device:', model.device)
                    change_nid = nf._node_mapping.tousertensor().cuda(rank)[-1]
                    Print('rank:', rank, 'to change nid:', change_nid)
                    pred = model(nf, v = change_nid)
                    loss = loss_fn(pred, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    Print('rank:', rank, 'before training nid emb:', nf._node_frames[0]._frame._columns['features'][-2:])
                    
                iter += 1
                if epoch==0 and iter==1:
                    cacher.auto_cache(args.dataset, "metis", world_size, rank, ['features'])
            if cacher.log:
                miss_rate = cacher.get_miss_rate()
                print('Epoch miss rate: {:.4f}'.format(miss_rate))
            print(f'cur_epoch {epoch} finished on rank {rank}', '===='*20)
            # move gnn to next right gpu
            model_idx = (model_idx+1)%world_size
            print(f'rank: {rank} will move model to {model_idx}')
            # model = model.to(model_idx)
            print(f'rank: {rank} after moving gnn, model now on device{model.device}')
    if rank == 0:
        print(prof.key_averages().table(sort_by='cuda_time_total'))
    





def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='GNN Training')
    parser.add_argument('-d', '--dataset', default="/data/pagraph/gendemo", type=str, choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-mag'], help='training dataset name')
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
