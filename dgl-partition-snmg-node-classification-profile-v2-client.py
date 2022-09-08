import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import sklearn.metrics
import torch.multiprocessing as mp

import time, argparse, sys
from models import demo_model, graphsage
from utils import timer
from gpucache import cache
from partition import metis


# torch.set_printoptions(threshold=np.inf)


######################################################################
# Defining Training Procedure
# ---------------------------

def run(proc_id, devices, args):
    # print config parameters
    if proc_id == 0:
        print(args)
    total_epochs = args.epoch
    # Initialize distributed training context.
    dev_id = devices[proc_id]
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='12345')
    if torch.cuda.device_count() < 1:
        device = torch.device('cpu')
        torch.distributed.init_process_group(
            backend='gloo', init_method=dist_init_method, world_size=len(devices), rank=proc_id)
    else:
        torch.cuda.set_device(dev_id)
        device = torch.device('cuda:' + str(dev_id))
        torch.distributed.init_process_group(
            backend='gloo', init_method=dist_init_method, world_size=len(devices), rank=proc_id)
    
    # Define training and validation dataloader, copied from the previous tutorial
    # but with one line of difference: use_ddp to enable distributed data parallel
    # data loading.
    sampling_param = list(map(int, args.sampling.split('-')))
    sampler = dgl.dataloading.NeighborSampler(sampling_param,
        
        )
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to NodeDataLoader.
        graph,              # The graph
        train_nids,         # The node IDs to iterate over in minibatches
        sampler,            # The neighbor sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        use_ddp=True,       # Make it work with distributed data parallel
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=args.batch_size,    # Per-device batch size.
                            # The effective batch size is this number times the number of GPUs.
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=args.num_worker       # Number of sampler processes
    )
    valid_dataloader = dgl.dataloading.DataLoader(
        graph, valid_nids, sampler,
        device=device,
        use_ddp=False,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_worker,
    )

    # create the models
    # model = demo_model.Model(num_features, 128, num_classes).to(device)
    if args.model_name == 'graphsage':
        model = graphsage.SAGE(num_features, args.hidden_size, num_classes).to(device)
    elif args.model_name == 'demo':
        model = demo_model.Model(num_features, args.hidden_size, num_classes).to(device)

    # Wrap the model with distributed data parallel module.
    if device == torch.device('cpu'):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)
    
    # Define optimizer
    opt = torch.optim.Adam(model.parameters())
    
    best_accuracy = 0
    best_model_path = './model.pt'

    # cpu shared memory graph
    remote_g = dgl.contrib.graph_store.create_graph_from_store(args.dataset, "shared_mem")


    # gpu cache 初始化
    
    # 1. get local partitioned graph with: local_graph, local_nfeat, local_efeat, gpb, graphname 
    parti_results = metis.partition_graph(graph, args.dataset, args.num_gpu, len(sampling_param), proc_id)
    
    # gnid2onid=[a, b, c...] a表示原图的node id，remap后对应的id是0，依次类推; 
    # 我们把remap后从0开始的这些每个partition连续的id称为global id (gnid)，原始图的id称为original id(onid)
    # gpb.nid2localnid(gnid, partid)得到local node id;其中partid只能是当前proc分配到的partion的id，不能查询远程的partition
    # gpb.nid2partid(gnid)得到global node id -> partition id;
    # gpb.partid2nids(partid)得到partition id -> global node id；这里的global node id是连续的；
    # local_feat是包含了k-hop的结果
    local_graph, local_nfeat, local_efeat, gpb, graphname, gnid2onid, geid2oeid = parti_results
    
    
    """
    # understanding onid, gnid, local node id, local_nfeat, local_graph
    print(gnid2onid[0], gnid2onid[1], geid2oeid[0], geid2oeid[1], len(gnid2onid), len(geid2oeid))
    print('proc_id:', proc_id, gpb.nid2localnid(0, proc_id), gpb.nid2localnid(42693, proc_id), gpb.nid2partid(42690),gpb.nid2partid(42691), gpb.partid2nids(proc_id))
    print(gpb.metadata()) # [{'machine_id': 0, 'num_nodes': 42691, 'num_edges': 530820}, {'machine_id': 1, 'num_nodes': 41964, 'num_edges': 393048}, {'machine_id': 2, 'num_nodes': 41992, 'num_edges': 420392}, {'machine_id': 3, 'num_nodes': 42696, 'num_edges': 988226}]

    if proc_id==0:
        print(graph, local_graph) # Graph(num_nodes=169343, num_edges=2332486, ndata_schemes={'year': Scheme(shape=(1,), dtype=torch.int64), 'feat': Scheme(shape=(128,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)} edata_schemes={}) 
        #Graph(num_nodes=153373, num_edges=2102947, ndata_schemes={'orig_id': Scheme(shape=(), dtype=torch.int64), 'part_id': Scheme(shape=(), dtype=torch.int64), 'inner_node': Scheme(shape=(), dtype=torch.int32), '_ID': Scheme(shape=(), dtype=torch.int64)} edata_schemes={'inner_edge': Scheme(shape=(), dtype=torch.int8), 'orig_id': Scheme(shape=(), dtype=torch.int64), '_ID': Scheme(shape=(), dtype=torch.int64)})

        # 下面展示从子图访问和原图访问相同的点以及node id
        print(graph.ndata['feat'][121586]) # 从原图访问某个点，改点对应part0的0号点
        print(gpb.partid2nids(0)[0], gnid2onid[gpb.partid2nids(0)[0]], local_nfeat['_N/feat'][0],) # 从子图的属性中访问同一个点，打印gnid, onid,和feat
    """


    
    # 2. local gpu-cache
    n_local_nfeats, nfeat_dim = local_nfeat['_N/feat'].size()
    print(n_local_nfeats, nfeat_dim)
    local_cache_size = args.cache_size*1024*1024*1024 // nfeat_dim // 4 # # of node feats to cache on each GPU
    local_gpu_cacher = cache.GraphGPUCache(gpb, gnid2onid.tolist(), local_nfeat, proc_id, cache_size=local_cache_size)
    
    profile_begin = time.time()
    avg_time_transfer = [] # per iteration
    avg_time_sampling = [] # per iteration
    with torch.autograd.profiler.profile(enabled=(proc_id==0), use_cuda=True) as prof:
        for epoch in range(total_epochs):
            model.train()

            with tqdm.tqdm(train_dataloader) as tq:
                st = time.time()
                for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                    # assert torch.equal(input_nodes, mfgs[0].srcdata[dgl.NID])
                    # if 121586 in input_nodes:
                    #     print('121586 in mfgs with feat:', mfgs[0].srcdata['feat'][input_nodes.tolist().index(121586)]) # 说明mfgs中的点id是onid
                    avg_time_sampling.append(time.time() - st)
                    with torch.autograd.profiler.record_function('cpu-gpu-feat-label-load'):
                        with timer.Timer(device=f"cuda:{proc_id}") as t:
                            # inputs = mfgs[0].srcdata['feat']
                            inputs2 = local_gpu_cacher.construct_input_feats(mfgs)
                            # inputs3 = local_gpu_cacher.construct_input_feats_include_remote(mfgs)
                            # print('rank:', proc_id, torch.equal(inputs, inputs2), torch.equal(inputs, inputs3))
                            # print('inputs:',proc_id, inputs)
                            # print('inputs:',proc_id, inputs3)
                            # assert torch.equal(inputs, inputs2)
                            # assert torch.equal(inputs, inputs3)
                            labels = mfgs[-1].dstdata['label']

                    avg_time_transfer.append(t.elapsed_secs)
                    with torch.autograd.profiler.record_function('gpu-forward'):
                        predictions = model(mfgs, inputs2)
                    with torch.autograd.profiler.record_function('gpu-cal_loss'):
                        loss = F.cross_entropy(predictions, labels)
                        opt.zero_grad()
                    with torch.autograd.profiler.record_function('gpu-backward'):
                        loss.backward()
                    with torch.autograd.profiler.record_function('gpu-optimizer'):
                        opt.step()

                    accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy())

                    tq.set_postfix({'loss': '%.03f' % loss.item(), 'acc': '%.03f' % accuracy}, refresh=False)
                    st = time.time()
                    # sys.exit()

            model.eval()

            # Evaluate on only the first GPU.
            if proc_id == -1:
                predictions = []
                labels = []
                with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
                    for input_nodes, output_nodes, mfgs in tq:
                        inputs = mfgs[0].srcdata['feat']
                        labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                        predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                    predictions = np.concatenate(predictions)
                    labels = np.concatenate(labels)
                    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
                    print('Epoch {} Validation Accuracy {}'.format(epoch, accuracy))
                    if best_accuracy < accuracy:
                        best_accuracy = accuracy
                        torch.save(model.state_dict(), best_model_path)
            # break
    print('Training Total Time for {} Epochs : {:.4f}s'.format(total_epochs, time.time() - profile_begin))
    if proc_id == 0:
        print(prof.key_averages().table(sort_by='cuda_time_total'))
        print('Feats and labels transfer avg time per iteration:', sum(avg_time_transfer)/len(avg_time_transfer))
        print('Waiting sampples avg time per iteration:', sum(avg_time_sampling)/len(avg_time_sampling))


def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='GNN Training')
    parser.add_argument('-d', '--dataset', default="ogbn-arxiv", type=str, choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-mag'], help='training dataset name')
    parser.add_argument('-ngpu', '--num-gpu', default=2, type=int, help='# of gpus to train gnn with DDP')
    parser.add_argument('-s', '--sampling', default="10-10-10", type=str, help='neighborhood sampling method parameters')
    parser.add_argument('-hd', '--hidden-size', default=256, type=int, help='hidden dimension size')
    parser.add_argument('-bs', '--batch-size', default=1024, type=int, help='training batch size')
    parser.add_argument('-mn', '--model-name', default='graphsage', type=str, choices=['graphsage', 'gcn', 'demo'], help='GNN model name')
    parser.add_argument('-ep', '--epoch', default=3, type=int, help='total trianing epoch')
    parser.add_argument('-wkr', '--num-worker', default=0, type=int, help='sampling worker')
    parser.add_argument('-cs', '--cache-size', default=0, type=int, help='cache size in each gpu (GB)')
    return parser.parse_args(argv)


args = parse_args_func(None)

dataset = DglNodePropPredDataset(args.dataset)
graph, node_labels = dataset[0]
# Add reverse edges since ogbn-arxiv is unidirectional.
graph = dgl.add_reverse_edges(graph)
graph.ndata['label'] = node_labels[:, 0]
graph.create_formats_() #  避免每个子进程重复做COO到CSC/CSR格式的转化

node_features = graph.ndata['feat']
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()

idx_split = dataset.get_idx_split()
train_nids = idx_split['train']
valid_nids = idx_split['valid']
test_nids = idx_split['test']    # Test node IDs, not used in the tutorial though.

num_gpus = args.num_gpu

if __name__ == '__main__':
    mp.spawn(run, args=(list(range(num_gpus)), args), nprocs=num_gpus)
