from pickle import TRUE
from ogb.nodeproppred import NodePropPredDataset
import pandas as pd
import numpy as np
import os
import os.path as osp
import sys
import argparse

def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='data set preprocess')
    parser.add_argument('-n', '--name', default="ogbn-arxiv", type=str, choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-mag', 'ogbn-papers100M'], help='training dataset name')
    parser.add_argument('-l', '--len', default=0, type=int, help='feature length')
    parser.add_argument('-p', '--path', default="/data/cwj/pagraph/ogb/set", type=str, help='data store path')
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_args_func(None)
    directed = {'ogbn-arxiv': 1, 'ogbn-papers100M': 1, 'ogbn-products': 0}
    # label = pd.read_csv(osp.join('/home/qhy/gnn/repgnn/dataset/ogbn_arxiv/raw', 'node-label.csv.gz'), compression='gzip', header = None).values.T.astype(np.int64)
    # print(label)

    # setpath = '/data/cwj/pagraph/ogb/set'
    # curpath =  osp.dirname(__file__)
    setpath = args.path
    print('-> loading NodePropPredDataset from DGL library')
    dataset = NodePropPredDataset(name=args.name,root=setpath)
    print('-> end')
    data = dataset[0]
    node_num = data[0]['num_nodes']
    edge_num = len(data[0]['edge_index'][0])
    print(f'-> dataset name: {args.name} #vertex: {node_num} #edges: {edge_num}')

    labels = np.zeros(node_num,dtype=np.float32)
    for i in range(0,node_num):
        labels[i] = data[1][i]

    dataset_name = args.name.replace('-','_')
    # labels = np.load(osp.join(osp.join(setpath, dataset_name), 'raw/node-label.npz'))['node_label']
    # print(f'labels: {labels}')
    # 生成的
    savepath = osp.join(setpath, dataset_name) + f'{args.len}'
    if not osp.exists(savepath):
        os.mkdir(savepath)
        print('-> Created dir name:', savepath)
    labpath = osp.join(savepath,'labels.npy')
    np.save(labpath,labels)
    print(f'-> Save labels into {labpath}')

    ppath = osp.join(savepath,'pp.txt')
    with open(ppath,'w') as f:
        for i in range(0,edge_num):
            f.write(str(data[0]['edge_index'][0][i]) + "\t" + str(data[0]['edge_index'][1][i]) + "\n")
    print(f'-> Save pp files into {ppath}')

    featpath = osp.join(savepath,'feat.npy')
    cur_len = len(data[0]['node_feat'][0])
    new_array = np.empty(data[0]['node_feat'].shape,dtype="float32")
    new_array[:] = data[0]['node_feat']
    print('-> original features dim:', len(new_array[0]), 'dtype:', new_array.dtype)
    if args.len <= 0:
        pass
    elif cur_len < args.len:
        new_array = np.concatenate((new_array,np.ones((node_num,args.len - cur_len), dtype=new_array.dtype)),axis=1)
    else:
        new_array = new_array[:,0:args.len]
    np.save(featpath,new_array)
    print(f'-> Save feats (dim: {len(new_array[0])}) into {featpath}')
    print(f"-> Total feats memory size (MB):", new_array.size * new_array.itemsize // 1024 // 1024)

    print(f'-> ogb_fmt.py run to end, directed value ={directed[args.name]}')

    # print(data[0]['edge_index'][0])
    # print(data[0]['edge_index'][1])
    # print(len(data[0]['node_feat']))
    sys.exit(directed[args.name])