from pickle import TRUE
from ogb.nodeproppred import NodePropPredDataset
import pandas as pd
import numpy as np
import os.path as osp
import sys
import argparse

def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='data set preprocess')
    parser.add_argument('-n', '--name', default="ogbn-arxiv", type=str, choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-mag'], help='training dataset name')
    parser.add_argument('-p', '--path', default="/data/pagraph/ogb/set", type=str, help='data store path')
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_args_func(None)
    directed = dict()
    directed['ogbn-arxiv'] = 1
    directed['ogbn-products'] = 0
    # label = pd.read_csv(osp.join('/home/qhy/gnn/repgnn/dataset/ogbn_arxiv/raw', 'node-label.csv.gz'), compression='gzip', header = None).values.T.astype(np.int64)
    # print(label)

    # setpath = '/data/pagraph/ogb/set'
    # curpath =  osp.dirname(__file__)
    setpath = args.path
    dataset = NodePropPredDataset(name=args.name,root='/data/pagraph/ogb')
    data = dataset[0]
    node_num = data[0]['num_nodes']
    edge_num = len(data[0]['edge_index'][0])
    # num = [0] * node_num
    # flag = 0
    # count = 0
    # for i in range(0,edge_num):
    #     num[data[0]['edge_index'][0][i]] = num[data[0]['edge_index'][0][i]] + 1
    #     num[data[0]['edge_index'][1][i]] = num[data[0]['edge_index'][1][i]] + 1
    # for i in range(0,node_num):
    #     if num[i] == 0:
    #         flag = 1
    #         count = count + 1
    #         print(data[1][i])
    # print(count/node_num)
    # assert flag == 0
    labels = np.zeros(node_num,dtype=np.int32)
    for i in range(0,node_num):
        labels[i] = data[1][i]

    labpath = osp.join(setpath,'labels.npy')
    ppath = osp.join(setpath,'pp.txt')
    featpath = osp.join(setpath,'feat.npy')
    np.save(labpath,labels)
    with open(ppath,'w') as f:
        for i in range(0,edge_num):
            f.write(str(data[0]['edge_index'][0][i]) + "\t" + str(data[0]['edge_index'][1][i]) + "\n")
    np.save(featpath,np.array(data[0]['node_feat']))
    

    print('end')

    # print(data[0]['edge_index'][0])
    # print(data[0]['edge_index'][1])
    # print(len(data[0]['node_feat']))
    sys.exit(directed[args.name])