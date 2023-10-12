#  reference code from dgl==0.6.1 dgl.data.CoraFullDataset()
from tabnanny import filename_only
import numpy as np
import argparse
import json
import os, sys

import requests
import scipy.sparse as sp

import dgl
from ogb.nodeproppred import DglNodePropPredDataset
import os.path as osp

def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='data set preprocess')
    parser.add_argument('-n', '--name', default="ogbn_arxiv0", type=str, help='training dataset name')
    parser.add_argument('-l', '--len', default=0, type=int, help='feature length')
    parser.add_argument('-p', '--path', default="/data/cwj/pagraph/ogb/set", type=str, help='data store path')
    return parser.parse_args(argv)

def download_file(url, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f'-> makedirs {target_dir} done.')
    file_name = url.split('/')[-1]
    full_path = os.path.join(target_dir, file_name)
    if not os.path.exists(full_path):
        r = requests.get(url)
        if not r.status_code == 200:
            raise Exception(f'ERROR: {file_name} donwloand failed.')
        with open(full_path, 'wb') as f:
            f.write(r.content)
        print(f'-> download {full_path} done.')
    else:
        print(f'-> {full_path} has already exists.')
    return full_path


def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask

if __name__ == '__main__':
    args = parse_args_func(None)
    args.name = '_'.join(args.name.split('-'))
    target_dir = os.path.join(args.path, args.name)
    

    # generate pp.txt
    dataset = DglNodePropPredDataset('ogbn-arxiv')
    g, node_labels = dataset[0]
    g = dgl.add_reverse_edges(g) # 因为是无向图，转化为有向图
    src_lst, dst_lst = g.edges()
    with open(osp.join(target_dir, 'pp.txt'), 'w') as f:
        for srcid, dstid in zip(src_lst,dst_lst):
            f.write(str(srcid.item()) + "\t" + str(dstid.item()) + "\n")

    # generate feat.npy
    feats = g.ndata['feat'].numpy() # shared storage with torch and numpy, dtype=float32, shape = (169343, 128)
    np.save(osp.join(target_dir, 'feat.npy'), feats)

    # generate labels.npy
    g.ndata['label'] = node_labels[:, 0]
    labels = g.ndata['label'].numpy() # dtype=int64 shape= (169343,)
    np.save(osp.join(target_dir, 'labels.npy'), labels)

    # generate train/val/test.npy
    idx_split = dataset.get_idx_split()
    idx_train = idx_split['train']
    idx_val = idx_split['valid']
    idx_test = idx_split['test']

    node_num = len(node_labels)
    train_mask = _sample_mask(idx_train, labels.shape[0])
    val_mask = _sample_mask(idx_val, labels.shape[0])
    test_mask = _sample_mask(idx_test, labels.shape[0])
    node_types = np.zeros(node_num) # graph has num_nodes=169343, same as len(node_labels)
    idx_split = dataset.get_idx_split()
    
    output_train = os.path.join(target_dir, 'train.npy')
    output_val = os.path.join(target_dir, 'val.npy')
    output_test = os.path.join(target_dir, 'test.npy')
    np.save(output_train, train_mask)
    np.save(output_val, val_mask)
    np.save(output_test, test_mask)

    sys.exit(1) # 由于 add_reverse_edges 方法， g.edges()已经转化为了有向图，因此返回是1











