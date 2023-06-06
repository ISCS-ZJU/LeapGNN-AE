#  reference code from dgl==0.6.1 dgl.data.CoraFullDataset()
from tabnanny import filename_only
import numpy as np
import argparse
import json
import os, sys

import requests
import scipy.sparse as sp

from dgl.data import CoraFullDataset
import os.path as osp

def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='data set preprocess')
    parser.add_argument('-n', '--name', default="corafull0", type=str, help='training dataset name')
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


if __name__ == '__main__':
    args = parse_args_func(None)
    # directed = {'reddit0': 0} # indicated by reddit-G_full.json
    
    cora_dgl_link = 'https://data.dgl.ai/dataset/cora_full.zip'
    target_dir = os.path.join(args.path, args.name)
    
    # # download and extract into target_dir
    filepath = download_file(cora_dgl_link, target_dir) # just download, not used

    # generate pp.txt
    data = CoraFullDataset(raw_dir=target_dir)
    g = data[0]
    src_lst, dst_lst = g.edges()
    with open(osp.join(target_dir, 'pp.txt'), 'w') as f:
        for srcid, dstid in zip(src_lst,dst_lst):
            f.write(str(srcid.item()) + "\t" + str(dstid.item()) + "\n")

    # generate feat.npy
    feats = g.ndata['feat'].numpy() # shared storage with torch and numpy, dtype=float32, shape = (19793, 8710)
    np.save(osp.join(target_dir, 'feat.npy'), feats)

    # generate labels.npy
    labels = g.ndata['label'].numpy() # dtype=int64
    np.save(osp.join(target_dir, 'labels.npy'), labels)

    sys.exit(0) # 由于 g.edges()已经转化为了无向图，因此返回是0











