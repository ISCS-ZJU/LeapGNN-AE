
from utils import get_sub_graph
import sys
sys.path.append('./')
import data
import os

import dgl
from dgl import DGLGraph
import torch
import numpy as np
import scipy.sparse as spsp
import argparse


sys.path.append('./')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hash')
    parser.add_argument("--dataset", type=str, default=None,
                        help="path to the dataset folder")
    parser.add_argument("--num-hops", type=int, default=1,
                        help="num hops for the extended graph")
    parser.add_argument("--partition", type=int, default=2,
                        help="partition number")
    args = parser.parse_args()

    # load data
    # 加载全图topo（其中的点应该是从0开始编号的，否则mask怎么确定？）
    adj = spsp.load_npz(os.path.join(args.dataset, 'adj.npz'))
    dgl_g = DGLGraph(adj, readonly=True)  # 构建全图dglgraph
    train_mask, val_mask, test_mask = data.get_masks(args.dataset)  # 加载mask
    train_nid = np.nonzero(train_mask)[0].astype(np.int64)  # 得到node id
    # shuffle
    np.random.shuffle(train_nid)  # shuffle train id
    labels = data.get_labels(args.dataset)

    # save
    adj_file = os.path.join(args.dataset, 'adj.npz')
    mask_file = os.path.join(args.dataset, 'train.npy')
    label_file = os.path.join(args.dataset, 'labels.npy')
    partition_dataset = os.path.join(
        args.dataset, '{}naive'.format(args.partition))  # 存放分割后的graph
    try:
        os.mkdir(partition_dataset)
    except FileExistsError:
        pass

    chunk_size = int(len(train_nid) / args.partition)  # 每个part的node数量
    for pid in range(args.partition):
        start_ofst = chunk_size * pid
        if pid == args.partition - 1:
            end_ofst = len(train_nid)
        else:
            end_ofst = start_ofst + chunk_size
        part_nid = train_nid[start_ofst:end_ofst]  # 各部分的点id np array
        subadj, sub2fullid, subtrainid = get_sub_graph(
            dgl_g, part_nid, args.num_hops)  # 根据子图分配到的点，进行邻居采样，扩充Part
        sublabel = labels[sub2fullid[subtrainid]]
        # files
        subadj_file = os.path.join(
            partition_dataset,
            'subadj_{}.npz'.format(str(pid)))
        sub_trainid_file = os.path.join(
            partition_dataset,
            'sub_trainid_{}.npy'.format(str(pid)))
        sub_train2full_file = os.path.join(
            partition_dataset,
            'sub_train2fullid_{}.npy'.format(str(pid)))
        sub_label_file = os.path.join(
            partition_dataset,
            'sub_label_{}.npy'.format(str(pid)))
        spsp.save_npz(subadj_file, subadj)  # lnid -> adj
        np.save(sub_trainid_file, subtrainid)  # lnid -> trainid
        np.save(sub_train2full_file, sub2fullid)  # lnid->onid
        np.save(sub_label_file, sublabel)  # labels
