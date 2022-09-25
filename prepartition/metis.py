import sys
sys.path.append('./')
import data
import enum
import numpy as np
import pymetis
import argparse
import scipy.sparse as spsp
import os

import logging



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hash')
    parser.add_argument("--dataset", type=str, default=None,
                        help="path to the dataset folder")
    parser.add_argument("--partition", type=int, default=2,
                        help="partition number")
    args = parser.parse_args()

    # load data
    # 加载全图topo（其中的点应该是从0开始编号的，否则mask怎么确定？）
    adj = spsp.load_npz(os.path.join(args.dataset, 'adj.npz'))
    train_mask, val_mask, test_mask = data.get_masks(args.dataset)  # 加载mask
    train_nid = np.nonzero(train_mask)[0].astype(np.int64)  # 得到node id
    # [array([0, 2, 3, 4, 5, 7, 8]), array([1, 2, 3, 4, 5, 6, 9]), ..., array([1, 5])]
    adjacency_list = [np.nonzero(lst)[0] for lst in adj.toarray()]
    # print(adjacency_list)
    # n_cuts:边切分的次数，membership:[...]表示从Index=0的点开始被分配到的partition的id
    n_cuts, membership = pymetis.part_graph(
        args.partition, adjacency=adjacency_list)
    # print(n_cuts, membership) # 7 [0, 1, 0, 0, 0, 1, 1, 0, 1, 1]

    # 在每个partition中，按照in degree大小排序，然后写入{partition}metis文件夹中，每个文件是一个npy，存储降序的node id
    part_id = [[] for _ in range(args.partition)]
    for nid, pid in enumerate(membership):
        part_id[pid].append(nid)
    # print(part_id)
    for lst in part_id:
        lst.sort(key=lambda x: len(adjacency_list[x]), reverse=True) # 按照出度从大到小排序
    # print(part_id)

    # save
    save_folder = os.path.join(args.dataset, f"{args.partition}_metis")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        logging.info(f"mkdir {save_folder} done")
    for i, partnid in enumerate(part_id):
        np.save(os.path.join(save_folder, f'{i}.npy'), partnid)
        logging.info(f"write partition nid {save_folder}/{i}.npy file  done")


# nodes_part_0 = np.argwhere(np.array(membership) == 0).ravel() # [3, 5, 6]
# nodes_part_1 = np.argwhere(np.array(membership) == 1).ravel() # [0, 1, 2, 4]
