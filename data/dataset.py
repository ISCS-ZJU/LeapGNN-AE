from dgl.data import load_data
import argparse
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os.path as osp

def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='data set preprocess')
    parser.add_argument('-n', '--name', default="citeseer", type=str, choices=['citeseer', 'pubmed'], help='training dataset name')
    parser.add_argument('-l', '--len', default=0, type=int, help='feature length')
    parser.add_argument('-p', '--path', default="/data/cwj/pagraph/ogb/set", type=str, help='data store path')
    return parser.parse_args(argv)

def _pickle_load(pkl_file):
    if sys.version_info > (3, 0):
        return pkl.load(pkl_file, encoding='latin1')
    else:
        return pkl.load(pkl_file)
    
def _parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask

def _preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return np.array(features.todense())
if __name__ == '__main__':
    args = parse_args_func(None)
    directed = {'citeseer': 1, 'pubmed': 1}
    objnames = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    dataset = args.name
    root = osp.join(args.path,dataset+str(args.len))
    objects = []
    for name in objnames:
        with open("{}/ind.{}.{}".format(root, dataset, name), 'rb') as f:
            objects.append(_pickle_load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = _parse_index_file("{}/ind.{}.test.index".format(root, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = _preprocess_features(features)
    # for src,dst_list in graph.items():
    #     for dst in dst_list:
    #         print(str(src) + "\t" + str(dst) + "\n")

    # graph = nx.DiGraph(nx.from_dict_of_lists(graph))

    onehot_labels = np.vstack((ally, ty))
    onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
    labels = np.argmax(onehot_labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    print(f'-> len of train: {len(idx_train)}, len of val: {len(idx_val)}, len of test: {len(idx_test)}')

    train_mask = _sample_mask(idx_train, labels.shape[0])
    val_mask = _sample_mask(idx_val, labels.shape[0])
    test_mask = _sample_mask(idx_test, labels.shape[0])

    features = features.astype(np.float32)
    print(features.dtype)

    np.save(osp.join(root, 'train.npy'),train_mask)
    np.save(osp.join(root, 'val.npy'),val_mask)
    np.save(osp.join(root, 'test.npy'),test_mask)
    np.save(osp.join(root, 'feat.npy'),features)
    np.save(osp.join(root, 'labels.npy'),labels)
    with open(osp.join(root, 'pp.txt'),'w') as f:
        for src,dst_list in graph.items():
            for dst in dst_list:
                f.write(str(src) + "\t" + str(dst) + "\n")
    sys.exit(directed[args.name])
