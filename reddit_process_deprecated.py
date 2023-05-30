#  ref: https://github.com/williamleif/GraphSAGE
import numpy as np
import argparse
import json
import os, sys

def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='data set preprocess')
    parser.add_argument('-n', '--name', default="reddit", type=str, help='training dataset name')
    parser.add_argument('-l', '--len', default=0, type=int, help='feature length')
    parser.add_argument('-p', '--path', default="/data/cwj/pagraph/ogb/set", type=str, help='data store path')
    return parser.parse_args(argv)

def read_npy_file(file_path, convert32=False):
    data = np.load(file_path)
    if convert32:
        data = data.astype(np.float32)
    return data

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]



if __name__ == '__main__':
    args = parse_args_func(None)

    directed = {'reddit0': 0} # indicated by reddit-G_full.json
    
    datasetpath = f'./dist/repgnn_data/{args.name}'
    idmap_path = os.path.join(datasetpath, 'reddit-id_map.json') # node name to node id
    src_feats_path = os.path.join(datasetpath, 'reddit-feats.npy') # src node feats
    target_feats_path = os.path.join(datasetpath, 'feat.npy')
    src_labels_path = os.path.join(datasetpath, 'reddit-class_map.json') # src node labels
    target_labels_path = os.path.join(datasetpath, 'labels.npy')
    adj_path = os.path.join(datasetpath, 'reddit-adjlist.txt') # adj
    pp_path = os.path.join(datasetpath, 'pp.txt')


    # generate feats.npy
    # node order is in reddit-id_map.json, feats are in reddit-feats.npy
    idmap = read_json_file(idmap_path)
    keys_order = list(idmap.values())
    feats_array = read_npy_file(src_feats_path, convert32=True)
    np.save(target_feats_path, feats_array[keys_order]) # make feats order 0, 1, 2, ...
    print(f'-> SAVE {target_feats_path} DONE')

    # generate labels.npy
    labels_dict = read_json_file(src_labels_path)
    labels = np.array(list(labels_dict.values()))
    keys2id = [idmap[keyname] for keyname in labels_dict.keys()]
    keys_order = np.array(keys2id)
    np.save(target_labels_path, labels[keys_order]) # make labels order 0, 1, 2, ...
    print(f'-> SAVE {target_labels_path} DONE')

    # generate pp.txt
    adj_lst = read_txt_file(adj_path)
    with open(pp_path, 'w') as f:
        for line_idx in range(len(adj_lst)):
            if line_idx < 3:
                continue
            temp_adj = adj_lst[line_idx].split()
            src_node = temp_adj[0]
            for end_node in temp_adj[1:]:
                f.write(f'{idmap[src_node]}\t{idmap[end_node]}\n')
    print(f'-> SAVE {adj_path} DONE')
    
    sys.exit(directed[args.name])









