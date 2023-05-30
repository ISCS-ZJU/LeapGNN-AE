#  reference code from dgl==0.6.1 dgl.data.RedditDataset()
from tabnanny import filename_only
import numpy as np
import argparse
import json
import os, sys

import requests
import scipy.sparse as sp

def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='data set preprocess')
    parser.add_argument('-n', '--name', default="reddit", type=str, help='training dataset name')
    parser.add_argument('-l', '--len', default=0, type=int, help='feature length')
    parser.add_argument('-p', '--path', default="/data/cwj/pagraph/ogb/set", type=str, help='data store path')
    return parser.parse_args(argv)

# def read_npy_file(file_path, convert32=False):
#     data = np.load(file_path)
#     if convert32:
#         data = data.astype(np.float32)
#     return data

# def read_json_file(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data

# def read_txt_file(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#     return [line.strip() for line in lines]



def extract_archive(file, target_dir, overwrite=False):
    extracted_files = [os.path.join(target_dir, 'reddit_data.npz'), os.path.join(target_dir, 'reddit_graph.npz')]
    if os.path.exists(extracted_files[0]) and os.path.exists(extracted_files[1]):
        return
    print('Extracting file to {}'.format(target_dir))
    if file.endswith('.tar.gz') or file.endswith('.tar') or file.endswith('.tgz'):
        import tarfile
        with tarfile.open(file, 'r') as archive:
            archive.extractall(path=target_dir)
    elif file.endswith('.gz'):
        import gzip
        import shutil
        with gzip.open(file, 'rb') as f_in:
            with open(file[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    elif file.endswith('.zip'):
        import zipfile
        with zipfile.ZipFile(file, 'r') as archive:
            archive.extractall(path=target_dir)
    else:
        raise Exception('Unrecognized file type: ' + file)

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


def process(target_dir):
    # graph
    output_adj = os.path.join(target_dir, 'adj.npz')
    coo_adj = sp.load_npz(os.path.join(target_dir, "reddit_graph.npz"))
    sp.save_npz(output_adj, coo_adj)

    # features and labels
    reddit_data = np.load(os.path.join(target_dir, "reddit_data.npz"))
    features = reddit_data["feature"].astype(np.float32)
    labels = reddit_data["label"].astype(np.int64)
    print(features.shape, labels.shape)
    output_feat = os.path.join(target_dir, 'feat.npy')
    output_labels = os.path.join(target_dir, 'labels.npy')
    np.save(output_feat, features)
    np.save(output_labels, labels)

    # tarin/val/test indices
    node_types = reddit_data["node_types"]
    train_mask = (node_types == 1)
    val_mask = (node_types == 2)
    test_mask = (node_types == 3)
    output_train = os.path.join(target_dir, 'train.npy')
    output_val = os.path.join(target_dir, 'val.npy')
    output_test = os.path.join(target_dir, 'test.npy')
    np.save(output_train, train_mask)
    np.save(output_val, val_mask)
    np.save(output_test, test_mask)



if __name__ == '__main__':
    args = parse_args_func(None)
    # directed = {'reddit0': 0} # indicated by reddit-G_full.json
    
    reddit_dgl_link = 'https://data.dgl.ai/dataset/reddit.zip'
    target_dir = os.path.join(args.path, args.name)
    
    # download and extract into target_dir
    filepath = download_file(reddit_dgl_link, target_dir)
    extract_archive(filepath, target_dir, overwrite=True)
    process(target_dir)









