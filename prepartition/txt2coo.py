import scipy.sparse as spsp
import numpy as np
import os
import argparse

def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='txt2coo')
    parser.add_argument('-d', '--dataset', default="./dist/repgnn_data/uk-2007",
                        type=str, help='training dataset path')
    parser.add_argument('-n', '--name', default="in-2004",
                        type=str, help='training dataset name')
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_args_func(None)
    dataset = args.dataset
    name = args.name
    row = []
    col = []
    i = 0
    with open(os.path.join(dataset,f'{name}.graph-txt'),'r') as f:
        line = f.readline()
        while line:
            line = line.split()
            for cid in line:
                row.append(i)
                col.append(cid)
            line = f.readline()
            i = i + 1
    print(i,len(row))
    row = np.array(row,dtype=np.int32)
    col = np.array(col,dtype=np.int32)
    data = np.ones_like(col)
    adj = spsp.coo_matrix((data,(row,col)),shape=(i,i))
    spsp.save_npz(os.path.join(dataset,'adj.npz'), adj)