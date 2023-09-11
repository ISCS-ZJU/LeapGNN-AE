import scipy.sparse as spsp
import numpy as np
import os

if __name__ == '__main__':
    dataset = './dist/repgnn_data/twitter'
    name = 'twitter-2010'
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