import scipy.sparse as spsp
import numpy as np
import os

root = '/data/repgnn/'
name = 'ogbn_papers100M0'
dataset = os.path.join(root,f'{name}')
adj_path = os.path.join(dataset,'adj_bak.npz')
matrix = spsp.load_npz(adj_path)
matrix = matrix.tocsr()
with open(os.path.join(dataset,),'w') as f:
    