import numpy as np
import os
import torch
from dgl import DGLGraph

def mkdir_p(path):
    import errno
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise

def load_reddit():
    from dgl.data import RedditDataset

    database = {}
    reddit = RedditDataset(False)

    database['train_nid'] = torch.LongTensor(np.nonzero(reddit.train_mask)[0])
    database['val_nid'] = torch.LongTensor(np.nonzero(reddit.val_mask)[0])
    database['test_nid'] = torch.LongTensor(np.nonzero(reddit.test_mask)[0])

    features = torch.FloatTensor(reddit.features)
    labels = torch.LongTensor(reddit.labels)
    database['train_mask'] = torch.ByteTensor(reddit.train_mask.astype(np.uint8))
    database['val_mask'] = torch.ByteTensor(reddit.val_mask.astype(np.uint8))
    database['test_mask'] = torch.ByteTensor(reddit.test_mask.astype(np.uint8))

    database['n_feats'] = features.shape[1]
    database['n_classes'] = reddit.num_labels

    database['n_train_samples'] = database['train_mask'].sum().item()
    database['n_val_samples'] = database['val_mask'].sum().item()
    database['n_test_samples'] = database['test_mask'].sum().item()

    g = DGLGraph(reddit.graph, readonly=True)
    g.ndata['features'] = features
    g.ndata['labels'] = labels
    database['g'] = g

    return database