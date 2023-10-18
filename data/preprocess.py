"""
Preprocess dataset to fit the input
"""

import numpy as np
import scipy.sparse
import os
import sys
import argparse

def pp2adj(filepath, is_direct=True, delimiter='\t',
           outfile=None):
  """
  Convert (vertex vertex) tuple into numpy adj matrix
  adj matrix will be returned.
  If outfile is provided, also save it.
  """
  pp = np.loadtxt(filepath, delimiter=delimiter)
  src_node = pp[:,0].astype(np.int64)
  dst_node = pp[:,1].astype(np.int64)
  max_nid = max(np.max(src_node), np.max(dst_node))
  min_nid = min(np.min(src_node), np.min(dst_node))
  print('=> max_nid:{}'.format(max_nid), 'min_nid:{}'.format(min_nid))

  # get vertex and ege num info
  vnum = max_nid - min_nid + 1
  enum = len(src_node) if is_direct else len(src_node) * 2
  print('=> # of vertex: {} # of edges: {}'.format(vnum, enum))

  # scale node id from 0
  src_node -= min_nid
  dst_node -= min_nid

  # make coo sparse adj matrix
  if not is_direct:
    src_node, dst_node = np.concatenate((src_node, dst_node)), \
                         np.concatenate((dst_node, src_node))
  edge_weight = np.ones(enum, dtype=np.int64)
  coo_adj = scipy.sparse.coo_matrix(
    (edge_weight, (src_node, dst_node)),
    shape=(vnum, vnum)
  )
  # output to file
  if outfile is not None:
    scipy.sparse.save_npz(outfile, coo_adj)
  print(f'=> Saving adj.npz done into {outfile}')
  return coo_adj


def random_multi_file_feature(vnum, feat_size, part_num, dataset,part_type):
  """
  Generate random features using numpy
  Params:
    vnum:       feature num (aka. vertex num)
    feat_size:  feature dimension 
    outfile:    save to the file if provided
  Returns:
    numpy array obj with shape of [vnum, feat_size]
  """
  rng = np.random.default_rng()
  for i in range(part_num):
    part_file = os.path.join(dataset,f'dist_True/{part_num}_{part_type}/{i}.npy')
    node_num = np.load(part_file).shape[0]
    outfile = os.path.join(dataset, f'feat_{part_type}/feat{i}.npy')
    feat_mat = rng.random(size=(node_num, feat_size), dtype=np.float32)
    # feat_mat = np.random.random((node_num, feat_size)).astype(np.float32)
    if outfile:
      np.save(outfile, feat_mat)
  return feat_mat

def random_p3_feature(vnum, feat_size, part_num, dataset):
  """
  Generate random features using numpy
  Params:
    vnum:       feature num (aka. vertex num)
    feat_size:  feature dimension 
    outfile:    save to the file if provided
  Returns:
    numpy array obj with shape of [vnum, feat_size]
  """
  sub_feat_size = [int(feat_size / part_num) for _ in range(part_num - 1)]
  sub_feat_size.append(feat_size - (part_num - 1)*int(feat_size / part_num))
  rng = np.random.default_rng()
  for i in range(part_num):
    outfile = os.path.join(dataset, f'feat_p3/p3_feat{i}.npy')
    feat_mat = rng.random(size=(vnum, sub_feat_size[i]), dtype=np.float32)
    # feat_mat = np.random.random((node_num, feat_size)).astype(np.float32)
    if outfile:
      np.save(outfile, feat_mat)
  return feat_mat

def random_feature(vnum, feat_size, outfile=None):
  """
  Generate random features using numpy
  Params:
    vnum:       feature num (aka. vertex num)
    feat_size:  feature dimension 
    outfile:    save to the file if provided
  Returns:
    numpy array obj with shape of [vnum, feat_size]
  """
  feat_mat = np.random.random((vnum, feat_size)).astype(np.float32)
  if outfile:
    np.save(outfile, feat_mat)
  return feat_mat

def random_label(vnum, class_num, outfile=None):
  """
  Generate random labels from 0 - class_num for each node
  Params:
    vnum:       total node num
    class_num:  number of classes, start from 0
    outfile:    save to the file if provided
  Returns:
    numpy array obj with shape of (vnum,).
    Each element denotes corresponding nodes labels
  """
  labels = np.random.randint(class_num, size=vnum)
  if outfile:
    np.save(outfile, labels)
  return labels


def split_dataset(vnum, outdir=None):
  """
  Split dataset to train/val/test.
  train:val:test = 6.5:1:1.5 - similar to reddit
  if outdir is provided:
    save as outdir/train.npy,
            outdir/val.npy,
            outdir/test.npy
  Return:
    3 ndarrays with train, val, test mask.
    All of them is of (vnum,) size with 0 or 1 indicator. 
  """
  nids = np.arange(vnum)
  np.random.shuffle(nids)
  train_len = int(vnum * 0.65)
  val_len = int(vnum * 0.1)
  test_len = vnum - train_len - val_len
  print(f'\n=> Split dataset with train_len = {train_len}, val_len = {val_len}, test_len = {test_len}\n')
  # train mask
  train_mask = np.zeros(vnum, dtype=np.int64)
  train_mask[nids[0:train_len]] = 1
  # val mask
  val_mask = np.zeros(vnum, dtype=np.int64)
  val_mask[nids[train_len:train_len + val_len]] = 1
  # test mask
  test_mask = np.zeros(vnum, dtype=np.int64)
  test_mask[nids[-test_len:]] = 1
  # save
  if outdir is not None:
    np.save(os.path.join(outdir, 'train.npy'), train_mask)
    np.save(os.path.join(outdir, 'val.npy'), val_mask)
    np.save(os.path.join(outdir, 'test.npy'), test_mask)
  return train_mask, val_mask, test_mask


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Preprocess')

  parser.add_argument("--dataset", type=str, default=None,
                      help="dataset dir")

  parser.add_argument("--ppfile", type=str, default=None,
                      help='point-to-point graph filename')
  parser.add_argument("--directed", dest="directed", action='store_true')
  parser.set_defaults(directed=False)

  parser.add_argument("--gen-feature", dest='gen_feature', action='store_true')
  parser.set_defaults(gen_feature=False)
  parser.add_argument("--feat-multi-file", dest='feat_multi_file', action='store_true')
  parser.set_defaults(feat_multi_file=False)
  parser.add_argument("--p3-feature", dest='p3_feature', action='store_true')
  parser.set_defaults(p3_feature=False)
  parser.add_argument("--feat-size", type=int, default=600,
                      help='generated feature size if --gen-feature is specified')
  parser.add_argument("--part-num", type=int, default=4,
                      help='partition num of graph')
  parser.add_argument("--part-type", type=str, choices=['metis','pagraph'],default='pagraph',
                      help='partition type of graph(metis,pagraph)')
  
  parser.add_argument("--gen-label", dest='gen_label', action='store_true')
  parser.set_defaults(gen_label=False)
  parser.add_argument("--class-num", type=int, default=60,
                      help='generated class number if --gen-label is specified')
  
  parser.add_argument("--gen-set", dest='gen_set', action='store_true')
  parser.add_argument("--seed", type=int, default=2024)

  parser.set_defaults(gen_set=False)
  args = parser.parse_args()

  np.random.seed(args.seed)

  if not os.path.exists(args.dataset):
    # print('{}: No such a dataset folder'.format(args.dataset))
    # sys.exit(-1)
    os.mkdir(args.dataset)
    print(f'=> Created folder {args.dataset} to store results.')
  
  # generate adj
  adj_file = os.path.join(args.dataset, 'adj.npz')
  if args.ppfile is not None:
    print('=> Generating adj matrix according pp.txt in: {}...'.format(adj_file))
    adj = pp2adj(
      os.path.join(args.dataset, args.ppfile),
      is_direct=args.directed,
      outfile=adj_file
    )
  else:
    adj = scipy.sparse.load_npz(adj_file)
    print(f'=> Load adj_file done')
  vnum = adj.shape[0]
  del adj

  # generate features
  feat_file = os.path.join(args.dataset, 'feat.npy')
  if args.gen_feature:
    print('=> Generating random features (size: {}) in: {}...'.format(args.feat_size, feat_file))
    if args.feat_multi_file:
      if args.p3_feature:
        feat = random_p3_feature(vnum, args.feat_size, args.part_num, args.dataset)
      else:
        feat = random_multi_file_feature(vnum, args.feat_size, args.part_num, args.dataset,args.part_type)
    else:
      feat = random_feature(vnum, args.feat_size, outfile=feat_file)
  
  # generate labels
  label_file = os.path.join(args.dataset, 'labels.npy')
  if args.gen_label:
    print('=> Generating labels (class num: {}) in: {}...'.format(args.class_num, feat_file))
    labels = random_label(vnum, args.class_num,
                          outfile=label_file)
  
  # generate train/val/test set
  if args.gen_set:
    print('=> Generating train/val/test masks in: {}...'.format(args.dataset))
    split_dataset(vnum, outdir=args.dataset)
  
  print('Done.')