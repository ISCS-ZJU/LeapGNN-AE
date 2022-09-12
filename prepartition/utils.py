import os
import sys
import dgl
from dgl import DGLGraph
import torch
import numpy as np
import scipy.sparse as spsp

def Print(*content, debug = False):
  if debug:
    print(*content)

def get_sub_graph(dgl_g, train_nid, num_hops):
  Print()
  Print('part_onid', train_nid)
  nfs = []
  for nf in dgl.contrib.sampling.NeighborSampler(dgl_g, len(train_nid),
                                                 expand_factor = dgl_g.number_of_nodes(), # 这里是为了切分图，因此每个node id扩展了全部的邻居节点数
                                                #  expand_factor = 1, # 多个邻居中只采样一个
                                                 neighbor_type='in',
                                                 shuffle=False,
                                                 num_workers=16,
                                                 num_hops=num_hops,
                                                 seed_nodes=train_nid,
                                                 prefetch=False):
    nfs.append(nf)
  
  assert(len(nfs) == 1) # 因为sample的时候batch size就是1，因此只有一个batch被sample出来
  nf = nfs[0]
  full_edge_src = []
  full_edge_dst = []
  for i in range(nf.num_blocks):
    nf_src_nids, nf_dst_nids, _ = nf.block_edges(i, remap_local=False) # nodeflow-level id
    full_edge_src.append(nf.map_to_parent_nid(nf_src_nids)) # map to parent id == onid
    full_edge_dst.append(nf.map_to_parent_nid(nf_dst_nids))
  full_srcs = torch.cat(tuple(full_edge_src)).numpy()
  full_dsts = torch.cat(tuple(full_edge_dst)).numpy()
  Print("full_srcs onid", full_srcs)
  Print("full_dsts onid", full_dsts)
  # set up mappings
  sub2full = np.unique(np.concatenate((full_srcs, full_dsts)))
  Print("lnid->onid", sub2full) # 例如 [0 1 5 6 8] 表示当前Batch对应的sub graph中的0对应全局的0，2对应全局的5
  full2sub = np.zeros(np.max(sub2full) + 1, dtype=np.int64)
  full2sub[sub2full] = np.arange(len(sub2full), dtype=np.int64)
  Print("onid->lnid", full2sub) # 对应的映射：[0 1 0 0 0 2 3 0 4]
  # map to sub graph space
  sub_srcs = full2sub[full_srcs]
  Print("src_lnid", sub_srcs)
  sub_dsts = full2sub[full_dsts]
  Print("dst_lnid", sub_dsts)
  vnum = len(sub2full)
  enum = len(sub_srcs)
  data = np.ones(sub_srcs.shape[0], dtype=np.uint8) # 子图边上的数据都填充1
  coo_adj = spsp.coo_matrix((data, (sub_srcs, sub_dsts)), shape=(vnum, vnum)) # 使用子图的local id进行构建graph
  csr_adj = coo_adj.tocsr() # remove redundant edges
  enum = csr_adj.data.shape[0]
  csr_adj.data = np.ones(enum, dtype=np.uint8)
  Print('vertex#: {} edge#: {}'.format(vnum, enum)) # 去重后的点和边的数量，即子图中实际上的点和边的数量
  # train nid
  tnid = nf.layer_parent_nid(-1).numpy() # 最后一层，即train_nid层在原图上的onid
  Print('train_onid:', tnid)
  valid_t_max = np.max(sub2full)
  valid_t_min = np.min(tnid)
  tnid = np.where(tnid <= valid_t_max, tnid, valid_t_min)
  subtrainid = full2sub[np.unique(tnid)]
  Print(valid_t_max, valid_t_min, tnid, subtrainid)
  return csr_adj, sub2full, subtrainid


def node2graph(fulladj, nodelist, train_nids):
  g = dgl.DGLGraph(fulladj)
  subg = g.subgraph(nodelist)
  sub2full = subg.parent_nid.numpy()
  subadj = subg.adjacency_matrix_scipy(transpose=True, return_edge_ids=False)
  # get train vertices under subgraph scope
  subtrain = subg.map_to_subgraph_nid(train_nids).numpy()
  return subadj, sub2full, subtrain