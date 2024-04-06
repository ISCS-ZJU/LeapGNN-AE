import argparse
import os

import dgl
import dgl.function as fn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNFiLMLayer(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.1):
        super(GNNFiLMLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        # weights for different types of edges
        self.W = nn.Linear(in_size, out_size, bias=False)

        # hypernets to learn the affine functions for different types of edges
        self.film = nn.Linear(in_size, 2 * out_size, bias=False)

        # layernorm before each propogation
        self.layernorm = nn.LayerNorm(out_size)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, nf, feat_dict, i):
        # the input graph is a multi-relational graph, so treated as hetero-graph.

        # for each type of edges, compute messages and reduce them all
        messages = self.W(
            feat_dict
        )  # apply W_l on src feature
        film_weights = self.film(
            feat_dict
        )  # use dst feature to compute affine function paras
        gamma = film_weights[
            :, : self.out_size
        ]  # "gamma" for the affine function
        beta = film_weights[
            :, self.out_size :
        ]  # "beta" for the affine function
        messages = gamma * messages + beta  # compute messages
        messages = F.relu_(messages)
        nf.layers[i].data['message'] = messages  # store in ndata
        nf.block_compute(i,fn.copy_u('message', "m"),
                fn.sum("m", "h"),)
        feat_dict = {}
        feat_dict = self.dropout(
            self.layernorm(nf.layers[i+1].data["h"])
        )  # apply layernorm and dropout
        return feat_dict


class GNNFiLM(nn.Module):
    def __init__(
        self, in_size, hidden_size, out_size, num_layers, dropout=0.1
    ):
        super(GNNFiLM, self).__init__()
        self.film_layers = nn.ModuleList()
        self.film_layers.append(
            GNNFiLMLayer(in_size, hidden_size, dropout)
        )
        for _ in range(num_layers - 1):
            self.film_layers.append(
                GNNFiLMLayer(hidden_size, hidden_size, dropout)
            )
        self.predict = nn.Linear(hidden_size, out_size, bias=True)

        # count number of intermediate data
        self.total_comb_size = 0
        self.total_actv_size = 0

    def forward(self, nf):
        # 记录一下每层activation结果的size
        actv_size_after_each_block = []
        # nf.layers[0].data['activation'] = nf.layers[0].data['features']
        h = nf.layers[0].data['features']
        # h_dict = {
        #     ntype: g.nodes[ntype].data["feat"] for ntype in g.ntypes
        # }  # prepare input feature dict
        # h = nf.layers[i].data.pop('activation')
        for i, layer in enumerate(self.film_layers):
            h = layer(nf, h, i)
            actv_size_after_each_block.append(h.numel())
            # h = h[nf.map_from_parent_nid(i,nf.layer_parent_nid(i+1),remap_local=True)]
        h = self.predict(
            h
        )  # use the final embed to predict, out_size = num_classes
        h = torch.sigmoid(h)

        with torch.no_grad():
            # combine_size 包含每个block aggr的结果shape、和w乘积后结果的shape
            # 每个block执行一次comb，后面block同时要迁移前面block的数据量；
            # activation size 包含每个block actv的结果shape
            old_comb_size = 0
            old_actv_size = 0
            nf_nids = nf._node_mapping.tousertensor()
            offsets = nf._layer_offsets # 这里的layer含义不是Block，一个Block包含输入Layer和输出layer
            for blkid, layer in enumerate(self.film_layers):
                # aggr_results = len(nf_nids[offsets[blkid]: offsets[blkid+1]]) * self.layers[blkid].linear.in_features
                tensor_after_combine_and_w = len(nf_nids[offsets[blkid+1]: offsets[blkid+2]])*self.film_layers[blkid].out_size

                # cur_block_comb_size = aggr_results + tensor_after_combine_and_w
                cur_block_comb_size = tensor_after_combine_and_w
                self.total_comb_size += old_comb_size
                old_comb_size += cur_block_comb_size

                tensor_after_actv = 0
                tensor_after_actv = actv_size_after_each_block[blkid]
                self.total_actv_size += old_actv_size
                old_actv_size += tensor_after_actv
        curnf_total_comb_size, curnf_total_actv_size = self.total_comb_size, self.total_actv_size
        self.total_comb_size, self.total_actv_size = 0,0
        return h, curnf_total_comb_size, curnf_total_actv_size
