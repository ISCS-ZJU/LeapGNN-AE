import argparse
import os

import dgl
import dgl.function as fn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


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


class P3_GNNFiLM(nn.Module):
    def __init__(
        self, in_size, hidden_size, out_size, num_layers, dropout=0.1
    ):
        super(P3_GNNFiLM, self).__init__()
        self.film_layers = nn.ModuleList()
        self.film_layers.append(
            GNNFiLMLayer(in_size, hidden_size, dropout)
        )
        for _ in range(num_layers - 1):
            self.film_layers.append(
                GNNFiLMLayer(hidden_size, hidden_size, dropout)
            )

    def forward(self, nfs, rank):
        # nf.layers[0].data['activation'] = nf.layers[0].data['features']
        mp_out = []
        for nf in nfs:
            h = nf.layers[0].data['features']
            h = self.film_layers[0](nf, h, 0)
            mp_out.append(h.clone())

        nf = nfs[rank]
        num = len(mp_out)
        with torch.autograd.profiler.record_function('all_reduce hidden vectors'):
            for i in range(num):
                dist.all_reduce(mp_out[i],dist.ReduceOp.SUM)

        x = mp_out[rank]
        for i, layer in enumerate(self.film_layers):
            if i == 0:
                continue
            x = layer(nf, x, i)
        x = torch.sigmoid(x)
        return x
