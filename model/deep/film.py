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

    def forward(self, nf):
        # nf.layers[0].data['activation'] = nf.layers[0].data['features']
        h = nf.layers[0].data['features']
        # h_dict = {
        #     ntype: g.nodes[ntype].data["feat"] for ntype in g.ntypes
        # }  # prepare input feature dict
        # h = nf.layers[i].data.pop('activation')
        for i, layer in enumerate(self.film_layers):
            h = layer(nf, h, i)
            # h = h[nf.map_from_parent_nid(i,nf.layer_parent_nid(i+1),remap_local=True)]
        h = self.predict(
            h
        )  # use the final embed to predict, out_size = num_classes
        h = torch.sigmoid(h)
        return h
