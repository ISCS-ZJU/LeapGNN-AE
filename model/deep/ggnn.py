"""
Gated Graph Neural Network module for graph classification tasks
"""
import torch

from dgl.nn.pytorch import GatedGraphConv, GlobalAttentionPooling
from torch import nn
import dgl




"""Torch Module for Gated Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name, cell-var-from-loop
import torch as th
from torch import nn
from torch.nn import init

import dgl.function as fn


class GatedGraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 n_steps,
                 bias=True):
        super(GatedGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._n_steps = n_steps
        self.linears = nn.ModuleList(
            [nn.Linear(out_feats, out_feats)]
        )
        self.gru = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = init.calculate_gain('relu')
        self.gru.reset_parameters()
        for linear in self.linears:
            init.xavier_normal_(linear.weight, gain=gain)
            init.zeros_(linear.bias)

    def forward(self, nf):
        feat = nf.layers[0].data['features']
        zero_pad = feat.new_zeros((feat.shape[0], self._out_feats - feat.shape[1]))
        feat = th.cat([feat, zero_pad], -1)
        nf.layers[0].data['activation'] = feat

        for i in range(self._n_steps):
            h = nf.layers[i].data.pop('activation')
            nf.layers[i].data['h'] = h
            nf.apply_block(i,lambda edges: {'W_e*h': self.linears(edges.src['h'])})
            nf.block_compute(i,fn.copy_e('W_e*h', 'm'), fn.sum('m', 'a'))
            a = nf.layers[i].data.pop('a') # (N, D)
            h = self.gru(a, h)
        return h
    

class GlobalAttentionPooling(nn.Module):
    def __init__(self, gate_nn, feat_nn=None):
        super(GlobalAttentionPooling, self).__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn

    def forward(self, nf, graph, feat):
        gate = self.gate_nn(feat)
        assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."
        feat = self.feat_nn(feat) if self.feat_nn else feat
        gate = dgl.backend.backend.softmax(feat, 1)
        
        graph.ndata['r'] = feat * gate
        readout = sum_nodes(graph, 'r')
        graph.ndata.pop('r')

        return readout



class GraphClsGGNN(nn.Module):
    def __init__(self, annotation_size, out_feats, n_steps, num_cls):
        super(GraphClsGGNN, self).__init__()

        self.annotation_size = annotation_size
        self.out_feats = out_feats

        self.ggnn = GatedGraphConv(
            in_feats=out_feats,
            out_feats=out_feats,
            n_steps=n_steps,
        )

        pooling_gate_nn = nn.Linear(annotation_size + out_feats, 1)
        self.pooling = GlobalAttentionPooling(pooling_gate_nn)
        self.output_layer = nn.Linear(annotation_size + out_feats, num_cls)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, nf, labels=None):
        annotation = nf.layers[0].data['features'].float()

        assert annotation.size()[-1] == self.annotation_size

        # node_num = nf.layer_size(0)

        # zero_pad = torch.zeros(
        #     [node_num, self.out_feats - self.annotation_size],
        #     dtype=torch.float,
        #     device=annotation.device,
        # )

        # h1 = torch.cat([annotation, zero_pad], -1)
        out = self.ggnn(nf)

        out = torch.cat([out, annotation], -1)

        out = self.pooling(nf, out)

        logits = self.output_layer(out)
        preds = torch.argmax(logits, -1)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, preds
        return preds