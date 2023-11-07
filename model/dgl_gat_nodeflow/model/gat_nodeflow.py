import dgl.backend as F
import operator
import torch
import torch.nn as nn
import torch.nn.functional as TorchF
from math import sqrt
import dgl.function as fn

from .attention_block_nodeflow import Attention


class SrcMulEdgeMessageFunction(object):
    """This is a temporary workaround for dgl's built-in srcmuledge message function
    as it is currently incompatible with NodeFlow.
    """
    def __init__(self, src_field, edge_field, out_field):
        self.mul_op = operator.mul
        self.src_field = src_field
        self.edge_field = edge_field
        self.out_field = out_field

    def __call__(self, edges):
        sdata = edges.src[self.src_field]
        edata = edges.data[self.edge_field]
        # Due to the different broadcasting semantics of different backends,
        # we need to broadcast the sdata and edata to be of the same rank.
        rank = max(F.ndim(sdata), F.ndim(edata))
        sshape = F.shape(sdata)
        eshape = F.shape(edata)
        sdata = F.reshape(sdata, sshape + (1,) * (rank - F.ndim(sdata)))
        edata = F.reshape(edata, eshape + (1,) * (rank - F.ndim(edata)))
        ret = self.mul_op(sdata, edata)
        return {self.out_field : ret}

    @property
    def name(self):
        return "src_mul_edge"

    @property
    def use_edge_feature(self):
        """Return true if the message function uses edge feature data."""
        return True

class GraphAttentionNodeFlow(nn.Module):
    def __init__(self,
                 index,
                 in_dim,
                 out_dim,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 alpha=0.2,
                 residual=False,
                 activation=None,
                 aggregate='concat'):
        super(GraphAttentionNodeFlow, self).__init__()
        self.index = index
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)

        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x : x

        self.attention = Attention(self.index, in_dim, out_dim, num_heads, alpha)
        self.ret_bias = nn.Parameter(torch.Tensor(size=(num_heads, out_dim)))
        self.residual = residual
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, num_heads * out_dim)
            else:
                self.res_fc = None

        self.activation = activation
        assert aggregate in ['concat', 'mean']
        self.aggregate = aggregate

        self.__init_params()

    def __init_params(self):
        nn.init.xavier_uniform_(self.fc.weight.data, gain=sqrt((self.in_dim + self.num_heads * self.out_dim)
                                                               /(self.in_dim + self.out_dim)))
        self.ret_bias.data.fill_(0)
        if self.residual and self.res_fc is not None:
            nn.init.xavier_uniform_(self.res_fc.weight.data, gain=sqrt((self.in_dim + self.num_heads * self.out_dim)
                                                                       /(self.in_dim + self.out_dim)))
            self.res_fc.bias.data.fill_(0)

    def forward(self, nf, h):
        # prepare
        # shape reference
        # ---------------
        # V: # nodes, D: input feature size, H: # heads
        # D': out feature size
        h = self.feat_drop(h)                                      # (V, D)
        ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1))  # (V, H, D')

        scores, normalizer = self.attention(nf, ft)
        nf.layers[self.index].data['ft'] = ft
        nf.blocks[self.index].data['a_drop'] = self.attn_drop(scores)
        # nf.block_compute(self.index, SrcMulEdgeMessageFunction('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
        nf.block_compute(self.index, fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))

        # 3. apply normalizer
        ret = nf.layers[self.index + 1].data['ft'] / normalizer
        ret = ret + self.ret_bias

        dst_indices_in_nodeflow = nf.layer_nid(self.index+1)

        nf.layers[self.index].data.pop('ft')
        nf.layers[self.index + 1].data.pop('ft')

        # 4. residual
        if self.residual:
            dst_indices_in_src_layer = nf.map_from_parent_nid(
                self.index, nf.map_to_parent_nid(dst_indices_in_nodeflow)) - nf._layer_offsets[self.index]
            h = h[dst_indices_in_src_layer, :]
            if self.res_fc is not None:
                resval = self.res_fc(h).reshape((h.shape[0], self.num_heads, -1))  # (V, H, D')
            else:
                resval = torch.unsqueeze(h, 1)                                     # (V, 1, D')
            ret = resval + ret

        if self.aggregate == 'concat':
            ret = ret.flatten(1)
        else:
            ret = ret.mean(1)

        if self.activation is not None:
            ret = self.activation(ret)

        return ret

class GATNodeFlow(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 residual,
                 activation=TorchF.elu):
        super(GATNodeFlow, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        # input projection (no residual)
        self.gat_layers.append(GraphAttentionNodeFlow(
            0, in_dim, num_hidden[0], num_heads[0],
            feat_drop, attn_drop, residual=False,
            activation=self.activation, aggregate='concat'))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GraphAttentionNodeFlow(
                l, num_hidden[l-1] * num_heads[l-1], num_hidden[l], num_heads[l],
                feat_drop, attn_drop, residual=residual,
                activation=self.activation, aggregate='concat'))
        # output projection
        self.gat_layers.append(GraphAttentionNodeFlow(
            num_layers, num_hidden[-1] * num_heads[-2], num_classes, num_heads[-1],
            feat_drop, attn_drop, residual=residual,
            activation=None, aggregate='mean'))

    def forward(self, nf):
        h = nf.layers[0].data['features']
        for l in range(self.num_layers + 1):
            h = self.gat_layers[l](nf, h)

        return h
