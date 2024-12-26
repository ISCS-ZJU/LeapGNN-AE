import dgl.backend as F
import operator
import torch
import torch.nn as nn
import torch.nn.functional as TorchF
from math import sqrt
import dgl.function as fn
import torch.distributed as dist

__all__ = ['Attention']

def get_ndata_name(nf, index, name):
    """Return a node data name that does not exist in the given layer of the nodeflow.
    The given name is directly returned if it does not exist in the given graph.
    Parameters
    ----------
    nf : NodeFlow
    index : int
        Nodeflow layer index.
    name : str
        The proposed name.
    Returns
    -------
    str
        The node data name that does not exist.
    """
    while name in nf.layers[index].data:
        name += '_'
    return name

def get_edata_name(nf, index, name):
    """Return an edge data name that does not exist in the given block of the nodeflow.
    The given name is directly returned if it does not exist in the given graph.
    Parameters
    ----------
    nf : NodeFlow
    index : int
        Source layer index of the target nodeflow block.
    name : str
        The proposed name.
    Returns
    -------
    str
        The node data name that does not exist.
    """
    while name in nf.blocks[index].data:
        name += '_'
    return name

class EdgeSoftmaxNodeFlow(nn.Module):
    r"""Apply softmax over signals of incoming edges from layer l to
    layer l+1.
    For a node :math:`i`, edgesoftmax is an operation of computing
    .. math::
      a_{ij} = \frac{\exp(z_{ij})}{\sum_{j\in\mathcal{N}(i)}\exp(z_{ij})}
    where :math:`z_{ij}` is a signal of edge :math:`j\rightarrow i`, also
    called logits in the context of softmax. :math:`\mathcal{N}(i)` is
    the set of nodes that have an edge to :math:`i`.
    """
    def __init__(self, index):
        super(EdgeSoftmaxNodeFlow, self).__init__()

        # Index of the related nodeflow block, also the index of the source layer.
        self.index = index
        self._logits_name = "_logits"
        self._max_logits_name = "_max_logits"
        self._normalizer_name = "_norm"

    def forward(self, nf, logits):
        r"""Compute edge softmax.
        Parameters
        ----------
        nf : NodeFlow
        logits : torch.Tensor
            The input edge feature
        Returns
        -------
        Unnormalized scores : torch.Tensor
            This part gives :math:`\exp(z_{ij})`'s
        Normalizer : torch.Tensor
            This part gives :math:`\sum_{j\in\mathcal{N}(i)}\exp(z_{ij})`
        Notes
        -----
            * Input shape: :math:`(N, *, 1)` where * means any number of additional
              dimensions, :math:`N` is the number of edges.
            * Unnormalized scores shape: :math:`(N, *, 1)` where all but the last
              dimension are the same shape as the input.
            * Normalizer shape: :math:`(M, *, 1)` where :math:`M` is the number of
              nodes and all but the first and the last dimensions are the same as
              the input.
        """
        self._logits_name = get_edata_name(nf, self.index, self._logits_name)
        self._max_logits_name = get_ndata_name(nf, self.index + 1, self._max_logits_name)
        self._normalizer_name = get_ndata_name(nf, self.index + 1, self._normalizer_name)

        nf.blocks[self.index].data[self._logits_name] = logits

        # compute the softmax
        nf.block_compute(self.index, fn.copy_edge(self._logits_name, self._logits_name),
                         fn.max(self._logits_name, self._max_logits_name))
        # minus the max and exp
        nf.apply_block(self.index, lambda edges: {
            self._logits_name : torch.exp(edges.data[self._logits_name] - edges.dst[self._max_logits_name])})

        # pop out temporary feature _max_logits, otherwise get_ndata_name could have huge overhead
        nf.layers[self.index + 1].data.pop(self._max_logits_name)
        # compute normalizer
        nf.block_compute(self.index, fn.copy_edge(self._logits_name, self._logits_name),
                         fn.sum(self._logits_name, self._normalizer_name))

        return nf.blocks[self.index].data.pop(self._logits_name), \
               nf.layers[self.index + 1].data.pop(self._normalizer_name)

    def __repr__(self):
        return 'EdgeSoftmax()'

class Attention(nn.Module):
    def __init__(self,
                 index,
                 in_dim,
                 out_dim,
                 num_heads,
                 alpha=0.2,
                 src_atten_attr='a1',
                 dst_atten_attr='a2',
                 atten_attr='a'):
        super(Attention, self).__init__()
        self.index = index

        self.src_atten_attr = src_atten_attr
        self.dst_atten_attr = dst_atten_attr
        self.atten_attr = atten_attr

        self.num_heads = num_heads
        self.out_dim = out_dim
        self.attn_l = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
        self.attn_r = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
        self.attn_bias = nn.Parameter(torch.Tensor(size=(num_heads, 1)))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = EdgeSoftmaxNodeFlow(index)

        self.__init_params()

    def __init_params(self):
        nn.init.xavier_uniform_(self.attn_l.data, gain=sqrt((self.num_heads + self.out_dim)/(self.out_dim + 1)))
        nn.init.xavier_uniform_(self.attn_r.data, gain=sqrt((self.num_heads + self.out_dim)/(self.out_dim + 1)))
        self.attn_bias.data.fill_(0)

    def forward(self, nf, projected_feats):
        """
        This is the variant used in the original GAT paper.
        Shape reference
        ---------------
        V - # nodes, D - input feature size,
        H - # heads, D' - out feature size
        Parameters
        ----------
        nf : dgl.NodeFlow
        projected_feats : torch.tensor of shape (V, H, D')
        Returns
        -------
        scores: torch.tensor of shape (# edges, # heads, 1)
        normalizer: torch.tensor of shape (# nodes, # heads, 1)
        """
        projected_feats = projected_feats.transpose(0, 1)                  # (H, V, D')
        a1 = torch.bmm(projected_feats, self.attn_l).transpose(0, 1)       # (V, H, 1)

        dst_indices_in_nodeflow = nf.layer_nid(self.index+1)
        dst_indices_in_src_layer = nf.map_from_parent_nid(
            self.index, nf.map_to_parent_nid(dst_indices_in_nodeflow)) - nf._layer_offsets[self.index]
        a2 = torch.bmm(projected_feats[:, dst_indices_in_src_layer, :], self.attn_r).transpose(0, 1)

        nf.layers[self.index].data[self.src_atten_attr] = a1
        nf.layers[self.index + 1].data[self.dst_atten_attr] = a2

        # nf.apply_block(self.index, func=self.edge_attention, edges=nf.block_eid(self.index))
        nf.apply_block(self.index, func=self.edge_attention)
        nf.layers[self.index].data.pop(self.src_atten_attr)
        nf.layers[self.index + 1].data.pop(self.dst_atten_attr)
        return self.softmax(nf, nf.blocks[self.index].data[self.atten_attr])

    def edge_attention(self, edges):
        # an edge UDF to compute unnormalized attention values from src and dst
        a = self.leaky_relu(edges.src[self.src_atten_attr] + edges.dst[self.dst_atten_attr] + self.attn_bias)
        return {self.atten_attr : a}


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

class P3_GATSampling(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 residual=False,
                 ):
        super(P3_GATSampling, self).__init__()
        self.num_layers = n_layers
        self.num_heads = heads
        self.num_hidden = n_hidden
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        # input projection (no residual)
        self.gat_layers.append(GraphAttentionNodeFlow(
            0, in_feats, n_hidden, heads[0],
            feat_drop, attn_drop, residual=False,
            activation=self.activation, aggregate='concat'))
        # hidden layers
        for l in range(1, n_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GraphAttentionNodeFlow(
                l, n_hidden * heads[l-1], n_hidden, heads[l],
                feat_drop, attn_drop, residual=residual,
                activation=self.activation, aggregate='concat'))
        # output projection
        self.gat_layers.append(GraphAttentionNodeFlow(
            n_layers, n_hidden * heads[-2], n_classes, heads[-1],
            feat_drop, attn_drop, residual=residual,
            activation=None, aggregate='mean'))

    def forward(self, nfs, rank):
        mp_out = []
        for nf in nfs:
            h = nf.layers[0].data['features']
            h = self.gat_layers[0](nf,h).flatten(1)
            mp_out.append(h.clone())
        num = len(mp_out)
        with torch.autograd.profiler.record_function('all_reduce hidden vectors'):
            for i in range(num):
                dist.all_reduce(mp_out[i],dist.ReduceOp.SUM)

        h = mp_out[rank]
        nf = nfs[rank]
        for l in range(1, self.num_layers + 1):
            h = self.gat_layers[l](nf, h)

        return h