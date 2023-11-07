import dgl.function as fn
import torch
import torch.nn as nn
from math import sqrt

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
