import torch
import torch.nn as nn
import dgl
import dgl.function as fn


class NodeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, test=False):
        super(NodeUpdate, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.test = test

    def forward(self, node):
        h = node.data['h']
        if self.test:
            h = h * node.data['norm']
        h = self.linear(h)
        if self.activation:
            h = self.activation(h)
        return {'activation': h}


class GCNSampling(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCNSampling, self).__init__()
        self.n_layers = n_layers
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(NodeUpdate(in_feats, n_hidden, activation))
        # hidden layers
        for i in range(1, n_layers):
            self.layers.append(NodeUpdate(n_hidden, n_hidden, activation))
        # output layer
        self.layers.append(NodeUpdate(n_hidden, n_classes))
        
        # count number of intermediate data
        self.total_comb_size = 0
        self.total_actv_size = 0


    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']
        # 记录一下每层activation结果的size
        actv_size_after_each_block = []
        for i, layer in enumerate(self.layers): # 这里的layers数量就是block数量
            h = nf.layers[i].data.pop('activation')
            if i!=0:
                actv_size_after_each_block.append(h.numel())
            if self.dropout:
                h = self.dropout(h)
            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.mean(msg='m', out='h'),
                             layer)

        h = nf.layers[-1].data.pop('activation')
        
        
        with torch.no_grad():
            # combine_size 包含每个block aggr的结果shape、和w乘积后结果的shape
            # 每个block执行一次comb，后面block同时要迁移前面block的数据量；
            # activation size 包含每个block actv的结果shape
            old_comb_size = 0
            old_actv_size = 0
            nf_nids = nf._node_mapping.tousertensor()
            offsets = nf._layer_offsets # 这里的layer含义不是Block，一个Block包含输入Layer和输出layer
            for blkid, layer in enumerate(self.layers):
                aggr_results = len(nf_nids[offsets[blkid]: offsets[blkid+1]]) * self.layers[blkid].linear.in_features
                tensor_after_combine_and_w = len(nf_nids[offsets[blkid+1]: offsets[blkid+2]])*self.layers[blkid].linear.out_features

                # cur_block_comb_size = aggr_results + tensor_after_combine_and_w
                cur_block_comb_size = tensor_after_combine_and_w
                self.total_comb_size += old_comb_size
                old_comb_size += cur_block_comb_size

                tensor_after_actv = 0
                if layer.activation:
                    tensor_after_actv = actv_size_after_each_block[blkid]
                self.total_actv_size += old_actv_size
                old_actv_size += tensor_after_actv
        curnf_total_comb_size, curnf_total_actv_size = self.total_comb_size, self.total_actv_size
        self.total_comb_size, self.total_actv_size = 0,0
        return h, curnf_total_comb_size, curnf_total_actv_size


class GCNInfer(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation):
        super(GCNInfer, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(NodeUpdate(
            in_feats, n_hidden, activation, test=True))
        # hidden layers
        for i in range(1, n_layers):
            self.layers.append(NodeUpdate(
                n_hidden, n_hidden, activation, test=True))
        # output layer
        self.layers.append(NodeUpdate(n_hidden, n_classes, test=True))

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']

        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'),
                             layer)
        h = nf.layers[-1].data.pop('activation')
        return h
