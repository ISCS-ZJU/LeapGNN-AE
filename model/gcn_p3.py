import torch
import torch.nn as nn
import torch.distributed as dist
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


class P3_GCNSampling(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(P3_GCNSampling, self).__init__()
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

    def forward(self, nfs, rank):
        mp_out = []
        for nf in nfs:
            nf.layers[0].data['activation'] = nf.layers[0].data['features']
            h = nf.layers[0].data.pop('activation')
            if self.dropout:
                h = self.dropout(h)
            nf.layers[0].data['h'] = h
            nf.block_compute(0,
                                fn.copy_src(src='h', out='m'),
                                fn.mean(msg='m', out='h'),
                                self.layers[0])
            h = nf.layers[1].data.pop('activation')
            mp_out.append(h.clone())

        nf = nfs[rank]
        num = len(mp_out)
        # out = list(torch.empty([num], dtype=mp_out[0].dtype).chunk(num))
        # dist.all_to_all(out,mp_out)
        # x = torch.sum(out)
        for i in range(num):
            dist.all_reduce(mp_out[i],dist.ReduceOp.SUM)
        x = mp_out[rank]
        nf.layers[1].data['activation'] = x

        for i, layer in enumerate(self.layers):
            if i == 0:
                continue
            h = nf.layers[i].data.pop('activation')
            if self.dropout:
                h = self.dropout(h)
            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                            fn.copy_src(src='h', out='m'),
                            fn.mean(msg='m', out='h'),
                            layer)

        h = nf.layers[-1].data.pop('activation')
        return h


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
