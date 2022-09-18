import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import torch
import dgl.function as fn


class NodeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, test=False, concat=False):
        super(NodeUpdate, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.concat = concat
        self.test = test

    def forward(self, node):
        h = node.data['h']  # 输入数据在data属性的'h'标签值中
        if self.test:
            h = h * node.data['norm']
        h = self.linear(h)  # 首先经过全连接层
        # skip connection
        if self.concat:
            # 把activation后的h和h连接起来
            h = torch.cat((h, self.activation(h)), dim=1)
        elif self.activation:
            h = self.activation(h)
        return {'activation': h}  # 返回点的activation值


class GCNSampling(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,  # 输出神经元个数
                 n_layers,
                 activation,  # 激活函数
                 dropout):
        super(GCNSampling, self).__init__()
        self.n_layers = n_layers
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.layers = nn.ModuleList()
        # input layer
        skip_start = (0 == n_layers-1)
        self.layers.append(NodeUpdate(in_feats, n_hidden,
                           activation, concat=skip_start))
        # hidden layers
        for i in range(1, n_layers):
            skip_start = (i == n_layers-1)  # 倒数第二层的concat=True
            self.layers.append(NodeUpdate(
                n_hidden, n_hidden, activation, concat=skip_start))
        # output layer
        self.layers.append(NodeUpdate(2*n_hidden, n_classes))  # 这一层用于做分类

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']  # nf的第0层

        for i, layer in enumerate(self.layers):
            print('layer:', i)
            h = nf.layers[i].data.pop('activation')
            if self.dropout:
                h = self.dropout(h)
            # nf的第0层的h值进行赋值为features的值，这样可以在NodeUpdate中进行更新
            nf.layers[i].data['h'] = h
            nf.block_compute(i,  # nf的第0层进行计算
                             # message function -> 直接传输不用做处理
                             fn.copy_src(src='h', out='m'),
                             # reduce function -> 在dim1上求均值
                             lambda node: {'h': node.mailbox['m'].mean(dim=1)},
                             layer)  # apply function -> 前面的message passing和reduce是汇聚信息，这里是利用gather的信息进行node h的更新

        h = nf.layers[-1].data.pop('activation')  # 输出的结果
        return h


class GCNInfer(nn.Module):
    # 看起来有两个不同：1. 没有了dropout 2. test=True，所以apply的时候会使用'norm'更新h
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
        skip_start = (0 == n_layers-1)
        self.layers.append(NodeUpdate(in_feats, n_hidden,
                           activation, test=True, concat=skip_start))
        # hidden layers
        for i in range(1, n_layers):
            skip_start = (i == n_layers-1)
            self.layers.append(NodeUpdate(
                n_hidden, n_hidden, activation, test=True, concat=skip_start))
        # output layer
        self.layers.append(NodeUpdate(2*n_hidden, n_classes, test=True))

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
