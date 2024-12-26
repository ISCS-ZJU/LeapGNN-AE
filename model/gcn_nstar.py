import torch
import torch.nn as nn
import dgl
import dgl.function as fn

from dgl.utils import toindex
import time

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
                 dropout,
                 nid2pid,
                 currank):
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
        
        self.n_hidden = n_hidden
        self.nid2pid = torch.from_numpy(nid2pid)
        self.currank = currank
        
        

    def forward(self, nf):
        st = time.time()
        with torch.no_grad():
            # 获取每个block的dst要从远程通信获得的 nid
            delete_dst_nid = []
            ori_nid = nf._node_mapping.tousertensor()
            for i, _ in enumerate(self.layers):
                edge_ids = nf.block_edges(i)[1]
                dst_nids = torch.unique_consecutive(ori_nid[edge_ids])
                print(dst_nids.shape, dst_nids[0])
                dst_bool = (self.nid2pid[dst_nids] == self.currank)
                print(dst_bool.shape, dst_bool[0])
                # 要从远程通信获得的元素的下标
                false_indices = torch.nonzero(~dst_bool, as_tuple=True)[0]
                # print(f'false_indices = {false_indices.shape} {false_indices[0]}')
                delete_dst_nid.append(false_indices)
        ed = time.time()
        print(f'make desicison time: {ed-st}')
            
        nf.layers[0].data['activation'] = nf.layers[0].data['features']

        for i, layer in enumerate(self.layers): # layer 对应于一个 block
            print('\n',f'i={i}')
            h = nf.layers[i].data.pop('activation')
            print(f'h size: {h.size()}')
            if i:
                # h = torch.cat((h,h[0,:].unsqueeze(0)), dim=0)
                h = torch.cat((h, h[:len(delete_dst_nid[i-1]), :].unsqueeze(0)), dim=0)
                print(f'h size: {h.size()}')
                
            if self.dropout:
                h = self.dropout(h)
            nf.layers[i].data['h'] = h
            
            # print(f'type(h) = {type(h)}, h.shape = {h.shape}, h')
            # print(f'nf.nodes = {nf._node_mapping.tousertensor().dtype}')
            # print(f'nf.edges = {nf._edge_mapping.tousertensor().dtype}')
            # print(f'nf.edges = {nf._edge_mapping.tousertensor().tolist()}')
            # print(f'nf.block_edges = {nf.block_edges(i)}')
            # print(f'nf.block_size = {nf.block_size(i)}') # 边数
            # print(f'nf.layer_size = {nf.layer_size(i+1)}') # src 点数量
            
            # nf._edge_mapping
        #     nf._edge_mapping = toindex(torch.tensor([19, 20, 21, -1,  0, 22, -1,  1, 23, 24, 25, -1,  2,  3,  4, 26, 27, -1,
        #  6,  7,  9, 30, 31, -1,  8, 11, 32, -1,  2,  3,  4, 26, 27, -1]))
        #     nf._node_mapping = toindex(torch.tensor([0, 1, 2, 3, 4, 5, 6, 8, 9, 0, 1, 2, 3, 5, 6, 3]))
            # nf._edge_mapping = toindex(torch.tensor([-1]))
            # nf._node_mapping = toindex(torch.tensor([-1]))
            # 修改 edge_mapping 和 _node_mapping 没用
            # 要修改 src_node_frame(i.e., nf._get_node_frame(i)) 和 dst_node_frame(i.e., nf._get_node_frame(i + 1)), uv_getter中的 src 和 dst
            
            
            st = time.time()
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.mean(msg='m', out='h'),
                             layer,
                            #  delete_dst_nid=[1] if i == 0 else [],
                            delete_dst_nid = delete_dst_nid[i],
                             n_hidden = self.n_hidden)
            ed = time.time()
            print(f'time = {ed-st} s')

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
                             layer,
                             delete_dst_nid=[])
        h = nf.layers[-1].data.pop('activation')
        return h
