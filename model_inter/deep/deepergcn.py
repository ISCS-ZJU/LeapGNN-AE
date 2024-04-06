# import __init__
import torch
from .gcn_lib.sparse.torch_vertex import GENConv
from .gcn_lib.sparse.torch_nn import norm_layer
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging
from torch_sparse import SparseTensor

class DeeperGCN(torch.nn.Module):
    def __init__(self, args):
        super(DeeperGCN, self).__init__()

        self.num_layers = args.n_layers + 1
        self.dropout = args.dropout
        self.block = args.block
        self.gpu = args.gpu

        self.checkpoint_grad = False

        in_channels = args.in_feats
        hidden_channels = args.hidden_size
        num_tasks = args.n_classes
        conv = args.conv
        aggr = args.gcn_aggr

        self.mlpoutput = hidden_channels

        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        y = args.y
        self.learn_y = args.learn_y

        self.msg_norm = args.msg_norm
        learn_msg_scale = args.learn_msg_scale

        norm = args.norm
        mlp_layers = args.mlp_layers

        if aggr in ['softmax_sg', 'softmax', 'power'] and self.num_layers > 7:
            self.checkpoint_grad = True
            self.ckp_k = self.num_layers // 2

        print('The number of layers {}'.format(self.num_layers),
              'Aggregation method {}'.format(aggr),
              'block: {}'.format(self.block))

        if self.block == 'res+':
            print('LN/BN->ReLU->GraphConv->Res')
        elif self.block == 'res':
            print('GraphConv->LN/BN->ReLU->Res')
        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')
        elif self.block == "plain":
            print('GraphConv->LN/BN->ReLU')
        else:
            raise Exception('Unknown block Type')

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.node_features_encoder = torch.nn.Linear(in_channels, hidden_channels)
        self.node_pred_linear = torch.nn.Linear(hidden_channels, num_tasks)

        for layer in range(self.num_layers):

            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              y=y, learn_y=self.learn_y,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              norm=norm, mlp_layers=mlp_layers)
            else:
                raise Exception('Unknown Conv Type')

            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))
        
        # count number of intermediate data
        self.total_comb_size = 0
        self.total_actv_size = 0

    def forward(self, nf):
        # 记录一下每层activation结果的size
        actv_size_after_each_block = []
        x = nf.layers[0].data['features']
        col, row, _ = nf.block_edges(0, remap_local=False)
        col = nf.map_from_parent_nid(0,nf.map_to_parent_nid(col),remap_local=True)
        row = nf.map_from_parent_nid(0,nf.map_to_parent_nid(row),remap_local=True)

        edge_index = SparseTensor(row=row,col=col,sparse_sizes=[x.shape[0],x.shape[0]])
        # .cuda(self.gpu)

        h = self.node_features_encoder(x)

        if self.block == 'res+':
            h = self.gcns[0](h, edge_index)

            if self.checkpoint_grad:

                for layer in range(1, self.num_layers):
                    h1 = self.norms[layer - 1](h)
                    h2 = F.relu(h1)
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)

                    if layer % self.ckp_k != 0:
                        res = checkpoint(self.gcns[layer], h2, edge_index)
                        h = res + h
                    else:
                        h = self.gcns[layer](h2, edge_index) + h
                    edge_index, h = self.nf_move_layer_to(h,nf,layer)

            else:
                for layer in range(1, self.num_layers):
                    h1 = self.norms[layer - 1](h)
                    h2 = F.relu(h1)
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)
                    h = self.gcns[layer](h2, edge_index) + h
                    actv_size_after_each_block.append(h.numel())
                    # print(f'layer: {layer}, len of actv_size_after_each_block: {len(actv_size_after_each_block)}')
                    edge_index, h = self.nf_move_layer_to(h,nf,layer)


            h = F.relu(self.norms[self.num_layers - 1](h))
            h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'res':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index)
                h2 = self.norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)
                edge_index, h = self.nf_move_layer_to(h,nf,layer)


        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')

        elif self.block == 'plain':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index)
                h2 = self.norms[layer](h1)
                h = F.relu(h2)
                h = F.dropout(h, p=self.dropout, training=self.training)
                edge_index, h = self.nf_move_layer_to(h,nf,layer)

        else:
            raise Exception('Unknown block Type')

        h = self.node_pred_linear(h)
        h = h[nf.map_from_parent_nid(self.num_layers - 1,nf.layer_parent_nid(self.num_layers),remap_local=True)]

        with torch.no_grad():
            # combine_size 包含每个block aggr的结果shape、和w乘积后结果的shape
            # 每个block执行一次comb，后面block同时要迁移前面block的数据量；
            # activation size 包含每个block actv的结果shape
            old_comb_size = 0
            old_actv_size = 0
            nf_nids = nf._node_mapping.tousertensor()
            offsets = nf._layer_offsets # 这里的layer含义不是Block，一个Block包含输入Layer和输出layer
            for blkid in range(1, self.num_layers):
                # print(f'blkid: {blkid}')
                # aggr_results = len(nf_nids[offsets[blkid]: offsets[blkid+1]]) * self.layers[blkid].linear.in_features
                tensor_after_combine_and_w = len(nf_nids[offsets[blkid+1]: offsets[blkid+2]])*self.mlpoutput

                # cur_block_comb_size = aggr_results + tensor_after_combine_and_w
                cur_block_comb_size = tensor_after_combine_and_w
                self.total_comb_size += old_comb_size
                old_comb_size += cur_block_comb_size

                tensor_after_actv = 0
                # if layer.activation:
                tensor_after_actv = actv_size_after_each_block[blkid-1]
                self.total_actv_size += old_actv_size
                old_actv_size += tensor_after_actv
        curnf_total_comb_size, curnf_total_actv_size = self.total_comb_size, self.total_actv_size
        self.total_comb_size, self.total_actv_size = 0,0
        return torch.log_softmax(h, dim=-1), curnf_total_comb_size, curnf_total_actv_size


        # return torch.log_softmax(h, dim=-1)

    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print('Final t {}'.format(ts))
            else:
                logging.info('Epoch {}, t {}'.format(epoch, ts))

        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print('Final p {}'.format(ps))
            else:
                logging.info('Epoch {}, p {}'.format(epoch, ps))

        if self.learn_y:
            ys = []
            for gcn in self.gcns:
                ys.append(gcn.sigmoid_y.item())
            if final:
                print('Final sigmoid(y) {}'.format(ys))
            else:
                logging.info('Epoch {}, sigmoid(y) {}'.format(epoch, ys))

        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print('Final s {}'.format(ss))
            else:
                logging.info('Epoch {}, s {}'.format(epoch, ss))

    def nf_move_layer_to(self, h, nf, i):
        col, row, _ = nf.block_edges(i, remap_local=False)
        col = nf.map_from_parent_nid(i,nf.map_to_parent_nid(col),remap_local=True)
        row = nf.map_from_parent_nid(i,nf.map_to_parent_nid(row),remap_local=True)
        h = h[nf.map_from_parent_nid(i - 1,nf.layer_parent_nid(i),remap_local=True)]
        edge_index = SparseTensor(row=row,col=col,sparse_sizes=[h.shape[0],h.shape[0]])
        # .cuda(self.gpu)
        # print(h,h.shape,edge_index,col,row)
        return edge_index, h