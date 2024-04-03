# import __init__
import torch
from .gcn_lib.sparse.torch_vertex import GENConv
from .gcn_lib.sparse.torch_nn import norm_layer
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging
from torch_sparse import SparseTensor
import torch.distributed as dist

class P3_DeeperGCN(torch.nn.Module):
    def __init__(self, args):
        super(P3_DeeperGCN, self).__init__()

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

    def forward(self, nfs, rank):
        x = []
        edge_index = []
        h_list = []
        for nf in nfs:
            x = nf.layers[0].data['features']
            col, row, _ = nf.block_edges(0, remap_local=False)
            col = nf.map_from_parent_nid(0,nf.map_to_parent_nid(col),remap_local=True)
            row = nf.map_from_parent_nid(0,nf.map_to_parent_nid(row),remap_local=True)

            edge_index.append(SparseTensor(row=row,col=col,sparse_sizes=[x.shape[0],x.shape[0]]).cuda(self.gpu))

            h_list.append(self.node_features_encoder(x))

        if self.block == 'res+':
            mp_out = []
            for i,nf in enumerate(nfs):
                h0 = self.gcns[0](h_list[i], edge_index[i])
                mp_out.append(h0.clone())

            nf = nfs[rank]
            num = len(mp_out)
            with torch.autograd.profiler.record_function('all_reduce hidden vectors'):
                for i in range(num):
                    dist.all_reduce(mp_out[i],dist.ReduceOp.SUM)

            h = mp_out[rank]

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
                    edge_index, h = self.nf_move_layer_to(h,nf,layer)


            h = F.relu(self.norms[self.num_layers - 1](h))
            h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'res':
            mp_out = []
            for i,nf in enumerate(nfs):
                h0 = F.relu(self.norms[0](self.gcns[0](h_list[i], edge_index[i])))
                h0 = F.dropout(h0, p=self.dropout, training=self.training)
                mp_out.append(h0.clone())

            nf = nfs[rank]
            num = len(mp_out)
            with torch.autograd.profiler.record_function('all_reduce hidden vectors'):
                for i in range(num):
                    dist.all_reduce(mp_out[i],dist.ReduceOp.SUM)

            h = mp_out[rank]

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index)
                h2 = self.norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)
                edge_index, h = self.nf_move_layer_to(h,nf,layer)


        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')

        elif self.block == 'plain':
            mp_out = []
            for i,nf in enumerate(nfs):
                h0 = F.relu(self.norms[0](self.gcns[0](h_list[i], edge_index[i])))
                h0 = F.dropout(h0, p=self.dropout, training=self.training)
                mp_out.append(h0.clone())

            nf = nfs[rank]
            num = len(mp_out)
            with torch.autograd.profiler.record_function('all_reduce hidden vectors'):
                for i in range(num):
                    dist.all_reduce(mp_out[i],dist.ReduceOp.SUM)

            h = mp_out[rank]


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

        return torch.log_softmax(h, dim=-1)

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
        edge_index = SparseTensor(row=row,col=col,sparse_sizes=[h.shape[0],h.shape[0]]).cuda(self.gpu)
        # print(h,h.shape,edge_index,col,row)
        return edge_index, h