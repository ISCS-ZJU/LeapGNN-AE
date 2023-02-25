import torch as th
import torch.nn as nn
import dgl
import dgl.function as fn

def u_add_v(edges):
    return {'e': edges.src['el'] + edges.dst['er']}

def e_div_v(edges):
    return {'a': edges.data['s'] / edges.dst['out_sum']}

class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats,
                 num_heads=2,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=None):
        super(GATLayer, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        # if residual:
        #     if in_feats != out_feats:
        #         self.res_fc = nn.Linear(in_feats, num_heads * out_feats, bias=False)
        #     else:
        #         self.res_fc = Identity()
        # else:
        self.register_buffer('res_fc', None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, nf, i):
        feat_src = nf.layers[i].data.pop('activation')
        feat_dst = nf.layers[i+1].data.pop('activation')
        h_src = self.feat_drop(feat_src)
        h_dst = self.feat_drop(feat_dst)
        feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
        feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        nf.layers[i].data['ft'] = feat_src
        nf.layers[i].data['el'] = el
        nf.layers[i+1].data['er'] = er
        # compute edge attention
        nf.apply_block(i,u_add_v)
        e = self.leaky_relu(nf.blocks[i].data.pop('e'))
        # compute softmax
        nf.blocks[i].data['s'] = e
        nf.blocks[i].data['s'] = th.exp(nf.blocks[i].data['s'])
        nf.block_compute(i,fn.copy_e('s', 'm'),
                         fn.sum('m', 'out_sum'))
        nf.apply_block(i,e_div_v)
        nf.blocks[i].data['a'] = self.attn_drop(nf.blocks[i].data['a'])
        # message passing
        nf.block_compute(i,fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = nf.layers[i+1].data['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_src).view(h_src.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst


class GATSampling(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 ):
        super(GATSampling, self).__init__()
        self.n_layers = n_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATLayer(
            in_feats, n_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, self.activation))
        # hidden layers
        for l in range(1, n_layers):
            # due to multi-head, the in_feats = n_hidden * num_heads
            self.gat_layers.append(GATLayer(
                n_hidden * heads[l-1], n_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, self.activation))
        # output projection
        self.gat_layers.append(GATLayer(
            n_hidden * heads[-2], n_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, None))

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']
        nf.layers[1].data['activation'] = nf.layers[1].data['features']
        for i in range(self.n_layers):
            h = self.gat_layers[i](nf,i).flatten(1)
            nf.layers[i+1].data['activation'] = h
            nf.layers[i+2].data['activation'] = h[nf.map_from_parent_nid(i+1,nf.map_to_parent_nid(nf.layer_nid(i+2)),remap_local=True)]
        # output projection
        logits = self.gat_layers[-1](nf, self.n_layers).mean(1)
        return logits
