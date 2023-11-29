import torch as th
import torch.nn as nn
import dgl
import dgl.function as fn
import torch

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
        
        # count number of intermediate data
        self.total_comb_size = 0
        self.total_actv_size = 0

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']
        nf.layers[1].data['activation'] = nf.layers[1].data['features']
        # 记录一下每层activation结果的size
        actv_size_after_each_block = []
        for i in range(self.n_layers):
            h = self.gat_layers[i](nf,i).flatten(1)
            nf.layers[i+1].data['activation'] = h
            actv_size_after_each_block.append(h.numel())
            nf.layers[i+2].data['activation'] = h[nf.map_from_parent_nid(i+1,nf.map_to_parent_nid(nf.layer_nid(i+2)),remap_local=True)]
            actv_size_after_each_block.append(h.numel())
        # output projection
        logits = self.gat_layers[-1](nf, self.n_layers).mean(1)

        with torch.no_grad():
            # combine_size 包含每个block aggr的结果shape、和w乘积后结果的shape
            # 每个block执行一次comb，后面block同时要迁移前面block的数据量；
            # activation size 包含每个block actv的结果shape
            old_comb_size = 0
            old_actv_size = 0
            nf_nids = nf._node_mapping.tousertensor()
            offsets = nf._layer_offsets # 这里的layer含义不是Block，一个Block包含输入Layer和输出layer
            for blkid, layer in enumerate(self.gat_layers):
                # aggr_results = len(nf_nids[offsets[blkid]: offsets[blkid+1]]) * self.gat_layers[blkid].fc.in_features
                tensor_after_combine_and_w = len(nf_nids[offsets[blkid+1]: offsets[blkid+2]])*self.gat_layers[blkid].fc.out_features

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
        return logits, curnf_total_comb_size, curnf_total_actv_size

        # return logits
