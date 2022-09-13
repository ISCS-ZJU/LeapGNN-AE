import dgl
import argparse
import torch
import numpy as np

import data
import os


def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='GNN Training')
    parser.add_argument('-d', '--dataset', default="/data/pagraph/gendemo", type=str, choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-mag'], help='training dataset name')
    parser.add_argument('-ngpu', '--num-gpu', default=2, type=int, help='# of gpus to train gnn with DDP')
    parser.add_argument('-s', '--sampling', default="10-10-10", type=str, help='neighborhood sampling method parameters')
    parser.add_argument('-hd', '--hidden-size', default=256, type=int, help='hidden dimension size')
    parser.add_argument('-bs', '--batch-size', default=1024, type=int, help='training batch size')
    parser.add_argument('-mn', '--model-name', default='graphsage', type=str, choices=['graphsage', 'gcn', 'demo'], help='GNN model name')
    parser.add_argument('-ep', '--epoch', default=3, type=int, help='total trianing epoch')
    parser.add_argument('-wkr', '--num-worker', default=0, type=int, help='sampling worker')
    parser.add_argument('-cs', '--cache-size', default=0, type=int, help='cache size in each gpu (GB)')
    return parser.parse_args(argv)




def main(args):
    # load dataset
    coo_adj, feat = data.get_graph_data(args.dataset)
    features = torch.FloatTensor(feat)
    # print(coo_adj) # (stnid, ednid), edege_data

    # construct graph
    graph = dgl.DGLGraph(coo_adj, readonly=True)
    graph_name = os.path.basename(args.dataset)
    vnum = graph.number_of_nodes()
    enum = graph.number_of_edges()
    feat_dim = features.size(1)

    print('=' * 30)
    print("Graph Name: {}\nNodes Num: {}\tEdges Num: {}\nFeature Size: {}"
            .format(graph_name, vnum, enum, feat_dim)
    )
    print('=' * 30)
    # 至此，graph的topo和feat都已经在cpu内存，并利用topo构建了DGLGraph
    

    # create server
    g = dgl.contrib.graph_store.create_graph_store_server(graph, graph_name,
        'shared_mem', args.num_gpu, False, edge_dir='in', port=8004)

    g.ndata['features'] = features
    # 至此，graph的topo和feat都已经在shared memory中，并赋给变量g._graph，可以通过graph server来访问

    g.run() # 监听server请求

if __name__ == '__main__':
    args = parse_args_func(None)
    main(args)
