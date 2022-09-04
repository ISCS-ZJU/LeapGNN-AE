from dgl.distributed.partition import partition_graph as dgl_partition
from dgl.distributed.partition import load_partition

def partition_graph(dglgraph, graphname, totalparts, nhops, rank):
    nmapping, emapping = dgl_partition(dglgraph, graphname, totalparts, num_hops=nhops, part_method='metis',
        out_path='parti_results/', reshuffle=True, return_mapping=True) # balance the numer of nodes in each patition
    print(nmapping, emapping) # mapping between shuffled node id and original node id to make node id in a partition is contiguous
    local_graph, node_feats, edge_feats, gpb, graph_name, _, _ = load_partition(f'parti_results/{graphname}.json', rank)
    return local_graph, node_feats, edge_feats, gpb, graph_name