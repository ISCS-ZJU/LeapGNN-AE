from dgl.distributed.partition import partition_graph as dgl_partition
from dgl.distributed.partition import load_partition
import os
import torch
import torch.distributed as dist

def partition_graph(dglgraph, graphname, totalparts, nhops, rank):
    # if not os.path.exists(f'parti_results/{graphname}.json'):
    #     nmapping, emapping = dgl_partition(dglgraph, graphname, totalparts, num_hops=nhops, part_method='metis',
    #         out_path='parti_results/', reshuffle=True, return_mapping=True) # balance the numer of nodes in each patition
    #     print(nmapping, emapping) # mapping between shuffled node id and original node id to make node id in a partition is contiguous

    # else:
    #     print(f'-> Partition has been done and generated file parti_results/{graphname}.json')
    # local_graph, node_feats, edge_feats, gpb, graph_name, _, _ = load_partition(f'parti_results/{graphname}.json', rank)
    # return local_graph, node_feats, edge_feats, gpb, graph_name, nmapping, emapping

    if rank==0:
        nmapping, emapping = dgl_partition(dglgraph, graphname, totalparts, num_hops=nhops, part_method='metis', out_path='parti_results/', reshuffle=True, return_mapping=True) # balance the numer of nodes in each patition
        print('-> Mapping:', nmapping, emapping) # mapping between shuffled node id and original node id to make node id in a partition is contiguous
        
    torch.distributed.barrier() # wait rank 0 graph partition done
    # recieve nmapping and emapping from rank 0
    if rank == 0:
        broad_msg = [nmapping, emapping]
    else:
        broad_msg = [None, None]
    dist.broadcast_object_list(broad_msg, src=0)
    nmapping, emapping = broad_msg

    local_graph, node_feats, edge_feats, gpb, graph_name, _, _ = load_partition(f'parti_results/{graphname}.json', rank)
    return local_graph, node_feats, edge_feats, gpb, graph_name, nmapping, emapping