import os.path as osp
import numpy as np



dataset = 'ogbn_papers100M0'
root_path = './dist/repgnn_data'
dataset_path = osp.join(root_path,dataset)



def write_node(node_num):
    node_type = np.zeros((node_num,1),dtype=int)
    node_weight = np.ones((node_num,1),dtype=int)
    node_id = np.arange(0,node_num,dtype=int).reshape(node_num,1)
    nodes = np.concatenate([node_type,node_weight,node_id],axis=1)
    np.savetxt(osp.join(dataset_path,'test_nodes.txt'),nodes,fmt='%d')

def write_edge():
    topo = np.loadtxt(osp.join(dataset_path,'pp.txt'),dtype=int)
    edge_num = topo.shape[0]
    edge_type = np.zeros((edge_num,1),dtype=int)
    edge_id = np.arange(0,edge_num,dtype=int).reshape(edge_num,1)
    edges = np.concatenate([topo,edge_type,edge_id],axis=1)
    np.savetxt(osp.join(dataset_path,'test_edges.txt'),edges,fmt='%d')
    return edge_num

def write_stat(node_num,edge_num):
    np.savetxt(osp.join(dataset_path,'test_stats.txt'),np.array([node_num,edge_num,1]).reshape(1,3),fmt='%d')

if __name__ == '__main__':
    train_mask = np.load(osp.join(dataset_path, 'train.npy'))
    node_num = train_mask.shape[0]
    # print(np.sort(np.load('/home/qhy/gnn/repgnn/dist/repgnn_data/citeseer0/dist_True/2_metis/0.npy')))
    write_node(node_num)
    edge_num = write_edge()
    write_stat(node_num,edge_num)

