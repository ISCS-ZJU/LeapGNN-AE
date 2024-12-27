import os
import time
import getpass
import argparse

def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='auto test')
    parser.add_argument('-n', '--name', default="overall",
                        type=str, help='test name')
    return parser.parse_args(argv)

all_server_cmd = dict()
all_client_cmd = dict()

all_server_cmd['overall'] = [
    'python3 servers_start.py --dataset in_2004 --cache_type static --partition_type metis --multi_feat_file True',
    'python3 servers_start.py --dataset in_2004 --cache_type p3 --partition_type metis --multi_feat_file True',
]

all_client_cmd['overall'] = [
    # default/ours
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset in_2004 --n_epochs 3 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset in_2004 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset in_2004 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset in_2004 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # navie
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 8 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 8 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 8 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 8 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 8 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 8 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # p3
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 5 --iter_stop 5 --dataset in_2004 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset in_2004 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 5 --iter_stop 5 --dataset in_2004 --n_epochs 3 --batch_size 8000 --hidden_size 128',

]

all_server_cmd['indv'] = [
    'python3 servers_start.py --dataset ogbn_products0 --cache_type static --partition_type metis',
]   

all_client_cmd['indv'] = [
    # hd 16 default ours1 ours1+2 ours+all
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 11 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 13 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 11 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 13 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 11 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 13 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # hd 128
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 11 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 13 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 11 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 13 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 11 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 13 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 128',

    #deep model
    'python3 clients_start.py --model_name deepergcn --sampling 2-2-2-2-2 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 512 --hidden_size 16',
    'python3 clients_start.py --model_name deepergcn --sampling 2-2-2-2-2 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 512 --hidden_size 16',
    'python3 clients_start.py --model_name deepergcn --sampling 2-2-2-2-2 --run_client_idx 11 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 512 --hidden_size 16',
    'python3 clients_start.py --model_name deepergcn --sampling 2-2-2-2-2 --run_client_idx 13 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 512 --hidden_size 16',
    'python3 clients_start.py --model_name film --sampling 2-2-2-2-2-2-2-2-2 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 512 --hidden_size 16',
    'python3 clients_start.py --model_name film --sampling 2-2-2-2-2-2-2-2-2 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 512 --hidden_size 16',
    'python3 clients_start.py --model_name film --sampling 2-2-2-2-2-2-2-2-2 --run_client_idx 11 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 512 --hidden_size 16',
    'python3 clients_start.py --model_name film --sampling 2-2-2-2-2-2-2-2-2 --run_client_idx 13 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 512 --hidden_size 16',

]

all_server_cmd['deep'] = [
    'python3 servers_start.py --dataset ogbn_arxiv0 --cache_type static --partition_type metis',
    'python3 servers_start.py --dataset ogbn_arxiv0 --cache_type p3 --partition_type metis',
]

all_client_cmd['deep'] = [
    'python3 clients_start.py --model_name deepergcn --sampling 2-2-2-2-2 --run_client_idx 9 --iter_stop 5 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 512 --hidden_size 16',
    'python3 clients_start.py --model_name deepergcn --sampling 2-2-2-2-2 --run_client_idx 6 --iter_stop 5 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 512 --hidden_size 16',
    'python3 clients_start.py --model_name deepergcn --sampling 2-2-2-2-2 --run_client_idx 8 --iter_stop 5 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 512 --hidden_size 16',
    'python3 clients_start.py --model_name deepergcn --sampling 2-2-2-2-2 --run_client_idx 5 --iter_stop 5 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 512 --hidden_size 16',
    'python3 clients_start.py --model_name film --sampling 2-2-2-2-2-2-2-2-2 --run_client_idx 9 --iter_stop 5 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 512 --hidden_size 16',
    'python3 clients_start.py --model_name film --sampling 2-2-2-2-2-2-2-2-2 --run_client_idx 6 --iter_stop 5 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 512 --hidden_size 16',
    'python3 clients_start.py --model_name film --sampling 2-2-2-2-2-2-2-2-2 --run_client_idx 8 --iter_stop 5 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 512 --hidden_size 16',
    'python3 clients_start.py --model_name film --sampling 2-2-2-2-2-2-2-2-2 --run_client_idx 5 --iter_stop 5 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 512 --hidden_size 16',
]

all_server_cmd['full_batch'] = [
    'python3 servers_start.py --dataset in_2004 --cache_type static --partition_type metis --multi_feat_file True',
]

all_client_cmd['full_batch'] = [
    'python3 clients_start.py --model_name gcn --sampling 10000000-10000000 --run_client_idx 9 --iter_stop 1 --dataset in_2004 --n_epochs 5 --batch_size 80000000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10000000-10000000 --run_client_idx 6 --iter_stop 1 --dataset in_2004 --n_epochs 5 --batch_size 80000000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10000000-10000000 --run_client_idx 15 --dataset in_2004 --n_epochs 1 --batch_size 80000000 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10000000-10000000 --run_client_idx 9 --iter_stop 1 --dataset in_2004 --n_epochs 3 --batch_size 80000000 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10000000-10000000 --run_client_idx 6 --iter_stop 1 --dataset in_2004 --n_epochs 5 --batch_size 80000000 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10000000-10000000 --run_client_idx 15 --dataset in_2004 --n_epochs 1 --batch_size 80000000 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10000000-10000000 --run_client_idx 9 --iter_stop 1 --dataset in_2004 --n_epochs 5 --batch_size 80000000 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10000000-10000000 --run_client_idx 6 --iter_stop 1 --dataset in_2004 --n_epochs 5 --batch_size 80000000 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10000000-10000000 --run_client_idx 15 --dataset in_2004 --n_epochs 1 --batch_size 80000000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10000000-10000000 --run_client_idx 9 --iter_stop 1 --dataset in_2004 --n_epochs 3 --batch_size 80000000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gcn --sampling 10000000-10000000 --run_client_idx 6 --iter_stop 1 --dataset in_2004 --n_epochs 5 --batch_size 80000000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gcn --sampling 10000000-10000000 --run_client_idx 15 --dataset in_2004 --n_epochs 1 --batch_size 80000000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10000000-10000000 --run_client_idx 9 --iter_stop 1 --dataset in_2004 --n_epochs 3 --batch_size 80000000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10000000-10000000 --run_client_idx 6 --iter_stop 1 --dataset in_2004 --n_epochs 5 --batch_size 80000000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10000000-10000000 --run_client_idx 15 --dataset in_2004 --n_epochs 1 --batch_size 80000000 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10000000-10000000 --run_client_idx 9 --iter_stop 1 --dataset in_2004 --n_epochs 3 --batch_size 80000000 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10000000-10000000 --run_client_idx 6 --iter_stop 1 --dataset in_2004 --n_epochs 5 --batch_size 80000000 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10000000-10000000 --run_client_idx 15 --dataset in_2004 --n_epochs 1 --batch_size 80000000 --hidden_size 128',
]

all_server_cmd['bs'] = [
    'python3 servers_start.py --dataset ogbn_products0 --cache_type static --partition_type metis',
]

all_client_cmd['bs'] = [
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 512 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 512 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 1024 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 1024 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 2048 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 2048 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8192 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8192 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 16384 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 16384 --hidden_size 16',
]

all_server_cmd['hd'] = [
    'python3 servers_start.py --dataset ogbn_products50 --cache_type static --partition_type metis',
    'python3 servers_start.py --dataset ogbn_products100 --cache_type static --partition_type metis',
    'python3 servers_start.py --dataset ogbn_products200 --cache_type static --partition_type metis',
    'python3 servers_start.py --dataset ogbn_products400 --cache_type static --partition_type metis',
    'python3 servers_start.py --dataset ogbn_products800 --cache_type static --partition_type metis',
]

all_client_cmd['hd'] = [
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products50 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products50 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products100 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products100 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products200 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products200 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products400 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products400 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products800 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products800 --n_epochs 5 --batch_size 8000 --hidden_size 16',
]

all_server_cmd['fanout'] = [
    'python3 servers_start.py --dataset ogbn_products0 --cache_type static --partition_type metis',
]

all_client_cmd['fanout'] = [
    'python3 clients_start.py --model_name gcn --sampling 5-5 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 5-5 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 20-20 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 20-20 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 40-40 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 40-40 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
]

all_server_cmd['machine'] = [
    'python3 servers_start.py --dataset ogbn_products0 --cache_type static --partition_type metis --servers_num 2',
    'python3 servers_start.py --dataset ogbn_products0 --cache_type static --partition_type metis --servers_num 3',
    'python3 servers_start.py --dataset ogbn_products0 --cache_type static --partition_type metis --servers_num 4',
    'python3 servers_start.py --dataset ogbn_products0 --cache_type static --partition_type metis --servers_num 5',
    'python3 servers_start.py --dataset ogbn_products0 --cache_type static --partition_type metis --servers_num 6',
]

all_client_cmd['machine'] = [
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16 --servers_num 2',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16 --servers_num 2',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16 --servers_num 3',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16 --servers_num 3',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16 --servers_num 4',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16 --servers_num 4',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16 --servers_num 5',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16 --servers_num 5',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16 --servers_num 6',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16 --servers_num 6',
]

all_server_cmd['merging'] = [
    'python3 servers_start.py --dataset ogbn_products0 --cache_type static --partition_type metis',
]

all_client_cmd['merging'] = [
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 7 --batch_size 8000 --hidden_size 16',
]

all_server_cmd['random'] = [
    'python3 servers_start.py --dataset in_2004 --cache_type static --partition_type metis --multi_feat_file True',
]
all_client_cmd['random'] = [
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 10 --iter_stop 5 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 10 --iter_stop 5 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 10 --iter_stop 5 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
]

all_server_cmd['gpu_util'] = [
    'python3 servers_start.py --dataset uk_2007 --cache_type static --partition_type metis --multi_feat_file True',
    'python3 servers_start.py --dataset uk_2007 --cache_type p3 --partition_type metis --multi_feat_file True',
]

all_client_cmd['gpu_util'] = [
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset uk_2007 --n_epochs 1 --batch_size 8000 --hidden_size 16 --gputil --util-interval 0.25',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset uk_2007 --n_epochs 1 --batch_size 8000 --hidden_size 16 --gputil --util-interval 0.25',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 5 --dataset uk_2007 --n_epochs 1 --batch_size 8000 --hidden_size 16 --gputil --util-interval 0.25',
]

all_server_cmd['hello'] = [
    'python3 servers_start.py --dataset ogbn_arxiv0 --cache_type static --partition_type metis',
]

all_client_cmd['hello'] = [
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset  ogbn_arxiv0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
]

def check_and_delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'Deleted file {file_path}')
        return 1
    else:
        return 0

if __name__ == '__main__':
    args = parse_args_func(None)
    server_cmd_lst = all_server_cmd[args.name]
    client_cmd_lst = all_client_cmd[args.name]
    username = getpass.getuser()
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../dist/server_done.txt')


    os.system('python3 servers_kill.py')
    time.sleep(10)
    os.system('python3 clients_kill.py')
    time.sleep(10)
    for server_cmdid, server_cmd in enumerate(server_cmd_lst):
        check_and_delete_file(file_path)
        returncode_server = os.system(server_cmd)
        print(f'{server_cmd}')
        print(f'returncode_server: {returncode_server}')
        if returncode_server == 0:
            # 检查server是否启动完成
            while True:
                time.sleep(30)
                exist = check_and_delete_file(file_path)
                if exist:
                    break

            cur_server_dataset = server_cmd.split("--dataset ")[1].split()[0]    
            cur_server_cache_type = server_cmd.split("--cache_type ")[1].split()[0]
            servers_num = -1
            if '--servers_num ' in server_cmd:
                servers_num = int(server_cmd.split("--servers_num ")[1].split()[0])   
            # 执行几条client命令
            for cur_client_cmd in client_cmd_lst:
                cur_client_dataset = cur_client_cmd.split("--dataset ")[1].split()[0]    
                cur_client_method = cur_client_cmd.split("--run_client_idx ")[1].split()[0]
                cur_clinet_num = -1
                if '--servers_num ' in cur_client_cmd:
                    cur_clinet_num = int(cur_client_cmd.split("--servers_num ")[1].split()[0])
                if cur_client_method == '4' or cur_client_method == '5':
                    cur_client_cache_type = 'p3'
                else:
                    cur_client_cache_type = 'static'    
                if cur_client_dataset == cur_server_dataset and cur_client_cache_type == cur_server_cache_type and servers_num == cur_clinet_num:
                    while True:
                        print(f'start run {cur_client_cmd}')
                        returncode_client = os.system(cur_client_cmd)
                        print(f'returncode is {returncode_client}')
                        if returncode_client == 0:
                            time.sleep(10)
                            os.system('python3 clients_kill.py')
                            time.sleep(10)
                            break
                        else:
                            print(f"Command {cur_client_cmd} failed, retrying in 1 minute...")
                            time.sleep(10)
                            os.system('python3 clients_kill.py')
                            time.sleep(10)
                            continue

            # kill server and remaining client
            os.system('python3 servers_kill.py')
            time.sleep(10)
            os.system('python3 clients_kill.py')
            time.sleep(10)
        else:
            print(f"Command {server_cmd} failed")

