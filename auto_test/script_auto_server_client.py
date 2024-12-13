import os
import time
import getpass

server_cmd_lst = [
    # # 18*4 bar
    
    # # hd 128
    # gcn
    # # static
    # 'python3 servers_start.py --dataset ogbn_arxiv0 --cache_type static --partition_type metis',
    # 'python3 servers_start.py --dataset ogbn_products0 --cache_type static --partition_type metis',
    # 'python3 servers_start.py --dataset uk_2007 --cache_type static --partition_type metis --multi_feat_file True',
    'python3 servers_start.py --dataset in_2004 --cache_type static --partition_type metis --multi_feat_file True',
    # 'python3 servers_start.py --dataset it --cache_type static --partition_type metis --multi_feat_file True',
    # # p3
    # 'python3 servers_start.py --dataset ogbn_arxiv0 --cache_type p3 --partition_type metis',
    # 'python3 servers_start.py --dataset ogbn_products0 --cache_type p3 --partition_type metis',
    # 'python3 servers_start.py --dataset uk_2007 --cache_type p3 --partition_type metis --multi_feat_file True',
    'python3 servers_start.py --dataset in_2004 --cache_type p3 --partition_type metis --multi_feat_file True',
    # 'python3 servers_start.py --dataset it --cache_type p3 --partition_type metis --multi_feat_file True',
]

client_cmd_lst = [
    # twitter
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --dataset twitter --iter_stop 0 --n_epochs 1 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --dataset twitter --iter_stop 0 --n_epochs 1 --batch_size 8000 --hidden_size 16',


    # # 18*4 bar
    # # arxiv
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 14 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 4096 --hidden_size 16',

    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 4096 --hidden_size 128',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 4096 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 4096 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 4096 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 4096 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 4096 --hidden_size 128',

    # #products    
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',

    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 128',

    # # in
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

    # # uk
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 16',

    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset uk_2007 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset uk_2007 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 9 --iter_stop 5 --dataset uk_2007 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 6 --iter_stop 5 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 128',


    



    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --dataset in_2004 --n_epochs 5 --batch_size 8000 --iter_stop 4 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --dataset in_2004 --n_epochs 5 --batch_size 4096 --iter_stop 4 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --dataset in_2004 --n_epochs 5 --batch_size 8000 --iter_stop 4 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --dataset in_2004 --n_epochs 5 --batch_size 4096 --iter_stop 4 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --iter_stop 4 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --iter_stop 2 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --iter_stop 4 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --iter_stop 2 --hidden_size 128',

    # naive
    # arxiv
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 8 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 8 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 8 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 8 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 8 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 8 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 128',

    # products
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 8 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 8 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 8 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 8 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 8 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 8 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # in
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 8 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 8 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 8 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 8 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 8 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 8 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # uk
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 8 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 8 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 8 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 8 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 8 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 8 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # it
    # 'python3 clients_start.py --model_name gcn --sampling 5-5 --run_client_idx 12 --dataset it --n_epochs 3 --iter_stop 5 --batch_size 1024 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 5-5 --run_client_idx 12 --dataset it --n_epochs 3 --iter_stop 5 --batch_size 1024 --hidden_size 16',
    # 'python3 clients_start.py --model_name graphsage --sampling 5-5 --run_client_idx 12 --dataset it --n_epochs 3  --iter_stop 5 --batch_size 1024 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 5-5 --run_client_idx 8 --dataset it --n_epochs 5 --batch_size 8000 --hidden_size 32',
    # 'python3 clients_start.py --model_name gat --sampling 5-5 --run_client_idx 8 --dataset it --n_epochs 5 --batch_size 8000 --hidden_size 32',
    # 'python3 clients_start.py --model_name graphsage --sampling 5-5 --run_client_idx 8 --dataset it --n_epochs 5 --batch_size 8000 --hidden_size 32',

    # # p3
    # # arxiv
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 128',

    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv128 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 5 --iter_stop 5 --dataset ogbn_arxiv128 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset ogbn_arxiv128 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 5 --iter_stop 5 --dataset ogbn_arxiv128 --n_epochs 3 --batch_size 8000 --hidden_size 128',

    

    # # products
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 5 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 5 --iter_stop 5 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 128',

    # # in
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset in_2004 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 5 --iter_stop 5 --dataset in_2004 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset in_2004 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 5 --iter_stop 5 --dataset in_2004 --n_epochs 3 --batch_size 8000 --hidden_size 128',

    # # uk
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset uk_2007 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 5 --iter_stop 5 --dataset uk_2007 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset uk_2007 --n_epochs 3 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 5 --iter_stop 5 --dataset uk_2007 --n_epochs 3 --batch_size 8000 --hidden_size 128',


    # # reddit
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 8 --dataset reddit0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 8 --dataset reddit0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 8 --dataset reddit0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 8 --dataset reddit0 --n_epochs 5 --batch_size 8000 --hidden_size 32',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 8 --dataset reddit0 --n_epochs 5 --batch_size 8000 --hidden_size 32',
    # 'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 8 --dataset reddit0 --n_epochs 5 --batch_size 8000 --hidden_size 32',

    # deep models
    # 'python3 clients_start.py --model_name deepergcn --sampling 2-2-2 --run_client_idx 6 --iter_stop 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name deepergcn --sampling 2-2-2 --run_client_idx 9 --iter_stop 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name film --sampling 2-2-2 --run_client_idx 6 --iter_stop 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name film --sampling 2-2-2 --run_client_idx 9 --iter_stop 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    #  p3
    # 'python3 clients_start.py --model_name deepergcn --sampling 2-2-2 --run_client_idx 5 --iter_stop 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # 'python3 clients_start.py --model_name film --sampling 2-2-2 --run_client_idx 5 --iter_stop 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 8000 --hidden_size 128',


    # 'python3 clients_start.py --model_name gcn --sampling 10000000 --run_client_idx 8 --dataset ogbn_arxiv0 --n_epochs 1 --batch_size 80000000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10000000 --run_client_idx 8 --dataset ogbn_arxiv0 --n_epochs 1 --batch_size 80000000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gcn --sampling 10000000-10000000 --run_client_idx 8 --dataset ogbn_arxiv0 --n_epochs 1 --batch_size 80000000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10000000-10000000 --run_client_idx 8 --dataset ogbn_arxiv0 --n_epochs 1 --batch_size 80000000 --hidden_size 128',

    # 'python3 clients_start.py --model_name gcn --sampling 10000000 --run_client_idx 8 --dataset in_2004 --n_epochs 1 --batch_size 80000000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10000000 --run_client_idx 8 --dataset in_2004 --n_epochs 1 --batch_size 80000000 --hidden_size 128',
    # 'python3 clients_start.py --model_name gcn --sampling 10000000-10000000 --run_client_idx 8 --dataset in_2004 --n_epochs 1 --batch_size 80000000 --hidden_size 16',
    # 'python3 clients_start.py --model_name gcn --sampling 10000000-10000000 --run_client_idx 8 --dataset in_2004 --n_epochs 1 --batch_size 80000000 --hidden_size 128',
]


def check_and_delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'Deleted file {file_path}')
        return 1
    else:
        return 0

if __name__ == '__main__':
    username = getpass.getuser()
    if username == 'weijian':
        file_path = '/home/weijian/gitclone/repgnn/dist/server_done.txt'
    elif username == 'qhy':
        file_path = '/home/qhy/gnn/repgnn/dist/server_done.txt'

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
            # 执行几条client命令
            for cur_client_cmd in client_cmd_lst:
                cur_client_dataset = cur_client_cmd.split("--dataset ")[1].split()[0]    
                cur_client_method = cur_client_cmd.split("--run_client_idx ")[1].split()[0]
                if cur_client_method == '4' or cur_client_method == '5':
                    cur_client_cache_type = 'p3'
                else:
                    cur_client_cache_type = 'static'    
                if cur_client_dataset == cur_server_dataset and cur_client_cache_type == cur_server_cache_type:
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

