import os
import time

server_cmd_lst = [
    # 'python3 servers_start.py --dataset ogbn_products50 --cache_type static',
    # 'python3 servers_start.py --dataset ogbn_products200 --cache_type static',
    # 'python3 servers_start.py --dataset ogbn_products400 --cache_type static',

    # large dataset
    # 'python3 servers_start.py --dataset twitter --cache_type static --partition_type metis --multi_feat_file True', # 补充之前缺少的几组实验
    # 'python3 servers_start.py --dataset ogbn_papers100M0 --cache_type static --partition_type metis --multi_feat_file True', # 补充之前缺少的几组实验
    # 'python3 servers_start.py --dataset twitter --cache_type p3 --partition_type metis --multi_feat_file True', # 补充之前缺少的几组实验
    # 'python3 servers_start.py --dataset ogbn_papers100M0 --cache_type p3 --partition_type metis --multi_feat_file True', # 补充之前缺少的几组实验

    # in-uk
    # 'python3 servers_start.py --dataset in_2004 --cache_type static --partition_type metis --multi_feat_file True', # 补充之前缺少的几组实验
    # 'python3 servers_start.py --dataset uk_2007 --cache_type static --partition_type metis --multi_feat_file True', # 补充之前缺少的几组实验
    # 'python3 servers_start.py --dataset in_2004 --cache_type p3 --partition_type metis --multi_feat_file True', # 补充之前缺少的几组实验
    # 'python3 servers_start.py --dataset uk_2007 --cache_type p3 --partition_type metis --multi_feat_file True', # 补充之前缺少的几组实验
    # 'python3 servers_start.py --dataset uk --cache_type static --partition_type pagraph --multi_feat_file',
    # 'python3 servers_start.py --dataset uk --cache_type p3 --partition_type pagraph --multi_feat_file',
    # 'python3 servers_start.py --dataset ogbn_papers100M0 --cache_type static --partition_type metis --multi_feat_file True', # 补充之前缺少的几组实验
    # 'python3 servers_start.py --dataset twitter --cache_type static --partition_type metis --multi_feat_file True', # 补充之前缺少的几组实验
    # 'python3 servers_start.py --dataset ogbn_papers100M0 --cache_type p3 --partition_type metis --multi_feat_file True', # 补充之前缺少的几组实验
    # 'python3 servers_start.py --dataset twitter --cache_type p3 --partition_type metis --multi_feat_file True', # 补充之前缺少的几组实验

    # 18*4 bar
    'python3 servers_start.py --dataset ogbn_arxiv0 --cache_type static --partition_type metis',
    'python3 servers_start.py --dataset ogbn_products0 --cache_type static --partition_type metis',
    'python3 servers_start.py --dataset in_2004 --cache_type static --partition_type metis --multi_feat_file True',
    'python3 servers_start.py --dataset uk_2007 --cache_type static --partition_type metis --multi_feat_file True',
    'python3 servers_start.py --dataset ogbn_arxiv0 --cache_type p3 --partition_type metis',
    'python3 servers_start.py --dataset ogbn_products0 --cache_type p3 --partition_type metis',
    'python3 servers_start.py --dataset in_2004 --cache_type p3 --partition_type metis --multi_feat_file True',
    'python3 servers_start.py --dataset uk_2007 --cache_type p3 --partition_type metis --multi_feat_file True',
]

client_cmd_lst = [
    # 18*4 bar
    # arxiv
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 16',

    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 32',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 32',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 32',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 32',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 32',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 32',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 32',

    #products    
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',

    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 32',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 32',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 32',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 32',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 32',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 32',

    # in
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 16',

    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 32',

    # uk
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 16',

    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 32',


    # p3
    # arxiv
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 32',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 32',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 2048 --hidden_size 32',

    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 4096 --hidden_size 32',

    

    # products
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 32',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 32',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 32',

    # in
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset in_2004 --n_epochs 5 --batch_size 4096 --hidden_size 32',

    # uk
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 16',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --iter_stop 2 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 32',
    'python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --hidden_size 32',



    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset in_2004 --n_epochs 5 --batch_size 2048 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --dataset in_2004 --n_epochs 5 --batch_size 2048 --iter_stop 4 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --dataset in_2004 --n_epochs 5 --batch_size 4096 --iter_stop 4 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --dataset in_2004 --n_epochs 5 --batch_size 2048 --iter_stop 4 --hidden_size 32',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --dataset in_2004 --n_epochs 5 --batch_size 4096 --iter_stop 4 --hidden_size 32',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --dataset uk_2007 --n_epochs 5 --batch_size 2048 --iter_stop 4 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --iter_stop 2 --hidden_size 16',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --dataset uk_2007 --n_epochs 5 --batch_size 2048 --iter_stop 4 --hidden_size 32',
    # 'python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 5 --dataset uk_2007 --n_epochs 5 --batch_size 4096 --iter_stop 2 --hidden_size 32',
]


def check_and_delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'Deleted file {file_path}')
        return 1
    else:
        return 0

if __name__ == '__main__':
    file_path = '/home/qhy/gnn/repgnn/dist/server_done.txt'

    client_cmd_num = [24,12,12,12,12,6,6,6]
    offset = 0
    for server_cmdid, server_cmd in enumerate(server_cmd_lst):
        returncode_server = os.system(server_cmd)
        print(f'returncode_server: {returncode_server}')
        if returncode_server == 0:
            # 检查server是否启动完成
            while True:
                time.sleep(60)
                exist = check_and_delete_file(file_path)
                if exist:
                    break
                
            # 执行几条client命令
            num = client_cmd_num[server_cmdid]
            client_cmdids = [i for i in range(offset,offset + num)]
            offset = offset + num
            for client_cmdid in client_cmdids:
                returncode_client = os.system(client_cmd_lst[client_cmdid])
                print(f'returncode is {returncode_client}')
                if returncode_client == 0:
                    time.sleep(10)
                    os.system('python3 clients_kill.py')
                    time.sleep(10)
                    continue
                else:
                    print(f"Command {client_cmd_lst[client_cmdid]} failed, retrying in 1 minute...")
                    time.sleep(10)
                    os.system('python3 clients_kill.py')
                    time.sleep(10)
            # kill server and remaining client
            os.system('python3 servers_kill.py')
            time.sleep(10)
            os.system('python3 clients_kill.py')
            time.sleep(10)
        else:
            print(f"Command {server_cmd} failed")

