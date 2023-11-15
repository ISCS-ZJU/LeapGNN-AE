import os
import time

server_cmd_lst = [
    'python3 servers_start.py --dataset ogbn_arxiv0 --cache_type static',
    'python3 servers_start.py --dataset ogbn_products0 --cache_type static',
    'python3 servers_start.py --dataset in_2004 --cache_type static',
    'python3 servers_start.py --dataset uk_2007 --cache_type static',
    # 'python3 servers_start.py --dataset ogbn_arxiv0 --cache_type p3',
    # 'python3 servers_start.py --dataset ogbn_products0 --cache_type p3',
    # 'python3 servers_start.py --dataset in_2004 --cache_type p3',
    # 'python3 servers_start.py --dataset uk_2007 --cache_type p3',
    # 'python3 servers_start.py --dataset ogbn_products0 --cache_type static',
]

client_cmd_lst = [
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 8000 --hidden_size 64',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 1 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 8000 --hidden_size 64',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 64',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 1 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 64',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset in_2004 --n_epochs 3 --batch_size 8000 --hidden_size 64',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 1 --dataset in_2004 --n_epochs 3 --batch_size 8000 --hidden_size 64',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset uk_2007 --n_epochs 3 --batch_size 8000 --hidden_size 64',
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 1 --dataset uk_2007 --n_epochs 3 --batch_size 8000 --hidden_size 64',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 8000 --hidden_size 64',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 64',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset in_2004 --n_epochs 3 --batch_size 8000 --hidden_size 64',
    # 'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset uk_2007 --n_epochs 3 --batch_size 8000 --hidden_size 64',
]


def check_and_delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'Deleted file {file_path}')
        return 1
    else:
        return 0

if __name__ == '__main__':
    file_path = '/home/weijian/gitclone/repgnn/dist/server_done.txt'

    groupsize = 2

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
                
            # 执行两条client命令
            for client_cmdid in range(server_cmdid*groupsize, (server_cmdid+1)*groupsize):
                returncode_client = os.system(client_cmd_lst[client_cmdid])
                if returncode_client == 0:
                    continue
                else:
                    print(f"Command {client_cmd_lst[client_cmdid]} failed, retrying in 1 minute...")
            # kill server and remaining client
            os.system('python3 servers_kill.py')
            time.sleep(10)
            os.system('python3 clients_kill.py')
            time.sleep(10)
        else:
            print(f"Command {server_cmd} failed")

