import os
import time

server_cmd_lst = [
    'python3 server_start.py --dataset ogbn_products50 --cache_type static'
    'python3 server_start.py --dataset ogbn_products200 --cache_type static'
    'python3 server_start.py --dataset ogbn_products400 --cache_type static'
    'python3 server_start.py --dataset ogbn_products800 --cache_type static'
]

client_cmd_lst = [
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products800 --n_epochs 3 --batch_size 8000 --hidden_size 16'
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products800 --n_epochs 5 --batch_size 8000 --hidden_size 16'
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products400 --n_epochs 3 --batch_size 8000 --hidden_size 16'
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products400 --n_epochs 5 --batch_size 8000 --hidden_size 16'
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products200 --n_epochs 3 --batch_size 8000 --hidden_size 16'
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products200 --n_epochs 5 --batch_size 8000 --hidden_size 16'
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products50 --n_epochs 3 --batch_size 8000 --hidden_size 16'
    'python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products50 --n_epochs 5 --batch_size 8000 --hidden_size 16'
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

    for server_cmdid, server_cmd in enumerate(server_cmd_lst):
        returncode_server = os.system(server_cmd)
        if returncode_server == 0:
            # 检查server是否启动完成
            while True:
                time.sleep(60)
                exist = check_and_delete_file(file_path)
                if exist:
                    break
                
            # 执行两条client命令
            client_cmdids = [server_cmdid*2, server_cmdid*2+1]
            for client_cmdid in client_cmdids:
                returncode_client = os.system(client_cmd_lst[client_cmdid])
                if returncode_b == 0:
                    break
                else:
                    print(f"Command {client_cmd_lst[client_cmdid]}failed, retrying in 1 minute...")
        else:
            print(f"Command {server_cmd}failed")

