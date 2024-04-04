import os
import time
import getpass

server_cmd_lst = [
    # # 18*4 bar
    # # hd 128
    # # static
    'python3 servers_start.py --dataset ogbn_arxiv0 --cache_type static --partition_type metis',
    # 'python3 servers_start.py --dataset ogbn_products0 --cache_type static --partition_type metis',
    # 'python3 servers_start.py --dataset in_2004 --cache_type static --partition_type metis --multi_feat_file True',
    # 'python3 servers_start.py --dataset uk_2007 --cache_type static --partition_type metis --multi_feat_file True',
    # # p3
    'python3 servers_start.py --dataset ogbn_arxiv0 --cache_type p3 --partition_type metis',
    # 'python3 servers_start.py --dataset ogbn_products0 --cache_type p3 --partition_type metis',
    # 'python3 servers_start.py --dataset in_2004 --cache_type p3 --partition_type metis --multi_feat_file True',
    # 'python3 servers_start.py --dataset uk_2007 --cache_type p3 --partition_type metis --multi_feat_file True',
]

client_cmd_lst = [
    # deep models
    'python3 clients_start.py --model_name deepergcn --sampling 2-2-2 --run_client_idx 6 --iter_stop 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name deepergcn --sampling 2-2-2 --run_client_idx 9 --iter_stop 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name film --sampling 2-2-2 --run_client_idx 6 --iter_stop 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name film --sampling 2-2-2 --run_client_idx 9 --iter_stop 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    # p3
    'python3 clients_start.py --model_name deepergcn --sampling 2-2-2 --run_client_idx 5 --iter_stop 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
    'python3 clients_start.py --model_name film --sampling 2-2-2 --run_client_idx 5 --iter_stop 3 --dataset ogbn_arxiv0 --n_epochs 5 --batch_size 8000 --hidden_size 128',
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

    client_cmd_num = [4,2]
    offset = 0
    for server_cmdid, server_cmd in enumerate(server_cmd_lst):
        returncode_server = os.system(server_cmd)
        print(f'returncode_server: {returncode_server}')
        if returncode_server == 0:
            # 检查server是否启动完成
            while True:
                time.sleep(30)
                exist = check_and_delete_file(file_path)
                if exist:
                    break
                
            # 执行几条client命令
            num = client_cmd_num[server_cmdid]
            client_cmdids = [i for i in range(offset,offset + num)]
            offset = offset + num
            for client_cmdid in client_cmdids:
                print(f'start run {client_cmd_lst[client_cmdid]}')
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

