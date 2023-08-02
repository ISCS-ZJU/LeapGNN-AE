# 运行本文件要求集群中个机器上，repgnn的代码路径一致
import os, sys
import subprocess
import yaml
import asyncio
import multiprocessing
import shutil
import argparse

# 确定当前机器所要使用的ip, 网卡名和rank
def get_ip_interface_rank(cluster_servers, localip_interfaces):
    for localip, netname in localip_interfaces:
        for r, ip in enumerate(cluster_servers):
            if localip == ip.strip():
                return localip, netname, r
    raise Exception(
        f"ERROR: Local IPs {localip_interfaces} are not in cluster_servers {cluster_servers}."
    )


# 读取配置文件
def parse_test_config(confpath):
    with open(confpath, 'r') as fh:
        data = yaml.safe_load(fh)
        # cluster info
        cluster_servers = data['cluster_servers']
        cluster_build_port = data['cluster_build_port']
        grpc_port = data['grpc_port']
        client_file_to_run = data['client_files'][data['run_client_idx']]

        # training info
        model_name = data['model_name']
        batch_size = data['batch_size']
        sampling = data['sampling']
        n_epochs = data['n_epochs']
        dataset = data['dataset']
        learning_rate = data['learning_rate']
        hidden_size = data['hidden_size']

        # statistic
        log = data['log']
        evalu = data['eval']

        # ssh info
        ssh_pswd = data['ssh_pswd']
    return (
        len(cluster_servers),
        cluster_servers,
        cluster_build_port,
        grpc_port,
        client_file_to_run,
        model_name,
        batch_size,
        sampling,
        n_epochs,
        learning_rate,
        hidden_size,
        dataset,
        log,
        evalu,
        ssh_pswd,
    )

def remote_run_command(ssh_pswd, serverip, cmd):
    ssh_cmd = f'sshpass -p {ssh_pswd} ssh {serverip} "bash -c \'{cmd}\'"'
    print(f"-> Start running ' {cmd} ' on remote {serverip} machine.")
    # 在远程机器执行命令
    result = subprocess.run(ssh_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)
    print(result.stderr)
    return result

def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Command Line Argument Parser')

    parser.add_argument('--model_name', type=str, default='',
                        help='Name of the model')
    parser.add_argument('--batch_size', type=int, default=-1,
                        help='Batch size for training')
    parser.add_argument('--sampling', type=str, default='',
                        help='Sampling fanout with layers')
    parser.add_argument('--n_epochs', type=int, default=-1,
                        help='Number of training epochs')
    parser.add_argument('--hidden_size', type=int, default=-1,
                        help='Size of the hidden layer')
    parser.add_argument('--run_client_idx', type=int, default=-1,
                        help = 'The chosen client file to run.')
    parser.add_argument('--dataset', type=str, default='',
                        help = 'Dataset name.')
    parser.add_argument('--cluster_servers', type=str, default='',
                        help='cluster IPs')
    args = parser.parse_args()

    return args

# 读取 auto test config 文件
auto_test_file = './test_config.yaml'
(
    world_size,
    cluster_servers,
    cluster_build_port,
    grpc_port,
    client_file_to_run,
    model_name,
    batch_size,
    sampling,
    n_epochs,
    learning_rate,
    hidden_size,
    dataset,
    log,
    evalu,
    ssh_pswd,
) = parse_test_config(auto_test_file)

# 覆写 yaml 中定义的值，方便批量执行client命令
args = parse_command_line_args()
if args.model_name != '':
    model_name = args.model_name
if args.batch_size != -1:
    batch_size = args.batch_size
if args.sampling != '':
    sampling = args.sampling
if args.n_epochs != -1:
    n_epochs = args.n_epochs
if args.hidden_size != -1:
    hidden_size = args.hidden_size
if args.run_client_idx != -1:
    with open(auto_test_file, 'r') as fh:
        data = yaml.safe_load(fh)
    client_file_to_run = data['client_files'][args.run_client_idx]
if args.dataset != '':
    dataset = args.dataset
if args.cluster_servers != '':
    cluster_servers = eval(args.cluster_servers)
    world_size = len(cluster_servers)

ip_interface_rank = {}  # key: serverip value:(interface, rank)
cur_dir = os.path.dirname(os.path.abspath(__file__))
for serverip in cluster_servers:
    cmd = f'sshpass -p {ssh_pswd} ssh {serverip} "cd {cur_dir} && python3 -c \'from getip import *; print(get_local_ip4_and_interface())\'"'
    output = subprocess.check_output(cmd, shell=True)
    ips_lst = eval(output.decode())  # convert output from byte to string and eval() it
    # ensure ip, interface, and rank
    ip, interface, rank = get_ip_interface_rank(cluster_servers, ips_lst)
    ip_interface_rank[ip] = (interface, rank)


# python client 的文件路径
client_file_to_run = os.path.abspath(os.path.join("../", client_file_to_run))
client_dir = os.path.dirname(client_file_to_run)
dataset_dir = os.path.join(client_dir, f'./dist/repgnn_data/{dataset}')

# 在远程服务器上后台异步运行 gnn training client
processes = []
for serverip in cluster_servers:
    interface, rank = ip_interface_rank[serverip]
    env_cmd = f"source `which conda | xargs readlink -f | xargs dirname | xargs dirname`/bin/activate && conda activate repgnn && cd {client_dir} && export GLOO_SOCKET_IFNAME={interface}"
    cmd = f"{env_cmd} && time python3 {client_file_to_run} -mn {model_name} -bs {batch_size} -s {sampling} -ep {n_epochs} -lr {learning_rate} --dist-url 'tcp://{cluster_servers[0]}:{cluster_build_port}' --world-size {world_size} --rank {rank} --grpc-port {serverip}:{grpc_port} -d {dataset_dir} -hd {hidden_size}"
    if log == True:
        cmd += " --log"
    if evalu == True:
        cmd += " --eval"
    # 获取运行本文件时添加的额外命令行参数
    # cmd += ' '
    # cmd += ' '.join(sys.argv[1:])
    # # remove 
    # asyncio.run(remote_run_command(ssh_pswd, serverip, cmd))
    p = multiprocessing.Process(target=remote_run_command, args=(ssh_pswd, serverip, cmd))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

# 把远程服务器上 logs 目录中的 *.log 文件都拷贝到本地
logs_dir = os.path.join(os.path.dirname(cur_dir), 'logs')
logs_file = logs_dir + '/*.log'
dest_logs_dir = os.path.dirname(logs_dir)
for serverip, (_, rank) in ip_interface_rank.items():
    dest_logs_path = os.path.join(dest_logs_dir, f"logs_rank{rank}")
    if os.path.exists(dest_logs_path):
        shutil.rmtree(dest_logs_path)
    os.mkdir(dest_logs_path)
    cmd = f'sshpass -p {ssh_pswd} scp -o StrictHostKeyChecking=no {serverip}:{logs_file} {dest_logs_dir}/logs_rank{rank}/'
    os.system(cmd)