# 运行本文件要求集群中个机器上，repgnn的代码路径一致
import os, sys
import subprocess
import yaml
import asyncio
import argparse
import multiprocessing

# 读取配置文件
def parse_server_config(confpath):
    with open(confpath, 'r') as fh:
        data = yaml.safe_load(fh)
        # cluster info
        cluster_servers = data['cluster_servers']
        # ssh info
        ssh_pswd = data['ssh_pswd']
        # server config file to run
        server_config_file = data['server_files'][data['run_server_idx']]
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        server_config_file = os.path.join(cur_dir, f"../dist/conf/{server_config_file}")
        # dataset
        dataset = data['dataset']
        dataset_path = os.path.join('./repgnn_data/', dataset)
        # cache type
        cache_type = data['cache_type']
        multi_feat_file = data['multi_feat_file']
        partition_type = data['partition_type']
        # log
        statistic = data['log']

    return (
        cluster_servers,
        ssh_pswd,
        server_config_file,
        dataset_path,
        cache_type,
        statistic,
        multi_feat_file,
        partition_type
    )

def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Command Line Argument Parser')

    parser.add_argument('--cache_type', type=str, default='',
                        help='cache type name')
    parser.add_argument('--dataset', type=str, default='',
                        help='training dataset name')
    args = parser.parse_args()

    return args

# 远程异步执行命令
async def remote_run_command(ssh_pswd, serverip, cmd, remote_log_file):
    ssh_cmd = f'sshpass -p {ssh_pswd} ssh {serverip} "{cmd} > {remote_log_file} 2>&1 &"'
    print(f"-> Start running {cmd} on remote {serverip} machine.")
    # 在远程机器异步执行后台命令
    process = await asyncio.create_subprocess_shell(
        ssh_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    return process

def remote_run_command_sync(ssh_pswd, serverip, cmd):
    ssh_cmd = f'sshpass -p {ssh_pswd} ssh {serverip} "bash -c \'{cmd}\'"'
    print(f"--> Start running ' {cmd} ' on remote {serverip} machine.")
    # ssh到远程机器并执行命令
    os.system(ssh_cmd)

# 读取 auto test config 文件
auto_test_file = './test_config.yaml'
(
    cluster_servers,
    ssh_pswd,
    server_config_file,
    dataset_path,
    cache_type,
    statistic,
    multi_feat_file,
    partition_type,
) = parse_server_config(auto_test_file)

# 覆写 yaml 中定义的值
args = parse_command_line_args()
if args.cache_type != '':
    cache_type = args.cache_type
if args.dataset != '':
    dataset = args.dataset
    dataset_path = os.path.join('./repgnn_data/', dataset)

# 复制 go server config file 到集群其他机器，确保server配置相同
for serverip in cluster_servers:
    scp_cmd = f"sshpass -p {ssh_pswd} scp -o StrictHostKeyChecking=no {server_config_file} {serverip}:{server_config_file}"
    try:
        subprocess.call(scp_cmd, shell=True)
        print(
            f"-> {server_config_file} has been sent to {serverip} on the same dir."
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to sent {server_config_file} to {serverip}. {e}")

# 在远程服务器上后台异步运行 server.go
exec_dir = os.path.abspath('../dist')
log_dir = os.path.abspath('../logs')
cache_group = ','.join(cluster_servers)

print(f'\n=> 开始分割 graph，准备分布式训练')
for serverip in cluster_servers:
    remote_log_file = os.path.join(log_dir, f"server_output_{serverip}.log")
    # 在每个节点异步执行 分图算法
    processes = []
    cmd = f'source `which conda | xargs readlink -f | xargs dirname | xargs dirname`/bin/activate && conda activate repgnn && cd {os.path.abspath("../")} && python3 prepartition/{partition_type}.py --partition {len(cluster_servers)} --dataset ./dist/{dataset_path}'
    p = multiprocessing.Process(target=remote_run_command_sync, args=(ssh_pswd, serverip, cmd))
    p.start()
    processes.append(p)
    for p in processes:
        p.join()

print(f'\n=> 开始运行 server.go，运行分布式服务端')
for serverip in cluster_servers:
    # 在每个节点异步执行 go run server.go 
    cmd = f'cd {exec_dir} && nohup `which go` run server.go -dataset {dataset_path} -cachegroup "{cache_group}" -cachetype {cache_type} -partition_type "{partition_type}"'
    if statistic:
        cmd += f' -statistic'
    if multi_feat_file:
        cmd += f' -multi_feat_file'
    asyncio.run(remote_run_command(ssh_pswd, serverip, cmd, remote_log_file))
    # remote_run_command(ssh_pswd, serverip, cmd, remote_log_file)
