# 运行本文件要求集群中个机器上，repgnn的代码路径一致
import os, sys
import subprocess
import yaml
import asyncio


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
        # log
        statistic = data['log']

    return (
        cluster_servers,
        ssh_pswd,
        server_config_file,
        dataset_path,
        cache_type,
        statistic
    )


# 远程异步执行命令
async def remote_run_command(ssh_pswd, serverip, cmd, remote_log_file):
    ssh_cmd = f'sshpass -p {ssh_pswd} ssh {serverip} "{cmd} > {remote_log_file} 2>&1 &"'
    print(f"-> Start running {cmd} on remote {serverip} machine.")
    # 在远程机器异步执行后台命令
    process = await asyncio.create_subprocess_shell(
        ssh_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    return process


# 读取 auto test config 文件
auto_test_file = './test_config.yaml'
(
    cluster_servers,
    ssh_pswd,
    server_config_file,
    dataset_path,
    cache_type,
    statistic,
) = parse_server_config(auto_test_file)

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
for serverip in cluster_servers:
    # 在每个节点异步执行 go run server.go 
    cmd = f'cd {exec_dir} && nohup `which go` run server.go -dataset {dataset_path} -cachegroup "{cache_group}" -cachetype {cache_type}'
    if statistic:
        cmd += f' -statistic'
    remote_log_file = os.path.join(log_dir, f"server_output_{serverip}.log")
    asyncio.run(remote_run_command(ssh_pswd, serverip, cmd, remote_log_file))
    # remote_run_command(ssh_pswd, serverip, cmd, remote_log_file)
