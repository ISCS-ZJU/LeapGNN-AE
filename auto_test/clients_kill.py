# 运行本文件要求集群中个机器上，repgnn的代码路径一致
import os, sys
import subprocess
import yaml
import asyncio
import getpass


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

    return (
        cluster_servers,
        ssh_pswd,
        server_config_file
    )


# 远程异步执行命令
async def remote_run_command(ssh_pswd, serverip, cmd, ):
    ssh_cmd = f'sshpass -p {ssh_pswd} ssh {serverip} "{cmd}"'
    print(f"Start running {cmd} on remote machine.")
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
) = parse_server_config(auto_test_file)

# 在远程服务器上后台异步运行 server.go
for serverip in cluster_servers:
    # 在每个节点异步执行 go run server.go
    cmd = f"ps -ef | grep {getpass.getuser()}" + " | grep python | grep repgnn | grep -v 'grep' | awk '{print \$2}' | xargs kill -9"
    asyncio.run(remote_run_command(ssh_pswd, serverip, cmd))
