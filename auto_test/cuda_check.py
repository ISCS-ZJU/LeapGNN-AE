import os, sys
import subprocess
import yaml
import asyncio
import multiprocessing
import shutil
import argparse
cluster_servers = ['yq4','yq5','yq6','yq7']

for serverip in cluster_servers:
    print(f'server:{serverip}')
    cmd = f'sshpass  ssh -p {2022} {serverip} "nvidia-smi"'
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)
    print(result.stderr)
    cmd = f'sshpass  ssh -p {2022} {serverip} "free -h"'
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)
    print(result.stderr)