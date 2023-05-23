import git
import os

# 执行本文件需要到git仓库根目录，否则get_modifed_files函数执行报错

def get_modifed_files(gitpath):
    repo = git.Repo(gitpath)
    modified_files = [item.a_path for item in repo.index.diff(None) if item.change_type != "D"]
    # print(modified_files)
    return modified_files

def scp_modifed_files_to_another_mathine_under_same_path(remote_ip, pswd):
    files_lst = get_modifed_files(os.getcwd())
    print(files_lst)
    instruct = input(f'将传输上述文件到{remote_ip}同名路径下，请确认文件路径是否正确[y/n]：')
    if instruct=='y':
        for rip in remote_ip.split():
            for filename in files_lst:
                os.system(f'sshpass -p {pswd} scp {filename} {rip}:{os.path.join(os.getcwd(), filename)}')
                print(f'-> scp {filename} {rip}:{os.path.join(os.getcwd(), filename)} 执行结束')
            print()
    elif instruct == 'n':
        print('未开始传输，强制结束')
    else:
        print('非法输入，请选择y/n')

remote_IP = input('请输入目的机器域名或ip (如果多个机器使用空格隔开) ：')
pswd = input('请输入ssh remote_IP 的密码：')
scp_modifed_files_to_another_mathine_under_same_path(remote_ip=remote_IP, pswd=pswd)