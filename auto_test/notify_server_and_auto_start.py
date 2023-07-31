import subprocess
import os

def tail(file_path, n=5):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return lines[-n:]

def check_and_execute_script(file_path, target_string, script_path):
    while True:
        # 使用inotifywait监视文件的变化，等待文件有新内容写入
        subprocess.run(['inotifywait', '-e', 'modify', file_path])

        # 获取文件的最后一行
        last_line = tail(file_path)
        print(last_line, ''.join(last_line))

        # 检查是否包含目标字符串
        if target_string in ''.join(last_line):
            print("目标字符串已出现！开始执行脚本...")
            # 执行指定脚本
            subprocess.run(['bash', script_path])
            break

if __name__ == "__main__":
    # 要监视的文件路径
    monitored_file_path = "/home/weijian/gitclone/repgnn/logs/server_output_10.214.241.227.log"

    # 目标字符串
    target_string = "启动完成"

    # 要执行的脚本路径
    batch_run_script = "./batch_run.sh"

    check_and_execute_script(monitored_file_path, target_string, batch_run_script)
