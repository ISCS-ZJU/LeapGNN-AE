import os
import argparse

def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='log analys')
    parser.add_argument('-d', '--dir', default="./logs",
                        type=str, help='log dir')
    return parser.parse_args(argv)

def extract_log_files(file_paths):
    # 有一个list变量，其中有很多字符串表示文件路径或者文件夹路径，抽取出其中所有以'.log'为结尾的文件路径，放到列表并返回
    log_files = []
    for path in file_paths:
        if os.path.isfile(path) and path.endswith('.log'):
            # if 'film' not in path and 'deep' not in path:
            #     continue
            log_files.append(path)
        elif os.path.isdir(path):
            sub_files = [os.path.join(path, f) for f in os.listdir(path)]
            log_files.extend(extract_log_files(sub_files))
    return log_files

if __name__ == '__main__':
    args = parse_args_func(None)
    analys_list = extract_log_files([args.dir])
    analys_list = sorted(analys_list, key=lambda x: os.path.basename(x))
    gpu_utils = {}
    max_len = 0
    for file_path in analys_list:
        with open(file_path,'r') as f:
            lines=f.readlines()
            if 'util_interval=' in lines[0]:
                util_interval = float(lines[0].split('util_interval=')[1].split(',')[0])
            gpu_util = eval(lines[-1].split('gpu util:')[1])
            gpu_utils[os.path.basename(file_path)] = gpu_util
            max_len = max(max_len, len(gpu_util))
    if os.path.exists(f'./{os.path.basename(args.dir)}.csv'):
        os.system("rm " + os.path.join(".", f'{os.path.basename(args.dir)}.csv'))
    with open(f'./{os.path.basename(args.dir)}.csv','w') as f:
        f.write('name,')
        for i in range(max_len):
            f.write(f'{util_interval*i},')
        f.write('\n')
        for name,gpu_util in gpu_utils.items():
            f.write(name)
            for util in gpu_util:
                f.write(',' + str(util))
            f.write(', 0' * (max_len - len(gpu_util)))
            f.write('\n')

    print(f'-> write to {os.path.basename(args.dir)}.csv done.')
    