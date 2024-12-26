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
    datas = {}
    for file_path in analys_list:
        start_num = int(file_path.split('trainer')[1].split('_')[0])
        jp_times = [start_num]
        with open(file_path,'r') as f:
            lines=f.readlines()
            for line in lines:
                if ' new jp_times is ' in line:
                    jp_times.append(int(line.split(' new jp_times is ')[1].split(' ')[0]))
            ep_times = eval(lines[-1].split('all_epoch_time:')[1])
            jp_time = jp_times[-1] + 1
            while len(jp_times) < len(ep_times):
                jp_times.append(jp_time)
            datas[os.path.basename(file_path)] = (jp_times, ep_times)
            max_len = max(max_len, len(ep_times))
    if os.path.exists(f'./{os.path.basename(args.dir)}.csv'):
        os.system("rm " + os.path.join(".", f'{os.path.basename(args.dir)}.csv'))
    with open(f'./{os.path.basename(args.dir)}.csv','w') as f:
        f.write(f'epoch id,{",".join(str(i) for i in range(max_len))}\n')
        for name,(jp_times,ep_times) in datas.items():
            f.write(f'jp times,{",".join(str(i) for i in jp_times)}')
            f.write(', ' * (max_len - len(jp_times)) + '\n')
            f.write(f'ep times,{",".join(str(i) for i in ep_times)}')
            f.write(', ' * (max_len - len(ep_times)) + '\n')

    print(f'-> write to {os.path.basename(args.dir)}.csv done.')
    