import re
import pandas as pd
import os
import yaml
import numpy as np
import argparse
import math
table_offset = 7

def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='log analys')
    parser.add_argument('-d', '--dir', default="./logs",
                        type=str, help='log dir')
    return parser.parse_args(argv)

# def parse_config(confpath):
#     with open(confpath, 'r') as fh:
#         data = yaml.safe_load(fh)
#         return data['files_to_analys']

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

def find_ep_number(string):
    pattern = r"ep(\d+)"
    match = re.search(pattern, string)
    if match:
        number = int(match.group(1))
        return number
    else:
        return None

def divide_by_epoch_num(value, epoch_num):
    if isinstance(value, (int, float)):
        return value / epoch_num
    else:
        return value

# analys_list = parse_config('./log_analys.yaml')

pattens = {
    'total_local_feats_gather_time' : 1,
    'total_remote_feats_gather_time' : 1,
    'total epochs time' : table_offset,
    '  fetch feat  ' : table_offset,
    'fetch feat from cache server' : table_offset,
    'sync before compute' : table_offset,
    'gpu-compute with optimizer.step' : table_offset,
    'gpu-compute' : table_offset,
    'model transfer' : table_offset,
    'miss_num' : 8,
    'try_num' : 9,
    'wait sampler total time' : 1,
    'sync for each sub_iter': table_offset,
    'train data prepare': table_offset,
    'topology transfer': table_offset,
    'all_reduce hidden vectors': table_offset,
    'get_nfs': table_offset
}

columns_mapp = {
    'wait sampler total time' : 'sampling stall (s)',
    'total epochs time' :'total epochs time (s)',
    'fetch feat from cache server' : 'fetch feats from cache server',
    'gpu-compute with optimizer.step' : 'GPU computing (s)',
    # 'gpu-compute' : 'GPU computing (s)',
    'total_local_feats_gather_time' : 'fetch local time(s)',
    'total_remote_feats_gather_time' : 'fetch remote time(s)',
    'try_num' : '# client-server request nodes',
    'miss_num' : '# local missed',
    'model transfer' : 'model transfer',
    'sync before compute' : 'sync before bkwd',
    'train data prepare' : 'train data prepare for jp'

}  # 字典，键为原始列名，值为目标列名

order = [
    'name',
    'total epochs time (s)',
    'sampling stall (s)',
    '  fetch feat  ',
    'fetch feats from cache server',
    'fetch local time(s)',
    'fetch remote time(s)',
    'GPU computing (s)',
    'sync before bkwd',
    'sync for each sub_iter',
    'others(grpc call etc) (s)',
    'topology transfer',
    'train data prepare for jp',
    'model transfer',
    'all_reduce hidden vectors',
    '# client-server request nodes',
    '# local missed',
    'miss-rate',
]

if __name__ == '__main__':
    # number = re.findall("[\d,.]+",str)
    args = parse_args_func(None)
    datas = []
    hash_map = {}
    # 把 analys_list 中的文件夹中的.log日志文件抽取出来并排序，方便观看
    # analys_list = extract_log_files(analys_list)
    analys_list = extract_log_files([args.dir])
    analys_list = sorted(analys_list, key=lambda x: os.path.basename(x))
    for file_path in analys_list:
        iter_num = -1
        avg_epoch_time_jpless = -1
        new_jp_times = -1
        # 确定epoch num，从而后续计算平均一个epoch的时间，方便比较
        epoch_num = find_ep_number(file_path)
        with open(file_path,'r') as f:
            data = dict()
            data['name'] = file_path.split('/')[-1]
            lines = f.readlines()
            for line in lines:
                for patten, offset in pattens.items():
                    if patten in line:
                        if patten == 'gpu-compute' and 'with optimizer.step' in line:
                            continue
                        remain = line.split(patten)[1].split()
                        # print(re.findall("[\d .]+",remain[offset])[0])
                        data[patten] = float(re.findall("[\d .]+",remain[offset])[0])
                        if 'ms' in remain[offset]:
                            data[patten] /= 1000
                if 'dataset=\'' in line:
                    dataset = line.split('dataset=\'')[1].split('\',')[0]
                if 'iter_num:' in line:
                    iter_num = int(float(line.split('iter_num:')[1]))
                if 'world_size=' in line:
                    world_size = int(line.split('world_size=')[1].split(',')[0])
                if 'batch_size=' in line:
                    batch_size =int(line.split('batch_size=')[1].split(',')[0])
                if 'avg_epoch_time of' in line:
                    avg_epoch_time_jpless = float(line.split("avg_epoch_time of [")[1].split(' ')[0][:-1])
                if 'new jp_times' in line:
                    new_jp_times = int(line.split('new jp_times is ')[1])
                if 'Got feature dim from server: ' in line:
                    data['feat_dim'] = int(line.split('Got feature dim from server: ')[1])
                if 'ori #nodes = ' in line:
                    data['ori_nodes'] = int(line.split('ori #nodes = ')[1].split(',')[0])
                    data['ori_edges'] = int(line.split('#edges = ')[1].split()[0])
                if 'nstart #nodes = ' in line:
                    data['nodes'] = int(line.split('nstart #nodes = ')[1].split(',')[0])
                    data['edges'] = int(line.split('#edges = ')[1].split()[0])
                    data['total epochs time'] = 0
                
            tol_node = np.sum(np.load(os.path.join(dataset,'train.npy')))
            exp_iter = math.ceil(tol_node//world_size/batch_size)
            if iter_num != -1:
                rate = exp_iter / iter_num 
            else:
                rate = 1

        if 'total epochs time' not in data.keys():
            continue
        # 后处理
        if 'model transfer' not in data.keys():
            data['model transfer'] = 0 # 填充默认值
        if 'gpu-compute' in data.keys():
            data['gpu-compute with optimizer.step'] = data.pop('gpu-compute') # 把 gpu-compute 的列等价成 gpu-compute with optimier.step
        
        if 'train data prepare' not in data.keys():
            data['train data prepare'] = 0
        if 'sync for each sub_iter' not in data.keys():
            data['sync for each sub_iter'] = 0
        
        if 'total_local_feats_gather_time' not in data.keys():
            data['total_local_feats_gather_time'] = 0
        if 'total_remote_feats_gather_time' not in data.keys():
            data['total_remote_feats_gather_time'] = 0
        if 'miss_num' not in data.keys():
            data['miss_num'] = 0
        if 'try_num' not in data.keys():
            data['try_num'] = 0
        if 'topology transfer' not in data.keys():
            data['topology transfer'] = 0
        if 'all_reduce hidden vectors' not in data.keys():
            data['all_reduce hidden vectors'] = 0
        if 'sync before compute' not in data.keys():
            data['sync before compute'] = 0
        if 'get_nfs' not in data.keys():
            data['get_nfs'] = 0
        if 'nodes' not in data.keys():
            data['nodes'] = 0
        if 'edges' not in data.keys():
            data['edges'] = 0
        if 'ori_nodes' not in data.keys():
            data['nodes'] = 0
        if 'ori_edges' not in data.keys():
            data['edges'] = 0
        if 'feat_dim' not in data.keys():
            data['feat_dim'] = 0
        
        
        
        # 如果是p3，那么总时间去掉 topology transfer
        if 'p3' in file_path:
            data['total epochs time'] -= data['get_nfs']


        
        # 如果是jpless，自动提取最小的时间作为total_epoch_time，其余的时间自动缩小相应倍数，打印最终跳跃次数
        if avg_epoch_time_jpless!= -1:
            print(file_path, 'final jp_times:', new_jp_times)
            scale = data['total epochs time'] / avg_epoch_time_jpless / epoch_num
            for k, v in data.items():
                if k != '# client-server request nodes' and k!= '# local missed' and k!='miss-rate' and k!='name':
                    data[k] = v / scale

        # 计算平均每个epoch的时间
        for k, v in data.items():
            data[k] = divide_by_epoch_num(v, epoch_num / rate)
        
        if data['try_num'] != 0:
            data['miss-rate'] = data['miss_num'] / data['try_num']
        else:
            data['miss-rate'] = '/'

        datas.append(data)
        hash_map[data['name']] = data
        # print(data)
    for data in datas:
        if 'naive' in data['name']:
            compute_file_name = 'default_' + data['name'].split('naive_')[1].split('.log')[0] + '_localFalse.log'
            # compute_file_name = 'jpgnn_trans_lessjp_dedup_True_' + data['name'].split('_',1)[1]
            if compute_file_name not in hash_map:
                continue
            if 'gpu-compute with optimizer.step' in hash_map[compute_file_name]:
                data['total epochs time'] += hash_map[compute_file_name]['gpu-compute with optimizer.step']
            elif 'GPU computing (s)' in hash_map[compute_file_name]:
                data['total epochs time'] += hash_map[compute_file_name]['GPU computing (s)']
        if 'neutronstar' in data['name']:
            compute_file_name = 'default_' + data['name'].split('neutronstar_')[1]
            if 'gpu-compute with optimizer.step' in hash_map[compute_file_name]:
                ori_compute_time = hash_map[compute_file_name]['gpu-compute with optimizer.step']
            elif 'GPU computing (s)' in hash_map[compute_file_name]:
                ori_compute_time = hash_map[compute_file_name]['GPU computing (s)']
            hd = int(compute_file_name.split('hd')[1].split('_')[0])
            ori_total_time = hash_map[compute_file_name]['total epochs time']
            fetch_cost_each_node = hash_map[compute_file_name]['total_remote_feats_gather_time']/hash_map[compute_file_name]['miss_num']
            data['total epochs time'] = ori_total_time - ori_compute_time + ori_compute_time*(data['nodes']+data['edges'])/(data['ori_nodes']+data['ori_edges']) + fetch_cost_each_node*(data['ori_nodes']-data['nodes'])*hd/hash_map[compute_file_name]['feat_dim']
    pf = pd.DataFrame(datas)
    # print(pf)
    pf.rename(columns=columns_mapp, inplace=True)  # 直接修改原table
    # pf['others(grpc call etc) (s)']=pf['total epochs time (s)'].sub(pf[['sampling stall (s)','fetch feats from cache server','GPU computing (s)','model transfer','sync before bkwd', 'sync for each sub_iter', 'train data prepare for jp']].sum(axis=1))
    # pf['others(grpc call) (s)'] = pf['total time/epoch (s)'] - pf['sampling stall (s)'] - pf['fetch feats from cache server'] - pf['GPU computing (s)'] - pf['model transfer'] - pf['syn']
    # if pf['# client-server request nodes'].bool():
    #     pf['miss-rate'] = pf['# local missed'].div(pf['# client-server request nodes'])
    # else:
    #     pf['miss-rate'] = '/'
    # pf = pf[order]
    # print(pf)
    if os.path.exists(f'./{os.path.basename(args.dir)}.csv'):
        os.system("rm " + os.path.join(".", f'{os.path.basename(args.dir)}.csv'))
    pf.to_csv(f'./{os.path.basename(args.dir)}.csv', float_format='%.5f',index=False)
    print(f'-> write to {os.path.basename(args.dir)}.csv done.')

