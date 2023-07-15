import re
import pandas as pd
import os
import yaml
table_offset = 7

def parse_config(confpath):
    with open(confpath, 'r') as fh:
        data = yaml.safe_load(fh)
        return data['files_to_analys']

analys_list = parse_config('./log_analys.yaml')

pattens = {
    'total_local_feats_gather_time' : 1,
    'total_remote_feats_gather_time' : 1,
    'total epochs time' : table_offset,
    'fetch feat from cache server' : table_offset,
    'sync before compute' : table_offset,
    'gpu-compute with optimizer.step' : table_offset,
    'gpu-compute' : table_offset,
    'model transfer' : table_offset,
    'miss_num' : 8,
    'try_num' : 9,
    'wait sampler total time' : 1,
    'sync for each sub_iter': table_offset,
    'train data prepare': table_offset
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
    'fetch feats from cache server',
    'fetch local time(s)',
    'fetch remote time(s)',
    'GPU computing (s)',
    'sync before bkwd',
    'sync for each sub_iter',
    'others(grpc call etc) (s)',
    'train data prepare for jp',
    'model transfer',
    '# client-server request nodes',
    '# local missed',
    'miss-rate'
]

if __name__ == '__main__':
    # number = re.findall("[\d,.]+",str)
    datas = []
    for file_path in analys_list:
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
        # 后处理
        if 'model transfer' not in data.keys():
            data['model transfer'] = 0 # 填充默认值
        if 'gpu-compute' in data.keys():
            data['gpu-compute with optimizer.step'] = data.pop('gpu-compute') # 把 gpu-compute 的列等价成 gpu-compute with optimier.step
        
        if 'train data prepare for jp' not in data.keys():
            data['train data prepare for jp'] = 0
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
        
        if data['try_num'] != 0:
            data['miss-rate'] = data['miss_num'] / data['try_num']
        else:
            data['miss-rate'] = '/'

        datas.append(data)
    pf = pd.DataFrame(datas)
    # print(pf)
    pf.rename(columns=columns_mapp, inplace=True)  # 直接修改原table
    pf['others(grpc call etc) (s)']=pf['total epochs time (s)'].sub(pf[['sampling stall (s)','fetch feats from cache server','GPU computing (s)','model transfer','sync before bkwd', 'sync for each sub_iter', 'train data prepare for jp']].sum(axis=1))
    # pf['others(grpc call) (s)'] = pf['total time/epoch (s)'] - pf['sampling stall (s)'] - pf['fetch feats from cache server'] - pf['GPU computing (s)'] - pf['model transfer'] - pf['syn']
    # if pf['# client-server request nodes'].bool():
    #     pf['miss-rate'] = pf['# local missed'].div(pf['# client-server request nodes'])
    # else:
    #     pf['miss-rate'] = '/'
    pf = pf[order]
    # print(pf)
    if os.path.exists('./data.xlsx'):
        os.system("rm " + os.path.join(".", 'data.xlsx'))
    writer = pd.ExcelWriter('./data.xlsx')  # 初始化一个writer
    pf.to_excel(writer, float_format='%.5f',index=False)  # table输出为excel, 传入writer
    writer._save()

