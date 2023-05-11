import re
import pandas as pd
import os
table_offset = 7

analys_list = [
    '/home/qhy/gnn/repgnn/logs/default_gcn_sampling_pubmed0_trainer2_bs8000_sl10-10.log',
    '/home/qhy/gnn/repgnn/logs/jpgnn_trans_multinfs_dedup_True_gcn_sampling_pubmed0_trainer2_bs8000_sl10-10.log'
]

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
    'wait sampler total time' : 1
}

columns_mapp = {
    'wait sampler total time' : 'sampling stall (s)',
    'total epochs time' :'total time/epoch (s)',
    'fetch feat from cache server' : 'fetch feats from cache server',
    'gpu-compute with optimizer.step' : 'GPU computing (s)',
    # 'gpu-compute' : 'GPU computing (s)',
    'total_local_feats_gather_time' : 'fetch local time(s)',
    'total_remote_feats_gather_time' : 'fetch remote time(s)',
    'try_num' : '# client-server request nodes',
    'miss_num' : '# local missed',
    'model transfer' : 'model transfer',
    'sync before compute' : 'syn',

}  # 字典，键为原始列名，值为目标列名

order = [
    'name',
    'total time/epoch (s)',
    'sampling stall (s)',
    'fetch feats from cache server',
    'GPU computing (s)',
    'fetch local time(s)',
    'fetch remote time(s)',
    'others(grpc call) (s)',
    '# client-server request nodes',
    '# local missed',
    'model transfer',
    'miss-rate',
    'syn'
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
        if 'model transfer' not in data.keys():
            data['model transfer'] = 0
        if 'gpu-compute' in data.keys():
            data['gpu-compute with optimizer.step'] = data.pop('gpu-compute')
        datas.append(data)
    pf = pd.DataFrame(datas)
    # print(pf)
    pf.rename(columns=columns_mapp, inplace=True)  # 直接修改原table
    pf['others(grpc call) (s)']=pf['total time/epoch (s)'].sub(pf[['sampling stall (s)','fetch feats from cache server','GPU computing (s)','model transfer','syn']].sum(axis=1))
    # pf['others(grpc call) (s)'] = pf['total time/epoch (s)'] - pf['sampling stall (s)'] - pf['fetch feats from cache server'] - pf['GPU computing (s)'] - pf['model transfer'] - pf['syn']
    pf['miss-rate'] = pf['# local missed'].div(pf['# client-server request nodes'])
    pf = pf[order]
    # print(pf)
    os.system("rm " + os.path.join(".", 'data.xlsx'))
    writer = pd.ExcelWriter('./data.xlsx')  # 初始化一个writer
    pf.to_excel(writer, float_format='%.5f',index=False)  # table输出为excel, 传入writer
    writer.save()

