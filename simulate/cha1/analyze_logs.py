import os, sys
import re
import copy
import pandas as pd

def extract_log_files(file_paths):
    # 有一个list变量，其中有很多字符串表示文件路径或者文件夹路径，抽取出其中所有以'.log'为结尾的文件路径，放到列表并返回
    log_files = []
    for path in file_paths:
        if os.path.isfile(path) and path.endswith('.log'):
            log_files.append(path)
        elif os.path.isdir(path):
            sub_files = [os.path.join(path, f) for f in os.listdir(path)]
            log_files.extend(extract_log_files(sub_files))
    return log_files


def analyze_logs(logf):
    if not os.path.exists(logf):
        print(f'ERROR: file {logf} does not exists.')
        sys.exit(-1)
    # get featdim
    featdim = -1
    with open(logf) as f:
        for line in f:
            if 'featdim=' in line:
                match = re.search(r'featdim=(\d+)', line)
                featdim = int(match.group(1))
    if '_arxiv0' in logf:
        featdim = 128
    elif '_products0' in logf:
        featdim = 100
    elif '_in_2004' in logf:
        featdim = 600
    elif '_uk_2007' in logf:
        featdim = 600
    print(f'Original featdim = {featdim}')
    assert featdim > 0, 'featdim error: featdim should be larger than zero'

    # 对所有machine上相应的数据量进行加和
    total_batches, total_tree_nodes, total_request_nodes = 0, 0, 0

    naive_total_model_optim_transfer, naive_total_aggr_transfer, naive_total_comb_transfer, naive_total_actv_transfer = 0, 0, 0, 0
    naive_total_transfer = 0 

    with open(logf) as f:
        for line in f:
            if 'rank=0' in line:
                # 采样点数和本地命中情况信息统计
                if 'number of training batches' in line:
                    total_batches += int(line.split('number of training batches:')[-1])
                elif 'total nodes:' in line:
                    total_tree_nodes += int(line.split('total nodes:')[-1])
                elif 'nodes hits on other trainers:' in line:
                    total_request_nodes += sum(eval(line.split('nodes hits on other trainers:')[-1]))
                # naive transfer方式下的传输数据量统计
                elif 'total data transfer:' in line:
                    naive_total_transfer += eval(line.split('total data transfer:')[-1])
                elif 'total model&opt size transfer:' in line:
                    naive_total_model_optim_transfer += eval(line.split("total model&opt size transfer:")[-1])
                elif 'total aggr size transfer:' in line:
                    naive_total_aggr_transfer += eval(line.split('total aggr size transfer:')[-1])
                elif 'total comb size transfer:' in line:
                    naive_total_comb_transfer += eval(line.split('total comb size transfer:')[-1])
                elif 'total actv size transfer:' in line:
                    naive_total_actv_transfer += eval(line.split('total actv size transfer:')[-1])


    # print(f'total number of batches: {total_batches}')
    # print(f'total number of tree nodes: {total_tree_nodes}, avg per batch tree ndata: {total_tree_nodes//total_batches}, avg per batch tree size: {total_tree_nodes/total_batches*featdim*4/1024/1024} MB')
    # print(f'total number of remote request nodes: {total_request_nodes}, avg per batch remote request ndata: {total_request_nodes // total_batches}, avg per batch remote request size: {total_request_nodes/total_batches*featdim*4/1024/1024} MB')
    # print(f"{'='*10}, naive transfer 数据迁移量统计, {'='*10}")
    # print(f'total model&optm transfer ndata: {naive_total_model_optim_transfer}, size: {naive_total_model_optim_transfer*4/1024/1024} MB, avg per batch size: {naive_total_model_optim_transfer*4/1024/1024/total_batches} MB')
    # print(f'total part aggr transfer ndata: {naive_total_aggr_transfer}, size: {naive_total_aggr_transfer*4/1024/1024} MB, avg per batch size: {naive_total_aggr_transfer*4/1024/1024/total_batches} MB')
    # print(f'total comb transfer ndata: {naive_total_comb_transfer}, size: {naive_total_comb_transfer*4/1024/1024} MB, avg per batch size: {naive_total_comb_transfer*4/1024/1024/total_batches} MB')
    # print(f'total actv transfer ndata: {naive_total_actv_transfer}, size: {naive_total_actv_transfer*4/1024/1024} MB, avg per batch size: {naive_total_actv_transfer*4/1024/1024/total_batches} MB')
    # print(f'total transfer ndata: {naive_total_transfer}, size: {naive_total_transfer*4/1024/1024} MB, avg per batch size: {naive_total_transfer*4/1024/1024/total_batches}')
    return naive_total_transfer, total_tree_nodes * featdim, naive_total_model_optim_transfer, naive_total_aggr_transfer, naive_total_comb_transfer, naive_total_actv_transfer
    
def get_gb(a):
    return a * 4 / 1024 / 1024 / 1024

if __name__ == '__main__':
    # logf = input('Please input the log file path+name to analyze: ')
    analys_list = extract_log_files(['/home/qhy/gnn/repgnn/simulate/cha1/logs/naive2'])
    # analys_list = ['/home/qhy/gnn/repgnn/simulate/cha1/logs/default_gcn_ogbn_arxiv0_trainer4_bs256_sl10-10_ep1_hd256.log']
    analys_list = sorted(analys_list, key=lambda x: os.path.basename(x))
    datas = []
    with open('./total.txt','w') as f:
        f.write('trans_num={\n')
        for file_path in analys_list:
            data = {}
            log_name = copy.deepcopy(file_path)
            model_name = log_name.split('_',2)[1]
            log_name = log_name.split('_',2)[2]
            dataset = log_name.split('_trainer',1)[0]
            log_name = log_name.split('_trainer',1)[1]
            bs = eval(log_name.split('_')[1].split('bs')[1])
            log_name = log_name.split('_',1)[1]
            hd = log_name.split('_hd')[-1].split('.')[0]
            # print(f'{model_name}_{dataset}_{bs}_{hd}')
            naive_total_transfer, default_transfer, model, aggr, comb, actv = analyze_logs(file_path)
            # f.write(f'\'{model_name}_{dataset}_{bs}_{hd}\' : naive:{get_gb(naive_total_transfer)}GB, default:{get_gb(default_transfer)}GB,  {naive_total_transfer > default_transfer},\n')
            # f.write(f'\'{model_name}_{dataset}_{bs}_{hd}\' : {naive_total_transfer},\n')
            data['model'] = model
            data['aggr'] = aggr
            data['comb'] = comb
            data['actv'] = actv
            data['name'] = f'\'{model_name}_{dataset}_{bs}_{hd}\''
            datas.append(data)
            f.write(f'\'{model_name}_{dataset}_{bs}_{hd}\' : {model}\t{aggr}\t{comb}\t{actv},\n')
        f.write('}\n')
    pf = pd.DataFrame(datas)
    if os.path.exists('./data.xlsx'):
        os.system("rm " + os.path.join(".", 'data.xlsx'))
    writer = pd.ExcelWriter('./data.xlsx')  # 初始化一个writer
    pf.to_excel(writer, float_format='%.5f',index=False)  # table输出为excel, 传入writer
    writer._save()