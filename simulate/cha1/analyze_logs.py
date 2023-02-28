import os, sys
import re

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
    print(f'Original featdim = {featdim}')
    assert featdim > 0, 'featdim error: featdim should be larger than zero'

    # 对所有machine上相应的数据量进行加和
    total_batches, total_tree_nodes, total_request_nodes = 0, 0, 0

    naive_total_model_optim_transfer, naive_total_aggr_transfer, naive_total_comb_transfer, naive_total_actv_transfer = 0, 0, 0, 0
    naive_total_transfer = 0 

    with open(logf) as f:
        for line in f:
            if 'rank=' in line:
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


    print(f'total number of batches: {total_batches}')
    print(f'total number of tree nodes: {total_tree_nodes}, avg per batch tree ndata: {total_tree_nodes//total_batches}, avg per batch tree size: {total_tree_nodes/total_batches*featdim*4/1024/1024} MB')
    print(f'total number of remote request nodes: {total_request_nodes}, avg per batch remote request ndata: {total_request_nodes // total_batches}, avg per batch remote request size: {total_request_nodes/total_batches*featdim*4/1024/1024} MB')
    print(f"{'='*10}, naive transfer 数据迁移量统计, {'='*10}")
    print(f'total model&optm transfer ndata: {naive_total_model_optim_transfer}, size: {naive_total_model_optim_transfer*4/1024/1024} MB, avg per batch size: {naive_total_model_optim_transfer*4/1024/1024/total_batches} MB')
    print(f'total part aggr transfer ndata: {naive_total_aggr_transfer}, size: {naive_total_aggr_transfer*4/1024/1024} MB, avg per batch size: {naive_total_aggr_transfer*4/1024/1024/total_batches} MB')
    print(f'total comb transfer ndata: {naive_total_comb_transfer}, size: {naive_total_comb_transfer*4/1024/1024} MB, avg per batch size: {naive_total_comb_transfer*4/1024/1024/total_batches} MB')
    print(f'total actv transfer ndata: {naive_total_actv_transfer}, size: {naive_total_actv_transfer*4/1024/1024} MB, avg per batch size: {naive_total_actv_transfer*4/1024/1024/total_batches} MB')
    print(f'total transfer ndata: {naive_total_transfer}, size: {naive_total_transfer*4/1024/1024} MB, avg per batch size: {naive_total_transfer*4/1024/1024/total_batches}')


if __name__ == '__main__':
    logf = input('Please input the log file path+name to analyze: ')
    analyze_logs(logf)