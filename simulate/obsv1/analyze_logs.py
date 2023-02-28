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

    total_batches, total_tree_nodes, total_request_nodes = 0, 0, 0
    same_machine_ratio_per_nf = []
    with open(logf) as f:
        for line in f:
            if 'rank=' in line:
                if 'number of training batches' in line:
                    total_batches += int(line.split('number of training batches:')[-1])
                elif 'total nodes:' in line:
                    total_tree_nodes += int(line.split('total nodes:')[-1])
                elif 'nodes hits on other trainers:' in line:
                    total_request_nodes += sum(eval(line.split('nodes hits on other trainers:')[-1]))
                elif 'percentage with target node per nf' in line:
                    same_machine_ratio_per_nf.append(float(line.split('percentage with target node per nf:')[-1]))

    print(f'total number of batches: {total_batches}')
    print(f'total number of tree nodes: {total_tree_nodes}, avg per batch tree ndata: {total_tree_nodes//total_batches}, avg per batch tree size: {total_tree_nodes/total_batches*featdim*4/1024/1024} MB')
    print(f'total number of remote request nodes: {total_request_nodes}, avg per batch remote request ndata: {total_request_nodes // total_batches}, avg per batch remote request size: {total_request_nodes/total_batches*featdim*4/1024/1024} MB')
    print(f'same machine percentage with target node per nf: {sum(same_machine_ratio_per_nf)/len(same_machine_ratio_per_nf)}')

if __name__ == '__main__':
    logf = input('Please input the log file path+name to analyze: ')
    analyze_logs(logf)