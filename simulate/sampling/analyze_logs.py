import os, sys

def analyze_logs(logf):
    if not os.path.exists(logf):
        print(f'ERROR: file {logf} does not exists.')
        sys.exit(-1)

    total_batches, total_tree_nodes, total_request_nodes = 0, 0, 0

    with open(logf) as f:
        for line in f:
            if 'rank=' in line:
                if 'number of training batches' in line:
                    total_batches += int(line.split('number of training batches:')[-1])
                elif 'total nodes:' in line:
                    total_tree_nodes += int(line.split('total nodes:')[-1])
                elif 'nodes hits on other trainers:' in line:
                    total_request_nodes += sum(eval(line.split('nodes hits on other trainers:')[-1]))

    print(f'total number of batches: {total_batches}')
    print(f'total number of tree nodes: {total_tree_nodes}')
    print(f'total number of remote request nodes: {total_request_nodes}')

if __name__ == '__main__':
    logf = input('Please input the log file path+name to analyze: ')
    analyze_logs(logf)