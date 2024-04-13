import os, sys
import numpy as np
import re

def extract_content_between_brackets(input_string):
    # 正则表达式模式，匹配方括号之间的内容
    pattern = r'\[(.*?)\]'
    # 使用 re.findall() 函数找到所有匹配的内容
    matches = re.findall(pattern, input_string)
    # 每组 'xx' 中的内容转化为 list
    ret = [list(eval(s)) for s in matches]
    return ret

def analyze_logs(logf):
    print(f'* analyse file {logf}:')
    if not os.path.exists(logf):
        print(f'ERROR: file {logf} does not exists.')
        sys.exit(-1)
    
    splitWord = 'load banlance after redistribution:'
    load_imbalance_after_redistribution = []
    with open(logf) as f:
        for line in f:
            if splitWord in line:
                target_str = line.split(splitWord)[1].strip()[1:-1]
                load_imbalance_after_redistribution = extract_content_between_brackets(target_str)

    # 计算每个 mini-batch 中 redistribution 后每个机器点数的极差 / 平均点数
    ratio_lst = []
    for ar in load_imbalance_after_redistribution:
        ratio = (max(ar) - min(ar)) / (sum(ar) / len(ar))
        ratio_lst.append(ratio)
    print(max(ratio_lst), sum(ratio_lst) / len(ratio_lst))
    return max(ratio_lst), sum(ratio_lst) / len(ratio_lst)

if __name__ == '__main__':
    logf = input('Please input the log file path to analyze: ')
    if os.path.isdir(logf):
        sub_files = [os.path.join(logf, f) for f in os.listdir(logf) if f.endswith('.log')]
        for sub_logf in sub_files:
            analyze_logs(sub_logf)
    else:
        analyze_logs(logf)