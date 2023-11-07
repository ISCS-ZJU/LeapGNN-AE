import matplotlib.pyplot as plt
import os, re
import yaml

import argparse

epoch_lst = [0, 1, 2, 3, 4, 5]
accuracy_lst = [20, 30, 40, 35, 45, 50]
color_lst = ['royalblue', 'crimson', 'green']

his_epoch_lst = []
his_accuracy_lst = []
his_legend_lst = []

def parse_args_func(argv):
    parser = argparse.ArgumentParser(description='draw accuracy curve along with epoch')
    parser.add_argument('--compare', action='store_true')
    return parser.parse_args(argv)
args = parse_args_func(None)

def parse_config(confpath):
    with open(confpath, 'r') as fh:
        data = yaml.safe_load(fh)
        return data['draw_acc']


def analyse_acc(filepath):
    sigstr = 'Test Accuracy'
    accuracy_lst = []
    epoch_lst = []
    with open(filepath, 'r') as f:
        for line in f:
            if sigstr in line:
                accuracy = re.search(r"Test Accuracy\s+(\d+(\.\d+)?)", line.strip()).group(1)
                accuracy_lst.append(float(accuracy)*100)
    epoch_lst = [idx for idx, _ in enumerate(accuracy_lst)]
    return accuracy_lst, epoch_lst

def concise(name):
    if name.endswith('default.log'):
        return 'Default'
    elif name.endswith('localFalse.log'):
        return 'Ours'
    elif name.endswith('localTrue.log'):
        return 'LS'
    else:
        return name

def draw_acc(target_filename, srcfilepath):
    fig = plt.figure(figsize=(6.4, 6.4)) # 画布大小调整
    fig.subplots_adjust(top=0.85, bottom=0.25) # 上下留白
    accuracy_lst, epoch_lst = analyse_acc(srcfilepath)
    if args.compare:
        # 把其他文件里的几个线条合并画在一个图中
        his_epoch_lst.append(epoch_lst)
        his_accuracy_lst.append(accuracy_lst)
        his_legend_lst.append(os.path.basename(srcfilepath))
        for tmp_epoch, tmp_accuracy, tmp_legend, color in zip(his_epoch_lst, his_accuracy_lst, his_legend_lst, color_lst):
            plt.plot(tmp_epoch, tmp_accuracy, label = concise(tmp_legend), color=color)
    else:
        # plt.clf() # when comment this line, all lines will show on the same figure
        plt.plot(epoch_lst, accuracy_lst)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    min_y_value = min(accuracy_lst) if not args.compare else min([min(lst) for lst in his_accuracy_lst]) -1
    max_y_value = max(accuracy_lst) if not args.compare else max([max(lst) for lst in his_accuracy_lst]) +1
    plt.ylim(min_y_value, max_y_value)  # Set the y-axis range
    plt.legend(fontsize=20)
    plt.tick_params(labelsize=20)
    plt.savefig(target_filename)
    plt.clf() # 清除窗口中之前的内容

if __name__ == '__main__':
    cur_dir = os.path.dirname(__file__)
    files_list = parse_config(f'{cur_dir}/log_analys.yaml')
    for srcfilepath in files_list:
        print(srcfilepath)
        filename = os.path.basename(srcfilepath)
        filedir = os.path.dirname(srcfilepath)
        target_filename = os.path.splitext(filename)[0] + '.png'
        target_path = os.path.join(filedir, target_filename)
        draw_acc(target_path, srcfilepath)