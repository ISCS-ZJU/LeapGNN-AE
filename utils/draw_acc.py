import matplotlib.pyplot as plt
import os, re
import yaml

epoch_lst = [0, 1, 2, 3, 4, 5]
accuracy_lst = [20, 30, 40, 35, 45, 50]

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


def draw_acc(target_filename, srcfilepath):
    accuracy_lst, epoch_lst = analyse_acc(srcfilepath)
    # plt.clf() # when comment this line, all lines will show on the same figure
    plt.plot(epoch_lst, accuracy_lst)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 100)  # Set the y-axis range
    plt.savefig(target_filename)

if __name__ == '__main__':
    cur_dir = os.path.dirname(__file__)
    files_list = parse_config(f'{cur_dir}/log_analys.yaml')
    for srcfilepath in files_list:
        filename = os.path.basename(srcfilepath)
        filedir = os.path.dirname(srcfilepath)
        target_filename = os.path.splitext(filename)[0] + '.png'
        target_path = os.path.join(filedir, target_filename)
        draw_acc(target_path, srcfilepath)