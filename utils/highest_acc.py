import os
import re

target_dir = '/home/weijian/gitclone/repgnn/logs'
sigstr = 'Test Accuracy'
target_files_lst = os.listdir(target_dir)
target_files_lst.sort()

file_name2highest_acc_dict = {}


for fn in target_files_lst:
    if fn.endswith('.log'):
        highest_acc, epoch_id = 0, 0
        with open(os.path.join(target_dir, fn), 'r') as fh:
            for line in fh:
                if sigstr in line:
                    accuracy = float(re.search(r"Test Accuracy\s+(\d+(\.\d+)?)", line.strip()).group(1))*100
                    if accuracy > highest_acc:
                        highest_acc, epoch_id = accuracy, line.split('Epoch:')[1].split(',')[0]
        file_name2highest_acc_dict[fn] = highest_acc
        # print 数据集	模型（sl参数）	机器数-bs	是否local	最高精度	最高精度出现的epoch	
        dataset = fn.split('_')[2]
        if dataset == 'ogbn':
            dataset += ('_' + fn.split('_')[3])
        model = fn.split('_')[1]
        sl = fn.split('sl')[1].split('_')[0]
        ntrainer = fn.split('trainer')[1].split('_')[0]
        bs = fn.split('bs')[1].split('_')[0]
        if 'local' in fn:
            try:
                local = fn.split('local')[1].split('_')[0]
            except Exception:
                local = fn.split('local')[1].split('.')[0]
        else:
            local = False
        print(dataset, model+'-'+sl, ntrainer+'-'+bs, local, highest_acc, epoch_id)



# 当 local 是 False 的时候，判断达到 True 时最高精度所花费的epoch id
for fn, highest_acc in file_name2highest_acc_dict.items():
    if 'localFalse' in fn:
        true_filename = fn.replace('False', 'True')
        h_acc_true = file_name2highest_acc_dict[true_filename]
        h_acc_false = highest_acc
        if h_acc_true < h_acc_false:
            # 打开 false 的文件，查看第几个 epoch 时已经达到或超越了 true 时的精度
            with open(os.path.join(target_dir, fn), 'r') as fh:
                for line in fh:
                    if sigstr in line:
                        cur_acc = float(re.search(r"Test Accuracy\s+(\d+(\.\d+)?)", line.strip()).group(1))*100
                        if cur_acc >= h_acc_true:
                            cur_epoch = line.split('Epoch:')[1].split(',')[0]
                            print(f'{fn} has achieved accuracy {cur_acc} at Epoch {cur_epoch}, which is higher than highest accuracy {h_acc_true} in {true_filename}.')
                            break

