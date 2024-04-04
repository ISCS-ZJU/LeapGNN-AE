import os
import re
import matplotlib.pyplot as plt

# 获取 logs 文件夹的路径
def get_logs_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs/gat_hd16_delay0.025')

# 计算列表中非零的比例
def get_busy_percent(lst):
    cnt = 0
    for it in lst:
        if it > 0:
            cnt += 1
    return round(cnt/len(lst), 2)

# 设置文件夹路径
logs_folder = get_logs_path()  # 请替换为实际的文件夹路径

# 用于存储所有文件的GPU利用率列表
file_gpu_util = {}

# 遍历文件夹中的所有.log文件
for filename in os.listdir(logs_folder):
    if filename.endswith('.log'):
        file_path = os.path.join(logs_folder, filename)
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if 'gpu util:' in line:
                    # 将字符串转换为实际的列表
                    gpu_util_list = eval(line.split('gpu util:')[1])
                    file_gpu_util[os.path.splitext(filename)[0]] = gpu_util_list

# 设置matplotlib的样式
# plt.style.use('seaborn-darkgrid')

# 绘制折线图
fig, ax = plt.subplots()

markers = ['.', 'x', '+', 'o']
linestyles = ['-', '--', '-.', ':']

# tmpf = open('tmp.txt', 'w+')

# 遍历所有文件的GPU利用率列表并绘制折线图
for i, (file_name, gpu_util) in enumerate(file_gpu_util.items()):
    gpu_util = gpu_util[50:500] # 截取部分进行展示
    plt.plot(gpu_util, label=file_name, linewidth=1, marker=markers[i], markersize=5)
    # 打印平均值
    print(f"{file_name}: Avg:{round(sum(gpu_util) / len(gpu_util), 2)}, Max:{max(gpu_util)}, Busy:{get_busy_percent(gpu_util)}")
    # print(gpu_util)
    # 纵向打印方便 zplot 画图
    # print(file_name, file=tmpf)
    # for utl in gpu_util:
    #     print(utl, file=tmpf)
# 设置图例
# ax.legend(fontsize=4)
ax.legend()

# 设置标题和坐标轴标签
# ax.set_title('GPU Utilization Over Time')
ax.set_xlabel('Timeline (s)')
ax.set_ylabel('GPU Utilization (%)')

# # 显示图表
# plt.show()

# 保存图表到本地
output_file_path = os.path.join(logs_folder, 'gpu_utilization.png')
plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
