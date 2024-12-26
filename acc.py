import os
import re
from collections import defaultdict

def extract_system_and_model(filename):
    # 定义系统和模型的匹配规则
    system_rules = {
        '_default.log': 'DGL',
        'localTrue': 'LO',
    }
    
    model_patterns = ['gcn', 'gat', 'sage']
    
    # 初始化默认值
    system = 'LeapGNN'
    model = None
    
    # 解析系统
    for pattern, sys in system_rules.items():
        if pattern in filename:
            system = sys
            break
    
    # 解析模型
    for model_pattern in model_patterns:
        if model_pattern in filename:
            model = model_pattern.upper()
            break
    
    return system, model

def extract_max_acc(log_dir):
    # 存储结果的字典，按 arxiv 和 reddit 分类
    results = defaultdict(lambda: defaultdict(list))
    
    # 遍历指定目录下的所有 .log 文件
    for filename in os.listdir(log_dir):
        if filename.endswith('.log'):
            file_path = os.path.join(log_dir, filename)
            
            # 检查是否为文件（防止目录也被处理）
            if os.path.isfile(file_path):
                # 根据文件名判断是 arxiv 还是 reddit
                if 'arxiv' in filename:
                    group = 'arxiv'
                elif 'reddit' in filename:
                    group = 'reddit'
                else:
                    continue  # 不符合条件的文件跳过
                
                # 提取系统和模型
                system, model = extract_system_and_model(filename)
                
                # 读取文件内容，查找 "Max acc:" 后的浮点数
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        match = re.search(r'Max acc:\s*(-?\d+(\.\d+)?)', line)
                        if match:
                            max_acc = float(match.group(1))
                            results[group][system].append((filename, model, max_acc))
                            break  # 假设每个文件中只有一行包含 "Max acc:"
    
    return results

def print_results(results):
    # 打印结果
    for group, systems in results.items():
        print(f"=== {group} ===")
        for system, entries in systems.items():
            print(f"System: {system}")
            for filename, model, max_acc in entries:
                print(f"  Model: {model}, Max Accuracy: {max_acc:.4f}")
        print()  # 空行分隔不同组

# 指定日志文件所在的目录
log_directory = 'log_acc'  # 请替换为你的日志文件所在目录路径

# 调用函数并打印结果
if __name__ == "__main__":
    max_acc_values = extract_max_acc(log_directory)
    print_results(max_acc_values)
