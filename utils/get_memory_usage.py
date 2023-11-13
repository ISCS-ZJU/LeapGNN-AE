import psutil

def convert_bytes(bytes):
    # 将字节数转换为更可读的格式（KB、MB、GB）
    if bytes < 1024:
        return f"{bytes} Bytes"
    elif bytes < 1024**2:
        return f"{bytes/1024:.2f} KB"
    elif bytes < 1024**3:
        return f"{bytes/1024**2:.2f} MB"
    else:
        return f"{bytes/1024**3:.2f} GB"


def get_cpu_memory_usage(prefix = None):
    # 获取CPU使用率和内存信息
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()

    # 格式化输出
    result = ''
    if prefix:
        result += f'----------{prefix}----------\n'
    result += f"CPU使用率: {cpu_percent}%\n"
    result += f"总内存: {convert_bytes(memory_info.total)}\n"
    result += f"可用内存: {convert_bytes(memory_info.available)}\n"
    result += f"已使用内存: {convert_bytes(memory_info.used)}\n"
    result += f"内存使用率: {memory_info.percent}%"

    return result