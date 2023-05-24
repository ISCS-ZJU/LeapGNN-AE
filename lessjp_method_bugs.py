import numpy as np

# 创建一个4x4的二维numpy数组
# arr = np.array([[5, 2, 3, 1],
#                 [4, 9, 7, 6],
#                 [8, 2, 1, 3],
#                 [6, 5, 4, 7]]

def find_first_nonconsecutive(arr):
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1] + 1:
            return i
    return -1  # 如果数组是连续的，返回-1表示没有不连续的数据

def avoid_colide(model_trace):
    for rowid, row in enumerate(model_trace):
        idx = find_first_nonconsecutive(row)
        if idx == -1:
            continue
        else:
            model_trace[rowid] = np.concatenate((row[idx:], row[:idx]))
    return model_trace

def cal_model_trace_by_minus1jp(arr):
    ret_arr = np.copy(arr)
    # 按行找到每行的最小值的索引
    min_indices = np.argmin(arr, axis=1)

    # 创建一个布尔数组，标记每列是否已经删除过元素
    deleted_columns = np.zeros(arr.shape[0], dtype=bool)

    # 存储每行剩下元素所在的列
    model_trace = []

    col_range = np.arange(arr.shape[1])
    # 遍历每行，去除最小值及对应的列，并记录剩下元素所在的列
    for row_idx, col_idx in enumerate(min_indices):
        # 如果该列已经删除过元素，则寻找下一个最小值所在的列
        while deleted_columns[col_idx]:
            arr[row_idx, col_idx] = np.iinfo(arr.dtype).max  # 将已删除的列的元素置为无穷大
            col_idx = np.argmin(arr[row_idx])
        
        # 记录剩下元素所在的列
        remaining_cols = col_range[col_range!=col_idx]
        model_trace.append(remaining_cols)
        
        # 删除最小值及对应的列
        arr[row_idx, col_idx] = np.iinfo(arr.dtype).max  # 将最小值置为无穷大
        ret_arr[row_idx, col_idx] = np.iinfo(arr.dtype).max # 只把最终的去掉的一个值变为最大值，其他保持不变
        deleted_columns[col_idx] = True
        print(col_idx, deleted_columns)

        

    # 将记录剩下元素所在的列转换为二维数组
    model_trace = np.array(model_trace)
    return ret_arr, model_trace

arr = np.random.randint(0, 100, size=(4,4))
# arr = np.array([[29, 73, 1, 96],
#                 [56, 86, 68, 60],
#                 [27, 36, 40, 28],
#                 [39, 77, 64, 21]])
arr = np.array([[27, 50, 69, 68],
                [23, 84, 63, 35],
                [3, 74, 77, 58],
                [90, 85, 41, 32]]) # bug case
print('arr', arr)
arr, model_trace = cal_model_trace_by_minus1jp(arr)
print('arr:', arr, 'model_trace:', model_trace)
model_trace = avoid_colide(model_trace)
print(model_trace)

arr, model_trace = cal_model_trace_by_minus1jp(arr)
print('arr:', arr, 'model_trace:', model_trace)
model_trace = avoid_colide(model_trace)
print(model_trace)


