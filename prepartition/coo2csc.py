import scipy.sparse as spsp
from scipy.sparse import csc_array, coo_array
import os
import numpy as np
import psutil

def get_memory_usage(prefix):
    print(f'prefix: {prefix}')
    # 获取内存使用情况
    memory = psutil.virtual_memory()
    # 已用内存
    used_memory = memory.used
    # 剩余内存
    available_memory = memory.available

    # 打印结果
    print(f"    已用内存：{used_memory / (1024 ** 2)} MB")
    print(f"    剩余内存：{available_memory / (1024 ** 2)} MB")

def save_array_to_file(array, file_path):
    np.save(file_path, array)

def load_array_from_file(file_path):
    file_path += '.npy'
    return np.load(file_path)

def replace_minus_one(lst):
    n = len(lst)
    for idx in range(n-1, -1, -1):
        if lst[idx] == -1:
            lst[idx] = lst[idx+1]
    return lst

def coo2csc(adj_path):
    target_dir = '/home/weijian/gitclone/repgnn/dist/repgnn_data/uk/'
    get_memory_usage('start running ')
    if type(adj_path) == str:
        coo_matrix = spsp.load_npz(adj_path) # <class 'scipy.sparse._coo.coo_matrix'>
        print('load coo_matrix done')
    else:
        coo_matrix = adj_path
    nrow, ncol = coo_matrix.shape
    print(f'nrow = {nrow}, ncol = {ncol}')
    get_memory_usage('after load coo_matrix')
    row = coo_matrix.row
    col = np.copy(coo_matrix.col)
    data = coo_matrix.data

    save_array_to_file(row, target_dir+'row')
    save_array_to_file(data, target_dir+'data')
    save_array_to_file(col, target_dir+'col')
    del row
    del data
    del col

    del coo_matrix
    get_memory_usage('after del coo_matrix')

    col = load_array_from_file(target_dir+'col')
    # 对col升序排列，并按照这个顺序对三个数组进行排序
    sorted_indices = np.argsort(col)
    get_memory_usage('after argsort')
    del col
    get_memory_usage('del col')

    tmp_row = load_array_from_file(target_dir+'row')
    get_memory_usage('after load row')
    sorted_row = tmp_row[sorted_indices]
    save_array_to_file(sorted_row, target_dir+'row')
    del sorted_row, tmp_row
    get_memory_usage('del sorted_row')

    tmp_data = load_array_from_file(target_dir+'data')
    sorted_data = tmp_data[sorted_indices]
    save_array_to_file(sorted_data, target_dir+'data')
    del sorted_data, tmp_data
    get_memory_usage('del sorted_data')

    col = load_array_from_file(target_dir+'col')
    sorted_col = col[sorted_indices]
    del col
    get_memory_usage('del col')

    print('sorted row, col, data done')
    # print(f'sorted_row {sorted_row}, sorted_col {sorted_col}, sorted_data {sorted_data}')
    # 根据排序后的数据，生成 indices, indptr, data
    # indices, data = sorted_row, sorted_data
    indptr = []
    # 找第 value 列从data的 idx 下标开始，直到找完所有的列
    col2ptr = {}
    for idx, value in enumerate(sorted_col):
        if value not in col2ptr:
            col2ptr[value] = idx
    print('construct dict done')
    get_memory_usage('after construct col2ptr dict')
    # print(col2ptr)
    indptr = [col2ptr.get(colid, -1) for colid in range(ncol)]
    del col2ptr
    get_memory_usage('after del col2ptr dict')
    indptr.append(ncol)
    # print(f'indptr {indptr}')
    indptr = replace_minus_one(indptr)
    print('indptr generating done')
    # print(f'indptr {indptr}')
    save_array_to_file(indptr, target_dir+'indptr')

    # 用 indices, indptr, data 生成 csc
    # indptr = load_array_from_file(target_dir+'indptr')

    indices = load_array_from_file(target_dir+'row')
    get_memory_usage('after load row')

    data = load_array_from_file(target_dir+'data')
    get_memory_usage('after load data')

    csc_matrix = csc_array((data, indices, indptr), shape=(nrow, ncol))
    print('csc_matrix producing done')
    return csc_matrix

        


# row  = np.array([0, 3, 1, 0, 4, 4])
# col  = np.array([0, 3, 1, 3, 1, 4])
# data = np.array([4, 5, 7, 9, 1, 8])
# input_array = coo_array((data, (row, col)), shape=(5, 5))
# print(input_array.toarray())
# csc_matrix = coo2csc(input_array)
# print(csc_matrix.toarray())

adj_path = '/home/weijian/gitclone/repgnn/dist/repgnn_data/uk/adj.npz'
csc_matrix = coo2csc(adj_path)

dst_path = '/home/weijian/gitclone/repgnn/dist/repgnn_data/uk/adj_csc.npz'
spsp.save_npz(dst_path, csc_matrix)
