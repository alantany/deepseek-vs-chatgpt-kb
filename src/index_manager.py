"""
索引管理模块
"""

import os
import pickle

def save_index(file_name, chunks, index):
    """
    保存文档的chunks和索引
    """
    if not os.path.exists('indices'):
        os.makedirs('indices')
    
    # 保存chunks和index
    with open(f'indices/{file_name}.pkl', 'wb') as f:
        pickle.dump((chunks, index), f)
    
    # 更新文件列表
    file_list_path = 'indices/file_list.txt'
    if os.path.exists(file_list_path):
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()
    else:
        file_list = []
    
    if file_name not in file_list:
        file_list.append(file_name)
        with open(file_list_path, 'w') as f:
            f.write('\n'.join(file_list))

def load_all_indices():
    """
    加载所有保存的索引
    """
    file_indices = {}
    file_list_path = 'indices/file_list.txt'
    
    if os.path.exists(file_list_path):
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()
        
        for file_name in file_list:
            file_path = f'indices/{file_name}.pkl'
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    chunks, index = pickle.load(f)
                file_indices[file_name] = (chunks, index)
    
    return file_indices

def delete_index(file_name):
    """
    删除指定文件的索引
    """
    # 删除索引文件
    if os.path.exists(f'indices/{file_name}.pkl'):
        os.remove(f'indices/{file_name}.pkl')
    
    # 更新文件列表
    file_list_path = 'indices/file_list.txt'
    if os.path.exists(file_list_path):
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()
        
        if file_name in file_list:
            file_list.remove(file_name)
            with open(file_list_path, 'w') as f:
                f.write('\n'.join(file_list)) 