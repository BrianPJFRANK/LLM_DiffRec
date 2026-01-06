import numpy as np
from fileinput import filename
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
from torch.utils.data import Dataset

def data_load(train_path, valid_path, test_path):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0
    train_dict = {}

    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    n_user = uid_max + 1
    n_item = iid_max + 1
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')

    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]), \
        (train_list[:, 0], train_list[:, 1])), dtype='float64', \
        shape=(n_user, n_item))
    
    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # valid_groundtruth

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # test_groundtruth
    
    return train_data, valid_y_data, test_y_data, n_user, n_item


class DataDiffusion(Dataset):
    def __init__(self, data):
        # self.data = data
        self.data = data
        # Support scipy sparse matrices, numpy arrays, or tensors.
        self.is_sparse = sp.isspmatrix(self.data)
        
    def __getitem__(self, index):
        # item = self.data[index]
        # return item
        if self.is_sparse:
            # 獲取稀疏行並轉換為密集numpy數組
            row = self.data[index]
            # 確保轉換為密集數組
            dense = row.toarray().astype(np.float32).squeeze(0)  # shape: (n_item,)
            return torch.from_numpy(dense)
        elif isinstance(self.data, np.ndarray):
            return torch.from_numpy(self.data[index].astype(np.float32))
        elif isinstance(self.data, torch.Tensor):
            return self.data[index].float()
        else:
            raise TypeError(f"Unsupported data type for DataDiffusion: {type(self.data)}")

    def __len__(self):
        #return len(self.data)
        if hasattr(self.data, "shape"):
            return self.data.shape[0]
        return len(self.data)
