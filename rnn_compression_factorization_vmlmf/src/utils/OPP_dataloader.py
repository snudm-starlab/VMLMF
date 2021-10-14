################################################################################
# [VMLMF] Lowrank Matrix Factorization with Vector-Multiplication
# Project: Starlab 
#
# Authors: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# File: OPP_dataloader.py
# - utilities for processing OPP data
#
# Version : 1.0
# Date : Oct 14, 2021
# Main Contact: Hyojin Jeon
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
#
################################################################################
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_path, mode):
        import numpy as np
        self.X = np.load((data_path + '/' + 'X_' + mode + '.npy'))
        self.y = np.load((data_path + '/' + 'y_' + mode + '.npy'))
        #print(self.X.shape)
        #print(self.y.shape)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


def HAR_dataloader(batch_size, dataset_folder='./data/smalldata'):
    """
    get dataloader - dataloader the batch size of load data 
    @param batch_size
        batch_size of dataset
    @param dataset_folder
        file path to load dataset from
    @return train and test dataloader
    """

    dataset_train = CustomDataset(dataset_folder, 'train')
    train_loader = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=True, drop_last=True
                              )

    dataset_test = CustomDataset(dataset_folder, 'test')
    test_loader = DataLoader(dataset=dataset_test,
                             batch_size=batch_size,
                             shuffle=False
                             )

    return (train_loader, test_loader)
