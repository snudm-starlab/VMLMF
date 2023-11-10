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
# pylint: disable=R0902, R0913, R0914, C0413
"""
====================================
 :mod:`OPP_dataloader`
====================================
.. moduleauthor:: Hyojin Jeon  <tarahjjeon@snu.ac.kr>
설명
=====
Opportunity data를 로드하기 위한 모듈입니다.

"""
from torch.utils.data import Dataset, DataLoader
import numpy as np
class CustomDataset(Dataset):
    """Dataset class """
    def __init__(self, data_path, mode):
        """Initialize CustomDataset

        :param string data_path:filepath of dataset
        :param string mode:whether train or test data
        """
        self.x = np.load((data_path + '/' + 'X_' + mode + '.npy'))
        self.y = np.load((data_path + '/' + 'y_' + mode + '.npy'))

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def har_dataloader(batch_size, dataset_folder='./data/opp'):
    """get dataloader - dataloader the batch size of load data

    :param batch_size: integer batch_size of dataset
    :param dataset_folder: string file path to load dataset from
    :returns: train_loader, test dataloader:
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