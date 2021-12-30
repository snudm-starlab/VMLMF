################################################################################
# [VMLMF] Lowrank Matrix Factorization with Vector-Multiplication
# Project: Starlab
#
# Authors: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# File: UCI_dataloader.py
# - utilities for processing UCI dataset
#
# Version : 1.0
# Date : Oct 14, 2021
# Main Contact: Hyojin Jeon
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
#
################################################################################
# pylint: disable=C0103, E1101, C0114, R0902,C0116, R0914, R0913, C0123, W0613, W0102,C0413, E0401,R1721, W1514
"""
====================================
 :mod:`UCI_dataloader`
====================================
.. moduleauthor:: Hyojin Jeon  <tarahjjeon@snu.ac.kr>
설명
=====
UCI datda를 불러오기 위한 모듈입니다.

"""
from torch.utils.data import Dataset, DataLoader
import numpy as np

def load_X(X_signals_paths):
    """load signal data from path

    :param X_signals_path: string file path of X dataset
    :returns: numpy array of X data
    """
    X_signals = []

    for signal_type_path in X_signals_paths:
        #file = open(signal_type_path, 'r')
        with open(signal_type_path,'r') as file:
            # Read dataset from disk, dealing with text files' syntax
            X_signals.append(
                [np.array(serie, dtype=np.float32) for serie in [
                    row.replace('  ', ' ').strip().split(' ') for row in file
                ]]
            )
            file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))


def load_y(y_path):
    """load label data from path

    :param y_path: string file path of y dataset
    :returns: numpy array of y data
    """
    #file = open(y_path, 'r')
    with open(y_path,'r') as file:
        # Read dataset from disk, dealing with text file's syntax
        y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]],
            dtype=np.int32
        )

    # Substract 1 to each output class for friendly 0-based indexing
    return np.squeeze(y_ - 1)


class UCIDataset(Dataset):
    """Dataset class for UCI

    :param string data_path: file path of dataset
    :param string mode: select mode to load train or test dataset
    """
    def __init__(self, data_path, mode):
        """Initialize UCIDataset class"""
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]

        x_path = [data_path + mode + '/' + "Inertial Signals/" + signal \
            + mode + '.txt' for signal in INPUT_SIGNAL_TYPES]
        y_path = data_path + mode + '/' + 'y_' + mode + '.txt'
        self.X = load_X(x_path)
        self.y = load_y(y_path)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


def UCI_dataloader(batch_size, dataset_folder='./data/UCI HAR Dataset/'):
    """get dataloader - dataloader the batch size of load data

    :param batch_size: integer batch_size of dataset
    :param dataset_folder: string file path to load dataset from
    :returns: train_loader, test_loader
    """

    dataset_train = UCIDataset(dataset_folder, 'train')
    train_loader = DataLoader(dataset=dataset_train,
                                batch_size=64,
                                shuffle=True, drop_last=True
                                )

    dataset_test = UCIDataset(dataset_folder, 'test')
    test_loader = DataLoader(dataset=dataset_test,
                                batch_size=64,
                                shuffle=False
                                )

    return (train_loader, test_loader)
