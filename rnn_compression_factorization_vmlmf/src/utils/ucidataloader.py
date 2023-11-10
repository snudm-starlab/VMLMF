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
# pylint: disable=R0902, R0913, R0914, C0413
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

def load_x(x_signals_paths):
    """load signal data from path

    :param X_signals_path: string file path of X dataset
    :returns: numpy array of X data
    """
    x_signals = []

    for signal_type_path in x_signals_paths:
        #file = open(signal_type_path, 'r')
        with open(signal_type_path, 'r', encoding="utf-8") as file:
            # Read dataset from disk, dealing with text files' syntax
            x_signals.append(
                [np.array(serie, dtype=np.float32) for serie in [
                    row.replace('  ', ' ').strip().split(' ') for row in file
                ]]
            )
            file.close()

    return np.transpose(np.array(x_signals), (1, 2, 0))


def load_y(y_path):
    """load label data from path

    :param y_path: string file path of y dataset
    :returns: numpy array of y data
    """
    #file = open(y_path, 'r')
    with open(y_path, 'r', encoding="utf-8") as file:
        # Read dataset from disk, dealing with text file's syntax
        y_ = np.array(
            [row.replace('  ', ' ').strip().split(' ') for row in file],
            dtype=np.int32
        )


    # Substract 1 to each output class for friendly 0-based indexing
    return np.squeeze(y_ - 1)

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

class UCIDataset(Dataset):
    """Dataset class for UCI

    :param string data_path: file path of dataset
    :param string mode: select mode to load train or test dataset
    """
    def __init__(self, data_path, mode):
        """Initialize UCIDataset class"""

        x_path = [data_path + mode + '/' + "Inertial Signals/" + signal \
            + mode + '.txt' for signal in INPUT_SIGNAL_TYPES]
        y_path = data_path + mode + '/' + 'y_' + mode + '.txt'
        self.x = load_x(x_path)
        self.y = load_y(y_path)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def uci_dataloader(dataset_folder='./data/UCI HAR Dataset/'):
    """get dataloader - dataloader the batch size of load data

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