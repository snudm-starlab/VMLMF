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
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))


def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return np.squeeze(y_ - 1)


class UCIDataset(Dataset):
    def __init__(self, data_path, mode):
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
        import numpy as np
        x_path = [data_path + mode + '/' + "Inertial Signals/" + signal + mode + '.txt' for signal in
                    INPUT_SIGNAL_TYPES]
        y_path = data_path + mode + '/' + 'y_' + mode + '.txt'
        self.X = load_X(x_path)
        self.y = load_y(y_path)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


def UCI_dataloader(batch_size, dataset_folder='./data/UCI HAR Dataset/'):
    """
    get dataloader - dataloader the batch size of load data 
    @param batch_size
        batch_size of dataset
    @param dataset_folder
        file path to load dataset from
    @return train and test dataloader
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
