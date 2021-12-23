################################################################################
# [VMLMF] Lowrank Matrix Factorization with Vector-Multiplication
# Project: Starlab 
#
# Authors: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# File: preprocess_opp.py
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
import sys

sys.path.append('../')
from utils.sliding_window import *
import os
import zipfile
import argparse

from io import BytesIO
from pandas import Series
import pickle as cp

# number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 77

# File names of Train/Validation/Test data
OPPORTUNITY_DATA_FILES_DICT = {
    'training': [
        'S1-ADL1.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
        'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat', 'S2-Drill.dat',
        'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat', 'S3-Drill.dat',
        'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-Drill.dat'
        'S1-ADL2.dat'
    ],
    'test': [
        'S2-ADL4.dat', 'S2-ADL5.dat',
        'S3-ADL4.dat', 'S3-ADL5.dat'
    ]
}

# Hard coded threshold to normalize data from each sensor
NORM_MAX_THRESHOLDS = [3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500, 3000, 3000, 3000, 10000, 10000, 10000,
                       1500, 1500, 1500, 3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500, 3000, 3000, 3000,
                       10000, 10000, 10000, 1500, 1500, 1500, 3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
                       250, 25, 200, 5000, 5000, 5000, 5000, 5000, 5000, 10000, 10000, 10000, 10000, 10000, 10000, 250,
                       250, 25, 200, 5000, 5000, 5000, 5000, 5000, 5000, 10000, 10000, 10000, 10000, 10000, 10000, 250]

NORM_MIN_THRESHOLDS = [-3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000, -3000, -3000, -3000, -10000,
                       -10000, -10000, -1000, -1000, -1000, -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000,
                       -1000, -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000, -3000, -3000, -3000,
                       -10000, -10000, -10000, -1000, -1000, -1000, -250, -100, -200, -5000, -5000, -5000, -5000, -5000,
                       -5000, -10000, -10000, -10000, -10000, -10000, -10000, -250, -250, -100, -200, -5000, -5000,
                       -5000, -5000, -5000, -5000, -10000, -10000, -10000, -10000, -10000, -10000, -250]


def select_columns_opp(data):
    """
    Selection of the 77 columns employed in the OPPORTUNITY challenge
    @ param data: numpy integer matrix
        Sensor data (all features)
    @ return: numpy integer matrix
        Selection of features
    """
    # features to exclude
    features_delete = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                       27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 46, 47, 48, 49, 59, 60, 61, 62, 72, 73, 74, 75, 85, 86,
                       87, 88, 98, 99, 100, 101, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
                       148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166,
                       167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
                       186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204,
                       205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
                       224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242,
                       243, 244, 245, 246, 247, 248]
    return np.delete(data, features_delete, 1)


def normalize(data, max_list, min_list):
    """
        Normalizes all sensor channels
    :param data: numpy integer matrix
        Sensor data
    :param max_list: numpy integer array
        Array containing maximums values for every one of the 113 sensor channels
    :param min_list: numpy integer array
        Array containing minimum values for every one of the 113 sensor channels
    :return:
        Normalized sensor data
    """
    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list
    for i in np.arange(data.shape[1]):
        data[:, i] = (data[:, i] - min_list[i]) / diffs[i]
    #     Checking the boundaries
    data[data > 1] = 0.99
    data[data < 0] = 0.00
    return data


def refine_data(data):
    """
    remove rows which have nan values
    @ param data 
        data to refine
    @ return data with complete instance without NaN values
    """

    # instances that have NaN values
    instances_delete = []
    for idx, row in enumerate(data):
        cnt = 0
        if np.isnan(row).any():
            instances_delete.append(idx)
            continue

    print("*" * 32)
    print(f"# of rows to be deleted:{len(instances_delete)}")
    print("*" * 32)
    return np.delete(data, instances_delete, 0)


def divide_x_y(data, label):
    """
    Segments each sample into features and label

    @ param data: numpy integer matrix
        Sensor data
    @ param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    ! return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """

    data_x = data[:, :77]
    if label not in ['locomotion', 'gestures']:
        raise RuntimeError("Invalid label: '%s'" % label)
    if label == 'locomotion':
        data_y = data[:, 114]  # Locomotion label
    elif label == 'gestures':
        data_y = data[:, 77]  # Gestures label

    return data_x, data_y


def adjust_idx_labels(data_y, label):
    """
    Transforms original labels into the range [0, nb_labels-1]

    @ param data_y: numpy integer array
        Sensor labels
    @ param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    @ return: numpy integer array
        Modified sensor labels
    """

    if label == 'locomotion':  # Labels for locomotion are adjusted
        data_y[data_y == 4] = 3
        data_y[data_y == 5] = 4
    elif label == 'gestures':  # Labels for gestures are adjusted
        data_y[data_y == 406516] = 1
        data_y[data_y == 406517] = 2
        data_y[data_y == 404516] = 3
        data_y[data_y == 404517] = 4
        data_y[data_y == 406520] = 5
        data_y[data_y == 404520] = 6
        data_y[data_y == 406505] = 7
        data_y[data_y == 404505] = 8
        data_y[data_y == 406519] = 9
        data_y[data_y == 404519] = 10
        data_y[data_y == 406511] = 11
        data_y[data_y == 404511] = 12
        data_y[data_y == 406508] = 13
        data_y[data_y == 404508] = 14
        data_y[data_y == 408512] = 15
        data_y[data_y == 407521] = 16
        data_y[data_y == 405506] = 17
    return data_y


def check_data(data_set):
    """
    Try to access to the file and checks if dataset is in the data directory
       In case the file is not found try to download it from original location

    @ param data_set:
            Path with original OPPORTUNITY zip file
    @ return: data directory
    """
    data_dir, data_file = os.path.split(data_set)
    # When a directory is not provided, check if dataset is in the data directory
    if data_dir == "" and not os.path.isfile(data_set):
        new_path = os.path.join(os.path.split(__file__)[0], "data", data_set)
        if os.path.isfile(new_path) or data_file == 'OpportunityUCIDataset.zip':
            data_set = new_path

    # When dataset not found, try to download it from UCI repository
    if (not os.path.isfile(data_set)) and data_file == 'OpportunityUCIDataset.zip':
        print('... dataset path {} not found'.format(data_set))
        import urllib
        origin = (
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip'
        )
        if not os.path.exists(data_dir):
            print('... creating directory {0}'.format(data_dir))
            os.makedirs(data_dir)
        print('... downloading data from {0}'.format(origin))
        urllib.request.urlretrieve(origin, data_set)

    return data_dir


def process_dataset_file(data, label):
    """
    Function defined as a pipeline to process individual OPPORTUNITY files

    @ param data: numpy integer matrix
        Matrix containing data samples (rows) for every sensor channel (column)
    @ param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    @ return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into features (x) and labels (y)
    """

    # Select correct columns
    data = select_columns_opp(data)
    data = refine_data(data)


    # Colums are segmentd into features and labels
    data_x, data_y = divide_x_y(data, label)
    data_y = adjust_idx_labels(data_y, label)
    data_y = data_y.astype(int)

    # Perform linear interpolation
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T

    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0
    data_x = normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)
    return data_x, data_y


def generate_data(dataset, target_filename, label):
    """
    Function to read the OPPORTUNITY challenge raw data and process all sensor channels

    @ param dataset: string
        Path with original OPPORTUNITY zip file
    @ param target_filename: string
        Processed file
    @ param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized. The OPPORTUNITY dataset includes several annotations to perform
        recognition modes of locomotion/postures and recognition of sporadic gestures.
    """

    data_dir = check_data(dataset)

    if os.path.exists("./src/data/oppChallenge_gestures_validation.data"):
        return

    data_x = np.empty((0, NB_SENSOR_CHANNELS))
    data_y = np.empty((0))

    zf = zipfile.ZipFile(dataset)
    print('Processing dataset files ...')
    for datatype, data in OPPORTUNITY_DATA_FILES_DICT.items():
        print("data type:", datatype)
        data_x = np.empty((0, NB_SENSOR_CHANNELS))
        data_y = np.empty((0))
        for filename in data:
            filename = "OpportunityUCIDataset/dataset/{}".format(filename)
            try:
                data = np.loadtxt(BytesIO(zf.read(filename)))
                print('... file {0}'.format(filename))
                x, y = process_dataset_file(data, label)
                data_x = np.vstack((data_x, x))
                data_y = np.concatenate([data_y, y])
            except KeyError:
                print('ERROR: Did not find {0} in zip file'.format(filename))
        if datatype == 'training':
            X_train, y_train = data_x, data_y
        else:
            X_test, y_test = data_x, data_y
    # Dataset is segmented into train and test
    nb_training_samples = 557963

    print("Final datasets with size: | train {0} | test {1} | ".format(X_train.shape, X_test.shape))

    #obj = [(X_train, y_train), (X_test, y_test), (X_valid, y_valid)]
    obj = [(X_train, y_train), (X_test, y_test)]
    f = open(os.path.join(data_dir, target_filename), 'wb')
    cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
    f.close()


def get_args():
    '''

    This function parses and return arguments passed in
    
    @ return results of parsing arguments

    '''
    parser = argparse.ArgumentParser(
        description='Preprocess OPPORTUNITY dataset')
    # Add arguments
    parser.add_argument(
        '-i', '--input', type=str, help='OPPORTUNITY zip file', required=True)
    parser.add_argument(
        '-o', '--output', type=str, help='Processed data file', required=True)
    parser.add_argument(
        '-t', '--task', type=str.lower, help='Type of activities to be recognized', default="gestures",
        choices=["gestures", "locomotion"], required=False)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    dataset = args.input
    target_filename = args.output
    label = args.task
    # Return all variable values
    return dataset, target_filename, label


def load_dataset(filename):
    """
    load dataset and process it for training, validation, test
    """

    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.float64)
    y_test = y_test.astype(np.float64)


    return X_train, y_train, X_test, y_test 


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.float64)


if __name__ == '__main__':
    opp_uci_dataset_zip, output, l = get_args()
    generate_data(opp_uci_dataset_zip, output, l)

    # Hardcoded length of the sliding window mechanism employed to segment the data
    SLIDING_WINDOW_LENGTH = 24
    # Hardcoded step of the sliding window mechanism employed to segment the data
    SLIDING_WINDOW_STEP = 12

    # X_train, y_train, X_test, y_test= load_dataset('./src/data/oppChallenge_gestures.data')
    X_train, y_train, X_test, y_test = load_dataset('./src/data/oppChallenge_gestures.data')

    assert NB_SENSOR_CHANNELS == X_train.shape[1]
    # Sensor data is segmented using a sliding window mechanism
    X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    print(" ..after sliding window (train): inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
    X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    print(" ..after sliding window (test): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))

    np.save('./src/data/opp/X_test', X_test)
    np.save('./src/data/opp/y_test', y_test)
    np.save('./src/data/opp/X_train', X_train)
    np.save('./src/data/opp/y_train', y_train)
