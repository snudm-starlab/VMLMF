################################################################################
# Starlab RNN-compression with factorization method : Lowrank and group-lowrank rnn
#
# Author: Donghae Jang (jangdonghae@gmail.com), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# Version : 1.0
# Date : Nov 10, 2020
# Main Contact: Donghae Jang
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
#
################################################################################
# import copy
# Code for downloading and extract Opportunity dataset and UCI HAR dataset
import os
from subprocess import call

print("")

print("Downloading...")
if not os.path.exists("./src/data/UCI HAR Dataset.zip"):
    call(
        'wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip"',
        shell=True
    )
    print("Downloading done.\n")
else:
    print("UCI Dataset already downloaded. Did not download twice.\n")

if not os.path.exists("./src/data/OpportunityUCIDataset.zip"):
    call(
        'wget https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip',
        shell=True
    )
    print("Downloading done.\n")
else:
    print("OppurtunityDataset already downloaded. Did not download twice.\n")


print("Extracting...")
extract_directory = os.path.abspath("./src/data/UCI HAR Dataset")
if not os.path.exists(extract_directory):
    call(
        'unzip -nq "UCI HAR Dataset.zip"',
        shell=True
    )
    print("Extracting successfully done to {}.".format(extract_directory))
else:
    print("Dataset already extracted. Did not extract twice.\n")
