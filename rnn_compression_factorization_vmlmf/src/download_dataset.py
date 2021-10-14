################################################################################
# [VMLMF] Lowrank Matrix Factorization with Vector-Multiplication
# Project: Starlab 
#
# Authors: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# File: download_dataset.py
# - utilities for download Opportunity and UCI dataset from web archive
#
# Version : 1.0
# Date : Oct 14, 2021
# Main Contact: Hyojin Jeon
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
#
################################################################################
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
