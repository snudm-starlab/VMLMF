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
# pylint: disable=R0902, R0913, R0914, C0413
"""
====================================
 :mod:`download_dataset`
====================================
.. moduleauthor:: Hyojin Jeon  <tarahjjeon@snu.ac.kr>
설명
=====
데이터 다운로드를 위한 모듈입니다.

- 웹에서 UCI HAR Dataset.zip 파일과 Opportunity UCI Dataset.zip 파일을 다운로드 받습니다.
"""
import os
from subprocess import call

print("")

print("Downloading...")
if not os.path.exists("./src/data/UCI HAR Dataset.zip"):
    call(
        'wget "https://archive.ics.uci.edu/ml/machine-learning-databases\
            /00240/UCI HAR Dataset.zip"',
        shell=True
    )
    print("Downloading done.\n")
else:
    print("UCI Dataset already downloaded. Did not download twice.\n")

if not os.path.exists("./src/data/OpportunityUCIDataset.zip"):
    call(
        'wget https://archive.ics.uci.edu/ml/machine-learning-databases\
            /00226/OpportunityUCIDataset.zip',
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
    print(f"Extracting successfully done to {extract_directory}.")
else:
    print("Dataset already extracted. Did not extract twice.\n")