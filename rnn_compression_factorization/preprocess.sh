#!/bin/sh

python ./src/download_dataset.py
mkdir ./src/data
mv "OpportunityUCIDataset.zip"  ./src/data
mv "UCI HAR Dataset.zip" ./src/data
mv "UCI HAR Dataset" ./src/data
python preprocess_data.py -i data/OpportunityUCIDataset.zip -o oppChallenge_gestures.data
python preprocess_HAR2.py