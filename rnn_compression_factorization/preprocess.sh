#!/bin/sh

python download_dataset.py
mkdir ./data
mv "OpportunityUCIDataset.zip"  ./data
mv "UCI HAR Dataset.zip" ./data
mv "UCI HAR Dataset" ./data
python preprocess_data.py -i data/OpportunityUCIDataset.zip -o oppChallenge_gestures.data
python preprocess_HAR2.py