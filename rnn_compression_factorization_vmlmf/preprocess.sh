#!/bin/sh

mkdir ./src/data/opp
python3 ./src/download_dataset.py
mv "OpportunityUCIDataset.zip"  ./src/data/
mv "UCI HAR Dataset.zip" ./src/data
mv "UCI HAR Dataset" ./src/data
python3 ./src/preprocess_opp.py -i ./src/data/OpportunityUCIDataset.zip -o oppChallenge_gestures.data
