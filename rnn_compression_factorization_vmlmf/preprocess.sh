#!/bin/sh

mkdir ./src/data/smalldata
python ./src/download_dataset.py
mv "OpportunityUCIDataset.zip"  ./src/data/smalldata
mv "UCI HAR Dataset.zip" ./src/data
mv "UCI HAR Dataset" ./src/data
python ./src/preprocess_Opportunity77.py -i ./src/data/smalldata/OpportunityUCIDataset.zip -o oppChallenge_gestures.data
