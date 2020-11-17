#!/bin/sh

python ./src/download_dataset.py
mkdir ./src/data
mv "OpportunityUCIDataset.zip"  ./src/data
mv "UCI HAR Dataset.zip" ./src/data
mv "UCI HAR Dataset" ./src/data
python ./src/preprocess_Opportunity.py -i ./src/data/OpportunityUCIDataset.zip -o oppChallenge_gestures.data