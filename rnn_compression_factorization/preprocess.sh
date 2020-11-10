#!/bin/sh

python download_dataset.py
python preprocess_data.py -i data/OpportunityUCIDataset.zip -o oppChallenge_gestures.data
python preprocess_HAR2.py