# import copy
import os
from subprocess import call

print("")

print("Downloading...")
if not os.path.exists("./data/UCI HAR Dataset.zip"):
    call(
        'wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip"',
        shell=True
    )
    print("Downloading done.\n")
else:
    print("UCI Dataset already downloaded. Did not download twice.\n")

if not os.path.exists("./data/OppurtunityUCIDataset.zip"):
    call(
        'wget https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip',
        shell=True
    )
    print("Downloading done.\n")
else:
    print("OppurtunityDataset already downloaded. Did not download twice.\n")


print("Extracting...")
extract_directory = os.path.abspath("./data/UCI HAR Dataset")
if not os.path.exists(extract_directory):
    call(
        'unzip -nq "UCI HAR Dataset.zip"',
        shell=True
    )
    print("Extracting successfully done to {}.".format(extract_directory))
else:
    print("Dataset already extracted. Did not extract twice.\n")
