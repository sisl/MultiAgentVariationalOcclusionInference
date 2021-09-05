# Code to split the dataset into train/validation/test.

import argparse
import os
import time
from collections import defaultdict

import math
import numpy as np
import csv

import hickle as hkl
import glob 
from sklearn.model_selection import train_test_split
import pdb
import random
from generate_data import getmap
from tqdm import tqdm
np.random.seed(123)

if __name__ == "__main__":
    ego_file = glob.glob('/data/INTERACTION-Dataset-DR-v1_1/processed_data/pkl/*ego*')
    y = np.arange(len(ego_file))
    ego_temp, ego_test_files, _, _ = train_test_split(ego_file, y, test_size=0.1, random_state=42) # test set : 10% of total data
    y = np.arange(len(ego_temp))
    ego_train_files, ego_val_files, _, _ = train_test_split(ego_temp, y, test_size=5./90., random_state=42) # validation set : 5% of total data
    
    ego_train_set_path = "/data/INTERACTION-Dataset-DR-v1_1/processed_data/train_test_split/ego_train_set.csv"
    ego_val_set_path = "/data/INTERACTION-Dataset-DR-v1_1/processed_data/train_test_split/ego_val_set.csv"
    ego_test_set_path = "/data/INTERACTION-Dataset-DR-v1_1/processed_data/train_test_split/ego_test_set.csv"
    train_set_path = "/data/INTERACTION-Dataset-DR-v1_1/processed_data/train_test_split/ref_train_set.csv"
    val_set_path = "/data/INTERACTION-Dataset-DR-v1_1/processed_data/train_test_split/ref_val_set.csv"
    test_set_path = "/data/INTERACTION-Dataset-DR-v1_1/processed_data/train_test_split/ref_test_set.csv"
    
    with open(ego_train_set_path, 'w') as f:
        wr = csv.writer(f, delimiter=',')
        for row in ego_train_files:
            wr.writerow([row])

    with open(ego_val_set_path, 'w') as f:
        wr = csv.writer(f, delimiter=',')
        for row in ego_val_files:
            wr.writerow([row])
    
    with open(ego_test_set_path, 'w') as f:
        wr = csv.writer(f, delimiter=',')
        for row in ego_test_files:
            wr.writerow([row])

    # Training
    ref_train_file = [glob.glob("_".join(filename.split('_')[:-3])+'_ref_*') for filename in ego_train_files]
    ref_train_file = sum(ref_train_file, [])

    with open(train_set_path, 'w') as f:
        wr = csv.writer(f, delimiter=',')
        for row in ref_train_file:
            wr.writerow([row])

    # Validation
    ref_val_file = [glob.glob("_".join(filename.split('_')[:-3])+'_ref_*') for filename in ego_val_files]
    ref_val_file = sum(ref_val_file, [])
    
    with open(val_set_path, 'w') as f:
        wr = csv.writer(f)
        for row in ref_val_file:
            wr.writerow([row])

    # Test
    ref_test_file = [glob.glob("_".join(filename.split('_')[:-3])+'_ref_*') for filename in ego_test_files]
    ref_test_file = sum(ref_test_file, [])
    
    with open(test_set_path, 'w') as f:
        wr = csv.writer(f)
        for row in ref_test_file:
            wr.writerow([row])

    print('ego_train_files', len(ego_train_files),len(set(ego_train_files)))
    print('ego_val_files', len(ego_val_files),len(set(ego_val_files)))
    print('ego_test_files', len(ego_test_files),len(set(ego_test_files)))
    print('ref_train_file', len(ref_train_file),len(set(ref_train_file)))
    print('ref_val_file', len(ref_val_file),len(set(ref_val_file)))
    print('ref_test_file', len(ref_test_file),len(set(ref_test_file)))

    print('train/val/test split done')
