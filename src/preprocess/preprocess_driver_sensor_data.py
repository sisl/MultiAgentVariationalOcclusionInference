# Code to downsample the training set and to form numpy arrays for the driver sensor dataset with the associated unique sources.

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

import os
import pdb
import pickle as pkl
import hickle as hkl
from tqdm import tqdm
import random

random.seed(123)
np.random.seed(seed=123)

dir_pickle_train = '/data/INTERACTION-Dataset-DR-v1_1/processed_data/driver_sensor_label_grids_train_data/'
dir_pickle_val = '/data/INTERACTION-Dataset-DR-v1_1/processed_data/driver_sensor_label_grids_val_data/'
dir_pickle_test = '/data/INTERACTION-Dataset-DR-v1_1/processed_data/driver_sensor_label_grids_test_data/'
dirs_pickle = [dir_pickle_train, dir_pickle_val, dir_pickle_test]
dir_dataset = '/data/INTERACTION-Dataset-DR-v1_1/processed_data/driver_sensor_dataset/'

if not os.path.isdir(dir_dataset):
    os.mkdir(dir_dataset)

count_train = 0
count_val = 0
count_test = 0

grids_train_list = []
states_train_list = []
sources_train_list = []

grids_val_list = []
states_val_list = []
sources_val_list = []

grids_test_list = []
states_test_list = []
sources_test_list = []

for i in range(0, len(dirs_pickle)):
    dir_pickle = dirs_pickle[i]

    if i == 0:
        # Random shuffle the directories for the training set.
        dirs_list = sorted(os.listdir(dir_pickle))
        random.shuffle(dirs_list)
    else:
        dirs_list = sorted(os.listdir(dir_pickle))

    for file in tqdm(dirs_list):
        path = os.path.join(dir_pickle, file)
        [grid_driver, ts, posx, posy, orientation, velx, vely, accx, accy, goalx, goaly, tgoal]  = pkl.load(open(path, 'rb'))
        state = np.vstack((posx,posy,orientation,velx,vely,accx,accy,goalx,goaly,tgoal)).T

        if grid_driver.shape == (0,):
            continue

        if i == 0:
            # Downsample the training set and ensure that at least 10 seconds of data exists for every observed trajectory.
            if count_train <= 70000 and state.shape[0] >= 10:
                count_train += 1

                states_train_list.append(state.astype('float32'))
                grids_train_list.append(grid_driver.astype('float32'))
                sources_train_list.append(np.repeat(np.array(file[:-4]), state.shape[0]))

                if count_train % 1000 == 0:
                    print(count_train)

        elif i == 1:
            # Ensure that at least 10 seconds of data exists for every observed trajectory.
            if state.shape[0] >= 10:

                states_val_list.append(state.astype('float32'))
                grids_val_list.append(grid_driver.astype('float32'))
                sources_val_list.append(np.repeat(np.array(file[:-4]), state.shape[0]))

                count_val += 1

        elif i == 2:
            # Ensure that at least 10 seconds of data exists for every observed trajectory.
            if state.shape[0] >= 10:
                states_test_list.append(state.astype('float32'))
                grids_test_list.append(grid_driver.astype('float32'))
                sources_test_list.append(np.repeat(np.array(file[:-4]), state.shape[0]))
                
                count_test += 1

    if i == 0:
        states_train = np.concatenate(states_train_list, axis=0)
        grids_train = np.concatenate(grids_train_list, axis=0)
        sources_train = np.concatenate(sources_train_list, axis=0)

        hkl.dump(states_train, os.path.join(dir_dataset, 'states_shuffled_train.hkl'), mode='w')
        hkl.dump(grids_train, os.path.join(dir_dataset, 'label_grids_shuffled_train.hkl'), mode='w')
        hkl.dump(sources_train, os.path.join(dir_dataset, 'sources_shuffled_train.hkl'), mode='w')                   
        
        # Get mean and std.
        mean = np.mean(states_train, axis=0)
        std = np.std(states_train, axis=0)
        print('mean: ', mean, 'std: ', std)

        del(states_train)
        del(grids_train)
        del(sources_train)
        del(states_train_list)
        del(grids_train_list)
        del(sources_train_list)

states_val = np.concatenate(states_val_list, axis=0)
grids_val = np.concatenate(grids_val_list, axis=0)
sources_val = np.concatenate(sources_val_list, axis=0)

states_test = np.concatenate(states_test_list, axis=0)
grids_test = np.concatenate(grids_test_list, axis=0)
sources_test = np.concatenate(sources_test_list, axis=0)

hkl.dump(states_val, os.path.join(dir_dataset, 'states_val.hkl'), mode='w')
hkl.dump(grids_val, os.path.join(dir_dataset, 'label_grids_val.hkl'), mode='w')
hkl.dump(sources_val, os.path.join(dir_dataset, 'sources_val.hkl'), mode='w')
del(states_val)
del(grids_val)
del(sources_val)

hkl.dump(states_test, os.path.join(dir_dataset, 'states_test.hkl'), mode='w')
hkl.dump(grids_test, os.path.join(dir_dataset, 'label_grids_test.hkl'), mode='w')
hkl.dump(sources_test, os.path.join(dir_dataset, 'sources_test.hkl'), mode='w')
del(states_test)
del(grids_test)
del(sources_test)