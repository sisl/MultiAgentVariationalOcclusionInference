# Code to form the state information for the driver sensor data and split into contiguous segments.

import numpy as np
from matplotlib import pyplot as plt
import hickle as hkl
import pickle as pkl
import pdb
import os
import fnmatch
import csv
from tqdm import tqdm

dir_train_set = '/data/INTERACTION-Dataset-DR-v1_1/processed_data/train_test_split/'
dir = '/data/INTERACTION-Dataset-DR-v1_1/processed_data/pkl/'
for split in ['train', 'val', 'test']:
    
    train_set = os.path.join(dir_train_set, 'ref_' + split + '_set.csv')
    train_set_files = []

    with open(train_set,'rt') as f:
      data = csv.reader(f)
      for row in data:
            train_set_files.append(row[0])

    dir_pickle = '/data/INTERACTION-Dataset-DR-v1_1/processed_data/driver_sensor_label_grids_' + split + '_data/'

    if not os.path.isdir(dir_pickle):
        os.mkdir(dir_pickle)

    grid_shape = (70,50)

    # Set-up count for number of files processed.
    count = 0

    for file in tqdm(os.listdir(dir)):
        file_path = os.path.join(dir, file)

        if file_path in train_set_files:
            count += 1
            f = open(file_path, 'rb')
            X = pkl.load(f)
            f.close()

            # Check if the data is empty. Skip if it is.
            if len(X) == 1:
                continue

            ts = []
            posx = []
            posy = []
            orientation = []
            velx = []
            vely = []
            accx = []
            accy = []
            grid_driver = []
            split_ids = []
 
            for i in range(1,len(X)):
                if ((i != 1) and (X[i][0] != (ts[-1] + 100))):
                    split_ids.append(i-1)
                ts.append(X[i][0])
                posx.append(X[i][2])
                posy.append(X[i][3])
                orientation.append(X[i][4])
                velx.append(X[i][5])
                vely.append(X[i][6])
                accx.append(X[i][7])
                accy.append(X[i][8])
                
                # Reduce grid shape to (20,30).
                grid_driver.append(X[i][9][25:-25,:30])
            
            start = 0
            if len(split_ids) > 0:
                for i in range(len(split_ids)):
                    if len(ts[start:split_ids[i]]) == 0:
                        print("the loop is a problem")
                        pdb.set_trace()
                    pkl.dump([np.array(grid_driver[start:split_ids[i]]),\
                    np.array(ts[start:split_ids[i]]),\
                    np.array(posx[start:split_ids[i]]),\
                    np.array(posy[start:split_ids[i]]),\
                    np.array(orientation[start:split_ids[i]]),\
                    np.array(velx[start:split_ids[i]]),\
                    np.array(vely[start:split_ids[i]]),\
                    np.array(accx[start:split_ids[i]]),\
                    np.array(accy[start:split_ids[i]])],\
                    open(os.path.join(dir_pickle, file[0:-4] + '_' + chr(i + 65) + '.pkl'), 'wb'))
                    start = split_ids[i]

            if len(ts[start:]) == 0:
                print("the last one is a problem")
                pdb.set_trace()
            ts = np.array(ts[start:])
            posx = np.array(posx[start:])
            posy = np.array(posy[start:])
            orientation = np.array(orientation[start:])
            velx = np.array(velx[start:])
            vely = np.array(vely[start:])
            accx = np.array(accx[start:])
            accy = np.array(accy[start:])
            grid_driver = np.array(grid_driver[start:])

            if len(split_ids) > 0:
                pkl.dump([grid_driver, ts, posx, posy, orientation, velx, vely, accx, accy], open(os.path.join(dir_pickle, file[0:-4] + '_' + chr(i + 1 + 65) + '.pkl'), 'wb'))     
            else:
                pkl.dump([grid_driver, ts, posx, posy, orientation, velx, vely, accx, accy], open(os.path.join(dir_pickle, file[0:-4] + '.pkl'), 'wb'))