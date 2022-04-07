# Code to save full occlusion inference pipeline. Code is adapted from: https://github.com/interaction-dataset/interaction-dataset.

try:
    import lanelet2
    use_lanelet2_lib = True
except:
    import warnings
    string = "Could not import lanelet2. It must be built and sourced, " + \
             "see https://github.com/fzi-forschungszentrum-informatik/Lanelet2 for details."
    warnings.warn(string)
    print("Using visualization without lanelet2.")
    use_lanelet2_lib = False
    from utils import map_vis_without_lanelet

import argparse
import os
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import hickle as hkl
import pickle as pkl
import pdb
import numpy as np
import csv

seed = 123

import numpy as np
np.random.seed(seed)
from matplotlib import pyplot as plt
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)

import io
import PIL.Image
from tqdm import tqdm
import time

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from collections import OrderedDict, defaultdict

os.chdir("../..")

from src.utils import dataset_reader
from src.utils import dataset_types
from src.utils import map_vis_lanelet2
from src.utils import tracks_save
# from src.utils import dict_utils
from src.driver_sensor_model.models_cvae import VAE
from src.utils.interaction_utils import *

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


def update_plot():
    global timestamp, track_dictionary, pedestrian_dictionary,\
     data, car_ids, sensor_grids, id_grids, label_grids, driver_sensor_data, driver_sensor_state_data, driver_sensor_state, driver_sensor_state_dict,\
     min_max_xy, models, results, mode, model

    # Update text and tracks based on current timestamp.
    if (timestamp < timestamp_min):
        pdb.set_trace()
    assert(timestamp <= timestamp_max), "timestamp=%i" % timestamp
    assert(timestamp >= timestamp_min), "timestamp=%i" % timestamp
    assert(timestamp % dataset_types.DELTA_TIMESTAMP_MS == 0), "timestamp=%i" % timestamp
    tracks_save_DR_goal_multimodal_average.update_objects_plot(timestamp, track_dict=track_dictionary, pedest_dict=pedestrian_dictionary,
                                    data=data, car_ids=car_ids, sensor_grids=sensor_grids, id_grids=id_grids, label_grids=label_grids, 
                                    driver_sensor_data = driver_sensor_data, driver_sensor_state_data=driver_sensor_state_data, driver_sensor_state = driver_sensor_state,
                                    endpoint=min_max_xy, models=models, results=results, mode=mode, model=model)

if __name__ == "__main__":

    # Provide data to be visualized.
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str, help="Name of the scenario (to identify map and folder for track "
                        "files)", nargs="?")
    parser.add_argument("track_file_number", type=int, help="Number of the track file (int)", default=0, nargs="?")
    parser.add_argument("load_mode", type=str, help="Dataset to load (vehicle, pedestrian, or both)", default="both",
                        nargs="?")
    parser.add_argument("--mode", type=str, help="Sensor fusion mode: evidential or average", nargs="?")
    parser.add_argument("--model", type=str, help="Name of the model: vae, gmm, kmeans", nargs="?")
    args = parser.parse_args()

    if args.load_mode != "vehicle" and args.load_mode != "pedestrian" and args.load_mode != "both":
        raise IOError("Invalid load command. Use 'vehicle', 'pedestrian', or 'both'")

    # Load test folder.
    error_string = ""
    tracks_dir = "/data/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles"
    maps_dir = "/data/INTERACTION-Dataset-DR-v1_1/maps"
    scenario_name = 'DR_USA_Intersection_GL'
    home = "/data/INTERACTION-Dataset-DR-v1_1/processed_data/pkl/"
    test_set_dir = '/data/INTERACTION-Dataset-DR-v1_1/processed_data/train_test_split'
    test_set_file = 'ego_test_set.csv'

    ego_files = []
    with open(os.path.join(test_set_dir, test_set_file)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            ego_files.append(row[0].split(home)[-1])
            
    number = 0
    ego_file = ego_files[number]

    # Load the models.
    models = dict()
    folder_vae = '/model/cvae/'
    name_vae = 'lstm_1_Adam_z_100_lr_0.001_rand_123_norm_True_kl_start_0_finish_1.0_center_10000.0_mutual_info_const_alpha_1.5_epochs_30_batch_256'

    models['vae'] = VAE(
        encoder_layer_sizes_p=[7, 5],
        n_lstms=1,
        latent_size=100,
        dim=4
        )

    models['vae'] = models['vae'].cuda()

    save_filename = os.path.join(folder_vae, name_vae) + 'epoch_30_vae.pt'

    with open(save_filename, 'rb') as f:
        state_dict = torch.load(f)
        models['vae'].load_state_dict(state_dict)
    models['vae'].eval()

    folder_kmeans = '/model/kmeans/'
    models['kmeans'] = pkl.load(open(os.path.join(folder_kmeans,"clusters_kmeans_K_100.pkl"), "rb" ) )

    folder_gmm = '/model/gmm/'
    models['gmm'] = pkl.load(open(os.path.join(folder_gmm, "GMM_K_100_reg_covar_0_001.pkl"), "rb" ) )

    if args.model == 'vae':
        folder_model = folder_vae
    elif args.model == 'gmm':
        folder_model = folder_gmm
    elif args.model == 'kmeans':
        folder_model = folder_kmeans

    # Get the driver sensor data.
    for ego_file in tqdm(ego_files):
        print(ego_file)
        ego_run = int(ego_file.split('_run_')[-1].split('_ego_')[0])
        if scenario_name[-2:] == 'GL':
            middle = scenario_name
            track_file_number = ego_file.split("_run_")[0][-3:]
            if os.path.exists(os.path.join(home, ego_file)):
                data = pkl.load(open(os.path.join(home, ego_file), 'rb'))
            else:
                continue
            
        driver_sensor_data = dict()
        driver_sensor_state_data = dict()
        driver_sensor_state = dict()
        for file in os.listdir(os.path.join(home)):
            if scenario_name[-2:] == 'GL':
                file_split = file.split("_run_")
                if track_file_number != file_split[0][-3:]:
                    continue
                [run, vtype, _, vid] = file_split[-1][:-4].split('_')
            run = int(run)
            vid = int(vid)

            if (run == ego_run) and (vtype == 'ref'):
                driver_sensor_data[vid] = pkl.load(open(os.path.join(home, file), 'rb'))
                for item in driver_sensor_data[vid][1:]:
                    timestamp = item[0]
                    x = item[2]
                    y = item[3]
                    orientation = item[4]
                    vx = item[5]
                    vy = item[6]
                    ax = item[7]
                    ay = item[8]
                    if vid in driver_sensor_state_data.keys():
                        driver_sensor_state_data[vid].append(np.array([timestamp, x, y, orientation, vx, vy, ax, ay]))
                    else:
                        driver_sensor_state_data[vid] = [np.array([timestamp, x, y, orientation, vx, vy, ax, ay])]

        lanelet_map_ending = ".osm"
        lanelet_map_file = maps_dir + "/" + scenario_name + lanelet_map_ending
        scenario_dir = tracks_dir + "/" + scenario_name
        track_file_prefix = "vehicle_tracks_"
        track_file_ending = ".csv"
        track_file_name = scenario_dir + "/" + track_file_prefix + str(track_file_number).zfill(3) + track_file_ending
        pedestrian_file_prefix = "pedestrian_tracks_"
        pedestrian_file_ending = ".csv"
        pedestrian_file_name = scenario_dir + "/" + pedestrian_file_prefix + str(track_file_number).zfill(3) + pedestrian_file_ending
        if not os.path.isdir(tracks_dir):
            error_string += "Did not find track file directory \"" + tracks_dir + "\"\n"
        if not os.path.isdir(maps_dir):
            error_string += "Did not find map file directory \"" + tracks_dir + "\"\n"
        if not os.path.isdir(scenario_dir):
            error_string += "Did not find scenario directory \"" + scenario_dir + "\"\n"
        if not os.path.isfile(lanelet_map_file):
            error_string += "Did not find lanelet map file \"" + lanelet_map_file + "\"\n"
        if not os.path.isfile(track_file_name):
            error_string += "Did not find track file \"" + track_file_name + "\"\n"
        if not os.path.isfile(pedestrian_file_name):
            flag_ped = 0
        else:
            flag_ped = 1
        if error_string != "":
            error_string += "Type --help for help."
            raise IOError(error_string)

        # Load and draw the lanelet2 map, either with or without the lanelet2 library.
        lat_origin = 0.  # Origin is necessary to correctly project the lat lon values in the osm file to the local.
        lon_origin = 0.
        print("Loading map...")
        fig, axes = plt.subplots(1, 1)
        min_max_xy = map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, lat_origin, lon_origin)

        # Expand map size.
        min_max_xy[0:2] -= 25.0
        min_max_xy[2:] += 25.0

        # Load the tracks.
        print("Loading tracks...")
        track_dictionary = None
        pedestrian_dictionary = None
        if args.load_mode == 'both':
            track_dictionary = dataset_reader.read_tracks(track_file_name)
            if flag_ped:
                pedestrian_dictionary = dataset_reader.read_pedestrian(pedestrian_file_name)

        elif args.load_mode == 'vehicle':
            track_dictionary = dataset_reader.read_tracks(track_file_name)
        elif args.load_mode == 'pedestrian':
            pedestrian_dictionary = dataset_reader.read_pedestrian(pedestrian_file_name)

        timestamp_min = 1e9
        timestamp_max = 0

        # Set the time to the run and get the sensor vehicles.
        ego_id = data[1][1]
        car_ids = {}
        sensor_grids = {}
        label_grids = {}
        id_grids = {}
        for i in range(1,len(data)):
            timestamp_min = min(timestamp_min, data[i][0])
            timestamp_max = max(timestamp_max, data[i][0])

            if data[i][0] in car_ids.keys():
                car_ids[data[i][0]].append(data[i][1])
            else:
                car_ids[data[i][0]] = [data[i][1]]

            if data[i][1] == ego_id:
                sensor_grids[data[i][0]] = data[i][-2]
                id_grids[data[i][0]] = data[i][-3]
                label_grids[data[i][0]] = data[i][-4]

        args.start_timestamp = timestamp_min

        # Results.
        results = dict()
        results['ego_sensor'] = []
        results['ego_label'] = []
        results['vae'] = []
        results['timestamp'] = []
        results['source'] = []
        results['run_time'] = []
        results['all_latent_classes'] = []
        results['ref_local_xy'] = []
        results['ego_local_xy'] = []
        results['alpha_p'] = []
        results['ego_sensor_dst'] = []
        results['endpoint'] = []
        results['res'] = []
        mode = args.mode
        model = args.model

        print("Saving...")
        timestamp = args.start_timestamp
        
        while timestamp < timestamp_max:
            update_plot()
            timestamp += dataset_types.DELTA_TIMESTAMP_MS
        
        # Clear all variables for the next scenario
        del(timestamp_min)
        del(timestamp_max)
        del(timestamp)

        for i in range(len(results['timestamp'])):
            results['source'].append(ego_file[:-4])

        # Save the data for each run.
        folder_model_new = os.path.join(folder_model, 'full_pipeline_' + model + '_' + mode)
        if not os.path.isdir(folder_model_new):
            os.mkdir(folder_model_new)
        pkl.dump(results, open(os.path.join(folder_model_new, ego_file[:-4] + '_ego_results.pkl'), "wb"))

