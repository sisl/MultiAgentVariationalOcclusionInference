# Code to visualize full occlusion inference pipeline. Code is adapted from: https://github.com/interaction-dataset/interaction-dataset.

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
from collections import OrderedDict, defaultdict

from src.utils import dataset_reader
from src.utils import dataset_types
from src.utils import map_vis_lanelet2
from src.utils import tracks_vis
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
    global fig, timestamp, title_text, track_dictionary, patches_dict, text_dict, axes, pedestrian_dictionary,\
     data, car_ids, sensor_grids, id_grids, label_grids, grids_dict, driver_sensor_data, driver_sensor_state_data, driver_sensor_state, driver_sensor_state_dict,\
     min_max_xy, models, results, mode, model

    # Update text and tracks based on current timestamp.
    if (timestamp < timestamp_min):
        pdb.set_trace()
    assert(timestamp <= timestamp_max), "timestamp=%i" % timestamp
    assert(timestamp >= timestamp_min), "timestamp=%i" % timestamp
    assert(timestamp % dataset_types.DELTA_TIMESTAMP_MS == 0), "timestamp=%i" % timestamp
    title_text.set_text("\nts = {}".format(timestamp))
    tracks_vis_DR_goal_multimodal_average.update_objects_plot(timestamp, patches_dict, text_dict, axes, track_dict=track_dictionary, pedest_dict=pedestrian_dictionary,
                                    data=data, car_ids=car_ids, sensor_grids=sensor_grids, id_grids=id_grids, label_grids=label_grids, grids_dict=grids_dict,
                                    driver_sensor_data = driver_sensor_data, driver_sensor_state_data=driver_sensor_state_data, driver_sensor_state = driver_sensor_state, driver_sensor_state_dict=driver_sensor_state_dict,
                                    endpoint=min_max_xy, models=models, mode=mode, model=model)

    fig.canvas.draw()


def start_playback():
    global timestamp, timestamp_min, timestamp_max, playback_stopped
    playback_stopped = False
    plt.ion()
    while timestamp < timestamp_max and not playback_stopped:
        timestamp += dataset_types.DELTA_TIMESTAMP_MS
        start_time = time.time()
        update_plot()
        end_time = time.time()
        diff_time = end_time - start_time
        # plt.pause(max(0.001, dataset_types.DELTA_TIMESTAMP_MS / 1000. - diff_time))
    plt.ioff()


class FrameControlButton(object):
    def __init__(self, position, label):
        self.ax = plt.axes(position)
        self.label = label
        self.button = Button(self.ax, label)
        self.button.on_clicked(self.on_click)

    def on_click(self, event):
        global timestamp, timestamp_min, timestamp_max, playback_stopped

        if self.label == "play":
            if not playback_stopped:
                return
            else:
                start_playback()
                return
        playback_stopped = True
        if self.label == "<<":
            timestamp -= 10*dataset_types.DELTA_TIMESTAMP_MS
        elif self.label == "<":
            timestamp -= dataset_types.DELTA_TIMESTAMP_MS
        elif self.label == ">":
            timestamp += dataset_types.DELTA_TIMESTAMP_MS
        elif self.label == ">>":
            timestamp += 10*dataset_types.DELTA_TIMESTAMP_MS
        timestamp = min(timestamp, timestamp_max)
        timestamp = max(timestamp, timestamp_min)
        update_plot()

if __name__ == "__main__":

    # Provide data to be visualized.
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str, help="Name of the scenario (to identify map and folder for track "
                        "files)", nargs="?")
    parser.add_argument("track_file_number", type=int, help="Number of the track file (int)", default=0, nargs="?")
    parser.add_argument("load_mode", type=str, help="Dataset to load (vehicle, pedestrian, or both)", default="both",
                        nargs="?")
    parser.add_argument("--start_timestamp", type=int, nargs="?")
    parser.add_argument("--mode", type=str, default='evidential', help="Sensor fusion mode: evidential or average", nargs="?")
    parser.add_argument("--model", type=str, default='vae', help="Name of the model: vae, gmm, kmeans", nargs="?")
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

    for ego_file in tqdm(ego_files):
        print(ego_file)

        # Visualize the paper and appendix scenarios.
        if (ego_file != 'DR_USA_Intersection_GL_037_run_54_ego_vehicle_83.pkl') and (ego_file != 'DR_USA_Intersection_GL_021_run_60_ego_vehicle_108.pkl'):
            continue

        ego_run = int(ego_file.split('_run_')[-1].split('_ego_')[0])
        if scenario_name[-2:] == 'VA':
            track_file_number = int(ego_file.split(scenario_name)[-1][1:4])
            middle = scenario_name + "/" + scenario_name + "_00" + str(track_file_number)
            if os.path.exists(os.path.join(home, middle, ego_file)):
                data = hkl.load(os.path.join(home, middle, ego_file))
            else:
                continue
        elif scenario_name[-2:] == 'GL':
            middle = scenario_name
            track_file_number = ego_file.split("_run_")[0][-3:]
            if os.path.exists(os.path.join(home, ego_file)):
                data = pkl.load(open(os.path.join(home, ego_file), 'rb'))
            else:
                continue
            
        # Get the driver sensor data.
        driver_sensor_data = dict()
        driver_sensor_state_data = dict()
        driver_sensor_state = dict()
        for file in os.listdir(os.path.join(home)):
            if scenario_name[-2:] == 'VA':
                file_split = file.split(scenario_name + "_00" + str(track_file_number) + "_run_")
                [run, vtype, _, vid] = file_split[-1][:-4].split('_')
            elif scenario_name[-2:] == 'GL':
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

        # Create a figure.
        fig, axes = plt.subplots(1, 1)
        fig.canvas.set_window_title("Interaction Dataset Visualization")

        # Load and draw the lanelet2 map, either with or without the lanelet2 library.
        lat_origin = 0.  # Origin is necessary to correctly project the lat lon values in the osm file to the local.
        lon_origin = 0.
        print("Loading map...")
        if use_lanelet2_lib:
            projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))
            laneletmap = lanelet2.io.load(lanelet_map_file, projector)
            map_vis_lanelet2.draw_lanelet_map(laneletmap, axes)
        else:
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

        if (pedestrian_dictionary == None) or len(pedestrian_dictionary.items()) == 0:
            continue

        timestamp_min = 1e9
        timestamp_max = 0

        # Set the time to the run and get the sensor vehicles
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

        if ego_file == 'DR_USA_Intersection_GL_037_run_54_ego_vehicle_83.pkl':
            args.start_timestamp = 117500
        elif ego_file == 'DR_USA_Intersection_GL_021_run_60_ego_vehicle_108.pkl':
            args.start_timestamp = 168700
        else:
            args.start_timestamp = timestamp_min

        mode = args.mode
        model = args.model

        button_pp = FrameControlButton([0.2, 0.05, 0.05, 0.05], '<<')
        button_p = FrameControlButton([0.27, 0.05, 0.05, 0.05], '<')
        button_f = FrameControlButton([0.4, 0.05, 0.05, 0.05], '>')
        button_ff = FrameControlButton([0.47, 0.05, 0.05, 0.05], '>>')

        button_play = FrameControlButton([0.6, 0.05, 0.1, 0.05], 'play')
        button_pause = FrameControlButton([0.71, 0.05, 0.1, 0.05], 'pause')

        # Storage for track visualization.
        patches_dict = dict()
        text_dict = dict()
        grids_dict = dict()
        driver_sensor_state_dict = dict()

        # Visualize tracks.
        print("Plotting...")
        timestamp = args.start_timestamp
        title_text = fig.suptitle("")
        playback_stopped = True
        update_plot()
        plt.show()
        
        # Clear all variables for the next scenario.
        plt.clf()
        plt.close()
        del(timestamp_min)
        del(timestamp_max)
        del(timestamp)

