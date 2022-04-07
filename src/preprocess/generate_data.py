# INTERACTION dataset processing code. Some code snippets are adapted from: https://github.com/interaction-dataset/interaction-dataset.

import argparse
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

from utils import dataset_reader
from utils import dataset_types
# from utils import dict_utils
from utils import map_vis_lanelet2
from utils.grid_utils import SceneObjects, global_grid, AllObjects, generateLabelGrid, generateSensorGrid
from utils.dataset_types import Track, MotionState
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import defaultdict
import os
import pickle as pkl
from datetime import datetime
import glob
import random
from tqdm import tqdm
import time
np.random.seed(123)


def getstate(timestamp, track_dict, id):
    for key, value in track_dict.items():
        if key==id:
            return value.motion_states[timestamp]

def getmap(maps_dir, scenario_data):
    # load and draw the lanelet2 map, either with or without the lanelet2 library
    fig, axes = plt.subplots(1)
    lat_origin = 0.  # origin is necessary to correctly project the lat lon values in the osm file to the local
    lon_origin = 0.  # coordinates in which the tracks are provided; we decided to use (0|0) for every scenario
    lanelet_map_ending = ".osm"
    lanelet_map_file = maps_dir + "/" + scenario_data + lanelet_map_ending
    # print("Loading map...")
    if use_lanelet2_lib:
        projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))
        laneletmap = lanelet2.io.load(lanelet_map_file, projector)
        map_vis_lanelet2.draw_lanelet_map(laneletmap, axes)
    else:
        min_max_xy = map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, lat_origin, lon_origin)
    plt.close(fig)
    return min_max_xy

# Get data for the observed drivers.
def vis_state(vis_datas, ref_datas, object_id, label_grid_ego, prev_id_grid_ego, sensor_grid_ego, prev_sensor_grid_ego, track_dict, stamp, start_timestamp, track_pedes_dict=None, pedes_id=None): # For sensor vis & vis vis. Not for ego vis
    global vis_ax, vis_ay, vis_ids, ego_id, num_vis, res, gridglobal_x, gridglobal_y
    
    # Create a mask for where there is occupied space in the sensor grid that is not the ego vehicle.
    mask = np.where(np.logical_and(sensor_grid_ego==1., label_grid_ego[3]!=ego_id), True, False)
    
    # Get the visible car ids around the ego vehicle.
    vis_temp = np.array(np.unique(label_grid_ego[3,mask]),dtype=int)

    # Remove the pedestrians from the labels.
    vis_temp = vis_temp[vis_temp >= 0]
    
    # Create a mask for the previous timestamp sensor grid where there is occupied space that is not the ego vehicle.
    mask = np.where(np.logical_and(prev_sensor_grid_ego==1., prev_id_grid_ego!=ego_id), True, False)
    prev_vis_temp = np.array(np.unique(prev_id_grid_ego[mask]),dtype=int)
    prev_vis_temp = prev_vis_temp[prev_vis_temp >= 0]

    # Compute the number of new visible ids at this timestamp.
    num_new = 0
    # Loop through the visible ids at the current timestamp and add them to vis_ids. 
    for vis_t in vis_temp:
        if vis_t not in vis_ids:
            # Add new visible vehicle to this vis_ids array.
            vis_ids.append(vis_t)
            num_new += 1

    # All visible cars till the current timestamp.
    num_vis = len(vis_ids)
    vis_ms = []
    prev_vis_ms = []
    widths = []
    lengths = []

    for id in vis_ids:
        # Check if the ID is present in the current and previous timestamp. This is important for acceleration computations.
        if id in vis_temp and id in prev_vis_temp:
            vis_ms.append(getstate(stamp, track_dict, id))
            widths.append(track_dict[id].width)
            lengths.append(track_dict[id].length)
            prev_vis_ms.append(getstate(stamp-100, track_dict, id))
        else:
            # If the ID was not present in a previous timestamp, set to None.
            vis_ms.append(None)
            prev_vis_ms.append(None)
            widths.append(None)
            lengths.append(None)

    # If there are new visible cars, extend the lists for accelerations and data.
    if num_new != 0:
        vis_ax = vis_ax+ [0]*num_new
        vis_ay = vis_ay+ [0]*num_new
        vis_datas = vis_datas + [[]]*num_new
        ref_datas = ref_datas + [[]]*num_new

    # Compute the accelerations for all the visible vehicles.
    for k in range(num_vis):
        # Only compute the acceleration, if the there are two timestamps of data for the vehicle.
        if vis_ms[k] is not None:
            vis_ax[k] = (vis_ms[k].vx - prev_vis_ms[k].vx)  / 0.1
            vis_ay[k]  = (vis_ms[k].vy - prev_vis_ms[k].vy) / 0.1
    
    # Loop through all the visible vehicles.
    for k in range(num_vis):
        # Check if the vehicle is visible in at least two timestamps.
        if vis_ids[k] in vis_temp and vis_ids[k] in prev_vis_temp :

            # Generate label and sensor grids for the observed driver.
            label_grid_ref, center_ref, _, _, pre_local_x_ref, pre_local_y_ref = generateLabelGrid(stamp, track_dict, vis_ids[k], object_id, ego_flag=False, res=res, track_pedes_dict=track_pedes_dict, pedes_id=pedes_id)
            width = widths[k]
            length = lengths[k]
            sensor_grid_ref, _, _ = generateSensorGrid(label_grid_ref, pre_local_x_ref, pre_local_y_ref, vis_ms[k], width, length, res=res, ego_flag=False)

            # Mark the ego car as free.
            sensor_grid_ref[label_grid_ref[0] == 2] = 0
            label_grid_ref[0] = np.where(label_grid_ref[0]==2., 0. ,label_grid_ref[0])            

            # Store data for visible vehicles for the ego vehicle and the observed driver.            
            vis_step_data = [stamp, vis_ids[k], vis_ms[k].x, vis_ms[k].y, vis_ms[k].psi_rad, vis_ms[k].vx, vis_ms[k].vy, vis_ax[k], vis_ay[k], label_grid_ref[0], label_grid_ref[3], sensor_grid_ref, np.nan]
            ref_step_data = [stamp, vis_ids[k], vis_ms[k].x, vis_ms[k].y, vis_ms[k].psi_rad, vis_ms[k].vx, vis_ms[k].vy, vis_ax[k], vis_ay[k], label_grid_ref[0], sensor_grid_ref]
            ref_datas[k] = ref_datas[k]+[ref_step_data]
            vis_datas[k] = vis_datas[k]+[vis_step_data]

    # Note ref_datas may include unseen vehicles (or vehicles seen for just one timestamp).
    return vis_datas, ref_datas

def Dataprocessing():
    global vis_ids, vis_ax, vis_ay, ref_ax, ref_ay, ego_id, res, gridglobal_x, gridglobal_y

    main_folder = '/data/INTERACTION-Dataset-DR-v1_1/'
    scenarios = ['DR_USA_Intersection_GL']
    
    # Total number of track files.
    num_files = [60]
    for scene, nth_scene in zip(scenarios, num_files):
        for i in tqdm(range(nth_scene)):

            i_str = ['%03d' % i][0]
            filename = os.path.join(main_folder, 'recorded_trackfiles/'+scene+'/'+'vehicle_tracks_'+ i_str +'.csv')
            track_dict = dataset_reader.read_tracks(filename)
            filename_pedes = os.path.join(main_folder, 'recorded_trackfiles/'+scene+'/'+'pedestrian_tracks_'+ i_str +'.csv')
            
            if os.path.exists(filename_pedes):
                track_pedes_dict = dataset_reader.read_pedestrian(filename_pedes)
            else:
                track_pedes_dict = None
            
            run_count = 0

            maps_dir = os.path.join(main_folder, "maps")
            min_max_xy = getmap(maps_dir, scene)
            xminGPS = min_max_xy[0]
            xmaxGPS = min_max_xy[2]
            yminGPS = min_max_xy[1]
            ymaxGPS = min_max_xy[3]

            res = 1.
            gridglobal_x,gridglobal_y = global_grid(np.array([xminGPS,yminGPS]),np.array([xmaxGPS,ymaxGPS]),res)

            vehobjects, pedesobjects = AllObjects(track_dict, track_pedes_dict)
            
            processed_file = glob.glob(os.path.join(main_folder, '/Processed_data_new_goal/pkl/DR_USA_Intersection_GL_'+i_str+'*_ego_*'))
            processed_id = [ int(file.split('_')[-1][:-4]) for file in processed_file]

            num = 0
            sampled_key = [id for id in vehobjects if id not in processed_id][num:]
            run_count = num
            sampled_key = np.random.choice(sampled_key,np.minimum(100, len(sampled_key)),replace=False)
            
            for key, value in track_dict.items():
                assert isinstance(value, Track)
                if key in sampled_key:

                    start_time = datetime.now()  
                    ego_data = [['timestamp', 'car_id', 'x', 'y', 'orientation', 'vx','vy', 'ax', 'ay', 'label_grid', 'id_grid', 'sensor_grid', 'occluded_id']]
                    ref_data = [['timestamp', 'car_id', 'x', 'y', 'orientation', 'vx','vy', 'ax', 'ay', 'label_grid', 'sensor_grid']]

                    ego_id = int(key)
                    vis_ids = []

                    start_timestamp = value.time_stamp_ms_first
                    last_timestamp = value.time_stamp_ms_last

                    ref_datas = []
                    vis_datas = []

                    vis_ax = []
                    vis_ay = []

                    # Get accelerations.
                    for stamp in range(start_timestamp, last_timestamp, 100):
                        object_id, pedes_id = SceneObjects(track_dict, stamp, track_pedes_dict)
                        ego_ms = getstate(stamp, track_dict, ego_id)

                        if stamp == start_timestamp :
                            prev_ego_vx = ego_ms.vx
                            prev_ego_vy = ego_ms.vy

                        else:    
                            ego_ax = (ego_ms.vx - prev_ego_vx)  / 0.1
                            ego_ay = (ego_ms.vy - prev_ego_vy) / 0.1

                        # Get label grid.
                        label_grid_ego, center_ego, local_x_ego, local_y_ego, pre_local_x_ego, pre_local_y_ego = generateLabelGrid(stamp, track_dict, ego_id, object_id, ego_flag=True, res=res, track_pedes_dict=track_pedes_dict, pedes_id=pedes_id)

                        # Get sensor grid.
                        width = value.width
                        length = value.length
                        sensor_grid_ego, occluded_id, visible_id = generateSensorGrid(label_grid_ego, pre_local_x_ego, pre_local_y_ego, ego_ms, width, length, res=res, ego_flag=True)

                        # Convert the ego grid cells to occupied.
                        label_grid_ego[0] = np.where(label_grid_ego[0]==2., 1. ,label_grid_ego[0])

                        # Ignore the first timestamp because we do not have acceleration data.
                        if stamp != start_timestamp :

                            # Save ego data.
                            ego_step_data = [stamp, ego_id, ego_ms.x, ego_ms.y, ego_ms.psi_rad, ego_ms.vx, ego_ms.vy, ego_ax, ego_ay, label_grid_ego[0], label_grid_ego[3], sensor_grid_ego, occluded_id]
                            ego_data.append(ego_step_data)

                            vis_datas, ref_datas = vis_state(vis_datas, ref_datas, object_id, label_grid_ego, prev_id_grid_ego, sensor_grid_ego, prev_sensor_grid_ego, track_dict, stamp, start_timestamp, track_pedes_dict=track_pedes_dict, pedes_id=pedes_id)

                        # Get the previous time stamp information.
                        prev_id_grid_ego =  label_grid_ego[3]
                        prev_sensor_grid_ego = sensor_grid_ego
                        prev_ego_vx = ego_ms.vx
                        prev_ego_vy = ego_ms.vy

                    if vis_ids is not None:
                        ego_data = ego_data + sum(vis_datas,[])

                    # Save the ego information.
                    pkl_path = os.path.join(main_folder, 'processed_data/pkl/')
                    if not os.path.exists(pkl_path):
                        os.makedirs(pkl_path)
                    ego_filename = scene+'_'+i_str+'_run_'+str(run_count)+'_ego_vehicle_'+str(key)
                    pkl.dump(ego_data, open(str(hkl_path) + ego_filename+'.pkl', 'wb'))

                    # Save only the visible reference drivers.
                    for k in range(num_vis):
                        ref_filename = scene+'_'+i_str+'_run_'+str(run_count)+'_ref_vehicle_'+str(vis_ids[k])
                        ref_d = ref_data + ref_datas[k]
                        pkl.dump(ref_d, open(pkl_path +ref_filename+'.pkl', 'wb'))

                    run_count += 1
                    end_time = datetime.now()
                    print(ego_filename, ', execution time:', end_time - start_time)

if __name__ == "__main__":
    Dataprocessing()
