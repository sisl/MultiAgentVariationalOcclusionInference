# Save the full pipeline occlusion inference output. Code is adapted from: https://github.com/interaction-dataset/interaction-dataset.

import matplotlib
import matplotlib.patches
import matplotlib.transforms
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, AnchoredOffsetbox
from scipy import ndimage
import skimage.transform
from PIL import Image
import pdb
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from copy import deepcopy

import os
import time

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
from tqdm import tqdm
import time

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import torch._utils

from src.utils.dataset_types import Track, MotionState
from src.utils.grid_utils import *
from src.utils.grid_fuse import *
from src.utils.utils_model import to_var
from src.driver_sensor_model.models_cvae import VAE
from src.utils.interaction_utils import *

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center


def polygon_xy_from_motionstate(ms, width, length):
    assert isinstance(ms, MotionState)
    lowleft = (ms.x - length / 2., ms.y - width / 2.)
    lowright = (ms.x + length / 2., ms.y - width / 2.)
    upright = (ms.x + length / 2., ms.y + width / 2.)
    upleft = (ms.x - length / 2., ms.y + width / 2.)
    return rotate_around_center(np.array([lowleft, lowright, upright, upleft]), np.array([ms.x, ms.y]), yaw=ms.psi_rad)


def polygon_xy_from_motionstate_pedest(ms, width, length):
    assert isinstance(ms, MotionState)
    lowleft = (ms.x - length / 2., ms.y - width / 2.)
    lowright = (ms.x + length / 2., ms.y - width / 2.)
    upright = (ms.x + length / 2., ms.y + width / 2.)
    upleft = (ms.x - length / 2., ms.y + width / 2.)
    return np.array([lowleft, lowright, upright, upleft])

def update_objects_plot(timestamp, track_dict=None, pedest_dict=None,
 data=None, car_ids=None, sensor_grids=None, id_grids=None, label_grids=None,
 driver_sensor_data=None, driver_sensor_state_data=None, driver_sensor_state=None, endpoint=None, models=None, results=None, mode='evidential', model='vae'):

    ego_id = data[1][1]

    update = False

    if track_dict is not None:

        # Plot and save the ego-vehicle first.
        assert isinstance(track_dict[ego_id], Track)
        value = track_dict[ego_id]
        if (value.time_stamp_ms_first <= timestamp <= value.time_stamp_ms_last):
            ms_ego = value.motion_states[timestamp]
            assert isinstance(ms_ego, MotionState)

            width = value.width
            length = value.length
            
            # Obtain the ego vehicle's grid.
            res = 1.0
            object_id, pedes_id = SceneObjects(track_dict, timestamp, track_pedes_dict=pedest_dict)
            local_x_ego, local_y_ego, _, _ = local_grid(ms_ego, width, length, res=res, ego_flag=True)
            label_grid_ego, center, x_local, y_local, pre_local_x, pre_local_y = generateLabelGrid(timestamp, track_dict, ego_id, object_id, ego_flag=True, res=res, track_pedes_dict=pedest_dict, pedes_id=pedes_id)
            sensor_grid_ego, occluded_id, visible_id = generateSensorGrid(label_grid_ego, pre_local_x, pre_local_y, ms_ego, width, length, ego_flag=True, res=res)
            visible_id += [ego_id]
            
            full_sensor_grid_dst, mask_unk = get_belief_mass(sensor_grid_ego, ego_flag=True)
            full_sensor_grid = pignistic(full_sensor_grid_dst)

        run_time = 0.

        # Initialize the variables to keep for later computation.
        if model != 'kmeans':
            all_latent_classes = []
            ref_local_xy_list = []
            ego_local_xy_list = []
            alpha_p_list = []
        sensor_grid_ego_dst = [full_sensor_grid_dst]

        if mode == 'average':
            average_mask = np.zeros(full_sensor_grid.shape)
            driver_sensor_grid = np.zeros(full_sensor_grid.shape)

        # Consider the rest of the agents.
        for key, value in track_dict.items():
            assert isinstance(value, Track)
            if ((value.time_stamp_ms_first <= timestamp <= value.time_stamp_ms_last)):
                ms = value.motion_states[timestamp]
                assert isinstance(ms, MotionState)

                width = value.width
                length = value.length

                # Consider all the visible drivers.
                if ((key in visible_id) and (key != ego_id)):
                    res = 1.0            
                    if key in driver_sensor_state_data.keys():
                        for state in driver_sensor_state_data[key]:
                            if state[0] == timestamp:
                                if key in driver_sensor_state.keys():

                                    # Make sure that the states are contiguous.
                                    if state[0] - driver_sensor_state[key][-1,0] == 100:
                                        driver_sensor_state[key] = np.concatenate((driver_sensor_state[key], np.reshape(state, (1,-1))))
                                    else:
                                        driver_sensor_state[key] = np.reshape(state, (1,-1))
                                else:
                                    driver_sensor_state[key] = np.reshape(state, (1,-1))
                                
                                # Perform occlusion inference if at least 1 second of observed driver state has been observed.
                                if ((driver_sensor_state[key].shape[0] == 10)):

                                    # Flag that the map is being updated.
                                    update = True
                                    
                                    start = time.time()

                                    x_local, y_local, _, _ = local_grid(ms, width, length, res=res, ego_flag=False, grid_shape=(20,30))

                                    # Merge grid with driver_sensor_grid.
                                    ref_local_xy = np.stack((x_local, y_local), axis=0)
                                    ego_local_xy = np.stack((local_x_ego, local_y_ego), axis=0)

                                    if model == 'kmeans':
                                        input_state = driver_sensor_state[key][:,1:]
                                        input_state = np.expand_dims(input_state.flatten(), 0)
                                        [kmeans, p_m_a_np] = models['kmeans']
                                        cluster_ids_y_val = kmeans.predict(input_state.astype('float32'))
                                        pred_maps = p_m_a_np[cluster_ids_y_val]
                                        pred_maps = np.reshape(pred_maps[0], (20,30))

                                    elif model == 'gmm':
                                        input_state = driver_sensor_state[key][:,1:]
                                        input_state = np.expand_dims(input_state.flatten(), 0)
                                        [gmm, p_m_a_np] = models['gmm']
                                        cluster_ids_y_val = gmm.predict(input_state.astype('float32'))
                                        alpha_p = gmm.predict_proba(input_state.astype('float32'))
                                        pred_maps = p_m_a_np[cluster_ids_y_val]
                                        pred_maps = np.reshape(pred_maps[0], (20,30))

                                        if len(all_latent_classes) == 0:
                                            all_latent_classes = [np.reshape(p_m_a_np, (100,20,30))]

                                    elif model == 'vae':
                                        input_state = preprocess(driver_sensor_state[key][:,1:])
                                        input_state = torch.unsqueeze(to_var(torch.from_numpy(input_state)), 0).float().cuda()

                                        models['vae'].eval()
                                        with torch.no_grad():
                                            pred_maps, alpha_p, _, _, z = models['vae'].inference(n=1, c=input_state, mode='most_likely')
                                            if len(all_latent_classes) == 0:
                                                recon_x_inf, _, _, _, _ = models['vae'].inference(n=100, c=input_state, mode='all')
                                                all_latent_classes = [recon_x_inf.cpu().numpy()]

                                        pred_maps = pred_maps[0][0].cpu().numpy()
                                        alpha_p = alpha_p.cpu().numpy()

                                    # Transfer the driver sensor model prediction to the ego vehicle's frame of reference.
                                    predEgoMaps = Transfer_to_EgoGrid(ref_local_xy, pred_maps, ego_local_xy, full_sensor_grid, endpoint=endpoint, res=res, mask_unk=mask_unk)

                                    if model != 'kmeans':
                                        if not np.all(predEgoMaps == 2):
                                            alpha_p_list.append(alpha_p[0])
                                            ref_local_xy_list.append(ref_local_xy)
                                            ego_local_xy_list.append(ego_local_xy)                                   

                                    # Fuse the driver sensor model prediction into the ego vehicle's grid.
                                    if mode == 'evidential':
                                        driver_sensor_grid_dst, _ = get_belief_mass(predEgoMaps, ego_flag=False, m=0.95)
                                        full_sensor_grid_dst, full_sensor_grid = dst_fusion(driver_sensor_grid_dst, full_sensor_grid_dst, mask_unk)
                                    elif mode == 'average':
                                        average_mask[predEgoMaps != 2] += 1.0
                                        driver_sensor_grid[predEgoMaps != 2] +=  predEgoMaps[predEgoMaps != 2]
                                    
                                    run_time += (time.time() - start)

                                    # Remove the oldest state.
                                    driver_sensor_state[key] = driver_sensor_state[key][1:]

                                break

        # Save data if the map is updated.
        if update:

            if mode == 'average':
                driver_sensor_grid[average_mask != 0] /= average_mask[average_mask != 0]
                full_sensor_grid[average_mask != 0] = driver_sensor_grid[average_mask != 0]

            sensor_grid_ego[sensor_grid_ego == 2] = 0.5
            results['ego_sensor'].append(sensor_grid_ego)
            results['ego_label'].append(label_grid_ego[0])
            results['vae'].append(full_sensor_grid)
            results['timestamp'].append(timestamp)
            results['run_time'].append(run_time)
            if model != 'kmeans':
                results['all_latent_classes'].append(all_latent_classes[0])
                results['ref_local_xy'].append(ref_local_xy_list)
                results['ego_local_xy'].append(ego_local_xy_list)
                results['alpha_p'].append(alpha_p_list)
            results['ego_sensor_dst'].append(sensor_grid_ego_dst[0])
            results['endpoint'].append(endpoint)
            results['res'].append(res)