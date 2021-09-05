# Code to transform the driver sensor OGMs to the ego vehicle's OGM frame of reference.

import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from utils.grid_utils import global_grid
import time
from scipy.spatial import cKDTree
import pdb

def mask_in_EgoGrid(global_grid_x, global_grid_y, ref_xy, ego_xy, pred_egoGrid, pred_maps, res, mask_unk=None, tolerance=1):

    # Consider only the unknown cells in pred_egoGrid (ego sensor grid before trasfering values).
    indices = np.where(mask_unk)
    ego_x = ego_xy[0][indices]
    ego_y = ego_xy[1][indices]
    ego_xy = [ego_x, ego_y]
    flat_indicies = indices[0]*pred_egoGrid.shape[1]+indices[1]

    # ref indx --> global indx
    ref_x_ind = np.floor(global_grid_x.shape[1]*(ref_xy[0]-x_min+res/2.)/(x_max-x_min+res)).astype(int) # column index
    ref_y_ind = np.floor(global_grid_y.shape[0]*(ref_xy[1]-y_min+res/2.)/(y_max-y_min+res)).astype(int) # row index

    ref_global_ind = np.vstack((ref_y_ind.flatten(), ref_x_ind.flatten())).T

    # ego indx --> global indx
    ego_x_ind = np.floor(global_grid_x.shape[1]*(ego_xy[0]-x_min+res/2.)/(x_max-x_min+res)).astype(int) # column index
    ego_y_ind = np.floor(global_grid_y.shape[0]*(ego_xy[1]-y_min+res/2.)/(y_max-y_min+res)).astype(int) # row index
    ego_global_ind = np.vstack((ego_y_ind.flatten(), ego_x_ind.flatten())).T

    # Look for the matching global_grid indices between the ref_grid and ego_grid.
    kdtree = cKDTree(ref_global_ind) 
    dists, inds = kdtree.query(ego_global_ind)

    pred_egoGrid_flat = pred_egoGrid.flatten()
    pred_maps_flat = pred_maps.flatten()

    # Back to the local grid indices. Tolerance should be an integer because kd tree is comparing indices.
    ego_ind = flat_indicies[np.where(dists<=tolerance)]
    ref_ind = inds[np.where(dists<=tolerance)] 

    # Assign the values for the corresponding cells.
    pred_egoGrid_flat[ego_ind] = pred_maps_flat[ref_ind]
    pred_egoGrid = pred_egoGrid_flat.reshape(pred_egoGrid.shape)
    return pred_egoGrid 

def Transfer_to_EgoGrid(ref_local_xy, pred_maps, ego_local_xy, ego_sensor_grid, endpoint, res=0.1, mask_unk=None):
    global x_min, x_max, y_min, y_max
    #####################################################################################################################################
    ## Goal : Transfer pred_maps (in driver sensor's grid) cell information to the unknown cells of ego car's sensor_grid
    ## Method : Use global grid as an intermediate (ref indx --> global indx --> ego indx).
    ## ref_local_xy (N, 2, w, h) & pred_maps (N, w, h)
    ## ego_xy (2, w', h') & & ego_sensor_grid (w', h')
    ## return pred_maps_egoGrid(N, w', h')
    ## * N : number of agents
    #####################################################################################################################################

    x_min = endpoint[0]
    x_max = endpoint[2]
    y_min = endpoint[1]
    y_max = endpoint[3]

    global_res = 1.0
    global_grid_x, global_grid_y = global_grid(np.array([x_min,y_min]),np.array([x_max,y_max]),global_res)

    if np.any(ref_local_xy[0] == None):
        pred_maps_egoGrid.append(None)

    else:
        pred_egoGrid = copy.copy(ego_sensor_grid) 
        pred_egoGrid = np.ones(ego_sensor_grid.shape)*2

        pred_egoGrid = mask_in_EgoGrid(global_grid_x, global_grid_y, ref_local_xy, ego_local_xy, pred_egoGrid, pred_maps, res, mask_unk)

    return pred_egoGrid