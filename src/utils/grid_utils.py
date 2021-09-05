# Code includes utilities for evidential fusion and OGMs.

import numpy as np
from utils.dataset_types import Track, MotionState
import pdb
import time
from matplotlib import pyplot as plt

# Outputs grid_dst: [2,w,h].
# The first channel denotes occupied and free belief masses.
def get_belief_mass(grid, ego_flag=True, m=None):
    grid_dst = np.zeros((2,grid.shape[0], grid.shape[1]))

    if ego_flag:
        mask_occ = grid == 1
        mask_free = grid == 0
        mask_unk = grid == 2

    else:
        mask_not_unk = grid != 2
        mask_unk = None

    if m is not None:
        mass = m
    elif ego_flag:
        mass = 1.0
    else:
        mass = 0.75

    if ego_flag:
        grid_dst[0,mask_occ] = mass
        grid_dst[1,mask_free] = mass

    else:
        grid_dst[0, mask_not_unk] = grid[mask_not_unk] * mass
        grid_dst[1, mask_not_unk] = (1.0-grid[mask_not_unk]) * mass

    return grid_dst, mask_unk

def dst_fusion(sensor_grid_dst, ego_grid_dst, mask_unk):

    fused_grid_dst = np.zeros(ego_grid_dst.shape)

    # predicted unknown mass
    ego_unknown = 1. - ego_grid_dst[0] - ego_grid_dst[1]
    
    # measurement masses: meas_m_free, meas_m_occ
    sensor_unknown = 1. - sensor_grid_dst[0] - sensor_grid_dst[1]
    
    # Implement DST rule of combination.
    K = np.multiply(ego_grid_dst[1], sensor_grid_dst[0]) + np.multiply(ego_grid_dst[0], sensor_grid_dst[1])
    
    fused_grid_dst[0] = np.divide((np.multiply(ego_grid_dst[0], sensor_unknown) + np.multiply(ego_unknown, sensor_grid_dst[0]) + np.multiply(ego_grid_dst[0], sensor_grid_dst[0])), (1. - K))
    fused_grid_dst[1] = np.divide((np.multiply(ego_grid_dst[1], sensor_unknown) + np.multiply(ego_unknown, sensor_grid_dst[1]) + np.multiply(ego_grid_dst[1], sensor_grid_dst[1])), (1. - K))
    
    pignistic_grid = pignistic(fused_grid_dst)

    return fused_grid_dst, pignistic_grid

def pignistic(grid_dst):
    grid = 0.5*grid_dst[0] + 0.5*(1.-grid_dst[1])
    return grid

def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center

# Create a grid of x and y values that have an associated x,y centre coordinate in the global frame.
def global_grid(origin,endpoint,res):

    xmin = min(origin[0],endpoint[0])
    xmax = max(origin[0],endpoint[0])
    ymin = min(origin[1],endpoint[1])
    ymax = max(origin[1],endpoint[1])

    x_coords = np.arange(xmin,xmax,res)
    y_coords = np.arange(ymin,ymax,res)

    gridx,gridy = np.meshgrid(x_coords,y_coords)
    return gridx.T,gridy.T

def local_grid(ms, width, length, res, ego_flag=True, grid_shape=None):

    center = np.array([ms.x, ms.y])

    if ego_flag:
        minx = center[0] - 10.
        miny = center[1] - 35.
        maxx = center[0] + 50.
        maxy = center[1] + 35.
    else:
        if grid_shape is not None:
            minx = center[0] + length/2.
            miny = center[1] - grid_shape[0]/2.
            maxx = center[0] + length/2. + grid_shape[1]
            maxy = center[1] + grid_shape[0]/2.
        else:
            minx = center[0] + length/2.
            miny = center[1] - 35.
            maxx = center[0] + length/2. + 50.
            maxy = center[1] + 35.

    x_coords = np.arange(minx,maxx,res)
    y_coords = np.arange(miny,maxy,res)

    mesh_x, mesh_y = np.meshgrid(x_coords,y_coords)
    pre_local_x = mesh_x
    pre_local_y = np.flipud(mesh_y)

    xy_local = rotate_around_center(np.vstack((pre_local_x.flatten(), pre_local_y.flatten())).T, center, ms.psi_rad)

    x_local = xy_local[:,0].reshape(mesh_x.shape)
    y_local = xy_local[:,1].reshape(mesh_y.shape)

    if grid_shape is not None:
        x_local = grid_reshape(x_local, grid_shape)
        y_local = grid_reshape(y_local, grid_shape)

    elif ego_flag:
        grid_shape = (int(70/res),int(60/res))

        x_local = grid_reshape(x_local, grid_shape)
        y_local = grid_reshape(y_local, grid_shape)

    return x_local, y_local, pre_local_x, pre_local_y

# Element in nd array closest to the scalar value v.
def find_nearest(n,v,v0,vn,res):
    idx = int(np.floor( n*(v-v0+res/2.)/(vn-v0+res) ))
    return idx

def grid_reshape(data, grid_shape):
    if len(data)==1:
        d = data[0]
        if d.shape[0]!= grid_shape[0] or d.shape[1]!= grid_shape[1]:
            data = [d[:grid_shape[0],:grid_shape[1]]]
        
    else:
        if len(data.shape) == 3:
            if data.shape[1] > grid_shape[0]:
                data = data[:,:grid_shape[0]]
            if data.shape[2] > grid_shape[1]:
                data = data[:,:,:grid_shape[1]]            
        
        elif len(data.shape) == 2:
            if data.shape[0]!= grid_shape[0] or data.shape[1]!= grid_shape[1]:
                data = data[:grid_shape[0],:grid_shape[1]]
    return data   

############## LabelGrid ###########################
# 0 = empty
# 1 = ego vehicle
# 2 = vehicle
# 3 = pedestrian
# Pedestrians have negative labels (e.g., P1 -> -1).
#####################################################
def generateLabelGrid(timestamp, track_dict, ego_id, object_id, ego_flag, res=1., grid_shape=None, track_pedes_dict=None, pedes_id=None):

    # Initialize the labels.
    labels = []
    boxes_vehicles = []
    boxes_persons = []
    dynamics_vehicles = []
    dynamics_persons = []

    # The global reference direction of all vehicles and people is in the direction of the ego vehicle.
    for key, value in track_dict.items():
        assert isinstance(value, Track)
        if key == ego_id:
            ms = value.motion_states[timestamp]
            assert isinstance(ms, MotionState)
            x_ego = ms.x
            y_ego = ms.y
            theta_ego = ms.psi_rad
            vx_ego = ms.vx
            vy_ego = ms.vy
            w_ego = value.width
            l_ego = value.length

    center = np.array([x_ego, y_ego])
    ms = getstate(timestamp, track_dict, ego_id)

    x_local, y_local, pre_local_x, pre_local_y = local_grid(ms, w_ego, l_ego, res, ego_flag=ego_flag, grid_shape=grid_shape)

    label_grid = np.zeros((4,x_local.shape[0],x_local.shape[1])) 
    label_grid[3] = np.nan

    for key, value in track_dict.items():
        assert isinstance(value, Track)
        if key in object_id:
            ms = value.motion_states[timestamp]
            assert isinstance(ms, MotionState)

            d = np.sqrt((ms.x-x_ego)**2 + (ms.y-y_ego)**2)
            if d < (np.sqrt(2.)*22.+2.*l_ego): 
                vx = ms.vx
                vy = ms.vy

                w = value.width
                l = value.length
                ID = key

                if key == ego_id:
                    w += res/4.

                coords = polygon_xy_from_motionstate(ms, w, l)
                mask = point_in_rectangle(x_local, y_local, coords)
                if ID == ego_id:
                    # Mark as occupied with ego vehicle.
                    label_grid[0,mask] = 2.

                else:
                    # Mark as occupied with vehicle.
                    label_grid[0,mask] = 1. #

                label_grid[3,mask] = value.track_id

                label_grid[1,mask] = vx
                label_grid[2,mask] = vy

    if track_pedes_dict != None:
        for key, value in track_pedes_dict.items():
            assert isinstance(value, Track)
            if key in pedes_id:
                ms = value.motion_states[timestamp]
                assert isinstance(ms, MotionState)

                d = np.sqrt((ms.x-x_ego)**2 + (ms.y-y_ego)**2)
                if d < (np.sqrt(2.)*22.+2.*l_ego): 
                    vx = ms.vx
                    vy = ms.vy

                    w = 1.5
                    l = 1.5
                    ID = key

                    coords = polygon_xy_from_motionstate_pedest(ms, w, l)
                    mask = point_in_rectangle(x_local, y_local, coords)
                    # Mark as occupied with vehicle.
                    label_grid[0,mask] = 1. #

                    label_grid[3,mask] = -1.0 * float(value.track_id[1:])

                    label_grid[1,mask] = vx
                    label_grid[2,mask] = vy

    if grid_shape == None:
        if ego_flag:
            grid_shape = (int(70/res),int(60/res))
        else:
            grid_shape = (int(70/res),int(50/res))

    label_grid = grid_reshape(label_grid, grid_shape)
    x_local = grid_reshape(x_local, grid_shape)
    y_local = grid_reshape(y_local, grid_shape)
    pre_local_x = grid_reshape(pre_local_x, grid_shape)
    pre_local_y = grid_reshape(pre_local_y, grid_shape)
    
    return label_grid, center, x_local, y_local, pre_local_x, pre_local_y

############## SensorGrid ##################
# 0:  empty
# 1 : occupied
# 2 :  unknown
#############################################
def generateSensorGrid(labels_grid, pre_local_x, pre_local_y, ms, width, length, res=1., ego_flag=True, grid_shape=None):
    center_ego = (ms.x, ms.y)
    theta_ego = ms.psi_rad
    occluded_id = []
    visible_id = []
    # All the angles of the LiDAR simulation.
    angle_res = 0.01 #0.002
    if ego_flag:
        angles = np.arange(0., 2.*np.pi+angle_res, angle_res)
    else:
        angles = np.arange(0., 2.*np.pi+angle_res, angle_res)

    # Get the maximum and minimum x and y values in the local grids.
    x_min = np.amin(pre_local_x)
    x_max = np.amax(pre_local_x)
    y_min = np.amin(pre_local_y)
    y_max = np.amax(pre_local_y)

    x_shape = pre_local_x.shape[0]
    y_shape = pre_local_y.shape[1]

    # Get the cells not occupied by the ego vehicle.
    mask = np.where(labels_grid[0]!=2, True,False)

    # No need to do ray tracing if no object on the grid.
    if np.all(labels_grid[0,mask]==0.):
        sensor_grid = np.zeros((x_shape, y_shape))
    
    else:
        # Generate a line from the center indices to the edge of the local grid: sqrt(2)*128./3. meters away (LiDAR distance).
        r = (np.sqrt(x_shape**2 + y_shape**2) + 10) * 1.0 * res
        x = (r*np.cos(angles)+center_ego[0]) # length of angles
        y = (r*np.sin(angles)+center_ego[1]) # length of angles

        sensor_grid = np.zeros((x_shape, y_shape))
        
        for i in range(x.shape[0]):

            if x[i] < center_ego[0]:
                    x_range = np.arange(center_ego[0],np.maximum(x[i], x_min-res),-res*angle_res)
            else:
                    x_range = np.arange(center_ego[0],np.minimum(x[i]+res, x_max+res),res*angle_res)

            # Find the corresponding ys.
            y_range = linefunction(center_ego[0],center_ego[1],x[i],y[i],x_range)

            y_temp = np.floor(y_shape*(x_range-x_min-res/2.)/(x_max-x_min+res)).astype(int) 
            x_temp = np.floor(x_shape*(y_range-y_min-res/2.)/(y_max-y_min+res)).astype(int)

            # Take only the indices inside the local grid.
            indices = np.where(np.logical_and(np.logical_and(np.logical_and((x_temp < x_shape), (x_temp >= 0)), (y_temp < y_shape)), (y_temp >= 0)))
            x_temp = x_temp[indices]
            y_temp = y_temp[indices]    

            # Found first occupied cell.
            labels_reduced = labels_grid[0,x_temp,y_temp]

            if len(labels_reduced)!=0 :
                unique_labels = np.unique(labels_reduced)

                # Check if there are any occupied cells.
                if np.any(unique_labels==1.):
                    ind = np.where(labels_reduced == 1) 
                    sensor_grid[x_temp[ind[0][0]:],y_temp[ind[0][0]:]] = 2
                    sensor_grid[x_temp[ind[0][0]], y_temp[ind[0][0]]] = 1
                else:
                    sensor_grid[x_temp, y_temp] = 0

    # No ego id included.
    unique_id = np.unique(labels_grid[3,:,:])
    unique_id= np.delete(unique_id, np.where(np.isnan(unique_id)))

    # Set the unknown area as free.
    for id in unique_id:
        mask = (labels_grid[3,:,:]==id)
        if np.any(sensor_grid[mask] == 1):
            # Set as occupied/visible.
            sensor_grid[mask] = 1
            visible_id.append(id)
        # Ignore ego vehicle.
        elif np.any(labels_grid[0,mask]== 2): # except ego car
            pass
        else:
            # Set as occluded/invisible.
            sensor_grid[mask] = 2
            occluded_id.append(id)

    # Set the ego vehicle as occupied.
    mask = (labels_grid[0,:,:] == 2)
    sensor_grid[mask] = 1

    if grid_shape == None:
        if ego_flag:
            grid_shape = (int(70/res),int(60/res))
        else:
            grid_shape = (int(70/res),int(50/res))

    sensor_grid = grid_reshape(sensor_grid, grid_shape)

    return sensor_grid, occluded_id, visible_id

def linefunction(velx,vely,indx,indy,x_range):
    m = (indy-vely)/(indx-velx)
    b = vely-m*velx
    return m*x_range + b 

def point_in_rectangle(x, y, rectangle):
    A = rectangle[0]
    B = rectangle[1]
    C = rectangle[2]

    M = np.array([x,y]).transpose((1,2,0))

    AB = B-A
    AM = M-A
    BC = C-B
    BM = M-B

    dotABAM = np.dot(AM,AB)
    dotABAB = np.dot(AB,AB)
    dotBCBM = np.dot(BM,BC)
    dotBCBC = np.dot(BC,BC)

    return np.logical_and(np.logical_and(np.logical_and((0. <= dotABAM), (dotABAM <= dotABAB)), (0. <= dotBCBM)), (dotBCBM <= dotBCBC)) # nxn

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

def getstate(timestamp, track_dict, id):
    for key, value in track_dict.items():
        if key==id:
            return value.motion_states[timestamp]

# All objects at time step t in interval [time_stamp_ms_start,time_stamp_ms_last].
def SceneObjects(track_dict, time_step, track_pedes_dict=None):
    object_id = []
    pedes_id = []
    if track_dict != None:
        for key, value in track_dict.items():
            assert isinstance(value, Track)

            if value.time_stamp_ms_first <= time_step <= value.time_stamp_ms_last:
                object_id.append(value.track_id)

    if track_pedes_dict != None:
        for key, value in track_pedes_dict.items():
            assert isinstance(value, Track)

            if value.time_stamp_ms_first <= time_step <= value.time_stamp_ms_last:
                pedes_id.append(value.track_id)
       
    return object_id, pedes_id

# All objects in the scene.
def AllObjects(track_dict, track_pedes_dict=None):
    object_id = []
    for key, value in track_dict.items():
        assert isinstance(value, Track)

        object_id.append(value.track_id)

    pedes_id = []
    if track_pedes_dict != None:
        for key, value in track_pedes_dict.items():
            assert isinstance(value, Track)

            pedes_id.append(value.track_id)   

    return object_id, pedes_id