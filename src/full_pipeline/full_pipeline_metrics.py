# Compute metrics on full pipeline.

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
from tqdm import tqdm

os.chdir("../..")

from src.utils.interaction_utils import *
from src.utils.grid_utils import *
from src.utils.grid_fuse import *
from src.utils.combinations import *

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


if __name__ == "__main__":

    model = 'vae' # Options: kmeans, gmm, vae.
    mode = 'evidential' # Options: evidential, average.
    print(model, mode)
    folder_model = '/models/' + model + '/full_pipeline_' + model + '_' + mode
    
    count = 0
    num_modes = 3

    ego_vanilla = []
    ego_label = []
    ego_pas = []
    timestamp = []
    source = []
    run_time = []

    ego_pas_multimodal = []
    check = 0

    for file in tqdm(os.listdir(folder_model)):
        if file[-3:] == 'pkl':

            count += 1

            results = pkl.load(open(os.path.join(folder_model, file), "rb"))

            if len(results['ego_vanilla_dst']) == 0:
                continue

            ego_vanilla_dst = results['ego_vanilla_dst']

            pop_indices = []

            if model != 'kmeans':
                all_latent_classes = [results['all_latent_classes'][0]]
                endpoint = [results['endpoint'][0]]
                res = [results['res'][0]]

                ref_local_xy = results['ref_local_xy']
                ego_local_xy = results['ego_local_xy']
                alpha_p = results['alpha_p']

                all_latent_classes = np.array(all_latent_classes[0])
                res = res[0]
                endpoint = endpoint[0]

                # Find the 3 likeliest modes for the fused grids.
                for i in range(len(alpha_p)):
                    if len(alpha_p[i]) > 0:
                        best_index, best_prob = BFS(alpha_p[i], num_modes)
                        if model == 'vae':
                            preds_pas = all_latent_classes[best_index.astype('int')][:,:,0,:,:]
                        elif model == 'gmm':
                            preds_pas = all_latent_classes[best_index.astype('int')]
                        multimodal = []
                        for k in range(num_modes):
                            full_vanilla_grid = pignistic(ego_vanilla_dst[i])
                            full_vanilla_grid_dst = deepcopy(ego_vanilla_dst[i])
                            mask_unk = full_vanilla_grid == 0.5
                            if mode == 'average':
                                average_mask = np.zeros(full_vanilla_grid.shape)
                                driver_vanilla_grid = np.zeros(full_vanilla_grid.shape)
                            for veh in range(preds_pas.shape[1]):
                                predEgoMaps = Transfer_to_EgoGrid(ref_local_xy[i][veh], preds_pas[k, veh], ego_local_xy[i][veh], full_vanilla_grid, endpoint=endpoint, res=res, mask_unk=mask_unk)

                                if mode == 'evidential':      
                                    driver_vanilla_grid_dst, _ = get_belief_mass(predEgoMaps, ego_flag=False, m=0.95)
                                    full_vanilla_grid_dst, full_vanilla_grid = dst_fusion(driver_vanilla_grid_dst, full_vanilla_grid_dst, mask_unk)
                                elif mode == 'average':
                                    average_mask[predEgoMaps != 2] += 1.0
                                    driver_vanilla_grid[predEgoMaps != 2] +=  predEgoMaps[predEgoMaps != 2]

                            if mode == 'average':
                                driver_vanilla_grid[average_mask != 0] /= average_mask[average_mask != 0]
                                full_vanilla_grid[average_mask != 0] = driver_vanilla_grid[average_mask != 0]

                            multimodal.append(full_vanilla_grid)
                        ego_pas_multimodal.append(multimodal)
                    else:
                        pop_indices.append(i)

                pop_indices = sorted(pop_indices, reverse=True)

                for i in pop_indices:
                    results['ego_vanilla'].pop(i)
                    results['ego_label'].pop(i)
                    results['vae'].pop(i)
                    results['run_time'].pop(i)

            ego_pas += results['vae']
            ego_vanilla += results['ego_vanilla']
            ego_label += results['ego_label']
            run_time += results['run_time']

    print(count)

    ego_vanilla = np.array(ego_vanilla, dtype='float32')
    ego_vanilla_dst = np.array(ego_vanilla_dst, dtype='float32')
    ego_label = np.array(ego_label, dtype='float32')
    ego_label[ego_label == 2] = 1.0
    ego_pas = np.array(ego_pas, dtype='float32')
    timestamp = np.array(timestamp)
    source = np.array(source)
    run_time = np.array(run_time)

    print('Run time: ', np.mean(run_time))    

    if model == 'kmeans':
         # Pick out only the occluded areas.
        mask_occluded = ego_vanilla == 0.5

        ego_pas_pred = (np.logical_and(ego_pas >= 0.6, mask_occluded)).astype(float)
        ego_pas_pred[ego_pas <= 0.4] = 0.0
        ego_pas_pred[np.logical_and(ego_pas < 0.6, ego_pas > 0.4)] = 0.5

        # Pick out only the actual predictions (not unknown).
        mask_not_unknown = ego_pas_pred != 0.5
        mask = np.logical_and(mask_not_unknown, mask_occluded)

        remove_inds = np.where(np.sum(mask, axis=(1,2)) == 0)
        ego_pas_pred = np.delete(ego_pas_pred, remove_inds, axis=0)
        ego_pas = np.delete(ego_pas, remove_inds, axis=0)
        ego_label = np.delete(ego_label, remove_inds, axis=0)
        ego_vanilla = np.delete(ego_vanilla, remove_inds, axis=0)
        mask = np.delete(mask, remove_inds, axis=0)

        ego_pas_pred[np.logical_not(mask)] = np.inf

        ego_pas[np.logical_not(mask)] = np.inf
        ego_label[np.logical_not(mask)] = np.inf
        ego_vanilla[np.logical_not(mask)] = np.inf
        
        acc_vanilla = np.mean(0.5 == ego_label[mask])
        mse_vanilla = np.mean((0.5 - ego_label[mask])**2)

        acc_pas = np.mean(ego_pas_pred[mask] == ego_label[mask])
        mse_pas = np.mean((ego_pas[mask] - ego_label[mask])**2)

        acc_occ_vanilla = np.mean(0.5 == 1.0)
        acc_free_vanilla = np.mean(0.5 == 0.0)

        mse_occ_vanilla = np.mean((0.5 - 1.0)**2)
        mse_free_vanilla = np.mean((0.5 - 0.0)**2)

        acc_occ_pas = np.mean(ego_pas_pred[mask][ego_label[mask] == 1] == 1.0)
        acc_free_pas = np.mean(ego_pas_pred[mask][ego_label[mask] == 0] == 0.0)

        mse_occ_pas = np.mean((ego_pas[mask][ego_label[mask] == 1] - 1.0)**2)
        mse_free_pas = np.mean((ego_pas[mask][ego_label[mask] == 0] - 0.0)**2)

        im_grids_vanilla, im_occ_grids_vanilla, im_free_grids_vanilla, im_ocl_grids_vanilla = MapSimilarityMetric(ego_vanilla, ego_label)

        im_vanilla = np.mean(im_grids_vanilla)
        im_occ_vanilla = np.mean(im_occ_grids_vanilla)
        im_free_vanilla = np.mean(im_free_grids_vanilla)
        im_occ_free_vanilla = np.mean(im_occ_grids_vanilla + im_free_grids_vanilla)

        im_grids_pas, im_occ_grids_pas, im_free_grids_pas, im_ocl_grids_pas = MapSimilarityMetric(ego_pas_pred, ego_label)
        im_pas = np.mean(im_grids_pas)
        im_occ_pas = np.mean(im_occ_grids_pas)
        im_free_pas = np.mean(im_free_grids_pas)
        im_occ_free_pas = np.mean(im_occ_grids_pas + im_free_grids_pas)

        acc_vanilla_std_error = np.std(0.5 == ego_label[mask])/np.sqrt(ego_label[mask].size)
        mse_vanilla_std_error = np.std((0.5 - ego_label[mask])**2)/np.sqrt(ego_label[mask].size)

        acc_pas_std_error = np.std(ego_pas_pred[mask] == ego_label[mask])/np.sqrt(ego_pas[mask].size)
        mse_pas_std_error = np.std((ego_pas[mask] - ego_label[mask])**2)/np.sqrt(ego_pas[mask].size)

        acc_occ_vanilla_std_error = np.std(0.5 == 1.0)
        acc_free_vanilla_std_error = np.std(0.5 == 0.0)

        mse_occ_vanilla_std_error = np.std((0.5 - 1.0)**2)
        mse_free_vanilla_std_error = np.std((0.5 - 0.0)**2)

        acc_occ_pas_std_error = np.std(ego_pas_pred[mask][ego_label[mask] == 1] == 1.0)/np.sqrt(ego_pas[mask][ego_label[mask] == 1].size)
        acc_free_pas_std_error = np.std(ego_pas_pred[mask][ego_label[mask] == 0] == 0.0)/np.sqrt(ego_pas[mask][ego_label[mask] == 0].size)

        mse_occ_pas_std_error = np.std((ego_pas[mask][ego_label[mask] == 1] - 1.0)**2)/np.sqrt(ego_pas[mask][ego_label[mask] == 1].size)
        mse_free_pas_std_error = np.std((ego_pas[mask][ego_label[mask] == 0] - 0.0)**2)/np.sqrt(ego_pas[mask][ego_label[mask] == 0].size)

        im_vanilla_std_error = np.std(im_grids_vanilla)/np.sqrt(len(im_grids_vanilla))
        im_occ_vanilla_std_error = np.std(im_occ_grids_vanilla)/np.sqrt(len(im_occ_grids_vanilla))
        im_free_vanilla_std_error = np.std(im_free_grids_vanilla)/np.sqrt(len(im_free_grids_vanilla))
        im_occ_free_vanilla_std_error = np.std(im_occ_grids_vanilla + im_free_grids_vanilla)/np.sqrt(len(im_occ_grids_vanilla))

        im_pas_std_error = np.std(im_grids_pas)/np.sqrt(len(im_grids_pas))
        im_occ_pas_std_error = np.std(im_occ_grids_pas)/np.sqrt(len(im_occ_grids_pas))
        im_free_pas_std_error = np.std(im_free_grids_pas)/np.sqrt(len(im_free_grids_pas))
        im_occ_free_pas_std_error = np.std(im_occ_grids_pas + im_free_grids_pas)/np.sqrt(len(im_occ_grids_pas))

        print("Metrics Vanilla OGM: ")
        print("Accuracy: ", acc_vanilla, "MSE: ", mse_vanilla, "IS full: ", im_vanilla, "IS occ + free: ", im_occ_free_vanilla)

        print("Metrics PaS: ")
        print("Accuracy: ", acc_pas, "MSE: ", mse_pas, "IS full: ", im_pas, "IS occ + free: ", im_occ_free_pas)

        print("Occupancy Metrics vanilla: ")
        print("Accuracy: ", acc_occ_vanilla, "MSE: ", mse_occ_vanilla, "IS: ", im_occ_vanilla)

        print("Free Metrics vanilla: ")
        print("Accuracy: ", acc_free_vanilla, "MSE: ", mse_free_vanilla, "IS: ", im_free_vanilla)

        print("Occupancy Metrics PaS: ")
        print("Accuracy: ", acc_occ_pas, "MSE: ", mse_occ_pas, "IS: ", im_occ_pas)

        print("Free Metrics PaS: ")
        print("Accuracy: ", acc_free_pas, "MSE: ", mse_free_pas, "IS: ", im_free_pas)

        print("Standard Error vanilla: ")
        print("Accuracy: ", acc_vanilla_std_error, "MSE: ", mse_vanilla_std_error, "IS full: ", im_vanilla_std_error, "IS occ + free: ", im_occ_free_vanilla_std_error)

        print("Standard Error PaS: ")
        print("Accuracy: ", acc_pas_std_error, "MSE: ", mse_pas_std_error, "IS full: ", im_pas_std_error, "IS occ + free: ", im_occ_free_pas_std_error)

        print("Occupancy Metrics vanilla: ")
        print("Accuracy: ", acc_occ_vanilla_std_error, "MSE: ", mse_occ_vanilla_std_error, "IS: ", im_occ_vanilla_std_error)

        print("Free Metrics vanilla: ")
        print("Accuracy: ", acc_free_vanilla_std_error, "MSE: ", mse_free_vanilla_std_error, "IS: ", im_free_vanilla_std_error)

        print("Occupancy Metrics PaS: ")
        print("Accuracy: ", acc_occ_pas_std_error, "MSE: ", mse_occ_pas_std_error, "IS: ", im_occ_pas_std_error)

        print("Free Metrics PaS: ")
        print("Accuracy: ", acc_free_pas_std_error, "MSE: ", mse_free_pas_std_error, "IS: ", im_free_pas_std_error)

    elif model != 'kmeans':
        ego_pas_multimodal = np.array(ego_pas_multimodal, dtype='float32')

        # Compute both the most likely metrics and the top 3 metrics.
        for num_modes in [1, 3]:

            acc_pas_multimodal = []
            acc_occ_pas_multimodal = []
            acc_free_pas_multimodal = []

            mse_pas_multimodal = []
            mse_occ_pas_multimodal = []
            mse_free_pas_multimodal = []

            acc_pas_multimodal_std = []
            acc_occ_pas_multimodal_std = []
            acc_free_pas_multimodal_std = []

            mse_pas_multimodal_std = []
            mse_occ_pas_multimodal_std = []
            mse_free_pas_multimodal_std = []

            im_pas_multimodal = []
            im_occ_free_pas_multimodal = []
            im_occ_pas_multimodal = []
            im_free_pas_multimodal = []

            divisor = np.zeros((ego_pas_multimodal.shape[0], num_modes))
            divisor_occ = np.zeros((ego_pas_multimodal.shape[0], num_modes))
            divisor_free = np.zeros((ego_pas_multimodal.shape[0], num_modes))

            divisor_multimodal_acc = np.zeros((ego_pas_multimodal.shape[0],))
            divisor_multimodal_occ_acc = np.zeros((ego_pas_multimodal.shape[0],))
            divisor_multimodal_free_acc = np.zeros((ego_pas_multimodal.shape[0],))

            divisor_multimodal_mse = np.zeros((ego_pas_multimodal.shape[0],))
            divisor_multimodal_occ_mse = np.zeros((ego_pas_multimodal.shape[0],))
            divisor_multimodal_free_mse = np.zeros((ego_pas_multimodal.shape[0],))

            for i in tqdm(range(ego_pas_multimodal.shape[0])):
                
                acc_pas = []
                acc_occ_pas = []
                acc_free_pas = []

                mse_pas = []
                mse_occ_pas = []
                mse_free_pas = []

                im_pas = []
                im_occ_free_pas = []
                im_occ_pas = []
                im_free_pas = []

                acc_pas_list = []
                acc_occ_pas_list = []
                acc_free_pas_list = []

                mse_pas_list = []
                mse_occ_pas_list = []
                mse_free_pas_list = []

                for k in range(num_modes):

                    # Pick out only the occluded areas.
                    mask_occluded = ego_vanilla[i] == 0.5

                    ego_pas = ego_pas_multimodal[i,k]

                    ego_pas_pred = (np.logical_and(ego_pas >= 0.6, mask_occluded)).astype(float)
                    ego_pas_pred[ego_pas <= 0.4] = 0.0
                    ego_pas_pred[np.logical_and(ego_pas < 0.6, ego_pas > 0.4)] = 0.5

                    # Pick out only the actual predictions (not unknown).
                    mask_not_unknown = ego_pas_pred != 0.5
                    mask = np.logical_and(mask_not_unknown, mask_occluded)

                    ego_pas_pred[np.logical_not(mask)] = np.inf

                    ego_pas[np.logical_not(mask)] = np.inf
                    ego_label[i][np.logical_not(mask)] = np.inf
                    ego_vanilla[i][np.logical_not(mask)] = np.inf

                    if ego_label[i][mask].size != 0:
                        divisor[i, k] += np.sum(mask)
                        acc_pas.append(np.sum(ego_pas_pred[mask] == ego_label[i][mask]))
                        mse_pas.append(np.sum((ego_pas[mask] - ego_label[i][mask])**2))
                        acc_pas_list.append(ego_pas_pred[mask] == ego_label[i][mask])
                        mse_pas_list.append((ego_pas[mask] - ego_label[i][mask])**2)

                    if np.sum(ego_label[i][mask] == 1) != 0:
                        divisor_occ[i, k] += np.sum(ego_label[i][mask] == 1)
                        acc_occ_pas.append(np.sum(ego_pas_pred[mask][ego_label[i][mask] == 1] == 1.0))
                        mse_occ_pas.append(np.sum((ego_pas[mask][ego_label[i][mask] == 1] - 1.0)**2))
                        acc_occ_pas_list.append(ego_pas_pred[mask][ego_label[i][mask] == 1] == 1.0)
                        mse_occ_pas_list.append((ego_pas[mask][ego_label[i][mask] == 1] - 1.0)**2)
            
                    if np.sum(ego_label[i][mask] == 0) != 0:
                        divisor_free[i, k] += np.sum(ego_label[i][mask] == 0)
                        acc_free_pas.append(np.sum(ego_pas_pred[mask][ego_label[i][mask] == 0] == 0.0))
                        mse_free_pas.append(np.sum((ego_pas[mask][ego_label[i][mask] == 0] - 0.0)**2))
                        acc_free_pas_list.append(ego_pas_pred[mask][ego_label[i][mask] == 0] == 0.0)
                        mse_free_pas_list.append((ego_pas[mask][ego_label[i][mask] == 0] - 0.0)**2)

                if len(acc_pas) != 0:
                    acc_pas_multimodal.append(max(acc_pas))
                    divisor_multimodal_acc[i] = divisor[i,np.argmax(acc_pas)]
                    acc_pas_multimodal_std += acc_pas_list[np.argmax(acc_pas)].tolist()
                    mse_pas_multimodal.append(min(mse_pas))
                    divisor_multimodal_mse[i] = divisor[i,np.argmin(mse_pas)]
                    mse_pas_multimodal_std += mse_pas_list[np.argmin(mse_pas)].tolist()

                if len(acc_occ_pas) != 0:
                    acc_occ_pas_multimodal.append(max(acc_occ_pas))
                    divisor_multimodal_occ_acc[i] = divisor_occ[i,np.argmax(acc_occ_pas)]
                    acc_occ_pas_multimodal_std += acc_occ_pas_list[np.argmax(acc_occ_pas)].tolist()
                    mse_occ_pas_multimodal.append(min(mse_occ_pas))
                    divisor_multimodal_occ_mse[i] = divisor_occ[i,np.argmin(mse_occ_pas)]
                    mse_occ_pas_multimodal_std += mse_occ_pas_list[np.argmin(mse_occ_pas)].tolist()

                if len(acc_free_pas) != 0:
                    acc_free_pas_multimodal.append(max(acc_free_pas))
                    divisor_multimodal_free_acc[i] = divisor_free[i,np.argmax(acc_free_pas)]
                    acc_free_pas_multimodal_std += acc_free_pas_list[np.argmax(acc_free_pas)].tolist()
                    mse_free_pas_multimodal.append(min(mse_free_pas))
                    divisor_multimodal_free_mse[i] = divisor_free[i,np.argmin(mse_free_pas)]
                    mse_free_pas_multimodal_std += mse_free_pas_list[np.argmin(mse_free_pas)].tolist()

            acc_pas_multimodal_mean = np.sum(np.array(acc_pas_multimodal))/np.sum(divisor_multimodal_acc)
            acc_occ_pas_multimodal_mean = np.sum(np.array(acc_occ_pas_multimodal))/np.sum(divisor_multimodal_occ_acc)
            acc_free_pas_multimodal_mean = np.sum(np.array(acc_free_pas_multimodal))/np.sum(divisor_multimodal_free_acc)

            acc_pas_multimodal_std_error = np.std(acc_pas_multimodal_std)/np.sqrt(len(acc_pas_multimodal_std))
            acc_occ_pas_multimodal_std_error = np.std(acc_occ_pas_multimodal_std)/np.sqrt(len(acc_occ_pas_multimodal_std))
            acc_free_pas_multimodal_std_error = np.std(acc_free_pas_multimodal_std)/np.sqrt(len(acc_free_pas_multimodal_std))

            mse_pas_multimodal_mean =  np.sum(np.array(mse_pas_multimodal))/np.sum(divisor_multimodal_mse)
            mse_occ_pas_multimodal_mean =  np.sum(np.array(mse_occ_pas_multimodal))/np.sum(divisor_multimodal_occ_mse)
            mse_free_pas_multimodal_mean = np.sum(np.array(mse_free_pas_multimodal))/np.sum(divisor_multimodal_free_mse)

            mse_pas_multimodal_std_error = np.std(mse_pas_multimodal_std)/np.sqrt(len(mse_pas_multimodal_std))
            mse_occ_pas_multimodal_std_error = np.std(mse_occ_pas_multimodal_std)/np.sqrt(len(mse_occ_pas_multimodal_std))
            mse_free_pas_multimodal_std_error = np.std(mse_free_pas_multimodal_std)/np.sqrt(len(mse_free_pas_multimodal_std))

            im_pas_multimodal_mean = np.mean(np.array(im_pas_multimodal))
            im_occ_free_pas_multimodal_mean = np.mean(np.array(im_occ_free_pas_multimodal))
            im_occ_pas_multimodal_mean = np.mean(np.array(im_occ_pas_multimodal))
            im_free_pas_multimodal_mean = np.mean(np.array(im_free_pas_multimodal))

            im_pas_multimodal_std_error = np.std(np.array(im_pas_multimodal))/np.sqrt(len(im_pas_multimodal))
            im_occ_free_pas_multimodal_std_error = np.std(np.array(im_occ_free_pas_multimodal))/np.sqrt(len(im_occ_free_pas_multimodal))
            im_occ_pas_multimodal_std_error = np.std(np.array(im_occ_pas_multimodal))/np.sqrt(len(im_occ_pas_multimodal))
            im_free_pas_multimodal_std_error = np.std(np.array(im_free_pas_multimodal))/np.sqrt(len(im_free_pas_multimodal))

            print("Top ", str(num_modes), " Metrics PaS: ")
            print("Accuracy: ", acc_pas_multimodal_mean, "MSE: ", mse_pas_multimodal_mean, "IS full: ", im_pas_multimodal_mean, "IS occ + free: ", im_occ_free_pas_multimodal_mean)

            print("Occupancy Metrics PaS: ")
            print("Accuracy: ", acc_occ_pas_multimodal_mean, "MSE: ", mse_occ_pas_multimodal_mean, "IS: ", im_occ_pas_multimodal_mean)

            print("Free Metrics PaS: ")
            print("Accuracy: ", acc_free_pas_multimodal_mean, "MSE: ", mse_free_pas_multimodal_mean, "IS: ", im_free_pas_multimodal_mean)

            print("Standard Error")
            print("Accuracy: ", acc_pas_multimodal_std_error, "MSE: ", mse_pas_multimodal_std_error, "IS full: ", im_pas_multimodal_std_error, "IS occ + free: ", im_occ_free_pas_multimodal_std_error)

            print("Occupancy Metrics PaS: ")
            print("Accuracy: ", acc_occ_pas_multimodal_std_error, "MSE: ", mse_occ_pas_multimodal_std_error, "IS: ", im_occ_pas_multimodal_std_error)

            print("Free Metrics PaS: ")
            print("Accuracy: ", acc_free_pas_multimodal_std_error, "MSE: ", mse_free_pas_multimodal_std_error, "IS: ", im_free_pas_multimodal_std_error)