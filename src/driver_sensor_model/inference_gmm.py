import torch
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import datasets, transforms

import numpy as np
np.random.seed(0)
from matplotlib import pyplot as plt
import hickle as hkl
import pickle as pkl
import pdb
import os
import multiprocessing as mp
from mpl_toolkits.axes_grid1 import ImageGrid

os.chdir("../..")

from src.utils.data_generator import *
from sklearn.mixture import GaussianMixture as GMM
from src.utils.interaction_utils import *
import time

from tqdm import tqdm

# Load data.
nt = 10
num_states = 7
grid_shape = (20, 30)
  
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

dir = '/data/INTERACTION-Dataset-DR-v1_1/processed_data/driver_sensor_dataset'

# Test on test data.
data_file_states = os.path.join(dir, 'states_test.hkl')
data_file_grids = os.path.join(dir, 'label_grids_test.hkl')
data_file_sources = os.path.join(dir, 'sources_test.hkl')

data_test = SequenceGenerator(data_file_state=data_file_states, data_file_grid=data_file_grids, source_file=data_file_sources, nt=nt,
                 batch_size=None, shuffle=False, sequence_start_mode='unique', norm=False)

test_loader = torch.utils.data.DataLoader(data_test,
        batch_size=len(data_test), shuffle=False,
        num_workers=mp.cpu_count()-1, pin_memory=True)

for batch_x_test, batch_y_test, sources_test in test_loader:
    batch_x_test, batch_y_test_orig = batch_x_test.to(device), batch_y_test.to(device)

batch_x_test = batch_x_test.cpu().data.numpy()
batch_y_test_orig = batch_y_test_orig.cpu().data.numpy()

# GMM
plt.ioff()

models = 'gmm'
dir_models = os.path.join('/models/', models)
if not os.path.isdir(dir_models):
    os.mkdir(dir_models)

Ks = [100] # [4, 10, 25, 50, 100, 150]
N = 3
ncols = {4:2, 10:2, 25:5, 50:5, 100:10, 150:10}
nrows = {4:2, 10:5, 25:5, 50:10, 100:10, 150:15}
acc_nums = []
mse_nums = []
im_nums = []
plot_train_flag = True
plot_metric_clusters_flag = False
plot_scatter_flag = False
plot_scatter_clusters_flag = False

for K in Ks:

    model, p_m_a_np = pkl.load(open(os.path.join(dir_models, "GMM_K_" + str(K) + "_reg_covar_0_001.pkl"), "rb" ) )
    
    cluster_centers_np = model.means_
    cluster_centers_np = np.reshape(cluster_centers_np, (K,nt,num_states))
    cluster_centers_np = unnormalize(cluster_centers_np)
    
    if plot_train_flag:
        plot_train(K, p_m_a_np, cluster_centers_np, dir_models, grid_shape)

    cluster_ids_y_test = model.predict(batch_x_test)
    cluster_ids_y_test_prob = model.predict_proba(batch_x_test)
    
    # Most likely class.
    grids_pred_orig = p_m_a_np[cluster_ids_y_test]

    grids_plot = np.reshape(p_m_a_np, (-1,grid_shape[0],grid_shape[1]))
    fig, axeslist = plt.subplots(ncols=ncols[K], nrows=nrows[K])
    for i in range(grids_plot.shape[0]):
        axeslist.ravel()[i].imshow(grids_plot[i], cmap=plt.gray())
        axeslist.ravel()[i].set_xticks([])
        axeslist.ravel()[i].set_xticklabels([])
        axeslist.ravel()[i].set_yticks([])
        axeslist.ravel()[i].set_yticklabels([])
        axeslist.ravel()[i].set_aspect('equal')
    plt.subplots_adjust(left = 0.5, hspace = 0, wspace = 0)
    plt.margins(0,0)
    fig.savefig(os.path.join(dir_models, 'test_' + str(K) + '.png'), pad_inches=0)
    plt.close(fig)

    grids_pred = deepcopy(grids_pred_orig)
    grids_pred[grids_pred_orig >= 0.6] = 1.0
    grids_pred[np.logical_and(grids_pred_orig < 0.6, grids_pred_orig > 0.4)] = 0.5
    grids_pred[grids_pred_orig <= 0.4] = 0.0

    acc_occ_free = np.mean(grids_pred == batch_y_test_orig)
    mse_occ_free = np.mean((grids_pred_orig - batch_y_test_orig)**2)
    
    im_grids, im_occ_grids, im_free_grids, im_ocl_grids = \
    MapSimilarityMetric(np.reshape(grids_pred, (-1,grid_shape[0],grid_shape[1])), np.reshape(batch_y_test_orig, (-1,grid_shape[0],grid_shape[1]))) # im_ocl_grids, 
    im = np.mean(im_grids)
    im_occ = np.mean(im_occ_grids)
    im_free = np.mean(im_free_grids)
    im_occ_free_grids = im_occ_grids + im_free_grids
    im_occ_free = np.mean(im_occ_grids + im_free_grids)

    acc_nums.append(acc)
    mse_nums.append(mse)
    im_nums.append(im)

    acc_occ = np.mean(grids_pred[batch_y_test_orig == 1] == batch_y_test_orig[batch_y_test_orig == 1])
    acc_free = np.mean(grids_pred[batch_y_test_orig == 0] == batch_y_test_orig[batch_y_test_orig == 0])
    
    mse_occ = np.mean((grids_pred_orig[batch_y_test_orig == 1] - batch_y_test_orig[batch_y_test_orig == 1])**2)
    mse_free = np.mean((grids_pred_orig[batch_y_test_orig == 0] - batch_y_test_orig[batch_y_test_orig == 0])**2)

    print("Metrics: ")
    
    print("Occupancy and Free Metrics: ")
    print("K: ", K, "Accuracy: ", acc_occ_free, "MSE: ", mse_occ_free, "IM: ", im_occ_free, "IM max: ", np.amax(im_occ_free_grids), "IM min: ", np.amin(im_occ_free_grids))

    print("Occupancy Metrics: ")
    print("Accuracy: ", acc_occ, "MSE: ", mse_occ, "IM: ", im_occ, "IM max: ", np.amax(im_occ_grids), "IM min: ", np.amin(im_occ_grids))

    print("Free Metrics: ")
    print("Accuracy: ", acc_free, "MSE: ", mse_free, "IM: ", im_free, "IM max: ", np.amax(im_free_grids), "IM min: ", np.amin(im_free_grids))

    num_occ = np.sum(batch_y_test_orig == 1, axis=-1)
    num_free = np.sum(batch_y_test_orig == 0, axis=-1)
    num_occ_free = np.sum(np.logical_or(batch_y_test_orig == 1, batch_y_test_orig == 0), axis=-1)

    im_nums_N = np.empty((batch_y_test_orig.shape[0], N))
    acc_occ_free_nums_N = np.empty((batch_y_test_orig.shape[0], N))
    mse_occ_free_nums_N = np.empty((batch_y_test_orig.shape[0], N))
    im_occ_free_nums_N = np.empty((batch_y_test_orig.shape[0], N))
    acc_occ_nums_N = np.empty((np.sum(num_occ > 0), N))
    mse_occ_nums_N = np.empty((np.sum(num_occ > 0), N))
    im_occ_nums_N = np.empty((batch_y_test_orig.shape[0], N))
    acc_free_nums_N = np.empty((np.sum(num_free > 0), N))
    mse_free_nums_N = np.empty((np.sum(num_free > 0), N))
    im_free_nums_N = np.empty((batch_y_test_orig.shape[0], N))

    for n in range(1,N+1):
        grids_pred_orig = p_m_a_np[cluster_ids_y_test_prob.argsort(1)[:,-n]]
        grids_pred = (grids_pred_orig >= 0.6).astype(float)
        grids_pred[grids_pred_orig <= 0.4] = 0.0
        grids_pred[np.logical_and(grids_pred_orig < 0.6, grids_pred_orig > 0.4)] = 0.5

        grids_pred_free = (grids_pred * (batch_y_test_orig == 0))
        grids_pred_free[batch_y_test_orig != 0] = 2.0

        acc_occ_free_grids = np.mean(grids_pred == batch_y_test_orig, axis=-1)
        acc_occ_grids = np.sum((grids_pred * (batch_y_test_orig == 1)) == 1., axis=-1)[num_occ > 0] / num_occ[num_occ > 0] * 1.0
        acc_free_grids = np.sum(grids_pred_free == 0., axis=-1)[num_free > 0] / num_free[num_free > 0] * 1.0
        
        mse_occ_free_grids = np.mean((grids_pred_orig - batch_y_test_orig)**2, axis=-1)
        mse_occ_grids = np.sum(((grids_pred_orig * (batch_y_test_orig == 1)) - batch_y_test_orig * (batch_y_test_orig == 1))**2, axis=-1)[num_occ > 0] / num_occ[num_occ > 0] * 1.0
        mse_free_grids = np.sum(((grids_pred_orig * (batch_y_test_orig == 0)) - batch_y_test_orig * (batch_y_test_orig == 0))**2, axis=-1)[num_free > 0] / num_free[num_free > 0] * 1.0

        im_grids, im_occ_grids, im_free_grids, im_ocl_grids = MapSimilarityMetric(np.reshape(grids_pred, (-1,grid_shape[0],grid_shape[1])), np.reshape(batch_y_test_orig, (-1,grid_shape[0],grid_shape[1])))
            
        acc_occ_free_nums_N[:,n-1] = acc_occ_free_grids
        mse_occ_free_nums_N[:,n-1] = mse_occ_free_grids
        im_occ_free_nums_N[:,n-1] = im_occ_grids + im_free_grids

        acc_occ_nums_N[:,n-1] = acc_occ_grids
        mse_occ_nums_N[:,n-1] = mse_occ_grids
        im_occ_nums_N[:,n-1] = im_occ_grids
    
        acc_free_nums_N[:,n-1] = acc_free_grids
        mse_free_nums_N[:,n-1] = mse_free_grids
        im_free_nums_N[:,n-1] = im_free_grids
    
    acc_occ_free_best = np.mean(np.amax(acc_occ_free_nums_N, axis=1))
    mse_occ_free_best = np.mean(np.amin(mse_occ_free_nums_N, axis=1))
    im_occ_free_best = np.mean(np.amin(im_occ_free_nums_N, axis=1))

    acc_occ_best = np.mean(np.amax(acc_occ_nums_N, axis=1))
    mse_occ_best = np.mean(np.amin(mse_occ_nums_N, axis=1))
    im_occ_best = np.mean(np.amin(im_occ_nums_N, axis=1))

    acc_free_best = np.mean(np.amax(acc_free_nums_N, axis=1))
    mse_free_best = np.mean(np.amin(mse_free_nums_N, axis=1))
    im_free_best = np.mean(np.amin(im_free_nums_N, axis=1)) 

    print("Top 3 Metrics: ")

    print("Occupancy and Free Metrics: ")
    print("Accuracy: ", acc_occ_free_best, "MSE: ", mse_occ_free_best, "IM: ", im_occ_free_best)

    print("Occupancy Metrics: ")
    print("Accuracy: ", acc_occ_best, "MSE: ", mse_occ_best, "IM: ", im_occ_best)

    print("Free Metrics: ")
    print("Accuracy: ", acc_free_best, "MSE: ", mse_free_best, "IM: ", im_free_best)

    hkl.dump(np.array([acc_best, acc_occ_best, acc_free_best, acc_occ_free_best,\
     mse_best, mse_occ_best, mse_free_best, mse_occ_free_best,\
      im_best, im_occ_best, im_free_best, im_occ_free_best]),\
       os.path.join(dir_models, 'top_3_metrics.hkl'), mode='w',)

    batch_x_np_test = batch_x_test
    batch_x_np_test = np.reshape(batch_x_np_test, (-1,nt,num_states))
    batch_x_np_test = unnormalize(batch_x_np_test)

    if plot_metric_clusters_flag:
        plot_metric_clusters(K, batch_x_np_test, cluster_ids_y_test, im_grids2, 'IM2', dir_models, sources_test)

    if plot_scatter_flag:
        plot_scatter(K, batch_x_np_test, im_grids, 'IM', dir_models)

    if plot_scatter_clusters_flag:
        plot_scatter_clusters(K, batch_x_np_test, cluster_ids_y_test, dir_models)
    plt.show()

# Standard error
acc_occ_std_error = np.std(grids_pred[batch_y_test_orig == 1] == 1)/np.sqrt(grids_pred[batch_y_test_orig == 1].size)
acc_free_std_error = np.std(grids_pred[batch_y_test_orig == 0] == 0)/np.sqrt(grids_pred[batch_y_test_orig == 0].size)
acc_occ_free_std_error = np.std(grids_pred[np.logical_or(batch_y_test_orig == 0, batch_y_test_orig == 1)] == batch_y_test_orig[np.logical_or(batch_y_test_orig == 0, batch_y_test_orig == 1)])/np.sqrt(batch_y_test_orig[np.logical_or(batch_y_test_orig == 0, batch_y_test_orig == 1)].size)

mse_occ_std_error = np.std((grids_pred_orig[batch_y_test_orig == 1] - 1)**2)/np.sqrt(grids_pred_orig[batch_y_test_orig == 0].size)
mse_free_std_error = np.std((grids_pred_orig[batch_y_test_orig == 0] - 0)**2)/np.sqrt(grids_pred_orig[batch_y_test_orig == 0].size)
mse_occ_free_std_error = np.std((grids_pred_orig[np.logical_or(batch_y_test_orig == 0, batch_y_test_orig == 1)] - batch_y_test_orig[np.logical_or(batch_y_test_orig == 0, batch_y_test_orig == 1)])**2)/np.sqrt(batch_y_test_orig[np.logical_or(batch_y_test_orig == 0, batch_y_test_orig == 1)].size)

im_occ_std_error = np.std(im_occ_grids)/np.sqrt(im_occ_grids.size)
im_free_std_error = np.std(im_free_grids)/np.sqrt(im_free_grids.size)
im_occ_free_std_error = np.std(im_occ_grids + im_free_grids)/np.sqrt(im_occ_grids.size)

print('Maximum standard error:')
print(np.amax([acc_occ_std_error, acc_free_std_error, acc_occ_free_std_error, mse_occ_std_error, mse_free_std_error, mse_occ_free_std_error]))
print(np.amax([im_occ_std_error, im_free_std_error, im_occ_free_std_error]))

# GMM latent space visualization.
all_latent_classes = np.reshape(p_m_a_np, (100,20,30))

fig = plt.figure(figsize=(10, 10))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(10, 10),
                 axes_pad=0.1,
                 )

for ax, im in zip(grid, all_latent_classes):
    ax.matshow(im, cmap='gray_r', vmin=0, vmax=1)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    
plt.savefig('models/gmm/all_latent_classes_gmm.png')
plt.show()