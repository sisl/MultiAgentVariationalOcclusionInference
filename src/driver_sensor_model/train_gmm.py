import torch
random = 123
torch.manual_seed(random)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import datasets, transforms

import numpy as np
np.random.seed(random)
from matplotlib import pyplot as plt
import hickle as hkl
import pickle as pkl
import pdb
import os
import multiprocessing as mp

os.chdir("../..")

from src.utils.data_generator import *
from sklearn.mixture import GaussianMixture as GMM
from tqdm import tqdm
import time

# Load data.
nt = 10

dir = '/data/INTERACTION-Dataset-DR-v1_1/processed_data/driver_sensor_dataset/'
models = 'gmm'
dir_models = os.path.join(dir, models)
if not os.path.isdir(dir_models):
	os.mkdir(dir_models)
data_file_states = os.path.join(dir, 'states_shuffled_train.hkl')
data_file_grids = os.path.join(dir, 'label_grids_shuffled_train.hkl')
data_file_sources = os.path.join(dir, 'sources_shuffled_train.hkl')

data_train = SequenceGenerator(data_file_state=data_file_states, data_file_grid=data_file_grids, source_file=data_file_sources, nt=nt,
                 batch_size=None, shuffle=True, sequence_start_mode='all', norm=False)
print("number of unique sources: ", len(np.unique(data_train.sources)))

train_loader = torch.utils.data.DataLoader(data_train,
        batch_size=len(data_train), shuffle=True,
        num_workers=mp.cpu_count()-1, pin_memory=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

for batch_x, batch_y, sources in train_loader:
	batch_x, batch_y = batch_x.to("cpu").data.numpy(), batch_y.to("cpu").data.numpy()
print(batch_x.shape, batch_y.shape)

# Test on validation data.
data_file_states = os.path.join(dir, 'states_val.hkl')
data_file_grids = os.path.join(dir, 'label_grids_val.hkl')
data_file_sources = os.path.join(dir, 'sources_val.hkl')

data_val = SequenceGenerator(data_file_state=data_file_states, data_file_grid=data_file_grids, source_file=data_file_sources, nt=nt,
                 batch_size=None, shuffle=False, sequence_start_mode='unique', norm=False)

val_loader = torch.utils.data.DataLoader(data_val,
        batch_size=len(data_val), shuffle=False,
        num_workers=mp.cpu_count()-1, pin_memory=True)

for batch_x_val, batch_y_val, sources_val in val_loader:
    batch_x_val, batch_y_val = batch_x_val.to("cpu").data.numpy(), batch_y_val.to("cpu").data.numpy()

print('unique values', np.unique(batch_y_val))

Ks = [100] # [4, 10, 25, 50, 100, 150]
acc_nums = []
acc_nums_free_occ = []
acc_nums_occ = []
acc_nums_free = []
acc_nums_ocl = []
plot_train = False
grid_shape = (20,30)

for K in tqdm(Ks):

	model = GMM(n_components=K, covariance_type='diag', reg_covar=1e-3)
	model.fit(batch_x)
	cluster_ids_x = model.predict(batch_x)
	cluster_centers = model.means_

	p_a_m_1 = np.zeros((K, batch_y.shape[1]))
	p_a_m_0 = np.zeros((K, batch_y.shape[1]))
	for i in range(batch_y.shape[1]):
		p_m_1 = np.sum(batch_y[:,i] == 1).astype(float)/batch_y.shape[0]
		p_m_0 = (np.sum(batch_y[:,i] == 0).astype(float)/batch_y.shape[0])
		for k in range(K):
			p_a_m_1[k,i] = np.sum(np.logical_and(cluster_ids_x == k, batch_y[:,i] == 1)).astype(float)/batch_y.shape[0]
			p_a_m_0[k,i] = np.sum(np.logical_and(cluster_ids_x == k, batch_y[:,i] == 0)).astype(float)/batch_y.shape[0]

		if p_m_1 != 0:
			p_a_m_1[:,i] = p_a_m_1[:,i]/p_m_1
		else:
			p_a_m_1[:,i] = p_a_m_1[:,i]/1.0

		if p_m_0 != 0:
			p_a_m_0[:,i] = p_a_m_0[:,i]/p_m_0
		else:
			p_a_m_0[:,i] = p_a_m_0[:,i]/1.0

	p_m_a_np = p_a_m_1/(p_a_m_1+p_a_m_0)

	pkl.dump([model, p_m_a_np], open( os.path.join(dir_models, "GMM_K_" + str(K) + "_reg_covar_0_001.pkl"), "wb" ) )

	cluster_centers_np = np.reshape(cluster_centers, (K,nt,7))
	cluster_centers_np = unnormalize(cluster_centers_np)

	if plot_train:
		fig, (ax1, ax2, ax3) = plt.subplots(3)
		fig.suptitle('State Clusters')

		for k in range(K):
			fig_occ, ax_occ = plt.subplots(1)
			image = np.flip(np.transpose(1.0-np.reshape(p_m_a_np[k], grid_shape), (1,0)), axis=0)
			ax_occ.imshow(image, cmap='gray')
			picture_file = os.path.join(dir_models, 'cluster_' + str(k) + '.png')
			plt.savefig(picture_file)
			fig_occ.clf()

			ax1.scatter(cluster_centers_np[k,:,1], cluster_centers_np[k,:,1], label=str(k))
			ax2.scatter(cluster_centers_np[k,:,1], cluster_centers_np[k,:,2], label=str(k))
			ax3.scatter(cluster_centers_np[k,:,1], cluster_centers_np[k,:,3], label=str(k))

		ax1.set_ylabel("Pos (m)")
		ax1.set_ylim(-5,120)
		ax1.set_xlim(-5,120)
		ax2.set_ylabel("Vel (m/s)")
		ax2.set_ylim(0,8)
		ax2.set_xlim(-5,120)
		ax3.set_xlabel("Position (m)")
		ax3.set_ylabel("Acc (m/s^2)")
		ax3.set_ylim(-3,3)
		ax3.set_xlim(-5,120)

		handles, labels = ax1.get_legend_handles_labels()
		fig.legend(handles, labels, loc='center right')

		picture_file = os.path.join(dir_models, 'state_clusters_gmm.png')
		fig.savefig(picture_file)

		del(p_m_a)
		del(cluster_centers_np)
		del(cluster_centers)
		del(cluster_ids)

	# Test on validation data.
	cluster_ids_y_val = model.predict(batch_x_val)

	grids_pred = p_m_a_np[cluster_ids_y_val]

	grids_pred[grids_pred >= 0.6] = 1.0
	grids_pred[grids_pred <= 0.4] = 0.0
	grids_pred[np.logical_and(grids_pred > 0.4, grids_pred < 0.6)] = 0.5
	grids_gt = batch_y_val

	acc = np.mean(grids_pred == grids_gt)
	acc_nums.append(acc)

	mask_occ = (batch_y_val == 1)
	acc_occ = np.mean(grids_pred[mask_occ] == grids_gt[mask_occ])
	acc_nums_occ.append(acc_occ)

	mask_free = (batch_y_val == 0)
	acc_free = np.mean(grids_pred[mask_free] == grids_gt[mask_free])
	acc_nums_free.append(acc_free)

	print("K: ", K, " Accuracy: ", acc, acc_occ, acc_free)

if plot_train:
	plt.scatter(Ks, acc_nums, label='acc')
	plt.scatter(Ks, acc_nums_occ, label='acc occupied')
	plt.scatter(Ks, acc_nums_free, label='acc free')
	plt.ylim(0,1)
	plt.legend()
	plt.title('Accuracy vs Number of Clusters for Driver Sensor Model')
	plt.xlabel('Number of Clusters')
	plt.ylabel('Accuracy')
	plt.savefig(os.path.join(dir_models, 'acc_clusters_gmm.png'))