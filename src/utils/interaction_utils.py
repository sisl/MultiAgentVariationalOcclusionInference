# Code for computing the IS metric and visualizing metrics. IS metric code is modified from: https://github.com/BenQLange/Occupancy-Grid-Metrics.

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import os
import imageio
from copy import deepcopy
import pdb
from tqdm import tqdm

# Metric visualization.
def plot_metric_clusters(K, states, cluster_ids, metric_grids, metric, dir, sources_val):
    patches = []
    color = plt.cm.jet(np.linspace(0,1,K))

    for c in range(K):
        patches.append(mpatches.Patch(color=color[c], label=c))

    unique_sources = np.unique(np.array(sources_val))

    for source in unique_sources:
        if source[-1] != scenario:
            continue

        mask = (np.array(sources_val) == source)

        batch_x_np_val_scenario = states[mask]
        cluster_ids_y_val_scenario = cluster_ids[mask]
        metric_grids_scenario = metric_grids[mask]

        if dict_scenario[scenario] == 0:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
            symbol = '.'
        else:
            symbol = 'x'

        for k in range(batch_x_np_val_scenario.shape[0]):
            ax1.scatter(batch_x_np_val_scenario[k,:,1], batch_x_np_val_scenario[k,:,1], color=color[cluster_ids_y_val_scenario[k]], label=str(cluster_ids_y_val_scenario[k]), linewidths=1.0, marker=symbol)
            ax2.scatter(batch_x_np_val_scenario[k,:,1], batch_x_np_val_scenario[k,:,2], color=color[cluster_ids_y_val_scenario[k]], label=str(cluster_ids_y_val_scenario[k]), linewidths=1.0, marker=symbol)
            ax3.scatter(batch_x_np_val_scenario[k,:,1], batch_x_np_val_scenario[k,:,3], color=color[cluster_ids_y_val_scenario[k]], label=str(cluster_ids_y_val_scenario[k]), linewidths=1.0, marker=symbol)
            ax4.scatter(batch_x_np_val_scenario[k,0,1], metric_grids_scenario[k], color=color[cluster_ids_y_val_scenario[k]], label=str(cluster_ids_y_val_scenario[k]), linewidths=1.0, marker=symbol)          

        if dict_scenario[scenario] == 1:
            ax1.set_ylabel("Pos (m)")
            ax1.set_ylim(-5,120)
            ax1.set_xlim(-5,120)
            ax2.set_ylabel("Vel (m/s)")
            ax2.set_ylim(0,8)
            ax2.set_xlim(-5,120)
            ax3.set_ylabel("Acc (m/s^2)")
            ax3.set_ylim(-3,3)
            ax3.set_xlim(-5,120)
            ax4.set_ylim(0,1)
            if metric == 'IM':
                ax4.set_ylim(0,50)
            if metric == 'IM2':
                ax4.set_ylim(0, 50)
            ax4.set_xlim(-5,120)
            ax4.set_xlabel("Position (m)")
            ax4.set_ylabel(metric)
            fig.tight_layout(pad=.05)

            picture_file = os.path.join(dir, 'val_examples_K_' + str(K) + '_scenario_' + scenario + '_' + metric + '.png')
            fig.savefig(picture_file)
            fig.clf()

        dict_scenario[scenario] += 1

def plot_train(K, p_m_a_np, cluster_centers_np, dir, grid_shape):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('State Clusters')

    for k in range(K):
        fig_occ, ax_occ = plt.subplots(1)
        image = np.flip(np.transpose(1.0-np.reshape(p_m_a_np[k], grid_shape), (1,0)), axis=0)
        ax_occ.imshow(image, cmap='gray')
        picture_file = os.path.join(dir, str(K) + '_cluster_' + str(k) + '.png')
        plt.savefig(picture_file)
        fig_occ.clf()

def plot_scatter(K, states, metric_vals, metric, dir):
    x = states[:,0,0]
    y = states[:,0,1]
    plt.figure()
    plt.scatter(x, y, marker='.', s=150, linewidths=1, c=metric_vals, cmap=plt.cm.coolwarm)
    cb = plt.colorbar()
    plt.clim(0,80)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('IM over First Driver State')
    picture_file = os.path.join(dir, str(K) + '_' + metric + '_first_driver_state.png')
    plt.savefig(picture_file)
    plt.clf()
    del(cb)

def plot_scatter_clusters(K, states, labels, dir):
    x = states[:,0,0]
    y = states[:,0,1]
    plt.figure()
    plt.scatter(x, y, marker='.', s=150, linewidths=1, c=labels, cmap=plt.cm.coolwarm)
    cb = plt.colorbar()
    plt.clim(0,K)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Clusters over First Driver State')
    picture_file = os.path.join(dir, str(K) + '_clusters_first_driver_state.png')
    plt.savefig(picture_file)
    plt.clf()
    del(cb)

def plot_multimodal_metrics(Ks, acc_nums, mse_nums, im_nums, acc_nums_best, mse_nums_best, im_nums_best, dir):
    plt.figure()
    plt.plot(Ks, acc_nums, label='Acc')
    plt.plot(Ks, acc_nums_best, label='Best 3 Acc')
    plt.title('Multimodality Performance Accuracy')
    plt.xlabel('K')
    plt.ylabel('Acc')
    plt.legend()
    picture_file = os.path.join(dir, 'multimodality_acc_best_3.png')
    plt.savefig(picture_file)

    plt.figure()
    plt.plot(Ks, mse_nums, label='MSE')
    plt.plot(Ks, mse_nums_best, label='Best 3 MSE')
    plt.title('Multimodality Performance MSE')
    plt.xlabel('K')
    plt.ylabel('MSE')
    plt.legend()
    picture_file = os.path.join(dir, 'multimodality_mse_best_3.png')
    plt.savefig(picture_file)

    plt.figure()
    plt.plot(Ks, im_nums, label='IM')
    plt.plot(Ks, im_nums_best, label='Best 3 IM')
    plt.title('Multimodality Performance IM')
    plt.xlabel('K')
    plt.ylabel('IM')
    plt.legend()
    picture_file = os.path.join(dir, 'multimodality_im_best_3.png')
    plt.savefig(picture_file)

# IS metric.
def MapSimilarityMetric(grids_pred, grids_actual):

    num_samples,_,_ = grids_pred.shape
    score, score_occupied, score_free, score_occluded = np.zeros((num_samples,)), np.zeros((num_samples,)), np.zeros((num_samples,)), np.zeros((num_samples,)) # score_occluded, np.zeros((num_samples,))
    for sample in range(num_samples): # tqdm
        occupied, free, occluded = computeSimilarityMetric(grids_actual[sample,:,:], grids_pred[sample,:,:])
        score[sample] += occupied
        score[sample] += occluded
        score[sample] += free

        score_occupied[sample] += occupied
        score_occluded[sample] += occluded
        score_free[sample] += free

    return score, score_occupied, score_free, score_occluded

def toDiscrete(m):
    """
    Args:
        - m (m,n) : np.array with the occupancy grid
    Returns:
        - discrete_m : thresholded m
    """

    y_size, x_size = m.shape
    m_occupied = np.zeros(m.shape)
    m_free = np.zeros(m.shape)
    m_occluded = np.zeros(m.shape)

    m_occupied[m == 1.0] = 1.0
    m_occluded[m == 0.5] = 1.0
    m_free[m == 0.0] = 1.0

    return m_occupied, m_free, m_occluded

def todMap(m):

    """
    Extra if statements are for edge cases.
    """

    y_size, x_size = m.shape
    dMap = np.ones(m.shape) * np.Inf
    dMap[m == 1] = 0.0

    for y in range(0,y_size):
        if y == 0:
            for x in range(1,x_size):
                h = dMap[y,x-1]+1
                dMap[y,x] = min(dMap[y,x], h)

        else:
            for x in range(0,x_size):
                if x == 0:
                    h = dMap[y-1,x]+1
                    dMap[y,x] = min(dMap[y,x], h)
                else:
                    h = min(dMap[y,x-1]+1, dMap[y-1,x]+1)
                    dMap[y,x] = min(dMap[y,x], h)

    for y in range(y_size-1,-1,-1):

        if y == y_size-1:
            for x in range(x_size-2,-1,-1):
                h = dMap[y,x+1]+1
                dMap[y,x] = min(dMap[y,x], h)

        else:
            for x in range(x_size-1,-1,-1):
                if x == x_size-1:
                    h = dMap[y+1,x]+1
                    dMap[y,x] = min(dMap[y,x], h)
                else:
                    h = min(dMap[y+1,x]+1, dMap[y,x+1]+1)
                    dMap[y,x] = min(dMap[y,x], h)

    return dMap

def computeDistance(m1,m2):

    y_size, x_size = m1.shape
    dMap = todMap(m2)

    d = np.sum(dMap[m1 == 1])
    num_cells = np.sum(m1 == 1)

    # If either of the grids does not have a particular class,
    # set to x_size + y_size (proxy for infinity - worst case Manhattan distance).
    # If both of the grids do not have a class, set to zero.
    if ((num_cells != 0) and (np.sum(dMap == np.Inf) == 0)):
        output = d/num_cells
    elif ((num_cells == 0) and (np.sum(dMap == np.Inf) != 0)):
        output = 0.0
    elif ((num_cells == 0) or (np.sum(dMap == np.Inf) != 0)):
        output = x_size + y_size

    if output == np.Inf:
        pdb.set_trace()

    return output

def computeSimilarityMetric(m1, m2):

    m1_occupied, m1_free, m1_occluded = toDiscrete(m1)
    m2_occupied, m2_free, m2_occluded = toDiscrete(m2)

    occupied = computeDistance(m1_occupied,m2_occupied) + computeDistance(m2_occupied,m1_occupied)
    occluded = computeDistance(m2_occluded, m1_occluded) + computeDistance(m1_occluded, m2_occluded)
    free = computeDistance(m1_free,m2_free) + computeDistance(m2_free,m1_free)

    return occupied, free, occluded