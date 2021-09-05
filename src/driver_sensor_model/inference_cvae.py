# Inference code for the CVAE driver sensor model. Code is adapted from: https://github.com/sisl/EvidentialSparsification.

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

from copy import deepcopy
import pdb
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

from src.utils.utils_model import to_var
from srs.driver_sensor_model.models_cvae import VAE
from src.utils.data_generator import *
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

def plot_bar(alpha,args):
    figure = plt.figure()
    plt.bar(range(args.latent_size), alpha.cpu().data.numpy())
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    plt.close(figure)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image)
    return image

def plot_scatter(pos_x, pos_y, acc):
    figure = plt.figure()
    plt.scatter(pos_x, pos_y, c=acc, vmin=-3.0, vmax=6.0)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.colorbar()
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    plt.close(figure)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image)
    return image

def inference(data_loader, vae, args, writer, num_val, nt, name, dir_models, tensorboard_flag=False):
    
    vae.eval()

    with torch.no_grad():
        for iteration, (y, x, sources) in enumerate(data_loader):
            # Reshape features.
            x = to_var(x).float().view(x.shape[0],1,20,30)
            y = to_var(y).float()
            y_full = unnormalize(y.cpu().data.numpy(), nt)
            pos_x = y_full[:,:,0]
            pos_y = y_full[:,:,1]
            orientation = y_full[:,:,2]
            cos_theta = np.cos(orientation)
            sin_theta = np.sin(orientation)
            acc_x = y_full[:,:,5]
            acc_y = y_full[:,:,6]

            # Project the acceleration on the orientation vector to get longitudinal acceleration.
            dot_prod = acc_x * cos_theta + acc_y * sin_theta
            sign = np.sign(dot_prod)
            acc_proj_x = dot_prod * cos_theta
            acc_proj_y = dot_prod * sin_theta

            acc_proj_sign = sign * np.sqrt(acc_proj_x**2 + acc_proj_y**2)

            batch_size = x.shape[0]

            # Get reconstruction from inference for the most likely class.
            start = time.time()
            recon_x_inf_most_likely, alpha_p, alpha_p_lin, full_c, z = vae.inference(n=1, c=y, mode='most_likely')

            if torch.min(alpha_p) < 1e-15:
                alpha_p = alpha_p.clone() + 1e-15

            # Form distributions out of alpha_q and alpha_p.
            p_dist = torch.distributions.one_hot_categorical.OneHotCategorical(probs=alpha_p)
            rec = torch.nn.functional.binary_cross_entropy(recon_x_inf_most_likely.view(-1), x.view(-1), size_average=False)
    
    grid_shape = (20,30)
    recon_x_inf_np = np.reshape(recon_x_inf_most_likely.cpu().data.numpy(), (-1, grid_shape[0], grid_shape[1]))
    x_np = np.reshape(x.cpu().data.numpy(), (-1, grid_shape[0], grid_shape[1]))

    recon_x_inf_np_pred = (recon_x_inf_np >= 0.6).astype(float)
    recon_x_inf_np_pred[recon_x_inf_np <= 0.4] = 0.0
    recon_x_inf_np_pred[np.logical_and(recon_x_inf_np < 0.6, recon_x_inf_np > 0.4)] = 0.5

    acc_occ_free = np.mean(recon_x_inf_np_pred == x_np)
    mse_occ_free = np.mean((recon_x_inf_np - x_np)**2)

    acc_occ = np.mean(recon_x_inf_np_pred[x_np == 1] == 1.0)
    acc_free = np.mean(recon_x_inf_np_pred[x_np == 0] == 0.0)

    mse_occ = np.mean((recon_x_inf_np[x_np == 1] - 1.0)**2)
    mse_free = np.mean((recon_x_inf_np[x_np == 0] - 0.0)**2)

    acc_occ_free_std_error = np.std(recon_x_inf_np_pred == x_np)/np.sqrt(x_np.size)
    mse_occ_free_std_error = np.std((recon_x_inf_np - x_np)**2)/np.sqrt(x_np.size)

    acc_occ_std_error = np.std(recon_x_inf_np_pred[x_np == 1] == 1.0)/np.sqrt(recon_x_inf_np_pred[x_np == 1].size)
    acc_free_std_error = np.std(recon_x_inf_np_pred[x_np == 0] == 0.0)/np.sqrt(recon_x_inf_np_pred[x_np == 0].size)

    mse_occ_std_error = np.std((recon_x_inf_np[x_np == 1] - 1.0)**2)/np.sqrt(recon_x_inf_np[x_np == 1].size)
    mse_free_std_error = np.std((recon_x_inf_np[x_np == 0] - 0.0)**2)/np.sqrt(recon_x_inf_np[x_np == 0].size)

    im_grids, im_occ_grids, im_free_grids, im_ocl_grids = MapSimilarityMetric(recon_x_inf_np_pred, x_np)
    
    im = np.mean(im_grids)
    im_occ = np.mean(im_occ_grids)
    im_free = np.mean(im_free_grids)
    im_occ_free = np.mean(im_occ_grids + im_free_grids)

    im_std_error = np.std(im_grids)/np.sqrt(im_grids.size)
    im_occ_std_error = np.std(im_occ_grids)/np.sqrt(im_occ_grids.size)
    im_free_std_error = np.std(im_free_grids)/np.sqrt(im_free_grids.size)
    im_occ_free_std_error = np.std(im_occ_grids + im_free_grids)/np.sqrt(im_occ_grids.size)

    print("Metrics: ")
    
    print("Occupancy and Free Metrics: ")
    print("Accuracy: ", acc_occ_free, "MSE: ", mse_occ_free, "IM: ", im_occ_free, "IM max: ", np.amax(im_occ_grids + im_free_grids), "IM min: ", np.amin(im_occ_grids + im_free_grids))

    print("Occupancy Metrics: ")
    print("Accuracy: ", acc_occ, "MSE: ", mse_occ, "IM: ", im_occ, "IM max: ", np.amax(im_occ_grids), "IM min: ", np.amin(im_occ_grids))

    print("Free Metrics: ")
    print("Accuracy: ", acc_free, "MSE: ", mse_free, "IM: ", im_free, "IM max: ", np.amax(im_free_grids), "IM min: ", np.amin(im_free_grids))

    print("Standard Error: ")

    print("Occupancy and Free Metrics: ")
    print("Accuracy: ", acc_occ_free_std_error, "MSE: ", mse_occ_free_std_error, "IM: ", im_occ_free_std_error)

    print("Occupancy Metrics: ")
    print("Accuracy: ", acc_occ_std_error, "MSE: ", mse_occ_std_error, "IM: ", im_occ_std_error)

    print("Free Metrics: ")
    print("Accuracy: ", acc_free_std_error, "MSE: ", mse_free_std_error, "IM: ", im_free_std_error)

    hkl.dump(np.array([acc, acc_occ, acc_free, acc_occ_free,\
     mse, mse_occ, mse_free, mse_occ_free,\
      im, im_occ, im_free, im_occ_free, np.amax(im_grids), np.amin(im_grids)]),\
       os.path.join(dir_models, name + '_metrics_most_likely_test.hkl'), mode='w',)

    alpha_p_np = alpha_p.cpu().data.numpy()

    # Repeat for N most likely classes.
    N = 3
    
    num_occ = np.sum(x_np == 1, axis=(1,2))
    num_free = np.sum(x_np == 0, axis=(1,2))
    num_occ_free = np.sum(np.logical_or(x_np == 1, x_np == 0), axis=(1,2))

    acc_nums_N = np.empty((x_np.shape[0], N))
    mse_nums_N = np.empty((x_np.shape[0], N))
    im_nums_N = np.empty((x_np.shape[0], N))

    acc_occ_free_nums_N = np.empty((x_np.shape[0], N))
    mse_occ_free_nums_N = np.empty((x_np.shape[0], N))
    im_occ_free_nums_N = np.empty((x_np.shape[0], N))
    
    acc_occ_nums_N = np.empty((np.sum(num_occ > 0), N))
    mse_occ_nums_N = np.empty((np.sum(num_occ > 0), N))
    im_occ_nums_N = np.empty((x_np.shape[0], N))
    
    acc_free_nums_N = np.empty((np.sum(num_free > 0), N))
    mse_free_nums_N = np.empty((np.sum(num_free > 0), N))
    im_free_nums_N = np.empty((x_np.shape[0], N))

    for n in range(1,N+1):

        grids_pred_orig, _, _, _, _ = vae.inference(n=1, c=y, mode='multimodal', k=n)
        grids_pred_orig = np.reshape(grids_pred_orig.cpu().data.numpy(), (-1, grid_shape[0], grid_shape[1]))

        grids_pred = (grids_pred_orig >= 0.6).astype(float)
        grids_pred[grids_pred_orig <= 0.4] = 0.0
        grids_pred[np.logical_and(grids_pred_orig < 0.6, grids_pred_orig > 0.4)] = 0.5

        acc_occ_free_grids = np.mean(grids_pred == x_np, axis=(1,2))
        grids_pred_free = (grids_pred * (x_np == 0))
        grids_pred_free[x_np != 0] = 2.0
        acc_occ_grids = np.sum((grids_pred * (x_np == 1)) == 1., axis=(1,2))[num_occ > 0] / num_occ[num_occ > 0] * 1.0
        
        # Adjustment for free because 0 everywhere.
        acc_free_grids = np.sum(grids_pred_free == 0., axis=(1,2))[num_free > 0] / num_free[num_free > 0] * 1.0
        
        mse_occ_free_grids = np.mean((grids_pred_orig - x_np)**2, axis=(1,2))
        mse_occ_grids = np.sum(((grids_pred_orig * (x_np == 1)) - x_np * (x_np == 1))**2, axis=(1,2))[num_occ > 0] / num_occ[num_occ > 0] * 1.0
        mse_free_grids = np.sum(((grids_pred_orig * (x_np == 0)) - x_np * (x_np == 0))**2, axis=(1,2))[num_free > 0] / num_free[num_free > 0] * 1.0

        im_grids, im_occ_grids, im_free_grids, im_ocl_grids = MapSimilarityMetric(grids_pred, x_np)

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

    acc_occ_free_best_std_error = np.std(np.amax(acc_occ_free_nums_N, axis=1))/np.sqrt(acc_occ_free_nums_N.shape[0]*20*30)
    mse_occ_free_best_std_error = np.std(np.amin(mse_occ_free_nums_N, axis=1))/np.sqrt(mse_occ_free_nums_N.shape[0]*20*30)
    im_occ_free_best_std_error = np.std(np.amin(im_occ_free_nums_N, axis=1))/np.sqrt(im_occ_free_nums_N.shape[0])

    acc_occ_best_std_error = np.std(np.amax(acc_occ_nums_N, axis=1))/np.sqrt(acc_occ_nums_N.shape[0]*20*30)
    mse_occ_best_std_error = np.std(np.amin(mse_occ_nums_N, axis=1))/np.sqrt(mse_occ_nums_N.shape[0]*20*30)
    im_occ_best_std_error = np.std(np.amin(im_occ_nums_N, axis=1))/np.sqrt(im_occ_nums_N.shape[0])

    acc_free_best_std_error = np.std(np.amax(acc_free_nums_N, axis=1))/np.sqrt(acc_free_nums_N.shape[0]*20*30)
    mse_free_best_std_error = np.std(np.amin(mse_free_nums_N, axis=1))/np.sqrt(mse_free_nums_N.shape[0]*20*30)
    im_free_best_std_error = np.std(np.amin(im_free_nums_N, axis=1))/np.sqrt(im_free_nums_N.shape[0])

    print("Top 3 Metrics: ")

    print("Occupancy and Free Metrics: ")
    print("Accuracy: ", acc_occ_free_best, "MSE: ", mse_occ_free_best, "IM: ", im_occ_free_best)

    print("Occupancy Metrics: ")
    print("Accuracy: ", acc_occ_best, "MSE: ", mse_occ_best, "IM: ", im_occ_best)

    print("Free Metrics: ")
    print("Accuracy: ", acc_free_best, "MSE: ", mse_free_best, "IM: ", im_free_best)

    print("Standard Error: ")

    print("Occupancy and Free Metrics: ")
    print("Accuracy: ", acc_occ_free_best_std_error, "MSE: ", mse_occ_free_best_std_error, "IM: ", im_occ_free_best_std_error)

    print("Occupancy Metrics: ")
    print("Accuracy: ", acc_occ_best_std_error, "MSE: ", mse_occ_best_std_error, "IM: ", im_occ_best_std_error)

    print("Free Metrics: ")
    print("Accuracy: ", acc_free_best_std_error, "MSE: ", mse_free_best_std_error, "IM: ", im_free_best_std_error)

    hkl.dump(np.array([acc_best, acc_occ_best, acc_free_best, acc_occ_free_best,\
     mse_best, mse_occ_best, mse_free_best, mse_occ_free_best,\
      im_best, im_occ_best, im_free_best, im_occ_free_best]),\
       os.path.join(dir_models, name + '_top_3_metrics_test.hkl'), mode='w',)

    pdb.set_trace()

    if tensorboard_flag:
        writer.add_image('Inferred Grids', torchvision.utils.make_grid(recon_x_inf_most_likely, nrow=int(recon_x_inf_most_likely.shape[0]/32)), torch.Tensor([0]).cuda())
        writer.add_image('Input Grids', torchvision.utils.make_grid(x, nrow=int(recon_x_inf_most_likely.shape[0]/32)), torch.Tensor([0]).cuda())
        writer.add_scalar('val/Loss/Reconstruction', (rec.data/x.size(0)).unsqueeze(-1), torch.Tensor([0]).cuda())

        y_plot = []
        for i in tqdm(range(acc_proj_sign.shape[0])):
            y_plot.append(plot_scatter(pos_x[i], pos_y[i], acc_proj_sign[i]))

        writer.add_image('Input States', torchvision.utils.make_grid(y_plot, nrow=32), torch.Tensor([0]).cuda())

        pred = []
        gt = []
        alpha_p_plot = []
        alpha_q_plot = []
        for i in [0,10,100,500,1000]:
            gt.append(torch.reshape(x[i], (-1,70,50)))
            pred.append(recon_x_inf_most_likely[i])
            alpha_p_plot.append(plot_bar(alpha_p[i], args))

        writer.add_image('Val/Occupancy Grid Ground Truth', torchvision.utils.make_grid(gt, nrow=len(gt)), torch.Tensor([0]).cuda())
        writer.add_image('Val/Most Likely Occupancy Grid Reconstructed', torchvision.utils.make_grid(pred, nrow=len(pred)), torch.Tensor([0]).cuda())                    

        writer.add_image('Val/alpha_p', torchvision.utils.make_grid(alpha_p_plot, nrow=len(alpha_p_plot)), torch.Tensor([0]).cuda())

    return rec.item()

def main(args):

    random = deepcopy(seed)

    ts = time.time()

    # Load data.
    nt = 10
    dir = '/data/INTERACTION-Dataset-DR-v1_1/processed_data/driver_sensor_dataset/'
    models = 'cvae'
    dir_models = os.path.join('/models', models)
    if not os.path.isdir(dir_models):
        os.mkdir(dir_models)
    print('Directory exists.')

    # Test data.
    data_file_states = os.path.join(dir, 'states_test.hkl')
    data_file_grids = os.path.join(dir, 'label_grids_test.hkl')
    data_file_sources = os.path.join(dir, 'sources_test.hkl')

    dataset_val = SequenceGenerator(data_file_state=data_file_states, data_file_grid=data_file_grids, source_file=data_file_sources, nt=nt, 
        batch_size=None, shuffle=False, sequence_start_mode='unique', norm=True)
    print(len(dataset_val))
    num_val = len(dataset_val)

    tracker_global_test = defaultdict(torch.cuda.FloatTensor)

    # name = 'lstm_1_Adam_z_100_lr_0.001_rand_123_norm_True_kl_start_0_finish_1.0_center_10000.0_mutual_info_const_alpha_1.5_epochs_30_batch_256'
    name = 'lstm_' + str(args.n_lstms) + '_Adam_z_' + str(args.latent_size) + '_lr_' + str(args.learning_rate) + '_rand_' + str(random) + '_norm_' + str(args.norm) + '_kl_start_0_finish_' + str(args.beta) + '_center_' + str(args.crossover) + '_mutual_info_' + args.mut_info + '_alpha_' + str(args.alpha) +'_epochs_' + str(args.epochs) + '_batch_' + str(args.batch_size)
    folder_name = str(ts) + "_" + name

    writer = SummaryWriter(os.path.join(dir_models, 'runs/' + str(ts) + '_inf_' + name))
                    
    if not os.path.exists(os.path.join(args.fig_root, folder_name)):
        if not(os.path.exists(os.path.join(args.fig_root))):
            os.mkdir(os.path.join(args.fig_root))
        os.mkdir(os.path.join(args.fig_root, folder_name))

    vae = VAE(
        encoder_layer_sizes_p=args.encoder_layer_sizes_p,
        n_lstms=args.n_lstms,
        latent_size=args.latent_size,
        dim=args.dim
        )
    
    vae = vae.cuda()

    save_filename = name + 'epoch_30_vae.pt'

    with open(os.path.join(dir_models, save_filename), 'rb') as f:
        print(f)
        state_dict = torch.load(f)
        vae.load_state_dict(state_dict, strict=False)
    vae.eval()

    # Set the batch size to be the full dataset, if possible, on the validation set.
    batch_size = len(dataset_val)
    data_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)

    loss_inf = inference(data_loader, vae, args, writer, num_val, nt, name, dir_models)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--kl_min", type=float, default=0.2)
    parser.add_argument("--encoder_layer_sizes_p", type=list, default=[7, 5])
    parser.add_argument("--n_lstms", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--crossover", type=float, default=10000.0)
    parser.add_argument("--mut_info", type=str, default='None')
    parser.add_argument("--norm", action='store_true')
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--latent_size", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=1000)
    parser.add_argument("--fig_root", type=str, default='figs')

    args = parser.parse_args()

    main(args)
