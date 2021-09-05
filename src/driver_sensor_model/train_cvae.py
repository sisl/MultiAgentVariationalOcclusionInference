# Training code for the CVAE driver sensor model. Code is adapted from: https://github.com/sisl/EvidentialSparsification and
# https://github.com/StanfordASL/Trajectron-plus-plus.

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
from src.driver_sensor_model.models_cvae import VAE
from src.utils.data_generator import *
import time

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

def plot_bar(alpha_p,alpha_q,args):
    figure = plt.figure()
    plt.bar(np.arange(args.latent_size)-0.1, alpha_p.cpu().data.numpy(), width=0.2, align='center', color='g', label='alpha_p')
    plt.bar(np.arange(args.latent_size)+0.1, alpha_q.cpu().data.numpy(), width=0.2, align='center', color='b', label='alpha_q')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', dpi=200)
    plt.close(figure)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image)
    return image

def plot_scatter(pos_x, pos_y, acc):
    figure = plt.figure()
    plt.scatter(pos_x, pos_y, c=acc)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.colorbar()
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    plt.close(figure)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image)
    return image

def kl_weight_schedule(iter, beta, crossover):
    start = 0.0
    finish = beta
    divisor = 10.0
    center_step = crossover
    steps_lo_to_hi = center_step/divisor
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    start = torch.tensor(start, device=device)
    finish = torch.tensor(finish, device=device)
    center_step = torch.tensor(center_step, device=device, dtype=torch.float)
    steps_lo_to_hi = torch.tensor(steps_lo_to_hi, device=device, dtype=torch.float)
    return start + (finish - start)*torch.sigmoid((torch.tensor(float(iter), device=device) - center_step) * (1./steps_lo_to_hi))

# Modified ELBO loss.
def loss_fn(recon_x, x, q_dist, p_dist, output_all_c, y, iter, args, mode='train'):
    # Load the arguments for the loss function.
    kl_min = args.kl_min
    beta = args.beta
    crossover = args.crossover
    alpha = args.alpha
    mut_info_mode = args.mut_info
    norm = args.norm

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    num_free = torch.sum(x == 0).double()
    num_occ = torch.sum(x == 1).double()
    num_batch = num_free + num_occ

    rec_weight_free = 1.0 - num_free/num_batch
    rec_weight_occ = 1.0 - num_occ/num_batch

    # Repeat from: latent_size x grid_flat to batch x latent_size x grid flat.
    if recon_x.shape[0] != x.shape[0]:
        recon_x = recon_x.flatten(start_dim=1).unsqueeze(0)
        recon_x = recon_x.repeat(x.shape[0], 1, 1)
        # Repeat from: batch_size x grid_flat to batch x latent_size x grid flat.
        x = x.flatten(start_dim=1).unsqueeze(1)
        x = x.repeat(1, recon_x.shape[1], 1)
    else:
        recon_x = recon_x.flatten(start_dim=1)
        x = x.flatten(start_dim=1)
    
    mask_free = (x == 0).float()
    mask_occ = (x == 1).float()

    bce = torch.nn.BCELoss(reduction='none')
    rec_loss = bce(recon_x, x)

    if norm:
        # Weighted sum using q probabilities along latent_size dimension.
        rec_loss_weighted = rec_loss*mask_occ*rec_weight_occ + rec_loss*mask_free*rec_weight_free
        rec_loss_weighted = rec_loss_weighted.sum(-1)
        
        # Check if we have all the latent classes or just one.
        if len(recon_x.shape) > 2:
            rec_loss_weighted = (rec_loss_weighted*q_dist.probs).sum(1).mean(0)
        else:
            rec_loss_weighted = rec_loss_weighted.mean(0)

    else:
        rec_loss_weighted = rec_loss.sum(-1)

        # Check if we have all the latent classes or just one.
        if len(recon_x.shape) > 2:
            rec_loss_weighted = (rec_loss_weighted*q_dist.probs).sum(1).mean(0)
        else:
            rec_loss_weighted = rec_loss_weighted.mean(0)

    # Check if we have all the latent classes or just one.
    if len(recon_x.shape) > 2:
    
        # Sum over grid dimension because the grid cells are independent.
        rec_loss_occ = ((rec_loss*mask_occ*rec_weight_occ).sum(-1)*q_dist.probs).sum(1).mean(0)
        rec_loss_free = ((rec_loss*mask_free*rec_weight_free).sum(-1)*q_dist.probs).sum(1).mean(0)

        rec_loss = (rec_loss.sum(-1)*q_dist.probs).sum(1).mean(0)
    
    else:

        # Sum over grid dimension because the grid cells are independent.
        rec_loss_occ = ((rec_loss*mask_occ*rec_weight_occ).sum(-1)).mean(0)
        rec_loss_free = ((rec_loss*mask_free*rec_weight_free).sum(-1)).mean(0)

        rec_loss = (rec_loss.sum(-1)).mean(0)

    if mode == 'train':
        kl_weight = kl_weight_schedule(iter, beta, crossover)

        kl_disc_loss = torch.distributions.kl.kl_divergence(q_dist, p_dist)

        if torch.any(torch.isinf(kl_disc_loss)):
            pdb.set_trace()

        kl_disc_loss = torch.mean(kl_disc_loss, dim=0, keepdim=False)

        if kl_min > 0:
            kl_lower_bounded = torch.clamp(kl_disc_loss, min=kl_min)

        dist = p_dist.__class__
        H_y = dist(probs=p_dist.probs.mean(dim=0)).entropy()
        mutual_info = (H_y - p_dist.entropy().mean(dim=0)).sum()

        if mut_info_mode == 'None':
            mi_weight = 0.0
        elif mut_info_mode == 'kl':
            mi_weight = kl_weight_schedule(iter, alpha, crossover)
        elif mut_info_mode == 'const':
            mi_weight = alpha

        loss = rec_loss_weighted + kl_weight*kl_disc_loss - mi_weight*mutual_info

    elif mode == 'inf':
        loss = rec_loss_weighted
        kl_disc_loss = 0.0
        kl_weight = 0.0
        mutual_info = 0.0
        mse_c_loss = 0.0

    return loss, rec_loss, kl_disc_loss, kl_weight, rec_loss_occ, rec_loss_free, mutual_info

def train(data_loader, vae, optimizer, args, writer, tracker_global_train, epoch, num_train):
    
    vae.train()
    for iteration, (y, x, sources) in enumerate(data_loader):
        # Reshape features.
        optimizer.zero_grad()
        start = time.time()
        x = to_var(x).float().view(x.shape[0],1,20,30)
        y = to_var(y).float()
        batch_size = x.shape[0]

        start_new = time.time()
        recon_x, alpha_q, alpha_p, alpha_q_lin, alpha_p_lin, output_all_c, z = vae(x, y)
        recon_x_inf_p_1, _, _, _, _ = vae.inference(n=1, c=y, mode='most_likely')

        # Add epsilon to q and p if elements of q are close to zero (KL will be Inf otherwise).
        if torch.min(alpha_q) < 1e-15:
            alpha_q = alpha_q.clone() + 1e-15
        if torch.min(alpha_p) < 1e-15:
            alpha_p = alpha_p.clone() + 1e-15

        # Form distributions out of alpha_q and alpha_p.
        q_dist = torch.distributions.one_hot_categorical.OneHotCategorical(probs=alpha_q)
        p_dist = torch.distributions.one_hot_categorical.OneHotCategorical(probs=alpha_p)

        start_new = time.time()
        M_N = batch_size/num_train*1.0
        loss, rec, kl, kl_weight, rec_loss_occ, rec_loss_free, mutual_info = loss_fn(recon_x, x, q_dist, p_dist, output_all_c, y, epoch*len(data_loader)+iteration, args)
        loss_inf, rec_inf, _, _, rec_loss_occ_inf, rec_loss_free_inf, _ = loss_fn(recon_x_inf_p_1, x, q_dist, p_dist, output_all_c, y, epoch*len(data_loader)+iteration, args, mode='inf')

        writer.add_scalar('train/Loss', (loss.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
        writer.add_scalar('train/Loss/KL', (kl.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
        writer.add_scalar('train/Loss/Mutual Information', (mutual_info.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
        writer.add_scalar('train/Loss/Reconstruction', (rec.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
        writer.add_scalar('train/Loss/Reconstruction Occ', (rec_loss_occ.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
        writer.add_scalar('train/Loss/Reconstruction Free', (rec_loss_free.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
        writer.add_scalar('train/Loss Inf', (loss_inf.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
        writer.add_scalar('train/Loss Inf/Reconstruction', (rec_inf.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
        writer.add_scalar('train/Loss Inf/Reconstruction Occ', (rec_loss_occ_inf.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
        writer.add_scalar('train/Loss Inf/Reconstruction Free', (rec_loss_free_inf.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
        writer.add_scalar('train/KL Weight', kl_weight, torch.Tensor([epoch*len(data_loader)+iteration]).cuda())

        tracker_global_train['loss'] = torch.cat((tracker_global_train['loss'], (loss.data).unsqueeze(-1)))
        tracker_global_train['it'] = torch.cat((tracker_global_train['it'], torch.Tensor([epoch*len(data_loader)+iteration]).cuda()))

        start_new = time.time()
        recon_x_inf_p = recon_x[torch.argmax(q_dist.probs, dim=-1)]
        recon_x_inf_p = recon_x_inf_p.view(x.shape)

        recon_x_inf_pred = (recon_x_inf_p >= 0.6).float()
        recon_x_inf_pred[recon_x_inf_p <= 0.4] = 0.0
        recon_x_inf_pred[(recon_x_inf_p < 0.6)*(recon_x_inf_p > 0.4)] = 0.5

        acc = torch.mean((recon_x_inf_pred == x).float())
        mse = torch.mean(torch.pow(recon_x_inf_p - x, 2))

        acc_occ = torch.mean((recon_x_inf_pred[x == 1] == 1).float())
        acc_free = torch.mean((recon_x_inf_pred[x == 0] == 0).float())

        mse_occ = torch.mean(torch.pow(recon_x_inf_p[x == 1] - 1, 2))
        mse_free = torch.mean(torch.pow(recon_x_inf_p[x == 0] - 0, 2))

        writer.add_scalar('train/Metrics/MSE', (mse.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
        writer.add_scalar('train/Metrics/Accuracy', (acc.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
        writer.add_scalar('train/Metrics/MSE Occ', (mse_occ.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
        writer.add_scalar('train/Metrics/MSE Free', (mse_free.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
        writer.add_scalar('train/Metrics/Accuracy Occ', (acc_occ.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
        writer.add_scalar('train/Metrics/Accuracy Free', (acc_free.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())

        if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
            print("Batch %04d/%i, Loss %9.4f"%(iteration, len(data_loader)-1, loss.data))
            print("recon_x", torch.max(recon_x_inf_p).data)
            print("recon", loss.data, "kl", kl.data, mutual_info.data)

            pred = []
            gt = []
            sample_numbers = np.random.choice(np.arange(y.shape[0]), size=5, replace=False)           
            for i in sample_numbers:
                gt.append(torch.reshape(x[i], (-1,20,30)))
                pred.append(torch.reshape(recon_x_inf_p[i], (-1,20,30))) 
            
            writer.add_image('Train/Occupancy Grid Ground Truth', torchvision.utils.make_grid(gt, nrow=len(gt)), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
            writer.add_image('Train/Occupancy Grid Reconstructed', torchvision.utils.make_grid(pred, nrow=len(pred)), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())                    

        loss.backward()
        optimizer.step()

def test(data_loader, vae, optimizer, args, writer, epoch, num_val, nt):
    
    vae.eval()

    num = 0.0
    loss_inf_total = torch.zeros(1).cuda()

    with torch.no_grad():
        for iteration, (y, x, sources) in tqdm(enumerate(data_loader)):
            # Reshape features.
            num += y.shape[0]
            x = to_var(x).float().view(x.shape[0],1,20,30)
            y = to_var(y).float()
            y_full = unnormalize(y.cpu().data.numpy(), nt)
            pos_x = y_full[:,:,0]
            pos_y = y_full[:,:,1]

            batch_size = x.shape[0]

            recon_x, alpha_q, alpha_p, alpha_q_lin, alpha_p_lin, output_all_c, z = vae(x, y)

            if torch.min(alpha_q) < 1e-15:
                alpha_q = alpha_q.clone() + 1e-15
            if torch.min(alpha_p) < 1e-15:
                alpha_p = alpha_p.clone() + 1e-15

            # Form distributions out of alpha_q and alpha_p.
            q_dist = torch.distributions.one_hot_categorical.OneHotCategorical(probs=alpha_q)
            p_dist = torch.distributions.one_hot_categorical.OneHotCategorical(probs=alpha_p)
            loss, rec, kl, kl_weight, rec_loss_occ, rec_loss_free, mutual_info = loss_fn(recon_x, x, q_dist, p_dist, output_all_c, y, epoch*len(data_loader)+iteration, args)

            # Get reconstruction from inference (all latent classes).
            sample_numbers = np.random.choice(np.arange(y.shape[0]), size=np.minimum(5, y.shape[0]), replace=False)
            if ((y.shape[0] > 5) and (num == y.shape[0])): 
                recon_x_inf_a, _, _, _, _ = vae.inference(n=args.latent_size, c=y[sample_numbers[0]].view(1, y.shape[1], y.shape[2]), mode='all')

            recon_x_inf_p, _, _, _, _ = vae.inference(n=1, c=y, mode='most_likely')

            loss_inf, rec_inf, _, _, rec_loss_occ_inf, rec_loss_free_inf, _= loss_fn(recon_x_inf_p, x, q_dist, p_dist, output_all_c, y, epoch*len(data_loader)+iteration, args, mode='inf')
            loss_inf_total += loss_inf*y.shape[0]

            # Most likely class.
            recon_x = recon_x[torch.argmax(q_dist.probs, dim=-1)]
            recon_x = recon_x.view(x.shape)
            
            if ((y.shape[0] > 5) and (num == y.shape[0])):
                recon_x_inf_a = recon_x_inf_a.view(args.latent_size, 1, x.shape[-2], x.shape[-1]) 
            recon_x_inf_p = recon_x_inf_p.view(x.shape)

            recon_x_inf_pred = (recon_x_inf_p >= 0.6).float()
            recon_x_inf_pred[recon_x_inf_p <= 0.4] = 0.0
            recon_x_inf_pred[(recon_x_inf_p < 0.6)*(recon_x_inf_p > 0.4)] = 0.5

            acc = torch.mean((recon_x_inf_pred == x).float())
            mse = torch.mean(torch.pow(recon_x_inf_p - x, 2))

            acc_occ = torch.mean((recon_x_inf_pred[x == 1] == 1).float())
            acc_free = torch.mean((recon_x_inf_pred[x == 0] == 0).float())

            mse_occ = torch.mean(torch.pow(recon_x_inf_p[x == 1] - 1, 2))
            mse_free = torch.mean(torch.pow(recon_x_inf_p[x == 0] - 0, 2))

            if ((y.shape[0] > 5) and (num == y.shape[0])):
                y_plot = []
                for i in sample_numbers:
                    y_plot.append(plot_scatter(pos_x[i], pos_y[i], range(len(pos_x[i]))))

                writer.add_image('Decoded Latent Classes/Input States', torchvision.utils.make_grid(y_plot, nrow=5), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
                writer.add_image('Decoded Latent Classes/a', torchvision.utils.make_grid(recon_x_inf_a, nrow=int(np.sqrt(args.latent_size))), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())

            writer.add_scalar('val/Loss', (loss.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
            writer.add_scalar('val/Loss/KL', (kl.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
            writer.add_scalar('val/Loss/Mutual Information', (kl.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
            writer.add_scalar('val/Loss/Reconstruction', (rec.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
            writer.add_scalar('val/Loss/Reconstruction Occ', (rec_loss_occ.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
            writer.add_scalar('val/Loss/Reconstruction Free', (rec_loss_free.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())

            writer.add_scalar('val/Loss Inf', (loss_inf.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
            writer.add_scalar('val/Loss Inf/Reconstruction', (rec_inf.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
            writer.add_scalar('val/Loss Inf/Reconstruction Occ', (rec_loss_occ_inf.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
            writer.add_scalar('val/Loss Inf/Reconstruction Free', (rec_loss_free_inf.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())

            writer.add_scalar('val/Metrics/MSE', (mse.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
            writer.add_scalar('val/Metrics/MSE Occ', (mse_occ.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
            writer.add_scalar('val/Metrics/MSE Free', (mse_free.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
            writer.add_scalar('val/Metrics/Accuracy', (acc.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
            writer.add_scalar('val/Metrics/Accuracy Occ', (acc_occ.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
            writer.add_scalar('val/Metrics/Accuracy Free', (acc_free.data).unsqueeze(-1), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())

            pred_p = []
            pred_q = []
            gt = []
            alpha_plot = []
            for i in sample_numbers:
                gt.append(torch.reshape(x[i], (-1,20,30)))
                pred_q.append(torch.reshape(recon_x[i], (-1,20,30)))
                pred_p.append(torch.reshape(recon_x_inf_p[i], (-1,20,30)))
                if (num == y.shape[0]):
                    alpha_plot.append(plot_bar(alpha_p[i], alpha_q[i], args))

            writer.add_image('Val/Occupancy Grid Ground Truth', torchvision.utils.make_grid(gt, nrow=len(gt)), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
            writer.add_image('Val/p Occupancy Grid Reconstructed', torchvision.utils.make_grid(pred_p, nrow=len(pred_q)), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())
            writer.add_image('Val/q Occupancy Grid Reconstructed', torchvision.utils.make_grid(pred_q, nrow=len(pred_p)), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())                    

            if (num == y.shape[0]):
                writer.add_image('Val/alpha', torchvision.utils.make_grid(alpha_plot, nrow=len(alpha_plot)), torch.Tensor([epoch*len(data_loader)+iteration]).cuda())

    loss_inf_total = loss_inf_total/float(num)    

    print("Batch %04d/%i, Loss %9.4f"%(iteration, len(data_loader)-1, loss.data))
    print("recon_x", torch.max(recon_x).data)
    print("recon", rec.data, "kl", kl.data)

    return loss_inf_total.item()

def gradient_vis(parameters, writer, num_epoch):
        for name, param in parameters:
            if ((param.requires_grad) and (type(param.grad)!=type(None))):
                writer.add_scalar('Gradients/'+str(name)+'_avg', param.grad.abs().mean(), num_epoch)
                writer.add_scalar('Gradients/'+str(name)+'_max', param.grad.abs().max(), num_epoch)
            elif(param.grad==None):
                print(name)

def main(args):

    random = deepcopy(seed)

    ts = time.time()

    # Load data.
    nt = 10
    
    dir = '/data/INTERACTION-Dataset-DR-v1_1/processed_data/driver_sensor_dataset/'
    models = 'cvae'
    dir_models = os.path.join('/models/', models)
    args.fig_root = dir_models
    if not os.path.isdir(dir_models):
        os.mkdir(dir_models)
    print('Directory exists.')
    data_file_states = os.path.join(dir, 'states_shuffled_train.hkl')
    data_file_grids = os.path.join(dir, 'label_grids_shuffled_train.hkl')
    data_file_sources = os.path.join(dir, 'sources_shuffled_train.hkl')

    datasets = OrderedDict()
    datasets['train'] = SequenceGenerator(data_file_state=data_file_states, data_file_grid=data_file_grids, source_file=data_file_sources, nt=nt,
                     batch_size=None, shuffle=True, sequence_start_mode='all', norm=True)
    print(len(datasets['train']))
    num_train = len(datasets['train'])

    # Test on validation data.
    data_file_states = os.path.join(dir, 'states_val.hkl')
    data_file_grids = os.path.join(dir, 'label_grids_val.hkl')
    data_file_sources = os.path.join(dir, 'sources_val.hkl')

    datasets['val'] = SequenceGenerator(data_file_state=data_file_states, data_file_grid=data_file_grids, source_file=data_file_sources, nt=nt,
                     batch_size=None, shuffle=False, sequence_start_mode='unique', norm=True)
    print(len(datasets['val']))
    num_val = len(datasets['val'])
    tracker_global_train = defaultdict(torch.cuda.FloatTensor)
    tracker_global_test = defaultdict(torch.cuda.FloatTensor)
    name = 'lstm_' + str(args.n_lstms) + '_Adam_z_' + str(args.latent_size) + '_lr_' + str(args.learning_rate) + '_rand_' + str(random) + '_norm_' + str(args.norm) + '_kl_start_0_finish_' + str(args.beta) + '_center_' + str(args.crossover) + '_mutual_info_' + args.mut_info + '_alpha_' + str(args.alpha) +'_epochs_' + str(args.epochs) + '_batch_' + str(args.batch_size)
    folder_name = str(ts) + "_" + name

    if not os.path.exists(os.path.join(args.fig_root, folder_name)):
        if not(os.path.exists(os.path.join(args.fig_root))):
            os.mkdir(os.path.join(args.fig_root))
        os.mkdir(os.path.join(args.fig_root, folder_name))

    writer = SummaryWriter(os.path.join(dir_models, 'runs/' + str(ts) + '_' + name))

    vae = VAE(
        encoder_layer_sizes_p=args.encoder_layer_sizes_p,
        n_lstms=args.n_lstms,
        latent_size=args.latent_size,
        dim=args.dim
        )

    vae = vae.cuda()
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    best_loss = -1.
    save_filename = name + 'best_p.pt'
    for epoch in tqdm(range(args.epochs)):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for split, dataset in datasets.items():

            print("split", split, epoch)
            if split == 'train':
                batch_size = args.batch_size
            else:
                batch_size = int(len(dataset)/10)

            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=split=='train', drop_last=True)
            
            if split == 'train':
                train(data_loader, vae, optimizer, args, writer, tracker_global_train, epoch, num_train)
                gradient_vis(vae.named_parameters(), writer, epoch)
            else:
                loss = test(data_loader, vae, optimizer, args, writer, epoch, num_val, nt)
                if epoch == args.epochs-1:
                    with open(os.path.join(dir_models, name + '_epoch_' + str(args.epochs) + '_vae.pt'), 'wb') as f:
                        torch.save(vae.state_dict(), f) 
                if ((epoch == 0) or (epoch == 5) or (loss < best_loss)):
                    best_loss = loss
                    with open(os.path.join(dir_models, save_filename), 'wb') as f:
                        torch.save(vae.state_dict(), f)

    # Plot losses.
    plt.plot(tracker_global_train['it'].data.cpu().numpy(), tracker_global_train['loss'].data.cpu().numpy())
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(dir_models, folder_name, "loss.png"))
    plt.clf()
    plt.close()

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
