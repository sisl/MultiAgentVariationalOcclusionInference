# Architecture for the CVAE driver sensor model. Code is adapted from: https://github.com/sisl/EvidentialSparsification and
# https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/modules.py.

seed = 123
import numpy as np
np.random.seed(seed)
import torch
import math

import torch.nn as nn
from src.utils.utils_model import to_var, sample_p
import pdb

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(False),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(False),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes_p, n_lstms, latent_size, dim):

        super().__init__()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        assert type(encoder_layer_sizes_p) == list
        assert type(latent_size) == int

        self.latent_size = latent_size
        self.label_size = encoder_layer_sizes_p[-1]

        self.encoder = Encoder(encoder_layer_sizes_p, n_lstms, latent_size, dim)
        self.decoder = Decoder(latent_size, self.label_size, dim)

    def forward(self, x, c=None):

        batch_size = x.size(0)

        # Encode the input.
        alpha_q, alpha_p, output_all_c = self.encoder(x, c)

        # Obtain all possible latent classes.
        z = torch.eye(self.latent_size).cuda()

        # Decode all latent classes.
        recon_x = self.decoder(z)

        return recon_x, alpha_q, alpha_p, self.encoder.linear_latent_q, self.encoder.linear_latent_p, output_all_c, z

    def inference(self, n=1, c=None, mode='sample', k=None):

        batch_size = n

        alpha_q, alpha_p, output_all_c = self.encoder(x=torch.empty((0,0)), c=c, train=False)

        if mode == 'sample':
            # Decode the mode sampled from the prior distribution.
            z = sample_p(alpha_p, batch_size=batch_size).view(-1,self.latent_size)
        elif mode == 'all':
            # Decode all the modes.
            z = torch.eye(self.latent_size).cuda()
        elif mode == 'most_likely':
            # Decode the most likely mode.
            z = torch.nn.functional.one_hot(torch.argmax(alpha_p, dim=1), num_classes=self.latent_size).float()
        elif mode == 'multimodal':
            # Decode a particular mode.
            z = torch.nn.functional.one_hot(torch.argsort(alpha_p, dim=1)[:,-k], num_classes=self.latent_size).float()

        recon_x = self.decoder(z)

        return recon_x, alpha_p, self.encoder.linear_latent_p, output_all_c, z

class Encoder(nn.Module):

    def __init__(self, layer_sizes_p, n_lstms, latent_size, dim):

        super().__init__()

        input_dim = 1
        self.VQVAEBlock = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(False),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim)
        )

        self.lstm = nn.LSTM(layer_sizes_p[0], 
                           layer_sizes_p[-1], 
                           num_layers=n_lstms, 
                           batch_first=True)

        self.linear_latent_q = nn.Linear(5*7*dim + layer_sizes_p[-1]*10, latent_size) # 10 is the time dimension.
        self.softmax_q = nn.Softmax(dim=-1)

        self.linear_latent_p = nn.Linear(layer_sizes_p[-1]*10, latent_size) # 10 is the time dimension.
        self.softmax_p = nn.Softmax(dim=-1)

    def forward(self, x=None, c=None, train=True):
        
        output_all, (full_c, _) = self.lstm(c)
        output_all_c = torch.reshape(output_all, (c.shape[0], -1))
        alpha_p_lin = self.linear_latent_p(output_all_c)
        alpha_p = self.softmax_p(alpha_p_lin)

        if train:
            
            full_x = self.VQVAEBlock(x)
            full_x = full_x.view(full_x.shape[0],-1)
            output_all_c = torch.cat((full_x, output_all_c), dim=-1)
            alpha_q_lin = self.linear_latent_q(output_all_c)
            alpha_q = self.softmax_q(alpha_q_lin)

        else:
            alpha_q_lin = None
            alpha_q = None

        return alpha_q, alpha_p, output_all_c


class Decoder(nn.Module):

    def __init__(self, latent_size, label_size, dim):

        super().__init__()
        self.latent_size = latent_size
        self.dim = dim
        input_dim = 1

        self.decode_linear = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size*dim),
            nn.ReLU(False),
            nn.Linear(self.latent_size*dim, self.latent_size*dim),
            nn.ReLU(False),
            )

        self.VQVAEBlock = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(False),
            nn.ConvTranspose2d(dim, dim, (4,5), (2,3), 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(False),
            nn.ConvTranspose2d(dim, input_dim, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, z):
        latent_size_sqrt = int(math.sqrt(self.latent_size))
        z_c = self.decode_linear(z)
        z_c = z_c.view(-1, self.dim, latent_size_sqrt, latent_size_sqrt)
        x = self.VQVAEBlock(z_c)
        return x