# Code for preprocessing and data generation. Code is adapted from: https://github.com/sisl/EvidentialSparsification.
import torch
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.utils.data as data
import torchvision

import numpy as np
np.random.seed(0)
import hickle as hkl
import pdb

def unnormalize(X, nt=1, norm=True):
    mean = np.array([9.9923511e+02, 9.9347858e+02, 1.3369133e-01, -3.7689242e-01, -6.0116798e-01, -1.3141568e-01, -1.2848811e-01])
    std = np.array([1.9491451e+01, 1.2455911e+01, 2.0100091e+00, 5.9114690e+00, 2.5912049e+00, 9.5518535e-01, 8.1346464e-01])

    if norm:
        X = X * std + mean
    return X.astype(np.float32)


def preprocess(X, norm=True):
    mean = np.array([9.9923511e+02, 9.9347858e+02, 1.3369133e-01, -3.7689242e-01, -6.0116798e-01, -1.3141568e-01, -1.2848811e-01])
    std = np.array([1.9491451e+01, 1.2455911e+01, 2.0100091e+00, 5.9114690e+00, 2.5912049e+00, 9.5518535e-01, 8.1346464e-01])
    if norm:
        X = (X - mean)/std
    return X.astype(np.float32) 

# Data generator that creates sequences for input.
class SequenceGenerator(data.Dataset):
    def __init__(self, data_file_state, data_file_grid, source_file, nt,
                 batch_size=8, shuffle=False, sequence_start_mode='all', norm=True):
        self.state = hkl.load(data_file_state)
        self.grid = hkl.load(data_file_grid)
        self.grid = np.reshape(self.grid, (self.grid.shape[0], -1))
        self.grid = self.grid.astype(np.float32)

        print(self.grid.shape)

        # Source for each grid so when creating sequences can ensure that consecutive frames are from same data run.
        self.sources = hkl.load(source_file)
        self.nt = nt
        self.norm = norm
        
        if batch_size == None:
            self.batch_size = self.state.shape[0]
        else:
            self.batch_size = batch_size
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode

        # Allow for any possible sequence, starting from any frame.
        if self.sequence_start_mode == 'all':
            self.possible_starts = np.array([i for i in range(self.state.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
        # Create sequences where each unique frame is in at most one sequence.
        elif self.sequence_start_mode == 'unique':
            curr_location = 0
            possible_starts = []
            while curr_location < self.state.shape[0] - self.nt + 1:
                if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                    possible_starts.append(curr_location)
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        self.N_sequences = len(self.possible_starts)        

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self.possible_starts[idx]
        batch_x = self.preprocess(self.state[idx:idx+self.nt])
        batch_y = self.grid[idx+self.nt-1]
        sources = self.sources[idx+self.nt-1]
        return batch_x, batch_y, sources

    def __len__(self):
        return self.N_sequences

    def preprocess(self, X):
        mean = np.array([9.9923511e+02, 9.9347858e+02, 1.3369133e-01, -3.7689242e-01, -6.0116798e-01, -1.3141568e-01, -1.2848811e-01])
        std = np.array([1.9491451e+01, 1.2455911e+01, 2.0100091e+00, 5.9114690e+00, 2.5912049e+00, 9.5518535e-01, 8.1346464e-01])

        if self.norm:
            X = (X - mean)/std
        return X.astype(np.float32)
