# Helper code for the CVAE driver sensor model. Code is adapted from: https://github.com/sisl/EvidentialSparsification.

seed = 123
import numpy as np
np.random.seed(seed)
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

from torch.autograd import Variable

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def idx2onehot(idx, n):

    assert idx.size(1) == 1
    assert torch.max(idx).data < n

    onehot = torch.zeros(idx.size(0), n).cuda()
    onehot.scatter_(1, idx.data, 1)
    onehot = to_var(onehot)
    
    return onehot

def sample_p(alpha, batch_size=1):
    zdist = torch.distributions.one_hot_categorical.OneHotCategorical(probs = alpha)
    return zdist.sample(torch.Size([batch_size]))
