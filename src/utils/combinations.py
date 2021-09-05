# Code for obtaining the top num_modes most likely combinations for the fused ego vehicle grid using breadth-first search.
# The algorithm we use is adapted from here: https://cs.stackexchange.com/questions/46910/efficiently-finding-k-smallest-elements-of-cartesian-product.

import numpy as np
from copy import deepcopy

def compute_best_cost(alpha, indices):
    cost = 1.0
    for i in range(len(alpha)):
        cost *= alpha[i][indices[i]]
    return cost

def covert_indices(alpha_indices, alpha_sorted, indices):
    best_indices = np.zeros(indices.shape)
    for i in range(len(alpha_sorted)):
        best_indices[i] = alpha_indices[i][indices[i]]
    return best_indices

# Takes as input sorted alpha.
def BFS(alpha, top=3):
    N = len(alpha)
    num_modes = len(alpha[0])
    
    alpha_sorted = []
    alpha_indices = []
    for i in range(len(alpha)):
        alpha_indices.append(np.argsort(alpha[i])[::-1])
        alpha_sorted.append(alpha[i][alpha_indices[-1]])
    
    Q = [np.zeros((N,)).astype('int')]
    best_index = np.array([])
    best_cost = np.array([])
    cost = np.array([compute_best_cost(alpha_sorted, Q[-1])])
    
    # Do BFS to obtain the top num_modes most likely products.
    while len(best_index) < top:
        idx = np.argmax(cost)
        if len(best_cost) == 0:
            best_cost = np.array([cost[idx]])
            best_index = np.array([covert_indices(alpha_indices, alpha_sorted, Q[idx])])
        else:
            best_cost = np.vstack((best_cost, cost[idx]))
            best_index = np.vstack((best_index, covert_indices(alpha_indices, alpha_sorted, Q[idx])))
        last = Q.pop(idx)
        cost = np.delete(cost, idx)
        if len(Q) == 0:
            for i in range(N):
                array = np.zeros((N,)).astype('int')
                array[i] = 1
                Q.append(array)
                if len(cost) == 0:
                    cost = np.array([compute_best_cost(alpha_sorted, Q[-1])])
                else:
                    cost = np.hstack((cost, compute_best_cost(alpha_sorted, Q[-1])))
        else:
            for i in range(N):
                array = deepcopy(last)
                if array[i] < num_modes-1:
                    array[i] += 1
                else:
                    continue
                Q.append(array)
                cost = np.hstack((cost, compute_best_cost(alpha_sorted, Q[-1])))
                if last[i] > 0:
                    break
    return best_index, best_cost