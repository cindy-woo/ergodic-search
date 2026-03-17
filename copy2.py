#!/usr/bin/env python
# coding: utf-8

import torch
# seed=42
# torch.manual_seed(seed)
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad #jacobian
from functools import partial
from functools import reduce
import torch.optim as optim

import timeit

import random
import pickle as pkl
import scipy

from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time 
import os

# Dynamics and target distribution
# The dynamics are defined as the constrained continuous time dynamical system
# The target distribution, for which information is distributed within the continuous search space
@torch.jit.script
def p(x):
    return torch.exp(-190.5 * torch.sum((x[:2] - 0.27)**2)) \
           + torch.exp(-180.5 * torch.sum((x[:2] - 0.75)**2))

@torch.jit.script
def f(x, u): # dynamics using discrete time eulerintegration
    xnew = x[:2] + u[:2]
    xnew = torch.clamp(xnew, 0, 1)
    return xnew, xnew


# Helper Functions
# Define a sigmoid function and a orthonormalizing factor $h_k$ for the ergodic metric
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def get_hk(k): # normalizing factor for basis function
    _hk = (2. * k + np.sin(2 * k)) / (4 * k + 1e-8)
    _hk[np.isnan(_hk)] = 1.
    return np.sqrt(np.prod(_hk))


# Ergodic metric and sample-weighted ergodic metric
def fk(x, k): # basis function
    return torch.prod(torch.cos(x * k))

#fk_vmap = lambda _x, _k: torch.func.vmap(fk, in_axes=(0,None))(_x, _k)
fk_vmap = lambda _x, _k: torch.func.vmap(fk)(_x, _k)

def get_ck(tr, k):
    ck = torch.mean(torch.cos(tr[:, None, :] * k[None, :, :]).prod(dim = -1), dim = 0)
    ck = np.copy(ck) / hk

    return ck

def get_ck_weighted(tr, k, lam, hk):
    weighted_fk = torch.prod(torch.cos(tr[:, None, :] * k[None, :, :]), dim = -1).T @ lam
    ck = weighted_fk / (hk + 1e-8)
    return ck

# ergodic metric + other costs----
def fourier_ergodic_loss(u, x0, phik, k):
    xf = x0.clone()
    tr = [xf]
    
    for i in range(u.shape[0]):
        x_new, _ = f(tr[-1], u[i, :2])
        tr.append(x_new)
    tr = torch.stack(tr[1:]) 

    lam = sigmoid(5 * u[:, 2])
    ck = get_ck_weighted(tr[:,:2], k, lam, hk)
    # ck = ck / (ck[0] + 1e-8)
    # barrier cost to ensure that x(t) stays within 
    barr_cost = 10 * torch.sum(torch.clamp_min(tr - 1, 0)**2 + torch.clamp_min(-tr, 0)**2)
    lam_barr_cost = 10 * torch.sum(torch.clamp_min(lam - 1, 0)**2 + torch.clamp_min(-lam, 0)**2)
    # Total loss = ergodic metric + control energy + smoothness penalty + state barriers + lambda sparsity + lambda Barrier
    loss = torch.sum(lamk * (phik - ck)**2) \
       + 0.001 * torch.mean(u[:, :2]**2) \
       + 0.001 * torch.sum((u[1:, :2] - u[:-1, :2])**2) \
       + 10 * torch.sum(torch.clamp_min(tr - 1, 0)**2 + torch.clamp_min(-tr, 0)**2) \
       + 0.001 * torch.sum(torch.abs(lam)) \
       + 10 * torch.sum(torch.clamp_min(lam - 1, 0)**2 + torch.clamp_min(-lam, 0)**2)
    # l1 cost to promote sparsity
    return loss


def optimize_trajectory(x0, phik, k, num_iters=1500, lr=1e-3):
    u = (0.01 * torch.randn((100, 3))).requires_grad_()
    optimizer = torch.optim.Adam([u], lr=lr)
    
    for i in range(num_iters):
        optimizer.zero_grad()
        loss = fourier_ergodic_loss(u, x0, phik, k)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 20 == 0:
            print(f"Iteration {i+1}, Loss: {loss.item()}")
    
    return u.detach()

def get_files_in_folder(path):
    files = []
    for file in os.listdir(path):
        item_path = os.path.join(path, file)
        if os.path.isfile(item_path): 
            files.append(item_path)
    return files

def load_files(file):
    ext = os.path.splitext(file)[1].lower()
    if ext == '.npy':
        arr = np.load(file)
    elif ext == '.txt':
        arr = np.loadtxt(file, dtype=np.float32)
    else:
        return None
    return torch.from_numpy(arr).float()

start_time = time.time()

# Load entropy and gaussian maps
entropy_maps_path = "/Users/cindy/Desktop/ergodic-search/entropy_maps"
entropy_maps = []
for file in (get_files_in_folder(entropy_maps_path)):
    entropy_file = load_files(file)
    if entropy_file != None:
        entropy_maps.append(entropy_file)

full_maps = torch.stack(entropy_maps, dim=0)
perm = torch.randperm(full_maps.shape[0])
maps = full_maps[perm[:3]]
maps = maps[0]
H, W = maps.shape


# Create consistent grid
xs = torch.linspace(0, 1, W)
ys = torch.linspace(0, 1, H)
X, Y = torch.meshgrid(xs, ys, indexing='xy')   
_s = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

# Compute the Fourier basis modes and the orthonormalization factors
x0 = torch.tensor([0.54, 0.3])
k1, k2 = np.meshgrid(np.arange(0, 20), np.arange(0, 20))
k = torch.tensor(np.stack([k1.ravel(), k2.ravel()], axis=-1), dtype=torch.float32).unsqueeze(0)
lamk = torch.exp(-0.8 * torch.norm(k, dim=1))
hk = torch.tensor([get_hk(ki) for ki in k.numpy()])

# phik, Fourier coefficients of the target distribution.
phik = (torch.cos(_s[:, None, :] * k).prod(dim=-1) * torch.stack([p(s) for s in _s])[:, None]).sum(dim=0)
phik_1 = phik.clone().detach().contiguous()
phik_1 = phik / (phik[0] + 1e-8)
fk_vals_all = torch.cos(_s[:, None, :] * k[None, :, :]).prod(dim=-1)

# Fix the reshape operation - use H and W instead of hardcoded 50
phik_recon = torch.matmul(fk_vals_all, phik_1).reshape(H, W)

optimized_u = optimize_trajectory(x0, phik_1, k)

# Generate trajectory
x = x0.clone()
tr = [x]
for step in optimized_u:
    x, _ = f(x, step[:2])
    tr.append(x)
tr = torch.stack(tr).cpu().detach().numpy()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# Visualization
plt.figure(figsize=(4, 4))
plt.contourf(X.numpy(), Y.numpy(), phik_recon.numpy(), cmap='viridis')
plt.scatter(tr[1:, 0], tr[1:, 1], s=10, c = 5 * sigmoid(5 * optimized_u[:, 2]), cmap='plasma')
plt.title("Information Map")

plt.figure(figsize=(3, 3))
plt.imshow(phik_recon.numpy(), extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
plt.scatter(tr[1:, 0], tr[1:, 1], s=5, c='red')  # smaller red dots
plt.title("Original Map and Trajectory")

plt.axis('square')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()