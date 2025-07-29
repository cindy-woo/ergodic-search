#!/usr/bin/env python
# coding: utf-8
import os
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

# Dynamics and target distribution
# The dynamics are defined as the constrained continuous time dynamical system
# The target distribution, for which information is distributed within the continuous search space
@torch.jit.script
def p(x):
    return torch.exp(-190.5 * torch.sum((x[:2] - 0.27)**2)) \
           + torch.exp(-180.5 * torch.sum((x[:2] - 0.75)**2))

@torch.jit.script
def f(x, u): # dynamics using discrete time eulerintegration
    xnew = x[:2] + 0.1*u[:2]
    xnew = torch.clamp(xnew, 0, 1)
    return xnew, xnew


# Helper Functions

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
def fourier_ergodic_loss(u, x0, phik, k, info_map):
    xf = x0.clone()
    tr = [xf]
    
    for i in range(u.shape[0]):
        x_new, _ = f(tr[-1], u[i, :2])
        tr.append(x_new)
    tr = torch.stack(tr[1:]) 

    lam = torch.clamp(torch.sigmoid(5 * u[:, 2]), 0.05, 1.0)
    ck = get_ck_weighted(tr[:,:2], k, lam, hk)
    phik_normed = (phik - phik.mean()) / (phik.std() + 1e-6)
    ck_normed = (ck - ck.mean()) / (ck.std() + 1e-6)
    H, W = info_map.shape
    indices_x = (tr[:, 0] * (W - 1)).long().clamp(0, W - 1)
    indices_y = (tr[:, 1] * (H - 1)).long().clamp(0, H - 1)
    info_values = info_map[indices_y, indices_x]
    reward_term = -0.05 * torch.sum(info_values)  # scale strength as needed

    # barrier cost to ensure that x(t) stays within 
    # Total loss = ergodic metric + control energy + smoothness penalty + state barriers + lambda sparsity + lambda Barrier
    loss = torch.sum(lamk * (phik_normed - ck_normed)**2) \
       + 0.001 * torch.mean(u[:, :2]**2) \
       + 0.001 * torch.sum((u[1:, :2] - u[:-1, :2])**2) \
       + 10 * torch.sum(torch.clamp_min(tr - 1, 0)**2 + torch.clamp_min(-tr, 0)**2) \
       + 0.001 * torch.sum(torch.abs(lam)) \
       + 10 * torch.sum(torch.clamp_min(lam - 1, 0)**2 + torch.clamp_min(-lam, 0)**2)
    loss += reward_term
    # l1 cost to promote sparsity
    return loss


def optimize_trajectory(x0, phik, k, info_map, num_iters=1500, lr=0.001):
    u = (0.05 * torch.randn((100, 3))).requires_grad_()
    optimizer = torch.optim.LBFGS([u], lr=lr, max_iter=20, history_size=10)
    for i in range(num_iters):
        def closure():
            optimizer.zero_grad()
            loss = fourier_ergodic_loss(u, x0, phik, k, info_map)
            if torch.isnan(loss):
                print("loss is NaN")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([u], max_norm=0.1)
            with torch.no_grad():
                u.clamp_(min=-2.0, max=2.0)
            return loss
        loss = optimizer.step(closure)
    
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

def phik_from_map(map_flattened, sample, k):
    fk_vals = torch.cos(sample[:, None, :] * k[None, :, :]).prod(dim=-1)
    return (fk_vals * map_flattened[:, None]).sum(dim=0) / (map_flattened.sum() + 1e-8)



# Get map information and set them into a list
# Change the maps path accordingly
entropy_maps_path = "/Users/cindy/Desktop/ergodic-search/entropy_maps"
# entropy_maps = (load_files(file) for file in (get_files_in_folder(entropy_maps_path)))
entropy_maps = []
for file in (get_files_in_folder(entropy_maps_path)):
    entropy_file = load_files(file)
    if entropy_file != None:
        entropy_maps.append(entropy_file)
gaussian_maps_path = "/Users/cindy/Desktop/ergodic-search/gaussian_maps"
# gaussian_maps = (load_files(file) for file in (get_files_in_folder(gaussian_maps_path)))
gaussian_maps = []
for file in (get_files_in_folder(gaussian_maps_path)):
    gaussian_file = load_files(file)
    if gaussian_file != None:
        gaussian_maps.append(gaussian_file)
full_maps = torch.stack(entropy_maps, dim=0)
maps = full_maps[:3]

N, H, W = maps.shape
xs = torch.linspace(0, 1, W)
ys = torch.linspace(0, 1, H)
X, Y = torch.meshgrid(xs, ys, indexing='xy')   
_s = torch.stack([ X.reshape(-1), Y.reshape(-1) ], dim=1)

_s_tensor = _s.clone().detach().float()
Z = torch.stack([p(s) for s in _s_tensor]).reshape(X.shape).detach().numpy()

# Compute the Fourier basis modes and the orthonormalization factors

x0 = torch.tensor([0.54, 0.3])
k1, k2 = np.meshgrid(np.arange(0, 20), np.arange(0, 20))
k = torch.tensor(np.stack([k1.ravel(), k2.ravel()], axis=-1), dtype=torch.float32)
lamk = torch.exp(-0.4 * (torch.norm(k, dim=1) ** 2))
hk = torch.tensor([get_hk(ki) for ki in k.numpy()])
fk_vals_all = torch.cos(_s[:, None, :] * k[None, :, :]).prod(dim=-1)




for i, info_map in enumerate(maps):
    info_map = info_map / (info_map.max() + 1e-8)
    phik = phik_from_map(info_map.flatten(), _s, k)
    optimized_u = optimize_trajectory(x0, phik, k, info_map)
    phik_recon = torch.matmul(fk_vals_all, phik).reshape(H, W)
    x = x0.clone()
    tr = [x]
    for step in optimized_u:
        x, _ = f(x, step[:2])
        tr.append(x)
        
    tr = torch.stack(tr).cpu().detach().numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(phik_recon.numpy(), extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    plt.contourf(X.numpy(), Y.numpy(), phik_recon.numpy(), cmap='viridis')
    plt.scatter(tr[1:, 0], tr[1:, 1], s=10, c = 5 * torch.sigmoid(5 * optimized_u[:, 2]), cmap='plasma')
    plt.title("Information Map")
    plt.figure(figsize=(3, 3))
    plt.imshow(phik_recon.numpy(), extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    plt.scatter(tr[1:, 0], tr[1:, 1], s=5, c='red')
    plt.scatter([x0[0]], [x0[1]], c='w', s=50, marker='X')
    plt.title("Original Map and Trajectory")
    plt.axis('square')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.draw()

plt.show()

