#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad
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
@torch.jit.script
def p(x):
    return torch.exp(-190.5 * torch.sum((x[:2] - 0.27)**2)) \
           + torch.exp(-180.5 * torch.sum((x[:2] - 0.75)**2))

@torch.jit.script
def f(x, u):
    xnew = x[:2] + u[:2]
    xnew = torch.clamp(xnew, 0, 1)
    return xnew, xnew

# Helper Functions
# def sigmoid(x):
#     return 1 / (1 + torch.exp(-x))

def get_hk(k):
    _hk = (2. * k + np.sin(2 * k)) / (4 * k + 1e-8)
    _hk[np.isnan(_hk)] = 1.
    return np.sqrt(np.prod(_hk))

# def fk(x, k):
#     return torch.prod(torch.cos(x * k))

# def get_ck(tr, k):
#     ck = torch.mean(torch.cos(tr[:, None, :] * k[None, :, :]).prod(dim = -1), dim = 0)
#     ck = np.copy(ck) / hk
#     return ck

# def get_ck_weighted(tr, k, lam, hk):
#     weighted_fk = torch.prod(torch.cos(tr[:, None, :] * k[None, :, :]), dim = -1).T @ lam
#     ck = weighted_fk / (hk + 1e-8)
#     return ck

def get_ck_weighted(tr, k_expanded, weights, hk):
    fk = torch.cos(tr[:, None, :] * k_expanded).prod(dim=-1)
    Z  = weights.sum() + 1e-8
    return (fk.T @ weights) / (Z * hk)

def fourier_ergodic_loss(u, x0, phik, k, goal):
    displacements = 0.1 * u[:, :2]
    tr = torch.cumsum(displacements, dim=0) + x0
    tr = tr.clamp(0.0, 1.0)
    # for i in range(u.shape[0]):
    #     x_new, _ = f(tr[-1], u[i, :2])
    #     tr.append(x_new)
    # tr = torch.stack(tr[1:]) 

    lam = torch.clamp(torch.sigmoid(5 * u[:, 2]), 0.05, 1.0)
    ck = get_ck_weighted(tr[:,:2], k, lam, hk)
    phik_normed = (phik - phik.mean()) / (phik.std() + 1e-6)
    ck_normed   = (ck   - ck.mean())   / (ck.std()   + 1e-6)
    disc = torch.exp(-2.0 * torch.arange(u.shape[0], device=u.device, dtype=torch.float32) / u.shape[0])
    
    ix = (tr[0, 0] * (W - 1)).long().clamp(0, W - 1)
    iy = (tr[0, 1] * (H - 1)).long().clamp(0, H - 1)
    info_values = info_map[iy, ix]
    reward_term = -0.30 * torch.sum(disc * info_values**3)

    loss = torch.sum(lamk * (phik_normed - ck_normed)**2) \
            + 0.001 * torch.mean(u[:, :2]**2) \
            + 0.001 * torch.sum((u[1:, :2] - u[:-1, :2])**2) \
            + 10 * torch.sum(torch.clamp_min(tr - 1, 0)**2 + torch.clamp_min(-tr, 0)**2) \
            + 0.001 * torch.sum(torch.abs(lam)) \
            + 10 * torch.sum(torch.clamp_min(lam - 1, 0)**2 + torch.clamp_min(-lam, 0)**2) \
            + reward_term
    x = x0.clone()
    x = x0.clone()
    for step in u:
        x, _ = f(x, step[:2])
    term = 0.0
    if goal is not None and 2.5 > 0:
        term = 2.0 * (x - goal).pow(2).sum()
    return loss + term

def optimize_trajectory(x0, phik, k, num_iters=1500):
    u = torch.empty((100, 3)).normal_(mean=0.0, std=0.01)
    u[:, 2].uniform_(-0.5, 0.5)
    # u = (0.01 * torch.randn((100, 3))).requires_grad_()
    u = torch.nn.Parameter(u)
    optimizer = torch.optim.LBFGS([u], lr=1e-3, max_iter=20, history_size=10)
    
    def closure():
        optimizer.zero_grad()
        max_val = torch.max(info_map)
        jy, ix = torch.where(info_map == max_val)
        gx = ix.float().mean() / max(W - 1, 1) 
        gy = jy.float().mean() / max(H - 1, 1)
        goal = torch.tensor([gx, gy], dtype=torch.float32, device=u.device)
        loss = fourier_ergodic_loss(u, x0, phik, k, goal)
        # if torch.isnan(loss) or torch.isinf(loss):
        #     return torch.tensor(1e8, requires_grad=True, device=loss.device)  # Return large loss to avoid NaN
        loss.backward()
        torch.nn.utils.clip_grad_norm_([u], max_norm=0.1)  # Gradient clipping
        return loss
    
    for i in range(num_iters):
        loss = optimizer.step(closure)
        with torch.no_grad():
            u.clamp_(min = -1.0, max = 1.0)
        if (i+1) % 20 == 0:
            print(f"Iteration {i+1}, Loss: {loss.item()}")
    print("u", u)
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


entropy_maps_path = "/Users/cindy/Desktop/ergodic-search/entropy_maps"
entropy_maps = []
for file in (get_files_in_folder(entropy_maps_path)):
    entropy_file = load_files(file)
    if entropy_file != None:
        entropy_maps.append(entropy_file)
# gaussian_maps_path = "/Users/cindy/Desktop/ergodic-search/gaussian_maps"
# gaussian_maps = []
# for file in (get_files_in_folder(gaussian_maps_path)):
#     gaussian_file = load_files(file)
#     if gaussian_file != None:
#         gaussian_maps.append(gaussian_file)
full_maps = torch.stack(entropy_maps, dim=0)
perm = torch.randperm(full_maps.shape[0])
maps = full_maps[perm[:]]

N, H, W = maps.shape
xs = torch.linspace(0, 1, W)
ys = torch.linspace(0, 1, H)
X, Y = torch.meshgrid(xs, ys, indexing='xy')   
_s = torch.stack([ X.reshape(-1), Y.reshape(-1) ], dim=1)

x0 = torch.tensor([0.54, 0.3])
k1, k2 = np.meshgrid(np.arange(0, 20), np.arange(0, 20))
k = torch.tensor(np.stack([k1.ravel(), k2.ravel()], axis=-1), dtype=torch.float32)
k_expanded = k.unsqueeze(0)
lamk = torch.exp(-0.15 * (torch.norm(k, dim=1) ** 2))
hk = torch.clamp(torch.tensor([get_hk(ki) for ki in k.numpy()]), min=1e-6)

fk_vals = torch.cos(_s[:, None, :] * k_expanded).prod(dim=-1)

start_time = time.time()

info_map = maps[0]
info_map = info_map / (info_map.max() + 1e-8)
maps_flattened = info_map.flatten()
phik = (fk_vals * maps_flattened[:, None]).sum(dim=0) / (maps_flattened.sum() + 1e-8)
phik_recon = torch.matmul(fk_vals, phik).reshape(H, W)
optimized_u = optimize_trajectory(x0, phik, k)
print("optimized_u", optimized_u)
x = x0.clone()
tr = [x]
for step in optimized_u:
    x, _ = f(x, step[:2])
    tr.append(x)
tr = torch.stack(tr).cpu().detach().numpy()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

plt.figure(figsize=(3, 3))
plt.contourf(X.numpy(), Y.numpy(), phik_recon.numpy(), cmap='viridis')
# plt.scatter(tr[1:, 0], tr[1:, 1], s=10, c = 5 * torch.sigmoid(5 * optimized_u[:, 2]), cmap='plasma')
plt.scatter(tr[1:, 0], tr[1:, 1], s=10, c = 'white', cmap='plasma')
plt.title("Information Map")
# plt.figure(figsize=(3, 3))
# plt.imshow(phik_recon.numpy(), extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
# plt.scatter(tr[1:, 0], tr[1:, 1], s=5, c='red')
# plt.title("Original Map and Trajectory")
plt.axis('square')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()