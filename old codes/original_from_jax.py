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
            # xf = x0.clone()
            # tr = [xf]
            # for step in u.detach():
            #     xf, _ = f(xf, step)
            #     tr.append(xf)
            # tr = torch.stack(tr).numpy()
            # plt.scatter(tr[:-1, 0], tr[:-1, 1], c=u.detach()[:, 2].numpy(), cmap='plasma')
            # plt.title("Trajectory Summary")
        #     plt.show()
    
    return u.detach()

start_time = time.time()

# Load a single map from the entropy_maps folder
entropy_maps_path = "/Users/cindy/Desktop/ergodic-search/entropy_maps"
map_files = [f for f in os.listdir(entropy_maps_path) if f.endswith('.npy')]
if not map_files:
    raise FileNotFoundError("No .npy files found in entropy_maps directory")

# Select a map (e.g., the first one or random)
map_to_load = os.path.join(entropy_maps_path, random.choice(map_files))
print(f"Loading map: {map_to_load}")
loaded_map = np.load(map_to_load)
map_tensor = torch.from_numpy(loaded_map).float()
H, W = map_tensor.shape

# Match the grid to the loaded map resolution
# Use indexing='xy' to ensure X varies along columns, Y along rows (standard Cartesian)
X, Y = torch.meshgrid(torch.linspace(0, 1, W, dtype=torch.float32), torch.linspace(0, 1, H, dtype=torch.float32), indexing='xy')
_s = torch.stack([X.ravel(), Y.ravel()]).T

# # Visualization of target distribution
# Z = map_tensor.detach().numpy()
# plt.imshow(Z, extent=[0, 1, 0, 1], origin='lower')
# plt.colorbar()
# plt.title("Target Distribution (Entropy Map)")
# plt.show()


# Compute the Fourier basis modes and the orthonormalization factors

x0 = torch.tensor([0.54, 0.3])
k1, k2 = np.meshgrid(np.arange(0, 20), np.arange(0, 20))
k = torch.tensor(np.stack([k1.ravel(), k2.ravel()], axis=-1), dtype=torch.float32)
lamk = torch.exp(-0.8 * torch.norm(k, dim=1))
hk = torch.tensor([get_hk(ki) for ki in k.numpy()])


# phik, Fourier coefficients of the target distribution.
# Efficient vectorized computation for the map data
fk_vals = torch.cos(_s[:, None, :] * k[None, :, :]).prod(dim=-1)
# Old version (without normalization):
# phik = (fk_vals * map_tensor.ravel()[:, None]).sum(dim=0)
# New version (normalized by grid count to approximate the integral - discrete -> continuous):
num_pixels = H * W
phik = (fk_vals * map_tensor.ravel()[:, None]).sum(dim=0) / num_pixels
phik_1 = phik / (phik[0] + 1e-8)
# Old version (without normalization):
phik_recon = torch.matmul(fk_vals, phik_1).reshape(W, H)
optimized_u = optimize_trajectory(x0, phik_1, k)


x = x0.clone()
tr = [x]
for step in optimized_u:
    x, _ = f(x, step[:2])
    tr.append(x)
tr = torch.stack(tr).cpu().detach().numpy()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

plt.figure(figsize=(4, 4))
# plt.imshow(phik_recon.numpy(), extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
plt.contourf(X.numpy(), Y.numpy(), phik_recon.numpy(), cmap='viridis')
plt.scatter(tr[1:, 0], tr[1:, 1], s=10, c = 5 * sigmoid(5 * optimized_u[:, 2]), cmap='plasma')
plt.title("Information Map")
# color = sigmoid(10 * optimized_u[:, 2]).detach().numpy()
plt.figure(figsize=(3, 3))
plt.imshow(phik_recon.numpy(), extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
plt.scatter(tr[1:, 0], tr[1:, 1], s=5, c='red')  # smaller red dots
plt.title("Original Map and Trajectory")

# color = len(optimized_u[:,2])*[0]
# color = torch.tensor(color, dtype=torch.float32, device=k.device)
# ck = get_ck_weighted(tr, k, color, hk)
# ck_recon = (fk_vals_all @ ck).reshape(X.shape).cpu().numpy()
# #ck_recon = ck_recon/ck_recon.sum()
# N = 25
# percent = 0.00000001
# color = torch.tensor(color, dtype=torch.float32, device=k.device)
# idx = sorted(range(len(optimized_u[:,2])), key = lambda sub: optimized_u[sub,2])[-25:]
# print(idx)
# for i in range(100):
#     if i in idx:
#         color[i] = 1
#     else:
#         color[i] = 0
# print(len(color))
# color[-1] = 0
# tr = torch.tensor(tr, dtype=torch.float32, device=k.device)
# tr = tr[1:, :2]
# # ck = get_ck_weighted(tr, k, color, hk)
# # ck_recon = (fk_vals_all @ ck).reshape(X.shape).cpu().numpy()
# plt.figure(figsize=(4,4))
# plt.contourf(X.numpy(), Y.numpy(), phik_recon.numpy(), cmap='viridis')
# plt.scatter(tr[:,0],tr[:,1], c=color, cmap = 'Reds')
# plt.title("RED")

# print("optimized_u[:, 2]", optimized_u[:, 2])
lam = sigmoid(5 * optimized_u[:, 2])
# lam = torch.clamp(sigmoid(5 * optimized_u[:, 2]), 0.05, 1.0).cpu().numpy()
# print("lam")
# print(lam)
# print(len(lam), "____")
red_idx = np.where(lam > 0.066)[0]
# print("red_idx", red_idx)
plt.figure(figsize=(4,4))
plt.contourf(X.numpy(), Y.numpy(), phik_recon.numpy(), cmap='viridis')
pts=tr[1:]
plt.scatter(pts[:, 0], pts[:, 1], s=10, c='white')
plt.scatter(pts[red_idx, 0], pts[red_idx, 1], s=10, c='red')
plt.title("Red and White")


plt.axis('square')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
