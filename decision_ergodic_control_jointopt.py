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

    lam = torch.clamp(sigmoid(10 * u[:, 2]), 0.05, 1.0)
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
            # plt.show()
    
    return u.detach()


# Visualization of target distribution
X, Y = torch.meshgrid(torch.linspace(0, 1, 50, dtype=torch.float32), torch.linspace(0, 1, 50, dtype=torch.float32))
_s = torch.stack([X.ravel(), Y.ravel()]).T

# def p(tensor):
#     if tensor.dim() == 1:
#         return torch.sin(torch.norm(tensor))  # No dim argument for 1D tensors
#     else:
#         return torch.sin(torch.norm(tensor, dim=1))


_s_tensor = _s.clone().detach().float()
Z = torch.stack([p(s) for s in _s_tensor]).reshape(X.shape).detach().numpy()
plt.contour(X, Y, Z)
plt.axis('square')


# Compute the Fourier basis modes and the orthonormalization factors

x0 = torch.tensor([0.54, 0.3])
k1, k2 = np.meshgrid(np.arange(0, 20), np.arange(0, 20))
k = torch.tensor(np.stack([k1.ravel(), k2.ravel()], axis=-1), dtype=torch.float32)
lamk = torch.exp(-0.8 * torch.norm(k, dim=1))
hk = torch.tensor([get_hk(ki) for ki in k.numpy()])

X, Y = torch.meshgrid(
    torch.linspace(0, 1, 50, dtype=torch.float32),
    torch.linspace(0, 1, 50, dtype=torch.float32),
    indexing='ij')
_s = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=-1), dtype=torch.float32)
# phik, Fourier coefficients of the target distribution.
phik = (torch.cos(_s[:, None, :] * k).prod(dim=-1) * torch.stack([p(s) for s in _s])[:, None]).sum(dim=0)
phik_1 = phik.clone().detach().contiguous()
phik_1 = phik / (phik[0] + 1e-8)
fk_vals_all = torch.cos(_s[:, None, :] * k[None, :, :]).prod(dim=-1)
phik_recon = torch.matmul(fk_vals_all, phik_1).reshape(50, 50)
optimized_u = optimize_trajectory(x0, phik_1, k)
x = x0.clone()
tr = [x]
for step in optimized_u:
    x, _ = f(x, step[:2])
    tr.append(x)
tr = torch.stack(tr).cpu().detach().numpy()

plt.figure(figsize=(4, 4))
#plt.imshow(phik_recon.numpy(), extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
plt.contourf(X.numpy(), Y.numpy(), phik_recon.numpy(), cmap='viridis')
#plt.scatter(tr[1:, 0], tr[1:, 1], s=10, c = 5 * sigmoid(5 * optimized_u[:, 2]), cmap='plasma')
color = sigmoid(10 * optimized_u[:, 2]).detach().numpy()
plt.figure(figsize=(3, 3))
plt.imshow(phik_recon.numpy(), extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
plt.scatter(tr[1:, 0], tr[1:, 1], s=5, c='red')  # smaller red dots
plt.title("Original Map and Trajectory")
plt.axis('square')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
