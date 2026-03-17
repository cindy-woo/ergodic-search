#!/usr/bin/env python
# coding: utf-8

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Dynamics and target distribution
# The dynamics are defined as the constrained continuous time dynamical system
# The target distribution, for which information is distributed within the continuous search space
@torch.jit.script
def p(x):
    return torch.exp(-190.5 * torch.sum((x[:2] - 0.27)**2)) \
           + torch.exp(-180.5 * torch.sum((x[:2] - 0.75)**2))

# dynamics using discrete time eulerintegration
@torch.jit.script
def f(x, u):
    xnew = x[:2] + 0.1 * u[:2]
    xnew = torch.clamp(xnew, 0, 1)
    return xnew, xnew


# Helper Functions
# Define an orthonormalizing factor h_k for the ergodic metric
# normalizing factor for basis function
def get_hk(k): 
    _hk = (2. * k + np.sin(2 * k)) / (4 * k + 1e-8)
    _hk[np.isnan(_hk)] = 1.
    return np.sqrt(np.prod(_hk))


# # Ergodic metric and sample-weighted ergodic metric
# def get_ck_weighted(tr, k_expanded, lam, hk):
#     weighted_fk = torch.prod(torch.cos(tr[:, None, :] * k_expanded), dim = -1).T @ lam
#     ck = weighted_fk / (hk + 1e-8)
#     return ck

def get_ck_weighted(tr, k_expanded, weights, hk):
    fk = torch.cos(tr[:, None, :] * k_expanded).prod(dim=-1)
    Z  = weights.sum() + 1e-8
    return (fk.T @ weights) / (Z * hk)


def fourier_ergodic_loss(u, x0, phik_normed, k_expanded, lamk, hk, info_map, w_vec, tau=None, goal_head=None, goal_head_w=1.0, reward_alpha=0.35, lam_sigma=5.0): 
    displacements = 0.1 * u[:, :2] 
    tr = torch.cumsum(displacements, dim=0) + x0 
    tr = tr.clamp(0.0, 1.0) 
    # sensor gating (per-timestep) 
    lam = torch.clamp(torch.sigmoid(lam_sigma * u[:, 2]), 0.05, 1.0) 
    weights = lam * w_vec 
    ck = get_ck_weighted(tr[:, :2], k_expanded, weights, hk) 
    ck_normed = (ck - ck.mean()) / (ck.std() + 1e-6)
    H, W = info_map.shape 
    ix = (tr[:, 0] * (W - 1)).long().clamp(0, W - 1) 
    iy = (tr[:, 1] * (H - 1)).long().clamp(0, H - 1) 
    info_values = info_map[iy, ix] 
    r_path = (info_values - info_values.mean()) / (info_values.std() + 1e-6) 
    reward_term = -reward_alpha * torch.sum(weights * r_path)
    head_term = 0.0 
    if tau is not None and goal_head is not None and goal_head_w > 0: 
        x_tau = tr[tau-1, :] 
        head_term = goal_head_w * (x_tau - goal_head).pow(2).sum() 
    box_term = 10 * torch.sum(torch.clamp_min(tr - 1, 0)**2 + torch.clamp_min(-tr, 0)**2) 
    loss = torch.sum(lamk * (phik_normed - ck_normed)**2) \
            + 0.001 * torch.mean(u[:, :2]**2) \
            + 0.001 * torch.sum((u[1:, :2] - u[:-1, :2])**2) \
            + 0.0001 * torch.sum(torch.abs(lam)) \
            + 10 * torch.sum(torch.clamp_min(lam - 1, 0)**2 + torch.clamp_min(-lam, 0)**2) \
            + box_term + reward_term + head_term
    return loss

def loss_with_goal(u, x0, phik_normed, k_expanded, lamk, hk, info_map, w_vec, tau=None, goal=None, goal_w=0.0):
    lambda_erg = fourier_ergodic_loss(u, x0, phik_normed, k_expanded, lamk, hk, info_map, w_vec, tau=tau, goal_head=goal)
    # rollout trajectory to get final state x_T
    x = x0.clone()
    for step in u:
        x, _ = f(x, step[:2])
    term = 0.0
    if goal is not None and goal_w > 0:
        term = goal_w * (x - goal).pow(2).sum()
    return lambda_erg + term

def optimize_trajectory(x0, phik, k_expanded, lamk, hk, info_map,
                        u_prev=None, T=100, tau=20, num_iters=1500, lr=1e-3):
    device = x0.device
    # ITOMP style replanning loop
    # If u_prev exists, warm-start from that. If not, random-init. Normally, only the inital trajectory will be None
    if u_prev is None:
        tail = torch.empty((T - tau, 3)).normal_(mean=0.0, std=0.01)
        tail[:, 2].uniform_(-0.5, 0.5)
        head = torch.empty((tau, 3)).normal_(mean=0.0, std=0.01)
        head[:, 2].uniform_(-0.5, 0.5)
    else:
        tail_prev = u_prev.detach().to(device=device, dtype=torch.float32)
        target_len = max(0, T - tau)
        if tail_prev.shape[0] >= target_len:
            tail = tail_prev[:target_len]
        else:
            padding = torch.empty((target_len - tail_prev.shape[0], 3)).normal_(mean=0.0, std=0.01)
            padding[:, 2].uniform_(-0.5, 0.5)
            tail = torch.cat([tail_prev, padding], dim=0)
        # hybrid head seed
        head = tail[:tau].clone()
        head[:, 2].zero_()
        head += 0.05 * torch.randn_like(head)
    head = torch.nn.Parameter(head)

    w_vec = make_w_vec(T, tau)
    phik_normed = (phik - phik.mean()) / (phik.std() + 1e-6)

    max_val = torch.max(info_map)
    jy, ix = torch.where(info_map == max_val)
    H, W = info_map.shape
    gx = ix.float().mean() / max(W - 1, 1)
    gy = jy.float().mean() / max(H - 1, 1)
    goal_head = torch.tensor([gx, gy], dtype=torch.float32, device=device)

    optimizer = torch.optim.LBFGS([head], lr=lr, max_iter=20, history_size=10)

    def u_builder(head, tail):
        return head if tail is None else torch.cat([head, tail], dim=0)
    def closure():
            optimizer.zero_grad(set_to_none=True)
            u = u_builder(head, tail)
            loss = loss_with_goal(u, x0, phik_normed, k_expanded, lamk, hk, info_map, w_vec, tau=tau, goal=goal_head, goal_w=2.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([head], max_norm=0.1)
            return loss
    for i in range(num_iters):
        loss = optimizer.step(closure)
        with torch.no_grad():
            head[:, :2].clamp_(min=-1.0, max=1.0)
            head[:, 2].clamp_(min=-1.0, max=1.0)
        if (i+1) % 20 == 0:
            print(f"Iteration {i+1}, Loss: {loss.item()}")

    u = u_builder(head, tail).detach()
    if u.shape[0] != T:
        if u.shape[0] > T:
            u = u[:T]
        else:
            padding = torch.zeros((T - u.shape[0], 3), device=u.device, dtype=torch.float32)
            padding[:, :] = u[-1, :].detach()
            u = torch.cat([u, padding], dim=0)
    return u

def check_itomp_consistency(u_prev, u_opt, T = 100, tau = 20):
    target_len = max(0, T - tau)
    if u_prev is None:
        return  # first cycle

    copied = u_opt[tau:tau+target_len]
    want   = u_prev[:target_len].detach()
    if not torch.allclose(copied, want, atol=1e-5, rtol=1e-5):
        print("Tail mismatch with previous u_prev[:T-tau]")
        print("copied - want", torch.norm(copied - want).item())

# compute phik from the information map
# input phik_map flattens values from the most recent information map from sample value, _s
def phik_from_map(map_flattened, sample, k_expanded):
    fk_vals = torch.cos(sample[:, None, :] * k_expanded).prod(dim=-1)
    return (fk_vals * map_flattened[:, None]).sum(dim=0) / (map_flattened.sum() + 1e-8)

def make_w_vec(T, tau, head_w=2.0, tail_w=0.10, beta=2.0, floor=0.05):
    w = torch.full((T,), tail_w, dtype=torch.float32)
    w[:tau] = head_w
    w *= torch.exp(-beta * torch.arange(T, dtype=torch.float32) / T)
    w = torch.clamp(w, min=floor * tail_w)
    return w

# returns trajectory as list of states
def replanning(maps, _s, k_expanded, lamk):
    u_prev = None
    x0 = torch.tensor([0.54, 0.3])
    N, H, W = maps.shape
    tau = 20

    trajectories, k_idx, phik_list = [], [], []

    for i, info_map in enumerate(maps):
        info_map = info_map / (info_map.max() + 1e-8)
        print(f"\n=== cycle {i} ===")
        # phik per map (safe to precompute per cycle)
        phik = phik_from_map(info_map.flatten(), _s, k_expanded)
        phik_list.append(phik.detach()) 

        u_optimized = optimize_trajectory(x0, phik, k_expanded, lamk, hk, info_map, u_prev = u_prev)

        # rollout for visualization & next-cycle seed
        x = x0.clone()
        executed_traj = [x]
        for step in u_optimized:
            x, _ = f(x, step[:2])
            executed_traj.append(x)
        executed_traj = torch.stack(executed_traj).cpu().detach().numpy()
        trajectories.append(executed_traj)

        x0 = torch.tensor(executed_traj[tau, :], dtype=torch.float32)
        u_prev = u_optimized[tau:].clone().detach().requires_grad_()

        # RED indices (unchanged)
        pts = executed_traj[1:, :2]
        sigma = 6.0
        lam = torch.sigmoid(sigma * u_optimized[:, 2].detach())
        ix = (pts[:, 0] * (W - 1)).astype(int).clip(0, W - 1)
        iy = (pts[:, 1] * (H - 1)).astype(int).clip(0, H - 1)
        info_vals = info_map[iy, ix]
        info_norm = (info_vals - info_vals.min()) / (info_vals.max() - info_vals.min() + 1e-8)
        score = (lam.cpu().numpy() * info_norm.cpu().numpy())
        K = 25
        topk_idx = np.argpartition(score, -K)[-K:]
        topk_idx = topk_idx[np.argsort(score[topk_idx])[::-1]]
        k_idx.append(topk_idx)

    return trajectories, k_idx, phik_list

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
# Compute the Fourier basis modes and the orthonormalization factors

# Get map information and set them into a list
# Change the maps path accordingly
entropy_maps_path = "/Users/cindy/Desktop/ergodic-search/entropy_maps"
entropy_maps = []
for file in (get_files_in_folder(entropy_maps_path)):
    entropy_file = load_files(file)
    if entropy_file != None:
        entropy_maps.append(entropy_file)
gaussian_maps_path = "/Users/cindy/Desktop/ergodic-search/gaussian_maps"
gaussian_maps = []
for file in (get_files_in_folder(gaussian_maps_path)):
    gaussian_file = load_files(file)
    if gaussian_file != None:
        gaussian_maps.append(gaussian_file)
full_maps = torch.stack(entropy_maps, dim=0)
perm = torch.randperm(full_maps.shape[0])
maps = full_maps[perm[:1]]
# all_maps = torch.stack(entropy_maps + gaussian_maps, dim=0)

# Sample grid to match the resolution
N, H, W = maps.shape
xs = torch.linspace(0, 1, W)
ys = torch.linspace(0, 1, H)
X, Y = torch.meshgrid(xs, ys, indexing='xy')   
_s = torch.stack([ X.reshape(-1), Y.reshape(-1) ], dim=1)


k1, k2 = np.meshgrid(np.arange(0, 20), np.arange(0, 20))
k = torch.tensor(np.stack([k1.ravel(), k2.ravel()], axis=-1), dtype=torch.float32)
k_expanded = k.unsqueeze(0)
lamk = torch.exp(-0.15 * (torch.norm(k, dim=1) ** 2))
hk = torch.clamp(torch.tensor([get_hk(ki) for ki in k.numpy()]), min=1e-6)
fk_vals_all = torch.cos(_s[:, None, :] * k_expanded).prod(dim=-1)
tau = 20

trajectory, k_idx, phik_list= replanning(maps, _s, k_expanded, lamk)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

pts = []
for t in trajectory:
    pts.append(t[1:, :2])

for i in range(len(trajectory)):
    tr = trajectory[i]
    pts_i = tr[1:, :2]
    topk = k_idx[i] 
    phik = phik_list[i]    

    phik_recon = torch.matmul(fk_vals_all, phik).reshape(H, W)

    plt.figure(figsize=(3, 3))
    plt.contourf(X.numpy(), Y.numpy(), phik_recon, cmap='viridis')
    plt.scatter(pts_i[:, 0], pts_i[:, 1], s=10, c='white')
    plt.scatter(pts_i[topk, 0], pts_i[topk, 1], s=15, c='red')
    plt.title("RED")

    tau = 20
    plt.figure(figsize=(3, 3))
    plt.imshow(phik_recon, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    plt.contourf(X.numpy(), Y.numpy(), phik_recon, cmap='viridis')
    plt.scatter(tr[1:tau, 0], tr[1:tau, 1], s=5, c='white')
    plt.scatter(tr[tau:, 0], tr[tau:, 1], s=5, c='red') 
    plt.scatter(tr[0, 0], tr[0, 1], c='w', s=50, marker='X')
    plt.title("Original Map and Trajectory")
    plt.axis('square')
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.tight_layout()
    plt.draw()

plt.show()


plt.show()
# pts = trajectory.cpu().numpy()
# plt.plot(pts[:,0], pts[:,1], '-o', markersize=3)