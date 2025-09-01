#!/usr/bin/env python
# coding: utf-8

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import random


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

# ergodic metric + other costs----
def fourier_ergodic_loss(u, x0, phik, k_expanded, lamk, hk, info_map, tau=None, head_w=1.0, tail_w=0.25):
    displacements = 0.1 * u[:, :2]
    tr = torch.cumsum(displacements, dim=0) + x0
    tr = tr.clamp(0.0, 1.0)

    T = u.shape[0]
    lam = torch.clamp(torch.sigmoid(5 * u[:, 2]), 0.05, 1.0)

    # determine the weights of head and tail for the amount of contributions to the trajectory
    w = torch.ones(T, device=u.device)
    if tau is not None:
        w[:tau] = head_w
        w[tau:] = tail_w

    # add exponetial discount
    disc = torch.exp(-2.0 * torch.arange(T, device=u.device, dtype=torch.float32) / T)
    w = w * disc

    ck = get_ck_weighted(tr[:, :2], k_expanded, lam * w, hk)

    phik_normed = (phik - phik.mean()) / (phik.std() + 1e-6)
    ck_normed   = (ck   - ck.mean())   / (ck.std()   + 1e-6)

    H, W = info_map.shape
    ix = (tr[:, 0] * (W - 1)).long().clamp(0, W - 1)
    iy = (tr[:, 1] * (H - 1)).long().clamp(0, H - 1)
    info_values = info_map[iy, ix]

    # emphasize head in the along-path reward too
    reward_term = -0.30 * torch.sum(w * info_values**3)

    loss = torch.sum(lamk * (phik_normed - ck_normed)**2) \
            + 0.001 * torch.mean(u[:, :2]**2) \
            + 0.001 * torch.sum((u[1:, :2] - u[:-1, :2])**2) \
            + 10 * torch.sum(torch.clamp_min(tr - 1, 0)**2 + torch.clamp_min(-tr, 0)**2) \
            + 0.0001 * torch.sum(torch.abs(lam)) \
            + 10 * torch.sum(torch.clamp_min(lam - 1, 0)**2 + torch.clamp_min(-lam, 0)**2) \
            + reward_term
    return loss


def optimize_trajectory(x0, phik, k_expanded, lamk, hk, info_map, u_prev=None, T = 100, tau = 9, num_iters=1500, lr=1e-3):
    # u = torch.empty((T, 3), dtype=torch.float32)
    # u[:, :2].normal_(mean=0.0, std=0.01)
    # u[:, 2].uniform_(-0.5, 0.5)
    # u.requires_grad_()

    # If u_prev exists, warm-start from that. If not, random-init. Normally, only the inital trajectory will be None
    # if u_prev is None:
    #     u = torch.empty((T, 3), dtype=torch.float32)
    #     u[:, :2].normal_(mean=0.0, std=0.01)
    #     u[:, 2].uniform_(-0.5, 0.5)
    #     u.requires_grad_()
    #     print("u for None", u.shape)
    # else:
    #     # u_prev is u from the previous trajectory, starting from the defined time step
    #     tau = 9
    #     u_add = torch.empty((tau, 3), dtype=torch.float32)
    #     u_add[:, :2].normal_(mean=0.0, std=0.01)
    #     u_add[:, 2].uniform_(-0.5, 0.5)
    #     u = torch.cat([u_prev, u_add], dim = 0)
    #     u = u.clone().detach().requires_grad_()
    #     print("u for else", u.shape)
    
    # ITOMP style replanning loop
    # If u_prev exists, warm-start from that. If not, random-init. Normally, only the inital trajectory will be None
    if u_prev is None:
        tail = torch.empty((T - tau, 3)).normal_(mean=0.0, std=0.01)
        tail[:, 2].uniform_(-0.5, 0.5)
        head = torch.empty((tau, 3)).normal_(mean=0.0, std=0.01)
        head[:, 2].uniform_(-0.5, 0.5)
    else:
        tail_prev = u_prev.detach().to(device = x0.device, dtype = torch.float32)
        # print("tail_prev shape", tail_prev.shape)
        target_len = max(0, T - tau)
        # print("target_len shape", target_len)
        if tail_prev.shape[0] >= target_len:
            tail = tail_prev[:target_len]
            # print("tail_prev.shape[0] >= target_len")
            # print("tail shape", tail.shape)
            # print("first 20 tail prev", tail_prev[:20])
            # print("first 20 tail", tail[:20])
        else:
            padding = torch.empty((target_len - tail_prev.shape[0], 3)).normal_(mean = 0.0, std = 0.01)
            padding[:, 2].uniform_(-0.5, 0.5)
            tail = torch.cat([tail_prev, padding], dim = 0)
            # print("padding on")
            # print("padding shape", padding.shape)
            # print("tail shape", tail.shape)
        # hybrid head seed
        head = tail[:tau].clone()
        head[:,2].zero_()
        head += 0.05 * torch.randn_like(head) # set noise, if big change in map -> crank up the noise
    head = torch.nn.Parameter(head)

    # LBFGS optimizer
    optimizer = torch.optim.LBFGS([head], lr=lr, max_iter=30, history_size=10)

    def u_builder(head, tail):
        if tail is None:
            return head
        else:
            return torch.cat([head, tail], dim = 0)

    for i in range(num_iters):
        def closure():
            optimizer.zero_grad()
            u = u_builder(head, tail)
            max_val = torch.max(info_map)
            jy, ix = torch.where(info_map == max_val)
            gx = ix.float().mean() / max(W - 1, 1) 
            gy = jy.float().mean() / max(H - 1, 1)
            goal = torch.tensor([gx, gy], dtype=torch.float32, device=u.device)
            loss = loss_with_goal(u, x0, phik, k_expanded, lamk, hk, info_map, tau=tau, head_w=1.0, tail_w=0.10, goal = goal, goal_w = 2.0)
            loss.backward()
            assert head.grad is not None and head.grad.abs().mean().item() > 0, "Head not receiving gradients."
            assert not (tail is not None and tail.requires_grad), "Tail should be frozen."
            torch.nn.utils.clip_grad_norm_([head], max_norm=0.1)
            return loss
        # if (i+1) % 100 == 0:
        #     if not torch.isfinite(loss).all().item():
        #         print("Warning: non-finite loss encountered.")
        loss = optimizer.step(closure)
        with torch.no_grad():
                head[:, :2].clamp_(min = -1.0, max = 1.0)
                head[:, 2].clamp_(min = -1.0, max = 1.0)
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
        u = u_builder(head, tail).detach()
        if u.shape[0] != T:
            if u.shape[0] > T:
                u = u[:T]
            else:
                padding = torch.zeros((T - u.shape[0], 3), device = u.device, dtype = torch.float32)
                padding[:, :] = u[-1, :].detach()
                u = torch.cat([u, padding], dim = 0)
    return u

def check_itomp_consistency(u_prev, u_opt, T = 100, tau = 9):
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

# combines ergodic loss over phik with a terminal goal penalty
def loss_with_goal(u, x0, phik, k_expanded, lamk, hk, info_map, tau=None, head_w=1.0, tail_w=0.25, goal=None, goal_w=0.0):
    lambda_erg = fourier_ergodic_loss(u, x0, phik, k_expanded, lamk, hk, info_map, tau=tau, head_w=head_w, tail_w=tail_w)
    # rollout trajectory to get final state x_T
    x = x0.clone()
    for step in u:
        x, _ = f(x, step[:2])
    term = 0.0
    if goal is not None and goal_w > 0:
        term = goal_w * (x - goal).pow(2).sum()
    return lambda_erg + term

# returns trajectory as list of states
def replanning(maps, _s, k_expanded, lamk, num_iters=1500):
    u_prev = None
    # phik_prev = None
    x0 = torch.tensor([0.54, 0.3])
    N, H, W = maps.shape
    tau = 30
    fk_vals_all = torch.cos(_s[:, None, :] * k_expanded).prod(dim=-1)
    # assert sample.shape[0] == H*W, f"expected {H*W} samples but got {sample.shape[0]}"

    # Each new plan with the tail of the previous u
    for i, info_map in enumerate(maps):
        info_map = info_map / (info_map.max() + 1e-8)
        print(f"\n=== cycle {i} ===")
        phik = phik_from_map(info_map.flatten(), _s, k_expanded)
        # if phik_prev is not None:
        #     print("change in phik", torch.norm(phik - phik_prev).item())
        # phik_prev = phik.detach()
        phik_recon = torch.matmul(fk_vals_all, phik).reshape(H, W)
        # max_val = torch.max(info_map)
        # jy, ix = torch.where(info_map == max_val)
        # gx = ix.float().mean() / max(W - 1, 1)
        # gy = jy.float().mean() / max(H - 1, 1)
        # goal = torch.tensor([gx, gy], dtype=x0.dtype, device=x0.device)
        # progress = i / float(N-1)
        # lam_goal_i = lambda_goal * (0.2 + 0.8*progress)
        # head of length tau is re-optimized each cycle
        # draw the remaining trajectory from the current position
        # if u_prev is None:
        #     # full-horizon initial plan
        #     u_optimized = optimize_trajectory(x0, phik, k, lamk, u_init = u_prev, num_iters = num_iters,
        #                                       loss_fn = lambda u_, x0_: loss_with_goal(u_, x0_, phik, k, lamk, lam_goal_i))
        # else:
        #     # sliding-window on head + untouched tail
        #     head = u_prev[:tau].clone().detach().requires_grad_()
        #     tail = u_prev[tau:]
        #     u_opt_seed  = optimize_trajectory(x0, phik, k, lamk, u_init = head ,num_iters = num_iters,
        #                                       loss_fn = lambda u_, x0_: loss_with_goal(u_, x0_, phik, k, lamk, lam_goal_i))
        #     u_optimized = torch.cat([u_opt_seed, tail], dim=0)
        u_prev_initial = u_prev
        u_optimized = optimize_trajectory(x0, phik, k_expanded, lamk, hk, info_map, u_prev = u_prev, tau = tau, num_iters=num_iters)
        check_itomp_consistency(u_prev_initial, u_optimized, T = 100, tau = tau)
        # Time-budget the optimizer: 
        # interrupt optimize_trajectory after a fixed number of iterations or wall-clock delta
        # then execute the first tau steps
        x = x0.clone()
        executed_traj = [x]
        for step in u_optimized:
            x, _ = f(x ,step[:2])
            executed_traj.append(x)
        executed_traj = torch.stack(executed_traj).cpu().detach().numpy()
        # for next planning step, start from the next 10 time step from the previous trajectory
        x_prev = x0.clone()
        x0 = torch.tensor(executed_traj[tau, :], dtype=torch.float32)
        u_prev = u_optimized[tau:].clone().detach().requires_grad_()
        # if remaining.numel()==0:
        #     u_prev = None
        # else:
        #     u_prev = remaining.clone().detach().requires_grad_()
        # print("u_prev 2", u_prev)
        # with torch.no_grad():
        #     lam = torch.clamp(torch.sigmoid(10 * u_optimized[:, 2]), 0.05, 1.0)
        #     print(f"  Lam min/max/mean: {lam.min():.3f}/{lam.max():.3f}/{lam.mean():.3f}")

        assert x0.shape == torch.Size([2])
        assert u_prev.shape[0] == u_optimized.shape[0] - tau

        # plt.figure(figsize=(3, 3))
        # plt.imshow(phik_recon.numpy(), extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
        # plt.contourf(X.numpy(), Y.numpy(), phik_recon.numpy(), cmap='viridis')
        # plt.scatter(executed_traj[1:, 0], executed_traj[1:, 1], s=10, c = 5 * torch.sigmoid(5 * u_optimized[:, 2]), cmap='plasma')
        # plt.title("Information Map")

        pts = executed_traj[1:, :2]
        T = pts.shape[0]
        sigma = 6.0
        lam = torch.sigmoid(sigma * u_optimized[:, 2].detach())
        ix = (pts[:, 0] * (W - 1)).astype(int).clip(0, W - 1)
        iy = (pts[:, 1] * (H - 1)).astype(int).clip(0, H - 1)
        info_vals = info_map[iy, ix]

        info_norm = (info_vals - info_vals.min()) / (info_vals.max() - info_vals.min() + 1e-8)
        K = 25
        score = (lam.cpu().numpy() * info_norm.cpu().numpy())
        topk_idx = np.argpartition(score, -K)[-K:]
        topk_idx = topk_idx[np.argsort(score[topk_idx])[::-1]]

        plt.figure(figsize=(3,3))
        plt.contourf(X.numpy(), Y.numpy(), phik_recon.numpy(), cmap='viridis')
        plt.scatter(pts[:,0], pts[:,1], s=10, c='white')
        plt.scatter(pts[topk_idx,0], pts[topk_idx,1], s=15, c='red')
        plt.title("RED")

        plt.figure(figsize=(3, 3))
        plt.imshow(phik_recon.numpy(), extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
        plt.contourf(X.numpy(), Y.numpy(), phik_recon.numpy(), cmap='viridis')
        plt.scatter(executed_traj[1:tau, 0], executed_traj[1:tau, 1], s=5, c='white')
        plt.scatter(executed_traj[tau:, 0], executed_traj[tau:, 1], s=5, c='red')
        plt.scatter([x_prev[0]], [x_prev[1]], c='w', s=50, marker='X')
        plt.title("Original Map and Trajectory")
        plt.axis('square')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.draw()
    return executed_traj

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
maps = full_maps[perm[:3]]
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


trajectory = replanning(maps, _s, k_expanded, lamk)
plt.show()
# pts = trajectory.cpu().numpy()
# plt.plot(pts[:,0], pts[:,1], '-o', markersize=3)