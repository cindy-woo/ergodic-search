#!/usr/bin/env python
# coding: utf-8

import torch
import os
import numpy as np
import matplotlib.pyplot as plt


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


# Ergodic metric and sample-weighted ergodic metric
def get_ck_weighted(tr, k_expanded, lam, hk):
    weighted_fk = torch.prod(torch.cos(tr[:, None, :] * k_expanded), dim = -1).T @ lam
    ck = weighted_fk / (hk + 1e-8)
    return ck

# ergodic metric + other costs----
def fourier_ergodic_loss(u, x0, phik, k_expanded, lamk, hk, info_map):
    # xf = x0.clone()
    # tr = [xf]
    
    # for i in range(u.shape[0]):
    #     x_new, _ = f(tr[-1], u[i, :2])
    #     tr.append(x_new)
    # tr = torch.stack(tr[1:]) 

    displacements = 0.1 * u[:, :2]
    tr = torch.cumsum(displacements, dim=0) + x0
    tr = tr.clamp(0.0, 1.0)
    lam = torch.clamp(torch.sigmoid(5 * u[:, 2]), 0.05, 1.0)
    ck = get_ck_weighted(tr[:,:2], k_expanded, lam, hk)
    phik_normed = (phik - phik.mean()) / (phik.std() + 1e-6)
    ck_normed = (ck - ck.mean()) / (ck.std() + 1e-6)
    H, W = info_map.shape
    indices_x = (tr[:, 0] * (W - 1)).long().clamp(0, W - 1)
    indices_y = (tr[:, 1] * (H - 1)).long().clamp(0, H - 1)
    info_values = info_map[indices_y, indices_x]
    reward_term = -0.05 * torch.sum(info_values)
    # barrier cost to ensure that x(t) stays within 
    # barr_cost = 10 * torch.sum(torch.clamp_min(tr - 1, 0)**2 + torch.clamp_min(-tr, 0)**2)
    # lam_barr_cost = 10 * torch.sum(torch.clamp_min(lam - 1, 0)**2 + torch.clamp_min(-lam, 0)**2)
    # Total loss = ergodic metric + control energy + smoothness penalty + state barriers + lambda sparsity + lambda Barrier
    loss = torch.sum(lamk * (phik_normed - ck_normed)**2) \
        + 0.001 * torch.mean(u[:, :2]**2) \
        + 0.001 * torch.sum((u[1:, :2] - u[:-1, :2])**2) \
        + 10 * torch.sum(torch.clamp_min(tr - 1, 0)**2 + torch.clamp_min(-tr, 0)**2) \
        + 0.0001 * torch.sum(torch.abs(lam)) \
        + 10 * torch.sum(torch.clamp_min(lam - 1, 0)**2 + torch.clamp_min(-lam, 0)**2) \
        + reward_term
    # l1 cost to promote sparsity
    return loss


def optimize_trajectory(x0, phik, k_expanded, lamk, lambda_goal, hk, info_map, u_init=None, T = 100, num_iters=1500, lr=1e-3):
    # If u_init exists, warm-start from that. If not, random-init.
    if u_init is None:
        u = torch.empty((T, 3), dtype=torch.float32)
        u[:, :2].normal_(mean=0.0, std=0.01)
        u[:, 2].uniform_(-0.5, 0.5)
        u.requires_grad_()
    else:
        # clone & require grad so LBFFGS will update it
        u = u_init.clone().detach().requires_grad_()
    # LBFGS optimizer
    optimizer = torch.optim.LBFGS([u], lr=lr, max_iter=30, history_size=10)

    for i in range(num_iters):
        def closure():
            optimizer.zero_grad()
            loss = loss_with_goal(u, x0, phik, k_expanded, lamk, lambda_goal, hk, info_map)
            if torch.isnan(loss):
                print("loss is NaN")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([u], max_norm=0.1)
            with torch.no_grad():
                u.clamp_(min=-2.0, max=2.0)
            return loss
        loss = optimizer.step(closure)
        with torch.no_grad():
            u[:, :2].clamp_(-1.0, 1.0)
            u[:, 2].clamp_(-1.0, 1.0)
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

# def optimize_trajectory_stomp(x0, phik, k, num_iters=150, num_rollouts=20, noise_std=0.05, stomp_lambda=1.0):
#     T, dim = 100, 3     #time step, control per step
#     u = torch.zeros((T, dim), dtype=torch.float32)   #set a normal control trajectory
#     normal = Normal(0.0, noise_std)
#     for it in range(num_iters):
#         eps = normal.sample((num_rollouts, T, dim))   # sample K noise trajectories in shape (K, T, dim)
#         u_perturbed = u.unsqueeze(0) + eps   # form K perturbed rollouts: u_i = u + eps_i
#         # evaluate each rollout’s total cost C_i
#         costs = torch.zeros(num_rollouts)   # create costs in the size of num_rollouts
#         for i in range(num_rollouts):
#             #only use ergodic loss as C_task, C_task = C_i
#             C_task = fourier_ergodic_loss(u_perturbed[i], x0, phik, k)
#             #### add smoothness or collision costs if needed
#             # costs[i] = α * C_smooth + β * C_collision + γ * C_task
#             costs[i] = C_task
#         # compute STOMP weights: w_i ∝ exp(−C_i / λ)
#         weights = torch.softmax(-costs / stomp_lambda, dim=0)  
#         # update u_new = u_old + ∑ w_i ε_i
#         du = torch.sum(weights.view(-1, 1, 1) * eps, dim=0)
#         u = u + du
#         if (it + 1) % 10 == 0:
#             print(costs)
#             print(f"STOMP iteration:{it+1}/{num_iters}, cost:{costs.mean().item():.4f}")
#     return u

# compute phik from the information map
# input phik_map flattens values from the most recent information map from sample value, _s
def phik_from_map(map_flattened, sample, k_expanded):
    fk_vals = torch.cos(sample[:, None, :] * k_expanded).prod(dim=-1)
    return (fk_vals * map_flattened[:, None]).sum(dim=0) / (map_flattened.sum() + 1e-8)

# combines ergodic loss over phik with a terminal goal penalty
def loss_with_goal(u, x0, phik, k_expanded, lamk, lambda_goal, hk, info_map):
    lambda_erg = fourier_ergodic_loss(u, x0, phik, k_expanded, lamk, hk, info_map)
    # rollout trajectory to get final state x_T
    x = x0.clone()
    for step in u:
        x, _ = f(x, step[:2])
    return lambda_erg + lambda_goal


# ITOMP style replanning loop
# returns trajectory as list of states
def replanning(x0, x_goal, maps, _s, k_expanded, lamk, lambda_goal, num_iters=1500):
    u_prev = None
    N, H, W = maps.shape
    # assert sample.shape[0] == H*W, f"expected {H*W} samples but got {sample.shape[0]}"

    # Each new plan with the tail of the previous u
    for i, info_map in enumerate(maps):
        executed_traj = [x0.clone()]
        info_map = info_map / (info_map.max() + 1e-8)
        phik = phik_from_map(info_map.flatten(), _s, k_expanded)
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
        u_optimized = optimize_trajectory(x0, phik, k_expanded, lamk, lambda_goal, hk, info_map, u_init = u_prev, num_iters = num_iters)
        # Time-budget the optimizer: 
        # interrupt optimize_trajectory after a fixed number of iterations or wall-clock delta
        # then execute the first tau steps
        for step in u_optimized:
            x0, _ = f(x0 ,step[:2])
            executed_traj.append(x0)
            if torch.dist(x0, x_goal) < 1e-2:
                print(f"Goal reached at step {i}th {step}")
                break
        executed_traj = torch.stack(executed_traj).cpu().detach().numpy()
        # for next planning step
        exec_steps = u_optimized.size(0)
        remaining = u_optimized[exec_steps:]
        # pad_len = max(0, tau - remaining.size(0))
        # padding = torch.zeros((pad_len, 3), dtype=torch.float32)
        # u_prev = torch.cat([remaining, padding], dim=0)
        if remaining.numel()==0:
            u_prev = None
        else:
            u_prev = remaining.clone().detach().requires_grad_()

        with torch.no_grad():
            lam = torch.clamp(torch.sigmoid(10 * u_optimized[:, 2]), 0.05, 1.0)
            # print(f"Replanning #{i+1}: Executed {exec_steps} steps, Remaining {remaining.size(0)}, Pad {pad_len}")
            print(f"  Lam min/max/mean: {lam.min():.3f}/{lam.max():.3f}/{lam.mean():.3f}")

        full_lam = 5 * torch.sigmoid(5 * u_optimized[:, 2])
        N_pts   = executed_traj[1:].shape[0]
        fk_vals_all = torch.cos(k_expanded * _s[:, None, :]).prod(dim=-1)
        # phi_coeffs = (fk_vals_all * map_flat[:,None]).sum(0) / (map_flat.sum()+1e-8)
        # phi_recon = (fk_vals_all @ phi_coeffs).reshape(H, W).cpu().numpy()
        phik_recon = torch.matmul(fk_vals_all, phik).reshape(H, W)
        
        plt.figure(figsize=(4, 4))
        plt.imshow(phik_recon.numpy(), extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
        plt.contourf(X.numpy(), Y.numpy(), phik_recon.numpy(), cmfap='viridis')
        plt.scatter(executed_traj[1:, 0], executed_traj[1:, 1], s=10, c = full_lam[:N_pts], cmap='plasma')
        plt.title("Information Map")
        plt.figure(figsize=(3, 3))
        plt.imshow(phik_recon.numpy(), extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
        plt.contourf(X.numpy(), Y.numpy(), phik_recon.numpy(), cmap='viridis')
        plt.scatter(executed_traj[1:, 0], executed_traj[1:, 1], s=5, c='red')
        plt.scatter([x0[0]], [x0[1]], c='w', s=50, marker='X')
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
maps = full_maps[:3]
# all_maps = torch.stack(entropy_maps + gaussian_maps, dim=0)

# Sample grid to match the resolution
N, H, W = maps.shape
xs = torch.linspace(0, 1, W)
ys = torch.linspace(0, 1, H)
X, Y = torch.meshgrid(xs, ys, indexing='xy')   
_s = torch.stack([ X.reshape(-1), Y.reshape(-1) ], dim=1)

lambda_goal = 10.0

x0 = torch.tensor([0.54, 0.3])
x_goal = torch.tensor([1.0, 1.0])
k1, k2 = np.meshgrid(np.arange(0, 20), np.arange(0, 20))
k = torch.tensor(np.stack([k1.ravel(), k2.ravel()], axis=-1), dtype=torch.float32)
k_expanded = k.unsqueeze(0)
lamk = torch.exp(-0.4 * (torch.norm(k, dim=1) ** 2))
hk = torch.tensor([get_hk(ki) for ki in k.numpy()])
fk_vals_all = torch.cos(_s[:, None, :] * k_expanded).prod(dim=-1)


trajectory = replanning(x0, x_goal, maps, _s, k_expanded, lamk, lambda_goal)
plt.show()
# pts = trajectory.cpu().numpy()
# plt.plot(pts[:,0], pts[:,1], '-o', markersize=3)