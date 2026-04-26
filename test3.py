#!/usr/bin/env python
# coding: utf-8

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import random
import time

SENSOR_ON_THRESHOLD = 0.70

# # Dynamics and target distribution
# # The dynamics are defined as the constrained continuous time dynamical system
# # The target distribution, for which information is distributed within the continuous search space
# # -----------------------------------------------------------------------
# # TUNING PARAMETER
# # -190.5 and -180.5 are the coefficients that control the sharpness of the peaks in the target distribution
# @torch.jit.script
# def p(x):
#     return torch.exp(-190.5 * torch.sum((x[:2] - 0.27)**2)) \
#            + torch.exp(-180.5 * torch.sum((x[:2] - 0.75)**2))

# dynamics using discrete time eulerintegration
# how state x changes based on input u
# -----------------------------------------------------------------------
# TUNING PARAMETER
# step size, 0.07, controls how much the state changes in response to the input u at each time step.
# high step size: bigger steps, more aggressive exploration but potentially less stable trajectories
@torch.jit.script
def f(x, u):
    xnew = x[:2] + 0.07 * u[:2]
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
# def get_ck_weighted(tr, k_expanded, lam, hk):
#     weighted_fk = torch.prod(torch.cos(tr[:, None, :] * k_expanded), dim = -1).T @ lam
#     ck = weighted_fk / (hk + 1e-8)
#     return ck

def get_ck_weighted(tr, k_expanded, weights, hk):
    fk = torch.cos(tr[:, None, :] * k_expanded).prod(dim=-1)
    Z  = weights.sum() + 1e-8
    return (fk.T @ weights) / (Z * hk)

def compute_sensor_lambda(u):
    return torch.clamp(torch.sigmoid(5.0 * u[:, 2]), 0.0, 1.0)

# ergodic metric + other costs----
# Primary loss function for trajectory optimization
# ergodic metric (how well the trajcetory matches the information distribution) + regularization + penalty
# Overall, the tells how well the path explores the interesting areas in ergodic metric,
# how much energy the robot used, how msoothly the robot moved, if the robot stayed within the map,
# how much the robot activated its sensors, and ow much valuable information the robot gathered
def fourier_ergodic_loss(u, x0, phik, k_expanded, lamk, hk, info_map, tau=None, head_w=1.0, tail_w=0.25):
    displacements = 0.07 * u[:, :2]
    tr = torch.cumsum(displacements, dim=0) + x0
    tr = tr.clamp(0.0, 1.0)

    T = u.shape[0]
    
    # Continuous relaxation of the sensor variable to [0, 1]
    lam = torch.sigmoid(u[:, 2])

    # Exponential discount: Heavily weights the immediate "head" of the trajectory
    # so the optimizer prioritizes the accuracy of the next executed steps.
    disc = torch.exp(-3.0 * torch.arange(T, device=u.device, dtype=torch.float32) / T)
    
    w = torch.ones(T, device=u.device)
    if tau is not None:
        w[:tau] = head_w
        w[tau:] = tail_w
    w = w * disc

    # Sample information values along the current trajectory
    H, W = info_map.shape
    ix = (tr[:, 0] * (W - 1)).long().clamp(0, W - 1)
    iy = (tr[:, 1] * (H - 1)).long().clamp(0, H - 1)
    info_values = info_map[iy, ix]

    # Calculate Ergodic Metric matching
    ck = get_ck_weighted(tr[:, :2], k_expanded, lam * w, hk)
    phik_normed = (phik - phik.mean()) / (phik.std() + 1e-6)
    ck_normed   = (ck   - ck.mean())   / (ck.std()   + 1e-6)
    ergodic_term = torch.sum(lamk * (phik_normed - ck_normed)**2)

    # 1. SPARSE SENSING PENALTY: L1 norm on lambda, penalized heavily in LOW info regions
    sparsity_cost = 0.5 * torch.sum(lam * (1.0 - info_values) * w)

    # 2. INFORMATION REWARD: Explicit reward for having the sensor ON in HIGH info regions
    reward_term = -1.5 * torch.sum(lam * (info_values**3) * w)

    # Kinematic costs and map boundaries
    control_cost = 0.005 * torch.mean(u[:, :2]**2)
    smoothness_cost = 0.001 * torch.sum((u[1:, :2] - u[:-1, :2])**2)
    barrier_cost = 10 * torch.sum(torch.clamp_min(tr - 1, 0)**2 + torch.clamp_min(-tr, 0)**2)

    loss = ergodic_term + sparsity_cost + reward_term + control_cost + smoothness_cost + barrier_cost
    return loss


def optimize_trajectory(x0, phik, k_expanded, lamk, hk, info_map, u_prev=None, T=100, tau=20, num_iters=1500, lr=1e-2):
    tau = int(min(max(tau, 1), T))
    H_map, W_map = info_map.shape
    
    # Calculate global goal to gently pull the trajectory tail
    max_val = torch.max(info_map)
    jy, ix = torch.where(info_map == max_val)
    gx = ix.float().mean() / max(W_map - 1, 1)
    gy = jy.float().mean() / max(H_map - 1, 1)
    goal = torch.tensor([gx, gy], dtype=torch.float32, device=x0.device)

    if u_prev is None:
        # First Cycle: Cold Start
        # Full 100-timestep optimization requires full iterations
        u_init = torch.empty((T, 3), device=x0.device, dtype=torch.float32)
        u_init[:, :2].normal_(mean=0.0, std=0.01)
        u_init[:, 2].uniform_(-0.5, 0.5)
        iters = num_iters
    else:
        # Subsequent Cycles: Warm Start (Receding Horizon)
        # Shift the previous trajectory left by tau, dropping the executed head
        u_seed = u_prev.detach().to(device=x0.device, dtype=torch.float32)
        shifted_head = u_seed[tau:] 
        
        # Generate a fresh tail of length tau to cap the 100-step horizon
        new_tail = torch.empty((tau, 3), device=x0.device, dtype=torch.float32)
        new_tail[:, :2].normal_(mean=0.0, std=0.01)
        new_tail[:, 2].uniform_(-0.5, 0.5)
        
        u_init = torch.cat([shifted_head, new_tail], dim=0)
        
        # Massive efficiency gain: 80% of the path is already optimized. 
        # We only need a fraction of the iterations to refine the head and settle the tail.
        iters = 75 

    u = torch.nn.Parameter(u_init)
    optimizer = torch.optim.Adam([u], lr=lr)

    for i in range(iters):
        optimizer.zero_grad()
        loss = loss_with_goal(
            u, x0, phik, k_expanded, lamk, hk, info_map,
            tau=tau, head_w=1.0, tail_w=0.10, goal=goal, goal_w=2.5
        )
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            # Clamp kinematics. Dimension 2 (lambda) relies on sigmoid, no hard clamp needed here.
            u[:, :2].clamp_(min=-1.0, max=1.0)
            
    # Pad to T just in case of dimension mismatch
    u_out = u.detach()
    if u_out.shape[0] < T:
        padding = torch.zeros((T - u_out.shape[0], 3), device=u.device, dtype=torch.float32)
        padding[:, :] = u_out[-1, :]
        u_out = torch.cat([u_out, padding], dim=0)
        
    return u_out[:T]

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

def rollout_states(x0, u):
    x = x0.clone()
    tr = [x]
    for step in u:
        x, _ = f(x, step[:2])
        tr.append(x)
    return torch.stack(tr)

def controls_from_states(states, dynamics_scale=0.07):
    deltas = states[1:] - states[:-1]
    u = torch.zeros((states.shape[0] - 1, 3), dtype=torch.float32, device=states.device)
    u[:, :2] = deltas / dynamics_scale
    u[:, :2].clamp_(min=-1.0, max=1.0)
    return u

def maps_are_same(map_a, map_b, atol=1e-6, rtol=1e-5):
    return torch.allclose(map_a, map_b, atol=atol, rtol=rtol)

# returns trajectory as list of states
def replanning(maps, _s, k_expanded, lamk, num_iters=1500, T=100, tau=20):
    x_init = torch.tensor([0.54, 0.3], dtype=torch.float32)
    x0 = x_init.clone()
    N, H, W = maps.shape
    trajectories, phik_list = [], []
    full_trajectories = []
    full_lambda_list = []
    head_len_list = []
    time_list = []
    last_map = None

    active_phik = None
    active_plan_u = None
    active_plan_states = None
    active_plan_cursor = 0
    active_plan_mode = "forward"

    def build_cycle_window_u(plan_u, cursor, horizon):
        rem = plan_u[cursor:].detach()
        if rem.shape[0] == 0:
            rem = plan_u[-1:, :].detach()
        if rem.shape[0] >= horizon:
            return rem[:horizon].clone().detach()
        pad = rem[-1:, :].repeat(horizon - rem.shape[0], 1)
        return torch.cat([rem, pad], dim=0).clone().detach()

    # Each new plan with the tail of the previous u
    for i, info_map in enumerate(maps):
        start_time = time.time()
        info_map = info_map / (info_map.max() + 1e-8)
        print(f"\n=== cycle {i} ===")
        map_changed = (last_map is None) or (not maps_are_same(info_map, last_map))

        if map_changed:
            print("Map changed: regenerating a new 100-step trajectory from current robot state.")
            active_phik = phik_from_map(info_map.flatten(), _s, k_expanded)
            active_plan_u = optimize_trajectory(
                x0, active_phik, k_expanded, lamk, hk, info_map,
                u_prev=None, T=T, tau=tau, num_iters=num_iters
            )
            active_plan_states = rollout_states(x0, active_plan_u).detach()
            active_plan_cursor = 0
            active_plan_mode = "forward"
            last_map = info_map.clone()
        else:
            print("Map unchanged: executing the next consecutive head segment from existing plan.")
            if active_plan_u is None or active_plan_states is None:
                active_phik = phik_from_map(info_map.flatten(), _s, k_expanded)
                active_plan_u = optimize_trajectory(
                    x0, active_phik, k_expanded, lamk, hk, info_map,
                    u_prev=None, T=T, tau=tau, num_iters=num_iters
                )
                active_plan_states = rollout_states(x0, active_plan_u).detach()
                active_plan_cursor = 0
                active_plan_mode = "forward"
            elif active_plan_cursor >= active_plan_u.shape[0]:
                active_plan_states = torch.flip(active_plan_states, dims=[0]).detach()
                active_plan_u = controls_from_states(active_plan_states).detach()
                active_plan_cursor = 0
                active_plan_mode = "reverse" if active_plan_mode == "forward" else "forward"
                print(f"Plan exhausted: switching to {active_plan_mode} execution.")

        # Build the per-cycle 100-step window from the CURRENT state and remaining controls.
        # This makes plot start = previous cycle executed-head end, and head/tail = next 20/80.
        cycle_plan_u = build_cycle_window_u(active_plan_u, active_plan_cursor, T)
        cycle_plan_states = rollout_states(x0, cycle_plan_u).detach()

        phik_list.append(active_phik.detach())
        full_trajectories.append(cycle_plan_states.cpu().detach().numpy())
        full_lambda_list.append(compute_sensor_lambda(cycle_plan_u).cpu().numpy())

        steps_left = active_plan_u.shape[0] - active_plan_cursor
        exec_len = min(tau, steps_left)
        u_head = active_plan_u[active_plan_cursor:active_plan_cursor + exec_len]
        active_plan_cursor += exec_len
        head_len_list.append(int(exec_len))

        executed_traj_t = rollout_states(x0, u_head)
        executed_traj = executed_traj_t.cpu().detach().numpy()
        end_time = time.time()
        time_list.append(end_time - start_time)
        trajectories.append(executed_traj)
        x0 = executed_traj_t[-1].detach()

    return trajectories, full_trajectories, full_lambda_list, head_len_list, phik_list, time_list

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
# Get the list of maps and stack into a tensor
# Change the maps path accordingly
# entropy_maps_path = "/home/younkyuw/Desktop/ergodic-search/entropy_maps"
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

# Build map sequence for testing
full_maps = torch.stack(entropy_maps)
n_cycles = 4

# ===========================================================================
# Option A: every cycle uses a NEW map (no immediate repeats).
# Uncomment this block to use all-new maps.
# perm = torch.randperm(full_maps.shape[0])
# maps = full_maps[perm[:n_cycles]]
# ===========================================================================
# Option B: current map may be SAME as previous map for some cycles.
# Uncomment this block (and comment Option A) to inject repeats.
repeat_prob = 0.5  # probability to reuse previous map
perm = torch.randperm(full_maps.shape[0])
seed_pool = full_maps[perm[:n_cycles]]
maps_list = [seed_pool[0]]
for i in range(1, n_cycles):
    if random.random() < repeat_prob:
        maps_list.append(maps_list[-1].clone())
    else:
        maps_list.append(seed_pool[i])
maps = torch.stack(maps_list, dim=0)
# ===========================================================================

# Sample grid of (0 to 1) to match the resolution
N, H, W = maps.shape
xs = torch.linspace(0, 1, W)
ys = torch.linspace(0, 1, H)
X, Y = torch.meshgrid(xs, ys, indexing='xy')   
_s = torch.stack([ X.reshape(-1), Y.reshape(-1) ], dim=1)

# Fourier Basis, k, for defining a 20x20 grid of frequency modes to represent a 
# sum of 400 cosine waves
k1, k2 = np.meshgrid(np.arange(0, 20), np.arange(0, 20))
k = torch.tensor(np.stack([k1.ravel(), k2.ravel()], axis=-1), dtype=torch.float32)

# --------------------------------------------------------------------
# TUNING PARAMETER
# frequency weighting for each k
# print("-0.8 * torch.norm(k, dim=1)", -0.8 * torch.norm(k, dim=1),"\n -0.15 * (torch.norm(k, dim=1) ** 2)", -0.15 * (torch.norm(k, dim=1) ** 2))
# this wll give me much more aggressive downweighting of higher frequencies, 
# focus on large global blobs. might miss out details
# lmak = torch.exp(-0.15 * (torch.norm(k, dim=1) ** 2))
# this gives more detailed but potentially jittery trajectories
lamk = torch.exp(-0.8 * torch.norm(k, dim=1))

# orthonormalization factor for each k
hk = torch.clamp(torch.tensor([get_hk(ki) for ki in k.numpy()]), min=1e-6)
#represent the Fourier basis modes (frequencies)
k_expanded = k.unsqueeze(0)
# Calculate the values of the Fourier basis functions at the sample points, _s, for all Fourier modes, k_expanded
fk_vals_all = torch.cos(_s[:, None, :] * k_expanded).prod(dim=-1)


trajectory, full_trajectory, full_lambda, head_len_per_cycle, phik_list, time_list = replanning(maps, _s, k_expanded, lamk)


print(f"Execution time: {time_list}")

num_cycles = len(trajectory)
fig, axes = plt.subplots(2, num_cycles, figsize=(4 * num_cycles, 8), squeeze=False)
sensor_cmap = LinearSegmentedColormap.from_list("sensor_white_red", ["#ffffff", "#ff0000"])
sc_last = None

for i in range(num_cycles):
    tr = trajectory[i]
    full_tr = full_trajectory[i]
    lam_i = full_lambda[i]
    head_len = head_len_per_cycle[i]
    full_pts = full_tr[1:, :2]
    head_pts = full_pts[:head_len]
    tail_pts = full_pts[head_len:]
    lam_head = lam_i[:head_len]
    lam_tail = lam_i[head_len:]
    phik = phik_list[i]    

    phik_recon = torch.matmul(fk_vals_all, phik).reshape(H, W)
    ax_ht = axes[0, i]
    ax_sa = axes[1, i]

    # Top row: Head/Tail
    ax_ht.imshow(phik_recon, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    ax_ht.contourf(X.numpy(), Y.numpy(), phik_recon, cmap='viridis')
    if tail_pts.shape[0] > 0:
        ax_ht.scatter(tail_pts[:, 0], tail_pts[:, 1], s=16, c='white', alpha=0.40)
    if head_pts.shape[0] > 0:
        ax_ht.scatter(head_pts[:, 0], head_pts[:, 1], s=34, c='red', edgecolors='black', linewidths=0.5)
    ax_ht.scatter(full_tr[0, 0], full_tr[0, 1], c='w', s=50, marker='X')
    ax_ht.scatter(tr[-1, 0], tr[-1, 1], c='yellow', s=35)
    ax_ht.set_title(f"Map {i + 1}: Head/Tail")
    ax_ht.set_aspect('equal')
    ax_ht.set_xlim(0, 1)
    ax_ht.set_ylim(0, 1)
    if i == 0:
        legend_handles_ht = [
            Line2D([0], [0], marker='o', linestyle='None', markersize=6, markerfacecolor='white', markeredgecolor='none', alpha=0.40, label='Tail'),
            Line2D([0], [0], marker='o', linestyle='None', markersize=7, markerfacecolor='red', markeredgecolor='black', label='Head'),
            Line2D([0], [0], marker='X', linestyle='None', markersize=7, markerfacecolor='white', markeredgecolor='white', label='Plan start'),
            Line2D([0], [0], marker='o', linestyle='None', markersize=6, markerfacecolor='yellow', markeredgecolor='none', label='Executed head end'),
        ]
        ax_ht.legend(handles=legend_handles_ht, loc='lower left', fontsize=7, framealpha=0.85)

    # Bottom row: Sensor Lambda (100 steps)
    ax_sa.imshow(phik_recon, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    ax_sa.contourf(X.numpy(), Y.numpy(), phik_recon, cmap='viridis')
    if tail_pts.shape[0] > 0:
        sc_last = ax_sa.scatter(
            tail_pts[:, 0], tail_pts[:, 1],
            s=14, c=lam_tail, cmap=sensor_cmap, vmin=0.0, vmax=1.0, alpha=0.55
        )
    if head_pts.shape[0] > 0:
        sc_last = ax_sa.scatter(
            head_pts[:, 0], head_pts[:, 1],
            s=34, c=lam_head, cmap=sensor_cmap, vmin=0.0, vmax=1.0,
            edgecolors='black', linewidths=0.5, alpha=0.95
        )
    sensor_on = lam_i >= SENSOR_ON_THRESHOLD
    if np.any(sensor_on):
        ax_sa.scatter(
            full_pts[sensor_on, 0],
            full_pts[sensor_on, 1],
            s=70,
            facecolors='none',
            edgecolors='cyan',
            linewidths=1.5,
        )
    ax_sa.scatter(full_tr[0, 0], full_tr[0, 1], c='w', s=50, marker='X')
    ax_sa.scatter(tr[-1, 0], tr[-1, 1], c='yellow', s=35)
    ax_sa.set_title(f"Map {i + 1}: Sensor Lambda (100 Steps)")
    ax_sa.set_aspect('equal')
    ax_sa.set_xlim(0, 1)
    ax_sa.set_ylim(0, 1)
    if i == 0:
        legend_handles_sa = [
            Line2D([0], [0], marker='o', linestyle='None', markersize=5, markerfacecolor='red', markeredgecolor='none', alpha=0.55, label='Tail (lambda color)'),
            Line2D([0], [0], marker='o', linestyle='None', markersize=7, markerfacecolor='red', markeredgecolor='black', label='Head (lambda color)'),
            Line2D([0], [0], marker='o', linestyle='None', markersize=8, markerfacecolor='none', markeredgecolor='cyan', label=f"Sensor ON (lambda >= {SENSOR_ON_THRESHOLD:.1f})"),
            Line2D([0], [0], marker='X', linestyle='None', markersize=7, markerfacecolor='white', markeredgecolor='white', label='Plan start'),
            Line2D([0], [0], marker='o', linestyle='None', markersize=6, markerfacecolor='yellow', markeredgecolor='none', label='Executed head end'),
        ]
        ax_sa.legend(handles=legend_handles_sa, loc='lower left', fontsize=7, framealpha=0.8)

fig.tight_layout()
if sc_last is not None:
    cbar = fig.colorbar(sc_last, ax=axes[1, :].tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Sensor activation lambda (white=OFF, red=ON)")


plt.show()
