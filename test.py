#!/usr/bin/env python
# coding: utf-8

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import time

SENSOR_ON_THRESHOLD = 0.60
HIGH_INFO_QUANTILE = 0.70
LOW_SPEED_THRESHOLD = 0.015
HOTSPOT_QUANTILE = 0.86
MAX_HOTSPOTS = 6

# Objective defaults (sweep can override select globals).
OCC_W = -0.35
SPEED_W = 0.55
SENSOR_BCE_W = 0.45
SENSOR_LOW_W = 0.10
SENSOR_BIN_W = 0.01
V_HIGH = 0.010
V_LOW = 0.055
COVERAGE_W = 0.55
HEAD_GOAL_W = 5.0
TERMINAL_GOAL_W = 0.50
GOAL_DISTANCE_WEIGHT = 0.60
ORDER_W_SCALE = 0.90
ORDER_BASE_WEIGHTS = (2.8, 1.8, 1.2)

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

def sample_info_values(points, info_map):
    H, W = info_map.shape
    ix = (points[:, 0] * (W - 1)).long().clamp(0, W - 1)
    iy = (points[:, 1] * (H - 1)).long().clamp(0, H - 1)
    return info_map[iy, ix], ix, iy

def normalize_info_map(info_map):
    map_min = torch.min(info_map)
    map_max = torch.max(info_map)
    return (info_map - map_min) / (map_max - map_min + 1e-8)

def extract_hotspots(info_map_norm, max_hotspots=MAX_HOTSPOTS, hotspot_quantile=HOTSPOT_QUANTILE, min_sep=0.12):
    H, W = info_map_norm.shape
    flat = info_map_norm.flatten()
    threshold = torch.quantile(flat, hotspot_quantile)
    candidate_ids = torch.nonzero(flat >= threshold, as_tuple=False).flatten()
    if candidate_ids.numel() == 0:
        candidate_ids = torch.topk(flat, k=min(max_hotspots, flat.numel())).indices

    candidate_scores = flat[candidate_ids]
    sorted_ids = candidate_ids[torch.argsort(candidate_scores, descending=True)]
    denom_x = float(max(W - 1, 1))
    denom_y = float(max(H - 1, 1))
    hotspots = []
    scores = []

    for idx in sorted_ids:
        idx_i = int(idx.item())
        y = idx_i // W
        x = idx_i % W
        point = torch.tensor([x / denom_x, y / denom_y], dtype=torch.float32, device=info_map_norm.device)
        if hotspots:
            min_dist = torch.stack([torch.norm(point - p) for p in hotspots]).min()
            if float(min_dist.item()) < min_sep:
                continue
        hotspots.append(point)
        scores.append(flat[idx_i])
        if len(hotspots) >= max_hotspots:
            break

    if not hotspots:
        idx_i = int(torch.argmax(flat).item())
        y = idx_i // W
        x = idx_i % W
        hotspots = [torch.tensor([x / denom_x, y / denom_y], dtype=torch.float32, device=info_map_norm.device)]
        scores = [flat[idx_i]]

    return torch.stack(hotspots, dim=0), torch.stack(scores, dim=0)

def choose_priority_goals(info_map, x0, hotspot_quantile=HOTSPOT_QUANTILE, max_hotspots=MAX_HOTSPOTS):
    info_map_norm = normalize_info_map(info_map)
    hotspots, hotspot_scores = extract_hotspots(
        info_map_norm,
        max_hotspots=max_hotspots,
        hotspot_quantile=hotspot_quantile,
    )
    # Ensure at least three targets.
    while hotspots.shape[0] < 3:
        hotspots = torch.cat([hotspots, hotspots[:1]], dim=0)
        hotspot_scores = torch.cat([hotspot_scores, hotspot_scores[:1]], dim=0)

    remaining_ids = list(range(hotspots.shape[0]))
    ordered_ids = []

    # h1: nearest strong hotspot from current start.
    start_xy = x0[:2]
    dists = torch.norm(hotspots - start_xy.unsqueeze(0), dim=1)
    rank_score = hotspot_scores - GOAL_DISTANCE_WEIGHT * dists
    h1_idx = int(torch.argmax(rank_score).item())
    ordered_ids.append(h1_idx)
    remaining_ids.remove(h1_idx)

    # h2/h3: greedy continuation from previous hotspot.
    prev_idx = h1_idx
    for _ in range(2):
        if not remaining_ids:
            ordered_ids.append(prev_idx)
            continue
        rem = torch.tensor(remaining_ids, dtype=torch.long, device=hotspots.device)
        d_prev = torch.norm(hotspots[rem] - hotspots[prev_idx].unsqueeze(0), dim=1)
        score_next = hotspot_scores[rem] - 0.6 * d_prev
        pick = int(rem[torch.argmax(score_next)].item())
        ordered_ids.append(pick)
        remaining_ids.remove(pick)
        prev_idx = pick

    ordered_goals = hotspots[torch.tensor(ordered_ids[:3], dtype=torch.long, device=hotspots.device)]
    return ordered_goals, hotspots

def compute_split_jump_metric(u, tau):
    T = u.shape[0]
    if T < 2:
        return {
            "split_jump_metric": 0.0,
            "split_vel_jump": 0.0,
            "split_acc_jump": 0.0,
            "split_lam_jump": 0.0,
        }
    tau_i = int(min(max(tau, 1), T - 1))
    lam = compute_sensor_lambda(u)
    vel_jump = torch.norm(u[tau_i - 1, :2] - u[tau_i, :2]).item()
    lam_jump = torch.abs(lam[tau_i - 1] - lam[tau_i]).item()
    if tau_i + 1 < T:
        acc_jump = torch.norm(u[tau_i + 1, :2] - 2.0 * u[tau_i, :2] + u[tau_i - 1, :2]).item()
    else:
        acc_jump = 0.0
    split_jump = float(vel_jump + 0.5 * acc_jump + lam_jump)
    return {
        "split_jump_metric": split_jump,
        "split_vel_jump": float(vel_jump),
        "split_acc_jump": float(acc_jump),
        "split_lam_jump": float(lam_jump),
    }

def compute_hotspot_distance_metrics(states, ordered_goals, tau):
    pts = states[1:, :2]
    if pts.shape[0] == 0:
        return {"head_dist_h1_at_tau": 0.0, "mean_hotspot_coverage_dist": 0.0}
    tau_idx = int(min(max(tau, 1), pts.shape[0])) - 1
    head_dist = torch.norm(pts[tau_idx] - ordered_goals[0]).item()
    cover_dists = []
    for goal in ordered_goals:
        d = torch.norm(pts - goal.unsqueeze(0), dim=1)
        cover_dists.append(torch.min(d).item())
    return {
        "head_dist_h1_at_tau": float(head_dist),
        "mean_hotspot_coverage_dist": float(np.mean(cover_dists)),
    }

def compute_behavior_metrics(
    executed_traj_t,
    u_head,
    info_map,
    sensor_on_threshold=SENSOR_ON_THRESHOLD,
    high_info_quantile=HIGH_INFO_QUANTILE,
    low_speed_threshold=LOW_SPEED_THRESHOLD,
):
    if u_head.shape[0] == 0:
        return {
            "high_info_dwell_ratio": 0.0,
            "high_info_dwell_area_norm": 0.0,
            "low_info_transit_efficiency": 0.0,
            "low_info_idle_rate": 0.0,
            "sensor_precision": 0.0,
            "sensor_recall": 0.0,
            "sensor_f1_high_info": 0.0,
            "avg_speed_high_info": 0.0,
            "avg_speed_low_info": 0.0,
        }

    pts = executed_traj_t[1:, :2]
    speed = torch.norm(executed_traj_t[1:, :2] - executed_traj_t[:-1, :2], dim=1)
    info_vals, _, _ = sample_info_values(pts, info_map)
    high_thr = torch.quantile(info_map.flatten(), high_info_quantile)
    high_mask = info_vals >= high_thr
    low_mask = ~high_mask

    lam = compute_sensor_lambda(u_head)
    sensor_on = lam >= sensor_on_threshold

    tp = (sensor_on & high_mask).sum().item()
    pred_on = sensor_on.sum().item()
    actual_high = high_mask.sum().item()

    precision = tp / max(pred_on, 1)
    recall = tp / max(actual_high, 1)
    f1 = (2.0 * precision * recall) / max((precision + recall), 1e-8)

    dwell_ratio = high_mask.float().mean().item()
    low_idle_rate = ((speed < low_speed_threshold) & low_mask).float().sum().item() / max(low_mask.sum().item(), 1)
    transit_eff = 1.0 - low_idle_rate
    map_high_area = (info_map >= high_thr).float().mean().item()
    dwell_area_norm = dwell_ratio / max(map_high_area, 1e-6)

    if high_mask.any():
        avg_speed_high = speed[high_mask].mean().item()
    else:
        avg_speed_high = 0.0
    if low_mask.any():
        avg_speed_low = speed[low_mask].mean().item()
    else:
        avg_speed_low = 0.0

    return {
        "high_info_dwell_ratio": float(dwell_ratio),
        "high_info_dwell_area_norm": float(dwell_area_norm),
        "low_info_transit_efficiency": float(transit_eff),
        "low_info_idle_rate": float(low_idle_rate),
        "sensor_precision": float(precision),
        "sensor_recall": float(recall),
        "sensor_f1_high_info": float(f1),
        "avg_speed_high_info": float(avg_speed_high),
        "avg_speed_low_info": float(avg_speed_low),
    }

# ergodic metric + other costs----
# Primary loss function for trajectory optimization
# ergodic metric (how well the trajcetory matches the information distribution) + regularization + penalty
# Overall, the tells how well the path explores the interesting areas in ergodic metric,
# how much energy the robot used, how msoothly the robot moved, if the robot stayed within the map,
# how much the robot activated its sensors, and ow much valuable information the robot gathered
def fourier_ergodic_loss(
    u,
    x0,
    phik,
    k_expanded,
    lamk,
    hk,
    info_map,
    tau=None,
    head_w=1.0,
    tail_w=0.25,
    hotspots=None,
    ordered_goals=None,
    split_vel_w=2.0,
    split_acc_w=0.8,
    split_lam_w=0.6,
):
    displacements = 0.07 * u[:, :2]
    tr = torch.cumsum(displacements, dim=0) + x0
    tr = tr.clamp(0.0, 1.0)

    T = u.shape[0]
    lam = compute_sensor_lambda(u)

    # determine the weights of head and tail for the amount of contributions to the trajectory
    w = torch.ones(T, device=u.device)
    if tau is not None:
        w[:tau] = head_w
        w[tau:] = tail_w

    # add exponetial discount
    # Lower discount aggressiveness keeps more tail influence for global coverage.
    disc = torch.exp(-1.0 * torch.arange(T, device=u.device, dtype=torch.float32) / T)
    w = w * disc

    ck = get_ck_weighted(tr[:, :2], k_expanded, lam * w, hk)

    phik_normed = (phik - phik.mean()) / (phik.std() + 1e-6)
    ck_normed   = (ck   - ck.mean())   / (ck.std()   + 1e-6)

    info_map_norm = normalize_info_map(info_map)
    info_norm, _, _ = sample_info_values(tr[:, :2], info_map_norm)
    speed = torch.norm(displacements, dim=1)

    # High-info dwell and low-info fast transit via target-speed shaping.
    occ_term = OCC_W * torch.mean(info_norm.pow(2))
    v_target = V_HIGH + (1.0 - info_norm) * (V_LOW - V_HIGH)
    speed_term = SPEED_W * torch.mean((speed - v_target).pow(2))

    # Sensor accuracy: turn on in high-info regions, off in low-info regions.
    q70 = torch.quantile(info_map_norm.flatten(), 0.70)
    target_lam = torch.sigmoid((info_norm - q70) / 0.08)
    eps = 1e-6
    sensor_bce = -(
        2.5 * target_lam * torch.log(lam + eps)
        + (1.0 - target_lam) * torch.log(1.0 - lam + eps)
    )
    sensor_bce_term = SENSOR_BCE_W * torch.mean(sensor_bce)
    sensor_low_term = SENSOR_LOW_W * torch.mean(lam * (1.0 - target_lam))
    sensor_bin_term = SENSOR_BIN_W * torch.mean(lam * (1.0 - lam))

    # Soft ordered visits for h1->h2->h3.
    ordered_visit_term = 0.0
    if ordered_goals is not None and ordered_goals.shape[0] >= 3:
        t1 = int(min(max(tau if tau is not None else T // 5, 1), T))
        t2 = int(min(max(t1 + 40, t1 + 1), T))
        segments = [(0, t1, ordered_goals[0]), (t1, t2, ordered_goals[1]), (t2, T, ordered_goals[2])]
        beta = 30.0
        seg_terms = []
        for s0, s1, goal in segments:
            if s1 <= s0:
                continue
            seg = tr[s0:s1, :2]
            if seg.shape[0] == 0:
                continue
            d2 = torch.sum((seg - goal.unsqueeze(0))**2, dim=1)
            soft_min = -torch.logsumexp(-beta * d2, dim=0) / beta
            seg_terms.append(soft_min)
        if seg_terms:
            seg_w = torch.tensor(ORDER_BASE_WEIGHTS[:len(seg_terms)], dtype=torch.float32, device=u.device) * ORDER_W_SCALE
            ordered_visit_term = torch.sum(seg_w * torch.stack(seg_terms))

    coverage_term = 0.0
    if hotspots is not None and hotspots.numel() > 0:
        dist2 = torch.sum((tr[:, None, :2] - hotspots[None, :, :])**2, dim=2)
        smooth_min = -torch.logsumexp(-25.0 * dist2, dim=0) / 25.0
        coverage_term = COVERAGE_W * torch.mean(smooth_min)

    # Head/tail split smoothness penalties.
    split_vel_term = 0.0
    split_acc_term = 0.0
    split_lam_term = 0.0
    if tau is not None and T >= 2:
        tau_i = int(min(max(tau, 1), T - 1))
        split_vel_term = split_vel_w * torch.sum((u[tau_i - 1, :2] - u[tau_i, :2])**2)
        split_lam_term = split_lam_w * (lam[tau_i - 1] - lam[tau_i]).pow(2)
        if tau_i + 1 < T:
            split_acc_term = split_acc_w * torch.sum((u[tau_i + 1, :2] - 2.0 * u[tau_i, :2] + u[tau_i - 1, :2])**2)

    loss = torch.sum(lamk * (phik_normed - ck_normed)**2) \
            + 0.001 * torch.mean(u[:, :2]**2) \
            + 0.001 * torch.sum((u[1:, :2] - u[:-1, :2])**2) \
            + 10 * torch.sum(torch.clamp_min(tr - 1, 0)**2 + torch.clamp_min(-tr, 0)**2) \
            + 0.000005 * torch.sum(torch.abs(lam)) \
            + 10 * torch.sum(torch.clamp_min(lam - 1, 0)**2 + torch.clamp_min(-lam, 0)**2) \
            + occ_term \
            + speed_term \
            + sensor_bce_term \
            + sensor_low_term \
            + sensor_bin_term \
            + ordered_visit_term \
            + split_vel_term \
            + split_acc_term \
            + split_lam_term \
            + coverage_term
    return loss

# -----------------------------------------------------------------------
# TUNING PARAMETER
# Num_iters, learning rate
def optimize_trajectory(
    x0,
    phik,
    k_expanded,
    lamk,
    hk,
    info_map,
    u_prev=None,
    T=100,
    tau=20,
    max_iters=1500,
    min_iters=120,
    patience=25,
    rel_improve_tol=1e-4,
    lr=1e-3,
    bridge_len=12,
    xy_noise_std=0.003,
    split_vel_w=2.0,
    split_acc_w=0.8,
    split_lam_w=0.6,
    return_diagnostics=False,
    return_debug_metrics=True,
):
    tau = int(min(max(tau, 1), T))
    bridge_len = int(max(0, bridge_len))
    opt_window = T if u_prev is None else min(T, tau + bridge_len)

    # Full-horizon optimize for initial call.
    if u_prev is None:
        head = torch.empty((opt_window, 3), device=x0.device, dtype=torch.float32)
        head[:, :2].normal_(mean=0.0, std=0.01)
        head[:, 2].uniform_(-0.5, 0.5)
        tail = None
        loss_tau = tau
    else:
        # Warm-start with head+bridge optimize window.
        u_seed = u_prev.detach().to(device=x0.device, dtype=torch.float32)
        if u_seed.shape[0] >= T:
            u_seed = u_seed[:T]
        else:
            pad = torch.empty((T - u_seed.shape[0], 3), device=x0.device, dtype=torch.float32)
            # -----------------------------------------------------------------------
            # TUNING PARAMETER
            # initial noise std 0.01
            pad[:, :2].normal_(mean=0.0, std=0.01)
            pad[:, 2].uniform_(-0.5, 0.5)
            u_seed = torch.cat([u_seed, pad], dim=0)

        head = u_seed[:opt_window].clone()
        if xy_noise_std > 0:
            warm_len = min(tau, head.shape[0])
            head[:warm_len, :2] += xy_noise_std * torch.randn_like(head[:warm_len, :2])
        tail = u_seed[opt_window:].clone().detach()
        loss_tau = tau
    head = torch.nn.Parameter(head)

    optimizer = torch.optim.Adam([head], lr=lr)
    def u_builder(head, tail):
        if tail is None:
            return head
        else:
            return torch.cat([head, tail], dim = 0)

    u_for_eval = u_builder(head, tail)
    with torch.no_grad():
        ordered_goals, hotspots = choose_priority_goals(info_map, x0)
        head_goal = ordered_goals[0]
        terminal_goal = ordered_goals[2]
    ordered_goals = ordered_goals.to(device=u_for_eval.device, dtype=torch.float32)
    head_goal = head_goal.to(device=u_for_eval.device, dtype=torch.float32)
    terminal_goal = terminal_goal.to(device=u_for_eval.device, dtype=torch.float32)
    hotspots = hotspots.to(device=u_for_eval.device, dtype=torch.float32)

    with torch.no_grad():
        initial_loss = float(
            loss_with_goal(
                u_for_eval, x0, phik, k_expanded, lamk, hk, info_map,
                tau=loss_tau, head_w=1.0, tail_w=0.75,
                goal=terminal_goal, goal_w=TERMINAL_GOAL_W,
                head_goal=head_goal, head_goal_w=HEAD_GOAL_W,
                hotspots=hotspots,
                ordered_goals=ordered_goals,
                split_vel_w=split_vel_w,
                split_acc_w=split_acc_w,
                split_lam_w=split_lam_w,
            ).item()
        )

    last_loss = initial_loss
    plateau_count = 0
    iterations_run = 0

    for i in range(max_iters):
        optimizer.zero_grad()
        u = u_builder(head, tail)
        loss = loss_with_goal(
            u, x0, phik, k_expanded, lamk, hk, info_map,
            tau=loss_tau, head_w=1.0, tail_w=0.75,
            goal=terminal_goal, goal_w=TERMINAL_GOAL_W,
            head_goal=head_goal, head_goal_w=HEAD_GOAL_W,
            hotspots=hotspots,
            ordered_goals=ordered_goals,
            split_vel_w=split_vel_w,
            split_acc_w=split_acc_w,
            split_lam_w=split_lam_w,
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            head[:, :2].clamp_(min=-1.0, max=1.0)
            head[:, 2].clamp_(min=-1.0, max=1.0)
        curr_loss = float(loss.item())
        rel_improve = (last_loss - curr_loss) / (abs(last_loss) + 1e-8)
        if rel_improve < rel_improve_tol:
            plateau_count += 1
        else:
            plateau_count = 0
        last_loss = curr_loss
        iterations_run = i + 1

        if iterations_run >= min_iters and plateau_count >= patience:
            print(f"Early stop at iter {iterations_run}: relative improvement plateau.")
            break

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

    debug = {}
    if return_debug_metrics:
        split_debug = compute_split_jump_metric(u, tau)
        states = rollout_states(x0, u)
        hotspot_debug = compute_hotspot_distance_metrics(states, ordered_goals, tau)
        behavior_debug = compute_behavior_metrics(states, u, info_map)
        debug.update(split_debug)
        debug.update(hotspot_debug)
        debug.update({
            "plan_dwell_ratio": behavior_debug["high_info_dwell_ratio"],
            "plan_dwell_area_norm": behavior_debug["high_info_dwell_area_norm"],
            "plan_low_idle_rate": behavior_debug["low_info_idle_rate"],
            "plan_sensor_f1_high_info": behavior_debug["sensor_f1_high_info"],
        })

    diagnostics = {
        "initial_loss": initial_loss,
        "final_loss": float(last_loss),
        "iterations_run": int(iterations_run),
        "early_stopped": bool(iterations_run < max_iters),
        "loss_tau": int(loss_tau),
        "used_warm_start": bool(u_prev is not None),
        "opt_window": int(opt_window),
        "bridge_len": int(bridge_len),
    }
    diagnostics.update(debug)

    if return_diagnostics:
        return u, diagnostics
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

# combines ergodic loss over phik with terminal + head goals
def loss_with_goal(
    u,
    x0,
    phik,
    k_expanded,
    lamk,
    hk,
    info_map,
    tau=None,
    head_w=1.0,
    tail_w=0.25,
    goal=None,
    goal_w=0.0,
    head_goal=None,
    head_goal_w=0.0,
    hotspots=None,
    ordered_goals=None,
    split_vel_w=2.0,
    split_acc_w=0.8,
    split_lam_w=0.6,
):
    lambda_erg = fourier_ergodic_loss(
        u, x0, phik, k_expanded, lamk, hk, info_map,
        tau=tau,
        head_w=head_w,
        tail_w=tail_w,
        hotspots=hotspots,
        ordered_goals=ordered_goals,
        split_vel_w=split_vel_w,
        split_acc_w=split_acc_w,
        split_lam_w=split_lam_w,
    )
    # rollout trajectory to get final state x_T
    x = x0.clone()
    x_head = None
    head_idx = None
    if tau is not None:
        head_idx = min(max(int(tau), 1), u.shape[0])
    for step in u:
        x, _ = f(x, step[:2])
        if head_idx is not None:
            head_idx -= 1
            if head_idx == 0 and x_head is None:
                x_head = x
    term = 0.0
    if goal is not None and goal_w > 0:
        term = goal_w * (x - goal).pow(2).sum()
    if head_goal is not None and head_goal_w > 0:
        if x_head is None:
            x_head = x
        term = term + head_goal_w * (x_head - head_goal).pow(2).sum()
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

def mean_abs_map_delta(map_a, map_b):
    return float(torch.mean(torch.abs(map_a - map_b)).item())

def build_warm_seed_from_remainder(active_plan_u, active_plan_cursor, T):
    remainder = active_plan_u[active_plan_cursor:].detach()
    if remainder.shape[0] == 0:
        remainder = active_plan_u[-1:, :].detach()
    if remainder.shape[0] >= T:
        return remainder[:T].clone().detach()
    pad = remainder[-1:, :].repeat(T - remainder.shape[0], 1)
    return torch.cat([remainder, pad], dim=0).clone().detach()

# returns trajectory as list of states
def replanning(
    maps,
    _s,
    k_expanded,
    lamk,
    T=100,
    tau=20,
    small_change_thr=0.03,
    large_change_thr=0.10,
    iters_warm=400,
    iters_full=1500,
    min_iters_warm=60,
    min_iters_full=120,
    patience=25,
    rel_improve_tol=1e-4,
    guard_min_improve=0.01,
):
    x_init = torch.tensor([0.54, 0.3], dtype=torch.float32)
    x0 = x_init.clone()
    N, H, W = maps.shape
    trajectories, phik_list = [], []
    full_trajectories = []
    full_lambda_list = []
    head_len_list = []
    time_list = []
    diagnostics = []
    last_map = None

    active_phik = None
    active_plan_u = None
    active_plan_states = None
    active_plan_cursor = 0
    active_plan_mode = "forward"

    # Each new plan with the tail of the previous u
    for i, info_map in enumerate(maps):
        start_time = time.time()
        info_map = info_map / (info_map.max() + 1e-8)
        print(f"\n=== cycle {i} ===")

        if active_plan_u is not None and active_plan_cursor >= active_plan_u.shape[0]:
            active_plan_states = torch.flip(active_plan_states, dims=[0]).detach()
            active_plan_u = controls_from_states(active_plan_states).detach()
            active_plan_cursor = 0
            active_plan_mode = "reverse" if active_plan_mode == "forward" else "forward"
            print(f"Plan exhausted: switching to {active_plan_mode} execution.")

        delta = float("inf") if last_map is None else mean_abs_map_delta(info_map, last_map)
        if delta <= 1e-6:
            tier = "unchanged"
        elif delta <= small_change_thr:
            tier = "small"
        elif delta < large_change_thr:
            tier = "medium"
        else:
            tier = "large"

        tier_initial = tier
        fallback_to_full = False
        rel_improve_warm = None
        opt_diag = {
            "initial_loss": None,
            "final_loss": None,
            "iterations_run": 0,
            "early_stopped": False,
            "used_warm_start": False,
            "head_dist_h1_at_tau": 0.0,
            "mean_hotspot_coverage_dist": 0.0,
            "split_jump_metric": 0.0,
            "split_vel_jump": 0.0,
            "split_acc_jump": 0.0,
            "split_lam_jump": 0.0,
        }

        # No optimization when map is unchanged and we already have a plan.
        if tier == "unchanged" and active_plan_u is not None and active_plan_states is not None:
            print("Map unchanged: execute next head without replanning.")
        else:
            active_phik = phik_from_map(info_map.flatten(), _s, k_expanded)
            do_warm = tier in ("small", "medium") and active_plan_u is not None

            if do_warm:
                warm_seed = build_warm_seed_from_remainder(active_plan_u, active_plan_cursor, T)
                warm_u, warm_diag = optimize_trajectory(
                    x0, active_phik, k_expanded, lamk, hk, info_map,
                    u_prev=warm_seed, T=T, tau=tau,
                    max_iters=iters_warm, min_iters=min_iters_warm,
                    patience=patience, rel_improve_tol=rel_improve_tol,
                    return_diagnostics=True
                )
                rel_improve_warm = (warm_diag["initial_loss"] - warm_diag["final_loss"]) / (abs(warm_diag["initial_loss"]) + 1e-8)
                quality_bad = (rel_improve_warm < guard_min_improve) or (warm_diag["final_loss"] > warm_diag["initial_loss"])
                if quality_bad:
                    fallback_to_full = True
                    tier = "large_fallback"
                    print("Warm-start quality guard triggered: escalating to full replan.")
                    active_plan_u, opt_diag = optimize_trajectory(
                        x0, active_phik, k_expanded, lamk, hk, info_map,
                        u_prev=None, T=T, tau=tau,
                        max_iters=iters_full, min_iters=min_iters_full,
                        patience=patience, rel_improve_tol=rel_improve_tol,
                        return_diagnostics=True
                    )
                else:
                    active_plan_u = warm_u
                    opt_diag = warm_diag
            else:
                if tier == "unchanged":
                    tier = "large_missing_plan"
                print("Running full replan from current robot state.")
                active_plan_u, opt_diag = optimize_trajectory(
                    x0, active_phik, k_expanded, lamk, hk, info_map,
                    u_prev=None, T=T, tau=tau,
                    max_iters=iters_full, min_iters=min_iters_full,
                    patience=patience, rel_improve_tol=rel_improve_tol,
                    return_diagnostics=True
                )

            active_plan_states = rollout_states(x0, active_plan_u).detach()
            active_plan_cursor = 0
            active_plan_mode = "forward"

        phik_list.append(active_phik.detach())
        full_trajectories.append(active_plan_states.cpu().detach().numpy())
        full_lambda_list.append(compute_sensor_lambda(active_plan_u.detach()).cpu().numpy())

        steps_left = active_plan_u.shape[0] - active_plan_cursor
        exec_len = min(tau, steps_left)
        u_head = active_plan_u[active_plan_cursor:active_plan_cursor + exec_len]
        active_plan_cursor += exec_len
        head_len_list.append(int(exec_len))

        executed_traj_t = rollout_states(x0, u_head)
        executed_traj = executed_traj_t.cpu().detach().numpy()
        end_time = time.time()
        cycle_time = end_time - start_time
        time_list.append(cycle_time)
        trajectories.append(executed_traj)
        x0 = executed_traj_t[-1].detach()
        last_map = info_map.clone()

        behavior = compute_behavior_metrics(executed_traj_t, u_head.detach(), info_map)

        diagnostics.append({
            "cycle": int(i),
            "delta": float(delta),
            "tier_initial": tier_initial,
            "tier_used": tier,
            "fallback_to_full": bool(fallback_to_full),
            "iterations_used": int(opt_diag["iterations_run"]),
            "initial_loss": opt_diag["initial_loss"],
            "final_loss": opt_diag["final_loss"],
            "rel_improve_warm": rel_improve_warm,
            "cycle_time_sec": float(cycle_time),
            "high_info_dwell_ratio": behavior["high_info_dwell_ratio"],
            "high_info_dwell_area_norm": behavior["high_info_dwell_area_norm"],
            "low_info_transit_efficiency": behavior["low_info_transit_efficiency"],
            "low_info_idle_rate": behavior["low_info_idle_rate"],
            "sensor_precision": behavior["sensor_precision"],
            "sensor_recall": behavior["sensor_recall"],
            "sensor_f1_high_info": behavior["sensor_f1_high_info"],
            "avg_speed_high_info": behavior["avg_speed_high_info"],
            "avg_speed_low_info": behavior["avg_speed_low_info"],
            "head_dist_h1_at_tau": float(opt_diag.get("head_dist_h1_at_tau", 0.0)),
            "mean_hotspot_coverage_dist": float(opt_diag.get("mean_hotspot_coverage_dist", 0.0)),
            "split_jump_metric": float(opt_diag.get("split_jump_metric", 0.0)),
            "split_vel_jump": float(opt_diag.get("split_vel_jump", 0.0)),
            "split_acc_jump": float(opt_diag.get("split_acc_jump", 0.0)),
            "split_lam_jump": float(opt_diag.get("split_lam_jump", 0.0)),
        })

    return trajectories, full_trajectories, full_lambda_list, head_len_list, phik_list, time_list, diagnostics

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

# Build random map sequence each run (different order every execution)
full_maps = torch.stack(entropy_maps)
perm = torch.randperm(full_maps.shape[0])
maps = full_maps[perm[]]

# Sample grid of (0 to 1) to match the resolution
_, H, W = maps.shape
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

trajectory, full_trajectory, full_lambda, head_len_per_cycle, phik_list, time_list, replanning_diagnostics = replanning(
    maps, _s, k_expanded, lamk
)

print(f"Execution time: {time_list}")
print("Cycle diagnostics (delta, tier, iters, fallback, head_d1, cov_d, dwell, idle, f1, split):")
for d in replanning_diagnostics:
    print(
        f"  cycle={d['cycle']}, delta={d['delta']:.6f}, tier={d['tier_used']}, "
        f"iters={d['iterations_used']}, fallback={d['fallback_to_full']}, "
        f"head_d1={d['head_dist_h1_at_tau']:.3f}, cov_d={d['mean_hotspot_coverage_dist']:.3f}, "
        f"dwell={d['high_info_dwell_ratio']:.3f}, idle={d['low_info_idle_rate']:.3f}, "
        f"f1={d['sensor_f1_high_info']:.3f}, split={d['split_jump_metric']:.3f}"
    )

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

    # Top row: head/tail structure
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

    # Bottom row: sensor activation over all 100 timesteps
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
