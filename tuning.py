#!/usr/bin/env python
# coding: utf-8

import csv
import itertools
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class PlannerConfig:
    sensor_on_threshold: float = 0.80
    high_info_quantile: float = 0.80
    goal_hotspot_quantile: float = 0.82
    goal_distance_weight: float = 0.18
    goal_w: float = 4.50
    head_goal_w: float = 16.00

    path_info_w: float = 1.25
    coverage_w: float = 1.00
    speed_w: float = 0.40
    low_info_fast_w: float = 0.55

    sensor_bce_w: float = 2.20
    sensor_off_w: float = 2.40
    sensor_smooth_w: float = 0.02

    v_high: float = 0.010
    v_low: float = 0.065

    tail_w_opt: float = 0.90
    discount_rate: float = 0.30
    seam_vel_w: float = 12.0
    seam_acc_w: float = 5.0
    bridge_len: int = 12
    warm_xy_noise_std: float = 0.003


@torch.jit.script
def f(x, u):
    xnew = x[:2] + 0.07 * u[:2]
    xnew = torch.clamp(xnew, 0, 1)
    return xnew, xnew


def get_hk(k):
    _hk = (2.0 * k + np.sin(2 * k)) / (4 * k + 1e-8)
    _hk[np.isnan(_hk)] = 1.0
    return np.sqrt(np.prod(_hk))


def get_ck_weighted(tr, k_expanded, weights, hk):
    fk = torch.cos(tr[:, None, :] * k_expanded).prod(dim=-1)
    z = weights.sum() + 1e-8
    return (fk.T @ weights) / (z * hk)


def compute_sensor_lambda(u):
    return torch.clamp(torch.sigmoid(5.0 * u[:, 2]), 0.0, 1.0)


def normalize_info_map(info_map):
    map_min = torch.min(info_map)
    map_max = torch.max(info_map)
    return (info_map - map_min) / (map_max - map_min + 1e-8)


def sample_info_values(points, info_map):
    h, w = info_map.shape
    ix = (points[:, 0] * (w - 1)).long().clamp(0, w - 1)
    iy = (points[:, 1] * (h - 1)).long().clamp(0, h - 1)
    return info_map[iy, ix], ix, iy


def extract_hotspots(info_map_norm, max_hotspots=5, hotspot_quantile=0.80, min_sep=0.10):
    h, w = info_map_norm.shape
    flat = info_map_norm.flatten()
    threshold = torch.quantile(flat, hotspot_quantile)
    candidate_ids = torch.nonzero(flat >= threshold, as_tuple=False).flatten()
    if candidate_ids.numel() == 0:
        candidate_ids = torch.topk(flat, k=min(max_hotspots, flat.numel())).indices

    candidate_scores = flat[candidate_ids]
    sorted_ids = candidate_ids[torch.argsort(candidate_scores, descending=True)]
    denom_x = float(max(w - 1, 1))
    denom_y = float(max(h - 1, 1))
    hotspots = []
    scores = []

    for idx in sorted_ids:
        idx_i = int(idx.item())
        y = idx_i // w
        x = idx_i % w
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
        y = idx_i // w
        x = idx_i % w
        hotspots = [torch.tensor([x / denom_x, y / denom_y], dtype=torch.float32, device=info_map_norm.device)]
        scores = [flat[idx_i]]

    return torch.stack(hotspots, dim=0), torch.stack(scores, dim=0)


def choose_priority_goal(info_map, x0, cfg: PlannerConfig, max_hotspots=4):
    info_map_norm = normalize_info_map(info_map)
    hotspots, scores = extract_hotspots(
        info_map_norm,
        max_hotspots=max(max_hotspots, 5),
        hotspot_quantile=cfg.goal_hotspot_quantile,
        min_sep=0.12,
    )
    start_xy = x0[:2].to(device=hotspots.device, dtype=torch.float32)
    d_start = torch.norm(hotspots - start_xy.unsqueeze(0), dim=1)

    # Head goal: mostly highest-info, lightly distance-aware.
    head_rank = 2.0 * scores - cfg.goal_distance_weight * d_start
    head_idx = int(torch.argmax(head_rank).item())
    head_goal = hotspots[head_idx]

    # Terminal goal: next strong hotspot, biased to be reachable from h1.
    top_idx = int(torch.argmax(scores).item())
    candidate_ids = [j for j in range(hotspots.shape[0]) if j != head_idx]
    if candidate_ids:
        rem = torch.tensor(candidate_ids, dtype=torch.long, device=hotspots.device)
        d_head = torch.norm(hotspots[rem] - head_goal.unsqueeze(0), dim=1)
        terminal_rank = 1.4 * scores[rem] - 0.35 * d_head
        terminal_goal = hotspots[int(rem[torch.argmax(terminal_rank)].item())]
    else:
        terminal_goal = hotspots[top_idx]

    # Keep strongest points first for coverage term.
    sorted_ids = torch.argsort(scores, descending=True)
    hotspot_pool = hotspots[sorted_ids]
    return head_goal, terminal_goal, hotspot_pool


def fourier_ergodic_loss(
    u,
    x0,
    phik,
    k_expanded,
    lamk,
    hk,
    info_map,
    cfg: PlannerConfig,
    tau=None,
    head_w=1.0,
    tail_w=0.25,
    hotspots=None,
):
    displacements = 0.07 * u[:, :2]
    tr = torch.cumsum(displacements, dim=0) + x0
    tr = tr.clamp(0.0, 1.0)

    t_steps = u.shape[0]
    lam = compute_sensor_lambda(u)

    w = torch.ones(t_steps, device=u.device)
    if tau is not None:
        w[:tau] = head_w
        w[tau:] = tail_w

    disc = torch.exp(-cfg.discount_rate * torch.arange(t_steps, device=u.device, dtype=torch.float32) / t_steps)
    w = w * disc

    ck = get_ck_weighted(tr[:, :2], k_expanded, w, hk)

    phik_normed = (phik - phik.mean()) / (phik.std() + 1e-6)
    ck_normed = (ck - ck.mean()) / (ck.std() + 1e-6)
    ergodic_term = torch.sum(lamk * (phik_normed - ck_normed) ** 2)

    info_map_norm = normalize_info_map(info_map)
    info_values, _, _ = sample_info_values(tr[:, :2], info_map_norm)

    path_info_term = -cfg.path_info_w * torch.mean(info_values.pow(2))
    speed = torch.norm(displacements, dim=1)
    v_target = cfg.v_high + (1.0 - info_values) * (cfg.v_low - cfg.v_high)
    speed_term = cfg.speed_w * torch.mean((speed - v_target).pow(2))
    low_fast_gap = torch.relu(0.80 * cfg.v_low - speed)
    low_fast_term = cfg.low_info_fast_w * torch.mean((1.0 - info_values).pow(2) * low_fast_gap.pow(2))

    coverage_term = 0.0
    if hotspots is not None and hotspots.numel() > 0:
        dist2 = torch.sum((tr[:, None, :2] - hotspots[None, :, :]) ** 2, dim=2)
        smooth_min = -torch.logsumexp(-18.0 * dist2, dim=0) / 18.0
        hotspot_info, _, _ = sample_info_values(hotspots, info_map_norm)
        hotspot_w = hotspot_info / (torch.sum(hotspot_info) + 1e-8)
        coverage_term = cfg.coverage_w * torch.sum(hotspot_w * smooth_min)

    q_hi = torch.quantile(info_map_norm.flatten(), cfg.high_info_quantile)
    target_lam = torch.sigmoid((info_values - q_hi) / 0.03)
    eps = 1e-6
    sensor_bce = -(
        2.8 * target_lam * torch.log(lam + eps)
        + (1.0 - target_lam) * torch.log(1.0 - lam + eps)
    )
    sensor_bce_term = cfg.sensor_bce_w * torch.mean(sensor_bce)
    sensor_off_term = cfg.sensor_off_w * torch.mean(lam * (info_values < q_hi).float())
    sensor_smooth_term = (
        cfg.sensor_smooth_w * torch.mean((lam[1:] - lam[:-1]).pow(2))
        if lam.shape[0] > 1
        else 0.0
    )

    loss = (
        ergodic_term
        + 0.001 * torch.mean(u[:, :2] ** 2)
        + 0.001 * torch.sum((u[1:, :2] - u[:-1, :2]) ** 2)
        + 0.008 * torch.sum((u[2:, :2] - 2.0 * u[1:-1, :2] + u[:-2, :2]) ** 2)
        + 10 * torch.sum(torch.clamp_min(tr - 1, 0) ** 2 + torch.clamp_min(-tr, 0) ** 2)
        + path_info_term
        + speed_term
        + low_fast_term
        + coverage_term
        + sensor_bce_term
        + sensor_off_term
        + sensor_smooth_term
    )

    if tau is not None and t_steps >= 3:
        tau_i = int(min(max(tau, 1), t_steps - 2))
        seam_vel = cfg.seam_vel_w * torch.sum((u[tau_i, :2] - u[tau_i - 1, :2]) ** 2)
        seam_acc = cfg.seam_acc_w * torch.sum((u[tau_i + 1, :2] - 2.0 * u[tau_i, :2] + u[tau_i - 1, :2]) ** 2)
        loss = loss + seam_vel + seam_acc

    return loss


def loss_with_goal(
    u,
    x0,
    phik,
    k_expanded,
    lamk,
    hk,
    info_map,
    cfg: PlannerConfig,
    tau=None,
    head_w=1.0,
    tail_w=0.25,
    goal=None,
    goal_w=0.0,
    head_goal=None,
    head_goal_w=0.0,
    hotspots=None,
):
    lambda_erg = fourier_ergodic_loss(
        u, x0, phik, k_expanded, lamk, hk, info_map, cfg,
        tau=tau, head_w=head_w, tail_w=tail_w, hotspots=hotspots
    )
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


def optimize_trajectory(
    x0,
    phik,
    k_expanded,
    lamk,
    hk,
    info_map,
    cfg: PlannerConfig,
    u_prev=None,
    t_horizon=100,
    tau=20,
    num_iters=600,
    lr=1e-3,
):
    tau = int(min(max(tau, 1), t_horizon))
    head_goal, terminal_goal, hotspots = choose_priority_goal(info_map, x0, cfg)
    head_goal = head_goal.to(device=x0.device, dtype=torch.float32)
    terminal_goal = terminal_goal.to(device=x0.device, dtype=torch.float32)
    hotspots = hotspots.to(device=x0.device, dtype=torch.float32)

    if u_prev is None:
        opt_window = t_horizon
        head = torch.empty((opt_window, 3), device=x0.device, dtype=torch.float32)
        head.zero_()
        seg1 = min(tau, opt_window)
        if seg1 > 0:
            ctrl1 = ((head_goal - x0) / (0.07 * seg1)).clamp(min=-1.0, max=1.0)
            head[:seg1, :2] = ctrl1
        if seg1 < opt_window:
            seg2 = max(opt_window - seg1, 1)
            ctrl2 = ((terminal_goal - head_goal) / (0.07 * seg2)).clamp(min=-1.0, max=1.0)
            head[seg1:, :2] = ctrl2
        head[:, :2] += 0.004 * torch.randn_like(head[:, :2])

        # Seed lambda from expected information along seeded path.
        rough_tr = torch.cumsum(0.07 * head[:, :2], dim=0) + x0
        rough_tr = rough_tr.clamp(0.0, 1.0)
        info_map_norm = normalize_info_map(info_map)
        rough_info, _, _ = sample_info_values(rough_tr, info_map_norm)
        q_hi = torch.quantile(info_map_norm.flatten(), cfg.high_info_quantile)
        lam_target = torch.sigmoid((rough_info - q_hi) / 0.03).clamp(1e-4, 1.0 - 1e-4)
        head[:, 2] = torch.log(lam_target / (1.0 - lam_target)) / 5.0
        tail = None
        loss_tau = tau
    else:
        u_seed = u_prev.detach().to(device=x0.device, dtype=torch.float32)
        if u_seed.shape[0] >= t_horizon:
            u_seed = u_seed[:t_horizon]
        else:
            pad = u_seed[-1:, :].repeat(t_horizon - u_seed.shape[0], 1)
            u_seed = torch.cat([u_seed, pad], dim=0)

        opt_window = min(t_horizon, tau + max(int(cfg.bridge_len), 0))
        head = u_seed[:opt_window].clone()
        warm_len = min(tau, head.shape[0])
        if warm_len > 0 and cfg.warm_xy_noise_std > 0:
            head[:warm_len, :2] += cfg.warm_xy_noise_std * torch.randn_like(head[:warm_len, :2])
        tail = u_seed[opt_window:].clone().detach()
        loss_tau = tau
    head = torch.nn.Parameter(head)

    optimizer = torch.optim.Adam([head], lr=lr)

    def u_builder(h, t):
        if t is None:
            return h
        return torch.cat([h, t], dim=0)

    for _ in range(num_iters):
        optimizer.zero_grad()
        u = u_builder(head, tail)
        loss = loss_with_goal(
            u, x0, phik, k_expanded, lamk, hk, info_map, cfg,
            tau=loss_tau, head_w=1.0, tail_w=cfg.tail_w_opt,
            goal=terminal_goal, goal_w=cfg.goal_w,
            head_goal=head_goal, head_goal_w=cfg.head_goal_w,
            hotspots=hotspots,
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            head[:, :2].clamp_(min=-1.0, max=1.0)
            head[:, 2].clamp_(min=-1.0, max=1.0)

    u = u_builder(head, tail).detach()
    if u.shape[0] != t_horizon:
        if u.shape[0] > t_horizon:
            u = u[:t_horizon]
        else:
            padding = torch.zeros((t_horizon - u.shape[0], 3), device=u.device, dtype=torch.float32)
            padding[:, :] = u[-1, :].detach()
            u = torch.cat([u, padding], dim=0)
    return u


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


def phik_from_map(map_flattened, sample, k_expanded):
    fk_vals = torch.cos(sample[:, None, :] * k_expanded).prod(dim=-1)
    return (fk_vals * map_flattened[:, None]).sum(dim=0) / (map_flattened.sum() + 1e-8)


def replanning(maps, _s, k_expanded, lamk, hk, cfg: PlannerConfig, num_iters=600, t_horizon=100, tau=20):
    x_init = torch.tensor([0.54, 0.3], dtype=torch.float32)
    x0 = x_init.clone()
    trajectories, phik_list = [], []
    full_trajectories = []
    full_lambda_list = []
    head_len_list = []
    cycle_maps = []
    time_list = []
    last_map = None

    active_phik = None
    active_plan_u = None
    active_plan_states = None
    active_plan_cursor = 0
    active_plan_mode = "forward"

    def build_cycle_window_u(plan_u, cursor, horizon, pad_with_last=False):
        rem = plan_u[cursor:].detach()
        if rem.shape[0] == 0:
            rem = plan_u[-1:, :].detach()
        if rem.shape[0] >= horizon:
            return rem[:horizon].clone().detach()
        if pad_with_last:
            pad = rem[-1:, :].repeat(horizon - rem.shape[0], 1)
        else:
            pad = torch.zeros((horizon - rem.shape[0], 3), device=plan_u.device, dtype=plan_u.dtype)
        return torch.cat([rem, pad], dim=0).clone().detach()

    for i, info_map in enumerate(maps):
        start_time = time.time()
        info_map = info_map / (info_map.max() + 1e-8)
        map_changed = (last_map is None) or (not maps_are_same(info_map, last_map))

        if map_changed:
            active_phik = phik_from_map(info_map.flatten(), _s, k_expanded)
            warm_seed = None
            if active_plan_u is not None and active_plan_states is not None:
                warm_seed = build_cycle_window_u(active_plan_u, active_plan_cursor, t_horizon, pad_with_last=True)
            active_plan_u = optimize_trajectory(
                x0, active_phik, k_expanded, lamk, hk, info_map, cfg,
                u_prev=warm_seed, t_horizon=t_horizon, tau=tau, num_iters=num_iters
            )
            active_plan_states = rollout_states(x0, active_plan_u).detach()
            active_plan_cursor = 0
            active_plan_mode = "forward"
            last_map = info_map.clone()
        else:
            if active_plan_u is None or active_plan_states is None:
                active_phik = phik_from_map(info_map.flatten(), _s, k_expanded)
                active_plan_u = optimize_trajectory(
                    x0, active_phik, k_expanded, lamk, hk, info_map, cfg,
                    u_prev=None, t_horizon=t_horizon, tau=tau, num_iters=num_iters
                )
                active_plan_states = rollout_states(x0, active_plan_u).detach()
                active_plan_cursor = 0
                active_plan_mode = "forward"
            elif active_plan_cursor >= active_plan_u.shape[0]:
                active_plan_states = torch.flip(active_plan_states, dims=[0]).detach()
                active_plan_u = controls_from_states(active_plan_states).detach()
                active_plan_cursor = 0
                active_plan_mode = "reverse" if active_plan_mode == "forward" else "forward"

        cycle_plan_u = build_cycle_window_u(active_plan_u, active_plan_cursor, t_horizon)
        cycle_plan_states = rollout_states(x0, cycle_plan_u).detach()

        phik_list.append(active_phik.detach())
        full_trajectories.append(cycle_plan_states.cpu().detach().numpy())
        full_lambda_list.append(compute_sensor_lambda(cycle_plan_u).cpu().numpy())
        cycle_maps.append(info_map.detach().clone())

        steps_left = active_plan_u.shape[0] - active_plan_cursor
        exec_len = min(tau, steps_left)
        u_head = active_plan_u[active_plan_cursor:active_plan_cursor + exec_len]
        active_plan_cursor += exec_len
        head_len_list.append(int(exec_len))

        executed_traj_t = rollout_states(x0, u_head)
        trajectories.append(executed_traj_t.cpu().detach().numpy())
        x0 = executed_traj_t[-1].detach()
        time_list.append(time.time() - start_time)

    return trajectories, full_trajectories, full_lambda_list, head_len_list, phik_list, cycle_maps, time_list


def build_map_sequence(full_maps, n_cycles=4, repeat_prob=0.5, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    perm = torch.randperm(full_maps.shape[0])
    seed_pool = full_maps[perm[:n_cycles]]
    maps_list = [seed_pool[0]]
    for i in range(1, n_cycles):
        if random.random() < repeat_prob:
            maps_list.append(maps_list[-1].clone())
        else:
            maps_list.append(seed_pool[i])
    return torch.stack(maps_list, dim=0)


def compute_run_metrics(full_trajectory, full_lambda, head_len_per_cycle, cycle_maps, cfg: PlannerConfig):
    head_peak_dist = []
    peak_min_dist = []
    high_dwell = []
    sensor_f1 = []
    low_info_on_rate = []

    for i in range(len(full_trajectory)):
        tr = torch.tensor(full_trajectory[i], dtype=torch.float32)
        pts = tr[1:, :2]
        lam = torch.tensor(full_lambda[i], dtype=torch.float32)
        info_map = normalize_info_map(cycle_maps[i])

        flat = info_map.flatten()
        peak_idx = int(torch.argmax(flat).item())
        h, w = info_map.shape
        py = peak_idx // w
        px = peak_idx % w
        peak_pt = torch.tensor([px / max(w - 1, 1), py / max(h - 1, 1)], dtype=torch.float32)

        head_len = max(1, min(head_len_per_cycle[i], pts.shape[0]))
        head_end = pts[head_len - 1]
        head_peak_dist.append(float(torch.norm(head_end - peak_pt).item()))

        peak_min_dist.append(float(torch.min(torch.norm(pts - peak_pt.unsqueeze(0), dim=1)).item()))

        info_vals, _, _ = sample_info_values(pts, info_map)
        high_thr = torch.quantile(info_map.flatten(), cfg.high_info_quantile)
        high_mask = info_vals >= high_thr
        low_mask = ~high_mask
        high_dwell.append(float(high_mask.float().mean().item()))

        sensor_on = lam >= cfg.sensor_on_threshold
        tp = int((sensor_on & high_mask).sum().item())
        pred = int(sensor_on.sum().item())
        actual = int(high_mask.sum().item())
        prec = tp / max(pred, 1)
        rec = tp / max(actual, 1)
        f1 = 2.0 * prec * rec / max(prec + rec, 1e-8)
        sensor_f1.append(float(f1))

        low_on = int((sensor_on & low_mask).sum().item())
        low_total = int(low_mask.sum().item())
        low_info_on_rate.append(low_on / max(low_total, 1))

    return {
        "head_to_peak_dist_mean": float(np.mean(head_peak_dist)),
        "peak_min_dist_mean": float(np.mean(peak_min_dist)),
        "high_info_dwell_ratio_mean": float(np.mean(high_dwell)),
        "sensor_f1_high_info_mean": float(np.mean(sensor_f1)),
        "low_info_on_rate_mean": float(np.mean(low_info_on_rate)),
    }


def _norm(vals):
    arr = np.asarray(vals, dtype=np.float64)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax - vmin < 1e-12:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def add_composite_scores(rows):
    head = _norm([r["head_to_peak_dist_mean"] for r in rows])
    peak = _norm([r["peak_min_dist_mean"] for r in rows])
    dwell = _norm([r["high_info_dwell_ratio_mean"] for r in rows])
    f1 = _norm([r["sensor_f1_high_info_mean"] for r in rows])
    low_on = _norm([r["low_info_on_rate_mean"] for r in rows])

    for i, r in enumerate(rows):
        score = (
            0.30 * (1.0 - head[i])
            + 0.25 * (1.0 - peak[i])
            + 0.20 * dwell[i]
            + 0.20 * f1[i]
            + 0.05 * (1.0 - low_on[i])
        )
        r["composite_score"] = float(score)


def plot_run(
    run_idx,
    output_path,
    trajectories,
    full_trajectory,
    full_lambda,
    head_len_per_cycle,
    cycle_maps,
    x_grid,
    y_grid,
    cfg: PlannerConfig,
):
    num_cycles = len(trajectories)
    fig, axes = plt.subplots(2, num_cycles, figsize=(4 * num_cycles, 8), squeeze=False)
    sensor_cmap = LinearSegmentedColormap.from_list("sensor_white_red", ["#ffffff", "#ff0000"])
    sc_last = None

    for i in range(num_cycles):
        tr = trajectories[i]
        full_tr = full_trajectory[i]
        lam_i = full_lambda[i]
        head_len = head_len_per_cycle[i]
        full_pts = full_tr[1:, :2]
        head_pts = full_pts[:head_len]
        tail_pts = full_pts[head_len:]
        lam_head = lam_i[:head_len]
        lam_tail = lam_i[head_len:]
        info_map = normalize_info_map(cycle_maps[i]).cpu().numpy()
        ax_ht = axes[0, i]
        ax_sa = axes[1, i]

        ax_ht.imshow(info_map, extent=[0, 1, 0, 1], origin="lower", cmap="viridis")
        ax_ht.contourf(x_grid.numpy(), y_grid.numpy(), info_map, cmap="viridis")
        if tail_pts.shape[0] > 0:
            ax_ht.scatter(tail_pts[:, 0], tail_pts[:, 1], s=16, c="white", alpha=0.40)
        if head_pts.shape[0] > 0:
            ax_ht.scatter(head_pts[:, 0], head_pts[:, 1], s=34, c="red", edgecolors="black", linewidths=0.5)
        ax_ht.scatter(full_tr[0, 0], full_tr[0, 1], c="w", s=50, marker="X")
        ax_ht.scatter(tr[-1, 0], tr[-1, 1], c="yellow", s=35)
        ax_ht.set_title(f"Map {i + 1}: Head/Tail")
        ax_ht.set_aspect("equal")
        ax_ht.set_xlim(0, 1)
        ax_ht.set_ylim(0, 1)
        if i == 0:
            legend_handles_ht = [
                Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor="white", markeredgecolor="none", alpha=0.40, label="Tail"),
                Line2D([0], [0], marker="o", linestyle="None", markersize=7, markerfacecolor="red", markeredgecolor="black", label="Head"),
                Line2D([0], [0], marker="X", linestyle="None", markersize=7, markerfacecolor="white", markeredgecolor="white", label="Plan start"),
                Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor="yellow", markeredgecolor="none", label="Executed head end"),
            ]
            ax_ht.legend(handles=legend_handles_ht, loc="lower left", fontsize=7, framealpha=0.85)

        ax_sa.imshow(info_map, extent=[0, 1, 0, 1], origin="lower", cmap="viridis")
        ax_sa.contourf(x_grid.numpy(), y_grid.numpy(), info_map, cmap="viridis")
        if tail_pts.shape[0] > 0:
            sc_last = ax_sa.scatter(
                tail_pts[:, 0], tail_pts[:, 1],
                s=14, c=lam_tail, cmap=sensor_cmap, vmin=0.0, vmax=1.0, alpha=0.55
            )
        if head_pts.shape[0] > 0:
            sc_last = ax_sa.scatter(
                head_pts[:, 0], head_pts[:, 1],
                s=34, c=lam_head, cmap=sensor_cmap, vmin=0.0, vmax=1.0,
                edgecolors="black", linewidths=0.5, alpha=0.95
            )
        sensor_on = lam_i >= cfg.sensor_on_threshold
        if np.any(sensor_on):
            ax_sa.scatter(
                full_pts[sensor_on, 0],
                full_pts[sensor_on, 1],
                s=70,
                facecolors="none",
                edgecolors="cyan",
                linewidths=1.5,
            )
        ax_sa.scatter(full_tr[0, 0], full_tr[0, 1], c="w", s=50, marker="X")
        ax_sa.scatter(tr[-1, 0], tr[-1, 1], c="yellow", s=35)
        ax_sa.set_title(f"Map {i + 1}: Sensor Lambda (100 Steps)")
        ax_sa.set_aspect("equal")
        ax_sa.set_xlim(0, 1)
        ax_sa.set_ylim(0, 1)
        if i == 0:
            legend_handles_sa = [
                Line2D([0], [0], marker="o", linestyle="None", markersize=5, markerfacecolor="red", markeredgecolor="none", alpha=0.55, label="Tail (lambda color)"),
                Line2D([0], [0], marker="o", linestyle="None", markersize=7, markerfacecolor="red", markeredgecolor="black", label="Head (lambda color)"),
                Line2D([0], [0], marker="o", linestyle="None", markersize=8, markerfacecolor="none", markeredgecolor="cyan", label=f"Sensor ON (lambda >= {cfg.sensor_on_threshold:.1f})"),
                Line2D([0], [0], marker="X", linestyle="None", markersize=7, markerfacecolor="white", markeredgecolor="white", label="Plan start"),
                Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor="yellow", markeredgecolor="none", label="Executed head end"),
            ]
            ax_sa.legend(handles=legend_handles_sa, loc="lower left", fontsize=7, framealpha=0.8)

    fig.suptitle(
        f"Run {run_idx:03d} | goal_w={cfg.goal_w}, head_goal_w={cfg.head_goal_w}, "
        f"coverage_w={cfg.coverage_w}, sensor_off_w={cfg.sensor_off_w}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if sc_last is not None:
        cbar = fig.colorbar(sc_last, ax=axes[1, :].tolist(), shrink=0.85, pad=0.02)
        cbar.set_label("Sensor activation lambda (white=OFF, red=ON)")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_entropy_maps(root: Path):
    candidates = [
        root / "entropy_maps",
        Path("/Users/cindy/Desktop/ergodic-search/entropy_maps"),
    ]
    folder = None
    for c in candidates:
        if c.exists():
            folder = c
            break
    if folder is None:
        raise FileNotFoundError("Could not find entropy_maps folder.")

    files = sorted([p for p in folder.iterdir() if p.is_file()])
    maps = []
    for p in files:
        if p.suffix.lower() == ".npy":
            arr = np.load(str(p))
        elif p.suffix.lower() == ".txt":
            arr = np.loadtxt(str(p), dtype=np.float32)
        else:
            continue
        maps.append(torch.from_numpy(arr).float())
    if not maps:
        raise RuntimeError(f"No map files found in {folder}")
    return torch.stack(maps)


def write_csv(path: Path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def generate_param_configs(base_cfg: PlannerConfig):
    # 54 combinations: 3 x 3 x 2 x 3
    grid = {
        "goal_w": [3.0, 4.5, 6.0],
        "head_goal_w": [10.0, 14.0, 18.0],
        "coverage_w": [0.80, 1.10],
        "sensor_off_w": [1.8, 2.4, 3.0],
    }
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        updates = dict(zip(keys, combo))
        yield replace(base_cfg, **updates)


def main():
    root = Path(__file__).resolve().parent
    out_dir = root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_maps = load_entropy_maps(root)
    env_seed = os.environ.get("MAP_SEQ_SEED")
    if env_seed is not None:
        map_seed = int(env_seed)
    else:
        map_seed = random.SystemRandom().randint(1, 10**9)
    maps = build_map_sequence(full_maps, n_cycles=4, repeat_prob=0.5, seed=map_seed)
    print(f"Map sequence seed for this tuning run: {map_seed}")

    _, h, w = maps.shape
    xs = torch.linspace(0, 1, w)
    ys = torch.linspace(0, 1, h)
    x_grid, y_grid = torch.meshgrid(xs, ys, indexing="xy")
    _s = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=1)

    k1, k2 = np.meshgrid(np.arange(0, 20), np.arange(0, 20))
    k = torch.tensor(np.stack([k1.ravel(), k2.ravel()], axis=-1), dtype=torch.float32)
    lamk = torch.exp(-0.8 * torch.norm(k, dim=1))
    hk = torch.clamp(torch.tensor([get_hk(ki) for ki in k.numpy()]), min=1e-6)
    k_expanded = k.unsqueeze(0)

    base_cfg = PlannerConfig()
    configs = list(generate_param_configs(base_cfg))
    print(f"Total parameter combinations: {len(configs)}")

    results = []
    for run_idx, cfg in enumerate(configs, start=1):
        print(f"[{run_idx:03d}/{len(configs)}] Running with goal_w={cfg.goal_w}, head_goal_w={cfg.head_goal_w}, coverage_w={cfg.coverage_w}, sensor_off_w={cfg.sensor_off_w}")
        t0 = time.time()
        trajectories, full_trajectory, full_lambda, head_len_per_cycle, _phik_list, cycle_maps, time_list = replanning(
            maps, _s, k_expanded, lamk, hk, cfg, num_iters=600, t_horizon=100, tau=20
        )
        wall_time = time.time() - t0

        metrics = compute_run_metrics(full_trajectory, full_lambda, head_len_per_cycle, cycle_maps, cfg)
        fig_name = f"run_{run_idx:03d}.png"
        fig_path = out_dir / fig_name
        plot_run(
            run_idx, fig_path, trajectories, full_trajectory, full_lambda, head_len_per_cycle,
            cycle_maps, x_grid, y_grid, cfg
        )

        row = {
            "run_id": run_idx,
            "figure_file": fig_name,
            "wall_time_sec": float(wall_time),
            "cycle_time_total_sec": float(np.sum(time_list)),
            **asdict(cfg),
            **metrics,
        }
        results.append(row)

    add_composite_scores(results)
    ranked = sorted(results, key=lambda r: r["composite_score"], reverse=True)
    for rank, r in enumerate(ranked, start=1):
        r["rank"] = rank

    write_csv(out_dir / "tuning_results_all_combinations.csv", results)
    write_csv(out_dir / "tuning_ranked_results.csv", ranked)

    best = ranked[0]
    print("\nTop 5 runs by composite score:")
    for r in ranked[:5]:
        print(
            f"  rank={r['rank']}, run={r['run_id']:03d}, score={r['composite_score']:.4f}, "
            f"head_dist={r['head_to_peak_dist_mean']:.4f}, peak_min={r['peak_min_dist_mean']:.4f}, "
            f"dwell={r['high_info_dwell_ratio_mean']:.4f}, f1={r['sensor_f1_high_info_mean']:.4f}, "
            f"low_on={r['low_info_on_rate_mean']:.4f}"
        )

    summary_path = out_dir / "best_config_summary.csv"
    write_csv(summary_path, [best])
    print(f"\nSaved comparison outputs to: {out_dir}")
    print(f"Best run figure: {best['figure_file']}")


if __name__ == "__main__":
    main()
