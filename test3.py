#!/usr/bin/env python
# coding: utf-8

import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


# -------------------------
# Planner / behavior knobs
# -------------------------
T_HORIZON = 100
HEAD_STEPS = 20
BRIDGE_LEN = 12

SENSOR_ON_THRESHOLD = 0.60
HIGH_INFO_QUANTILE = 0.70
HOTSPOT_QUANTILE = 0.84
MAX_HOTSPOTS = 6
HOTSPOT_MIN_SEP = 0.12

GOAL_DISTANCE_WEIGHT = 0.20
H2H3_DISTANCE_WEIGHT = 0.60
HEAD_GOAL_W = 12.0
TERM_GOAL_W = 2.5

PATH_INFO_W = 1.10
COVERAGE_W = 0.90
ORDER_WEIGHTS = (2.4, 1.6, 1.1)

SPEED_W = 0.38
LOW_INFO_FAST_W = 0.35
V_HIGH = 0.010
V_LOW = 0.060

SENSOR_BCE_W = 1.80
SENSOR_OFF_W = 2.10
SENSOR_BIN_W = 0.02
SENSOR_SMOOTH_W = 0.02
SENSOR_TEMP = 0.05

CONTROL_W = 0.001
SMOOTH_W = 0.001
CURV_W = 0.006
BARRIER_W = 10.0

TAIL_W_OPT = 0.90
DISCOUNT_RATE = 0.30
SEAM_VEL_W = 8.0
SEAM_ACC_W = 3.0
SEAM_LAM_W = 1.0

UNCHANGED_EPS = 1e-6
SMALL_CHANGE_THR = 0.03
LARGE_CHANGE_THR = 0.10
HEAD_QUALITY_DIST_THR = 0.13

LR = 1e-3
ITERS_FULL = 800
ITERS_WARM = 260
ITERS_REFINE = 140
MIN_ITERS_FULL = 120
MIN_ITERS_WARM = 60
PATIENCE = 25
REL_IMPROVE_TOL = 1e-4


# -------------------------
# Map sequence options
# -------------------------
N_CYCLES = 4
USE_REPEAT_MODE = True
REPEAT_PROB = 0.50
RUN_SEED = None  # set int for reproducibility; None => different each run


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


def extract_hotspots(info_map_norm, max_hotspots=MAX_HOTSPOTS, hotspot_quantile=HOTSPOT_QUANTILE, min_sep=HOTSPOT_MIN_SEP):
    h, w = info_map_norm.shape
    flat = info_map_norm.flatten()
    thr = torch.quantile(flat, hotspot_quantile)
    candidate_ids = torch.nonzero(flat >= thr, as_tuple=False).flatten()
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
        p = torch.tensor([x / denom_x, y / denom_y], dtype=torch.float32, device=info_map_norm.device)
        if hotspots:
            min_dist = torch.stack([torch.norm(p - q) for q in hotspots]).min()
            if float(min_dist.item()) < min_sep:
                continue
        hotspots.append(p)
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


def choose_ordered_goals(info_map, x0):
    info_map_norm = normalize_info_map(info_map)
    hotspots, scores = extract_hotspots(info_map_norm)
    start_xy = x0[:2].to(dtype=torch.float32, device=hotspots.device)

    d_start = torch.norm(hotspots - start_xy.unsqueeze(0), dim=1)
    h1_score = 2.0 * scores - GOAL_DISTANCE_WEIGHT * d_start
    h1_idx = int(torch.argmax(h1_score).item())

    ordered_ids = [h1_idx]
    remaining = [j for j in range(hotspots.shape[0]) if j != h1_idx]
    prev = h1_idx
    for _ in range(2):
        if not remaining:
            ordered_ids.append(prev)
            continue
        rem = torch.tensor(remaining, dtype=torch.long, device=hotspots.device)
        d_prev = torch.norm(hotspots[rem] - hotspots[prev].unsqueeze(0), dim=1)
        score_next = 1.4 * scores[rem] - H2H3_DISTANCE_WEIGHT * d_prev
        pick = int(rem[torch.argmax(score_next)].item())
        ordered_ids.append(pick)
        remaining.remove(pick)
        prev = pick

    ordered_goals = hotspots[torch.tensor(ordered_ids[:3], dtype=torch.long, device=hotspots.device)]
    sorted_ids = torch.argsort(scores, descending=True)
    hotspot_pool = hotspots[sorted_ids]
    return ordered_goals, hotspot_pool


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


def phik_from_map(map_flattened, sample, k_expanded):
    fk_vals = torch.cos(sample[:, None, :] * k_expanded).prod(dim=-1)
    return (fk_vals * map_flattened[:, None]).sum(dim=0) / (map_flattened.sum() + 1e-8)


def mean_abs_map_delta(map_a, map_b):
    return float(torch.mean(torch.abs(map_a - map_b)).item())


def build_warm_seed_from_remainder(active_plan_u, active_plan_cursor, horizon):
    remainder = active_plan_u[active_plan_cursor:].detach()
    if remainder.shape[0] == 0:
        remainder = active_plan_u[-1:, :].detach()
    if remainder.shape[0] >= horizon:
        return remainder[:horizon].clone().detach()
    pad = remainder[-1:, :].repeat(horizon - remainder.shape[0], 1)
    return torch.cat([remainder, pad], dim=0).clone().detach()


def evaluate_head_quality(active_plan_u, active_plan_cursor, x0, info_map, tau):
    u_window = build_warm_seed_from_remainder(active_plan_u, active_plan_cursor, T_HORIZON)
    states = rollout_states(x0, u_window)
    head_pts = states[1:tau + 1, :2]
    ordered_goals, _ = choose_ordered_goals(info_map, x0)
    d = torch.norm(head_pts - ordered_goals[0].unsqueeze(0), dim=1)
    return float(torch.min(d).item())


def seed_controls_from_goals(x0, ordered_goals, info_map, t_horizon, tau):
    u = torch.zeros((t_horizon, 3), dtype=torch.float32, device=x0.device)
    g1 = ordered_goals[0]
    g2 = ordered_goals[1]
    g3 = ordered_goals[2]

    seg1 = min(tau, t_horizon)
    seg2 = min(40, max(t_horizon - seg1, 0))
    seg3 = max(t_horizon - seg1 - seg2, 0)

    if seg1 > 0:
        ctrl1 = ((g1 - x0) / (0.07 * seg1)).clamp(min=-1.0, max=1.0)
        u[:seg1, :2] = ctrl1
    if seg2 > 0:
        ctrl2 = ((g2 - g1) / (0.07 * seg2)).clamp(min=-1.0, max=1.0)
        u[seg1:seg1 + seg2, :2] = ctrl2
    if seg3 > 0:
        ctrl3 = ((g3 - g2) / (0.07 * seg3)).clamp(min=-1.0, max=1.0)
        u[seg1 + seg2:, :2] = ctrl3

    u[:, :2] += 0.004 * torch.randn_like(u[:, :2])

    rough_tr = torch.cumsum(0.07 * u[:, :2], dim=0) + x0
    rough_tr = rough_tr.clamp(0.0, 1.0)
    info_map_norm = normalize_info_map(info_map)
    rough_info, _, _ = sample_info_values(rough_tr, info_map_norm)
    q_hi = torch.quantile(info_map_norm.flatten(), HIGH_INFO_QUANTILE)
    lam_target = torch.sigmoid((rough_info - q_hi) / SENSOR_TEMP).clamp(1e-4, 1.0 - 1e-4)
    u[:, 2] = torch.log(lam_target / (1.0 - lam_target)) / 5.0
    return u


def fourier_ergodic_loss(u, x0, phik, k_expanded, lamk, hk, info_map, tau=None, head_w=1.0, tail_w=TAIL_W_OPT, hotspots=None, ordered_goals=None):
    displacements = 0.07 * u[:, :2]
    tr = torch.cumsum(displacements, dim=0) + x0
    tr = tr.clamp(0.0, 1.0)
    t_steps = u.shape[0]

    lam = compute_sensor_lambda(u)
    w = torch.ones(t_steps, device=u.device)
    if tau is not None:
        w[:tau] = head_w
        w[tau:] = tail_w
    disc = torch.exp(-DISCOUNT_RATE * torch.arange(t_steps, device=u.device, dtype=torch.float32) / t_steps)
    w = w * disc

    ck = get_ck_weighted(tr[:, :2], k_expanded, lam * w, hk)
    phik_normed = (phik - phik.mean()) / (phik.std() + 1e-6)
    ck_normed = (ck - ck.mean()) / (ck.std() + 1e-6)
    ergodic_term = torch.sum(lamk * (phik_normed - ck_normed) ** 2)

    info_map_norm = normalize_info_map(info_map)
    info_values, _, _ = sample_info_values(tr[:, :2], info_map_norm)

    path_info_term = -PATH_INFO_W * torch.mean(info_values.pow(2))

    speed = torch.norm(displacements, dim=1)
    v_target = V_HIGH + (1.0 - info_values) * (V_LOW - V_HIGH)
    speed_term = SPEED_W * torch.mean((speed - v_target).pow(2))
    low_fast_gap = torch.relu(0.75 * V_LOW - speed)
    low_fast_term = LOW_INFO_FAST_W * torch.mean((1.0 - info_values).pow(2) * low_fast_gap.pow(2))

    coverage_term = 0.0
    if hotspots is not None and hotspots.numel() > 0:
        dist2 = torch.sum((tr[:, None, :2] - hotspots[None, :, :]) ** 2, dim=2)
        smooth_min = -torch.logsumexp(-20.0 * dist2, dim=0) / 20.0
        hotspot_info, _, _ = sample_info_values(hotspots, info_map_norm)
        hotspot_w = hotspot_info / (torch.sum(hotspot_info) + 1e-8)
        coverage_term = COVERAGE_W * torch.sum(hotspot_w * smooth_min)

    ordered_term = 0.0
    if ordered_goals is not None and ordered_goals.shape[0] >= 3:
        t1 = int(min(max(tau if tau is not None else t_steps // 5, 1), t_steps))
        t2 = int(min(max(t1 + 35, t1 + 1), t_steps))
        segs = [(0, t1, ordered_goals[0]), (t1, t2, ordered_goals[1]), (t2, t_steps, ordered_goals[2])]
        beta = 30.0
        seg_terms = []
        for s0, s1, goal in segs:
            if s1 <= s0:
                continue
            seg = tr[s0:s1, :2]
            d2 = torch.sum((seg - goal.unsqueeze(0)) ** 2, dim=1)
            soft_min = -torch.logsumexp(-beta * d2, dim=0) / beta
            seg_terms.append(soft_min)
        if seg_terms:
            seg_w = torch.tensor(ORDER_WEIGHTS[:len(seg_terms)], dtype=torch.float32, device=u.device)
            ordered_term = torch.sum(seg_w * torch.stack(seg_terms))

    q_hi = torch.quantile(info_map_norm.flatten(), HIGH_INFO_QUANTILE)
    target_lam = torch.sigmoid((info_values - q_hi) / SENSOR_TEMP)
    eps = 1e-6
    sensor_bce = -(
        2.5 * target_lam * torch.log(lam + eps)
        + (1.0 - target_lam) * torch.log(1.0 - lam + eps)
    )
    sensor_bce_term = SENSOR_BCE_W * torch.mean(sensor_bce)
    sensor_off_term = SENSOR_OFF_W * torch.mean(lam * (1.0 - target_lam))
    sensor_bin_term = SENSOR_BIN_W * torch.mean(lam * (1.0 - lam))
    sensor_smooth_term = SENSOR_SMOOTH_W * torch.mean((lam[1:] - lam[:-1]).pow(2)) if lam.shape[0] > 1 else 0.0

    seam_term = 0.0
    if tau is not None and t_steps >= 3:
        tau_i = int(min(max(tau, 1), t_steps - 2))
        seam_vel = SEAM_VEL_W * torch.sum((u[tau_i, :2] - u[tau_i - 1, :2]) ** 2)
        seam_acc = SEAM_ACC_W * torch.sum((u[tau_i + 1, :2] - 2.0 * u[tau_i, :2] + u[tau_i - 1, :2]) ** 2)
        seam_lam = SEAM_LAM_W * (lam[tau_i] - lam[tau_i - 1]).pow(2)
        seam_term = seam_vel + seam_acc + seam_lam

    loss = (
        ergodic_term
        + CONTROL_W * torch.mean(u[:, :2] ** 2)
        + SMOOTH_W * torch.sum((u[1:, :2] - u[:-1, :2]) ** 2)
        + CURV_W * torch.sum((u[2:, :2] - 2.0 * u[1:-1, :2] + u[:-2, :2]) ** 2)
        + BARRIER_W * torch.sum(torch.clamp_min(tr - 1, 0) ** 2 + torch.clamp_min(-tr, 0) ** 2)
        + path_info_term
        + speed_term
        + low_fast_term
        + coverage_term
        + ordered_term
        + sensor_bce_term
        + sensor_off_term
        + sensor_bin_term
        + sensor_smooth_term
        + seam_term
    )
    return loss


def loss_with_goal(u, x0, phik, k_expanded, lamk, hk, info_map, tau=None, head_w=1.0, tail_w=TAIL_W_OPT, goal=None, goal_w=0.0, head_goal=None, head_goal_w=0.0, hotspots=None, ordered_goals=None):
    l_erg = fourier_ergodic_loss(
        u, x0, phik, k_expanded, lamk, hk, info_map,
        tau=tau, head_w=head_w, tail_w=tail_w,
        hotspots=hotspots, ordered_goals=ordered_goals
    )
    x = x0.clone()
    x_head = None
    head_idx = min(max(int(tau), 1), u.shape[0]) if tau is not None else None
    for i, step in enumerate(u):
        x, _ = f(x, step[:2])
        if head_idx is not None and (i + 1) == head_idx:
            x_head = x
    term = 0.0
    if goal is not None and goal_w > 0:
        term = term + goal_w * torch.sum((x - goal) ** 2)
    if head_goal is not None and head_goal_w > 0:
        if x_head is None:
            x_head = x
        term = term + head_goal_w * torch.sum((x_head - head_goal) ** 2)
    return l_erg + term


def optimize_trajectory(
    x0,
    phik,
    k_expanded,
    lamk,
    hk,
    info_map,
    u_prev=None,
    t_horizon=T_HORIZON,
    tau=HEAD_STEPS,
    max_iters=ITERS_FULL,
    min_iters=MIN_ITERS_FULL,
    lr=LR,
):
    tau = int(min(max(tau, 1), t_horizon))
    ordered_goals, hotspots = choose_ordered_goals(info_map, x0)
    ordered_goals = ordered_goals.to(device=x0.device, dtype=torch.float32)
    hotspots = hotspots.to(device=x0.device, dtype=torch.float32)
    head_goal = ordered_goals[0]
    term_goal = ordered_goals[2]

    if u_prev is None:
        opt_window = t_horizon
        head = seed_controls_from_goals(x0, ordered_goals, info_map, t_horizon, tau)
        tail = None
    else:
        u_seed = u_prev.detach().to(device=x0.device, dtype=torch.float32)
        if u_seed.shape[0] >= t_horizon:
            u_seed = u_seed[:t_horizon]
        else:
            pad = u_seed[-1:, :].repeat(t_horizon - u_seed.shape[0], 1)
            u_seed = torch.cat([u_seed, pad], dim=0)
        opt_window = min(t_horizon, tau + BRIDGE_LEN)
        head = u_seed[:opt_window].clone()
        warm_len = min(tau, opt_window)
        head[:warm_len, :2] += 0.003 * torch.randn_like(head[:warm_len, :2])
        tail = u_seed[opt_window:].clone().detach()

    head = torch.nn.Parameter(head)
    optimizer = torch.optim.Adam([head], lr=lr)

    def build_u(h, t):
        return h if t is None else torch.cat([h, t], dim=0)

    with torch.no_grad():
        init_loss = float(
            loss_with_goal(
                build_u(head, tail),
                x0, phik, k_expanded, lamk, hk, info_map,
                tau=tau, head_w=1.0, tail_w=TAIL_W_OPT,
                goal=term_goal, goal_w=TERM_GOAL_W,
                head_goal=head_goal, head_goal_w=HEAD_GOAL_W,
                hotspots=hotspots, ordered_goals=ordered_goals
            ).item()
        )

    last_loss = init_loss
    plateau = 0
    iters_used = 0
    for i in range(max_iters):
        optimizer.zero_grad()
        u = build_u(head, tail)
        loss = loss_with_goal(
            u,
            x0, phik, k_expanded, lamk, hk, info_map,
            tau=tau, head_w=1.0, tail_w=TAIL_W_OPT,
            goal=term_goal, goal_w=TERM_GOAL_W,
            head_goal=head_goal, head_goal_w=HEAD_GOAL_W,
            hotspots=hotspots, ordered_goals=ordered_goals
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            head[:, :2].clamp_(min=-1.0, max=1.0)
            head[:, 2].clamp_(min=-2.0, max=2.0)

        curr_loss = float(loss.item())
        rel_improve = (last_loss - curr_loss) / (abs(last_loss) + 1e-8)
        if rel_improve < REL_IMPROVE_TOL:
            plateau += 1
        else:
            plateau = 0
        last_loss = curr_loss
        iters_used = i + 1
        if iters_used >= min_iters and plateau >= PATIENCE:
            break

    u_out = build_u(head, tail).detach()
    if u_out.shape[0] < t_horizon:
        pad = u_out[-1:, :].repeat(t_horizon - u_out.shape[0], 1)
        u_out = torch.cat([u_out, pad], dim=0)
    else:
        u_out = u_out[:t_horizon]

    diag = {
        "iters_used": int(iters_used),
        "init_loss": float(init_loss),
        "final_loss": float(last_loss),
        "opt_window": int(opt_window),
    }
    return u_out, diag


def replanning(maps, _s, k_expanded, lamk, hk, t_horizon=T_HORIZON, tau=HEAD_STEPS):
    x0 = torch.tensor([0.54, 0.30], dtype=torch.float32)
    trajectories = []
    full_trajectories = []
    full_lambda_list = []
    head_len_list = []
    phik_list = []
    cycle_maps = []
    time_list = []
    diagnostics = []

    last_map = None
    active_phik = None
    active_plan_u = None
    active_plan_states = None
    active_plan_cursor = 0
    active_plan_mode = "forward"

    for i, info_map in enumerate(maps):
        t0 = time.time()
        info_map = normalize_info_map(info_map)

        if active_plan_u is not None and active_plan_cursor >= active_plan_u.shape[0]:
            active_plan_states = torch.flip(active_plan_states, dims=[0]).detach()
            active_plan_u = controls_from_states(active_plan_states).detach()
            active_plan_cursor = 0
            active_plan_mode = "reverse" if active_plan_mode == "forward" else "forward"

        delta = float("inf") if last_map is None else mean_abs_map_delta(info_map, last_map)
        if delta <= UNCHANGED_EPS:
            tier = "unchanged"
        elif delta <= SMALL_CHANGE_THR:
            tier = "small"
        elif delta < LARGE_CHANGE_THR:
            tier = "medium"
        else:
            tier = "large"

        if active_phik is None:
            active_phik = phik_from_map(info_map.flatten(), _s, k_expanded)

        need_replan = active_plan_u is None
        reason = "bootstrap"
        u_prev_seed = None
        max_iters = ITERS_FULL
        min_iters = MIN_ITERS_FULL
        head_quality_dist = None

        if not need_replan:
            if tier == "unchanged":
                head_quality_dist = evaluate_head_quality(active_plan_u, active_plan_cursor, x0, info_map, tau)
                if head_quality_dist > HEAD_QUALITY_DIST_THR:
                    need_replan = True
                    reason = "unchanged_refine_head"
                    u_prev_seed = build_warm_seed_from_remainder(active_plan_u, active_plan_cursor, t_horizon)
                    max_iters = ITERS_REFINE
                    min_iters = MIN_ITERS_WARM
                else:
                    reason = "reuse_plan"
            else:
                need_replan = True
                reason = "map_changed"
                u_prev_seed = build_warm_seed_from_remainder(active_plan_u, active_plan_cursor, t_horizon)
                if tier in ("small", "medium"):
                    max_iters = ITERS_WARM
                    min_iters = MIN_ITERS_WARM
                else:
                    u_prev_seed = None
                    max_iters = ITERS_FULL
                    min_iters = MIN_ITERS_FULL

        opt_diag = {"iters_used": 0, "init_loss": 0.0, "final_loss": 0.0, "opt_window": 0}
        if need_replan:
            active_phik = phik_from_map(info_map.flatten(), _s, k_expanded)
            active_plan_u, opt_diag = optimize_trajectory(
                x0, active_phik, k_expanded, lamk, hk, info_map,
                u_prev=u_prev_seed, t_horizon=t_horizon, tau=tau,
                max_iters=max_iters, min_iters=min_iters, lr=LR
            )
            active_plan_states = rollout_states(x0, active_plan_u).detach()
            active_plan_cursor = 0
            active_plan_mode = "forward"

        cycle_plan_u = build_warm_seed_from_remainder(active_plan_u, active_plan_cursor, t_horizon)
        cycle_plan_states = rollout_states(x0, cycle_plan_u).detach()

        phik_list.append(active_phik.detach())
        full_trajectories.append(cycle_plan_states.cpu().numpy())
        full_lambda_list.append(compute_sensor_lambda(cycle_plan_u).cpu().numpy())
        head_len_list.append(int(min(tau, active_plan_u.shape[0] - active_plan_cursor)))
        cycle_maps.append(info_map.detach().clone())

        exec_len = min(tau, active_plan_u.shape[0] - active_plan_cursor)
        u_head = active_plan_u[active_plan_cursor:active_plan_cursor + exec_len]
        active_plan_cursor += exec_len

        executed = rollout_states(x0, u_head)
        trajectories.append(executed.cpu().numpy())
        x0 = executed[-1].detach()

        lam_head = compute_sensor_lambda(u_head)
        sensor_on_ratio = float((lam_head >= SENSOR_ON_THRESHOLD).float().mean().item()) if lam_head.numel() > 0 else 0.0
        control_energy = float(torch.mean(torch.sum(u_head[:, :2] ** 2, dim=1)).item()) if u_head.shape[0] > 0 else 0.0
        cycle_time = float(time.time() - t0)
        time_list.append(cycle_time)
        diagnostics.append({
            "cycle": int(i),
            "delta": float(delta),
            "tier": tier,
            "replanned": bool(need_replan),
            "reason": reason,
            "iters_used": int(opt_diag["iters_used"]),
            "cycle_time_sec": cycle_time,
            "head_quality_dist": float(head_quality_dist) if head_quality_dist is not None else -1.0,
            "sensor_on_ratio_head": sensor_on_ratio,
            "control_energy_head": control_energy,
            "plan_mode": active_plan_mode,
        })
        last_map = info_map.clone()

    return trajectories, full_trajectories, full_lambda_list, head_len_list, phik_list, cycle_maps, time_list, diagnostics


def get_files_in_folder(path):
    files = []
    for file in os.listdir(path):
        item_path = os.path.join(path, file)
        if os.path.isfile(item_path):
            files.append(item_path)
    return files


def load_files(file):
    ext = os.path.splitext(file)[1].lower()
    if ext == ".npy":
        arr = np.load(file)
    elif ext == ".txt":
        arr = np.loadtxt(file, dtype=np.float32)
    else:
        return None
    return torch.from_numpy(arr).float()


def build_map_sequence(full_maps, n_cycles=N_CYCLES, repeat_prob=REPEAT_PROB):
    perm = torch.randperm(full_maps.shape[0])
    base = full_maps[perm[:n_cycles]]
    if not USE_REPEAT_MODE:
        return base
    maps_list = [base[0]]
    for i in range(1, n_cycles):
        if random.random() < repeat_prob:
            maps_list.append(maps_list[-1].clone())
        else:
            maps_list.append(base[i])
    return torch.stack(maps_list, dim=0)


# -------------------------
# Data / setup
# -------------------------
if RUN_SEED is not None:
    torch.manual_seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    random.seed(RUN_SEED)

entropy_maps_path = "/Users/cindy/Desktop/ergodic-search/entropy_maps"
entropy_maps = []
for file in get_files_in_folder(entropy_maps_path):
    entropy_file = load_files(file)
    if entropy_file is not None:
        entropy_maps.append(entropy_file)

full_maps = torch.stack(entropy_maps)
maps = build_map_sequence(full_maps, n_cycles=N_CYCLES, repeat_prob=REPEAT_PROB)

_, H, W = maps.shape
xs = torch.linspace(0, 1, W)
ys = torch.linspace(0, 1, H)
X, Y = torch.meshgrid(xs, ys, indexing="xy")
_s = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

k1, k2 = np.meshgrid(np.arange(0, 20), np.arange(0, 20))
k = torch.tensor(np.stack([k1.ravel(), k2.ravel()], axis=-1), dtype=torch.float32)
lamk = torch.exp(-0.8 * torch.norm(k, dim=1))
hk = torch.clamp(torch.tensor([get_hk(ki) for ki in k.numpy()]), min=1e-6)
k_expanded = k.unsqueeze(0)
fk_vals_all = torch.cos(_s[:, None, :] * k_expanded).prod(dim=-1)


trajectory, full_trajectory, full_lambda, head_len_per_cycle, phik_list, cycle_maps, time_list, replanning_diag = replanning(
    maps, _s, k_expanded, lamk, hk, t_horizon=T_HORIZON, tau=HEAD_STEPS
)

print(f"Execution time per cycle (sec): {time_list}")
print("Cycle diagnostics:")
for d in replanning_diag:
    print(
        f"  cycle={d['cycle']}, tier={d['tier']}, replanned={d['replanned']} ({d['reason']}), "
        f"iters={d['iters_used']}, head_quality_dist={d['head_quality_dist']:.3f}, "
        f"sensor_on_ratio_head={d['sensor_on_ratio_head']:.3f}, "
        f"control_energy_head={d['control_energy_head']:.3f}, cycle_time={d['cycle_time_sec']:.3f}s"
    )

total_opt_iters = int(sum(d["iters_used"] for d in replanning_diag))
static_full_iters = int(len(replanning_diag) * ITERS_FULL)
iter_saving = 1.0 - (total_opt_iters / max(static_full_iters, 1))
print(f"\nTotal optimizer iterations used: {total_opt_iters}")
print(f"Static baseline iterations (full replan each cycle): {static_full_iters}")
print(f"Iteration saving vs static baseline: {100.0 * iter_saving:.1f}%")


def plot_trajectory_figure(background_mode="map"):
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

        if background_mode == "phik":
            phik_recon = torch.matmul(fk_vals_all, phik_list[i]).reshape(H, W).cpu().numpy()
            bg = phik_recon
        else:
            bg = cycle_maps[i].cpu().numpy()

        ax_ht = axes[0, i]
        ax_sa = axes[1, i]

        # Top row: Head / Tail
        ax_ht.imshow(bg, extent=[0, 1, 0, 1], origin="lower", cmap="viridis")
        ax_ht.contourf(X.numpy(), Y.numpy(), bg, cmap="viridis")
        if tail_pts.shape[0] > 0:
            ax_ht.scatter(tail_pts[:, 0], tail_pts[:, 1], s=16, c="white", alpha=0.40)
        if head_pts.shape[0] > 0:
            ax_ht.scatter(head_pts[:, 0], head_pts[:, 1], s=34, c="red", edgecolors="black", linewidths=0.5)
        ax_ht.scatter(full_tr[0, 0], full_tr[0, 1], c="w", s=50, marker="X")
        ax_ht.scatter(tr[-1, 0], tr[-1, 1], c="yellow", s=35)
        ax_ht.set_title("Map: Head/Tail")
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

        # Bottom row: Sensor Lambda (100 steps)
        ax_sa.imshow(bg, extent=[0, 1, 0, 1], origin="lower", cmap="viridis")
        ax_sa.contourf(X.numpy(), Y.numpy(), bg, cmap="viridis")
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

        sensor_on = lam_i >= SENSOR_ON_THRESHOLD
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
        ax_sa.set_title("Map: Sensor Lambda (100 Steps)")
        ax_sa.set_aspect("equal")
        ax_sa.set_xlim(0, 1)
        ax_sa.set_ylim(0, 1)
        if i == 0:
            legend_handles_sa = [
                Line2D([0], [0], marker="o", linestyle="None", markersize=5, markerfacecolor="red", markeredgecolor="none", alpha=0.55, label="Tail (lambda color)"),
                Line2D([0], [0], marker="o", linestyle="None", markersize=7, markerfacecolor="red", markeredgecolor="black", label="Head (lambda color)"),
                Line2D([0], [0], marker="o", linestyle="None", markersize=8, markerfacecolor="none", markeredgecolor="cyan", label=f"Sensor ON (lambda >= {SENSOR_ON_THRESHOLD:.1f})"),
                Line2D([0], [0], marker="X", linestyle="None", markersize=7, markerfacecolor="white", markeredgecolor="white", label="Plan start"),
                Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor="yellow", markeredgecolor="none", label="Executed head end"),
            ]
            ax_sa.legend(handles=legend_handles_sa, loc="lower left", fontsize=7, framealpha=0.8)

    if background_mode == "phik":
        fig.suptitle("Background: phik_recon", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()

    if sc_last is not None:
        # Put colorbar in its own axis so bottom-row plots don't get resized.
        cax = fig.add_axes([0.92, 0.15, 0.012, 0.70])
        cbar = fig.colorbar(sc_last, cax=cax)
        cbar.set_label("Sensor activation lambda (white=OFF, red=ON)")

    return fig


plot_trajectory_figure(background_mode="map")
plot_trajectory_figure(background_mode="phik")
plt.show()
