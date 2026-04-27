#!/usr/bin/env python3
# coding: utf-8

"""
Comparative study script based on the test4.py planning objective.

Methods:
1) objective_ergodic (our method)
2) frontier
3) probabilistic
4) greedy_info

Outputs:
- comparison_outputs/method_summary.csv
- comparison_outputs/cycle_metrics.csv
- comparison_outputs/baseline_outputs.txt
- comparison_outputs/fig_coverage_performance.png
- comparison_outputs/fig_runtime_comparison.png
- comparison_outputs/fig_ergodicity_vs_step.png
- comparison_outputs/fig_k_index_red_white.png
"""

from __future__ import annotations

import csv
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Force matplotlib/font cache paths to writable workspace locations.
_SCRIPT_DIR = Path(__file__).resolve().parent
_MPL_CACHE_DIR = _SCRIPT_DIR / ".mpl-cache"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE_DIR)
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# -------------------------
# Config
# -------------------------
SEED = 11
N_CYCLES = 10
USE_REPEAT_MODE = True
REPEAT_PROB = 0.50

T_HORIZON = 100
HEAD_STEPS = 20
BRIDGE_LEN = 7

HIGH_INFO_QUANTILE = 0.80
HOTSPOT_QUANTILE = 0.85
MAX_HOTSPOTS = 3
HOTSPOT_MIN_SEP = 0.12

GOAL_DISTANCE_WEIGHT = 0.70
H2H3_DISTANCE_WEIGHT = 0.60
HEAD_GOAL_W = 30.0
TERM_GOAL_W = 1.2

PATH_INFO_W = 0.95
COVERAGE_W = 0.23
ORDER_WEIGHTS = (2.0, 2.0, 2.0)
HEAD_DIRECT_W = 12.0

SPEED_W = 0.55
LOW_INFO_FAST_W = 1.10
V_HIGH = 0.020
V_LOW = 0.070

SENSOR_ON_THRESHOLD = 0.70
SENSOR_BCE_W = 2.40
SENSOR_OFF_W = 4.00
SENSOR_BIN_W = 0.02
SENSOR_SMOOTH_W = 0.02
SENSOR_TEMP = 0.025

CONTROL_W = 0.002
SMOOTH_W = 0.001
CURV_W = 0.004
BARRIER_W = 10.0
TAIL_W_OPT = 0.35
DISCOUNT_RATE = 1.30
SEAM_VEL_W = 10.0
SEAM_ACC_W = 4.0
SEAM_LAM_W = 1.5

LR = 1e-3
ITERS_FULL = 320
ITERS_WARM = 180
MIN_ITERS_FULL = 160
MIN_ITERS_WARM = 80
PATIENCE = 20
REL_IMPROVE_TOL = 1e-4

SMALL_CHANGE_THR = 0.03

# Baseline behavior knobs
FRONTIER_DIST_W = 0.80
FRONTIER_PREV_W = 0.35
PROB_ALPHA = 2.2
PROB_NOVELTY_BETA = 1.4
GREEDY_REVISIT_W = 0.00
GREEDY_INERTIA_W = 0.00

METHOD_ORDER = ["objective_ergodic", "frontier", "probabilistic", "greedy_info"]
METHOD_LABEL = {
    "objective_ergodic": "Objective-Based (test4)",
    "frontier": "Frontier-based",
    "probabilistic": "Probabilistic heuristic",
    "greedy_info": "Greedy info-max",
}
METHOD_COLOR = {
    "objective_ergodic": "#0E7490",
    "frontier": "#DC2626",
    "probabilistic": "#D97706",
    "greedy_info": "#7C3AED",
}


# -------------------------
# Dynamics and utilities
# -------------------------
@torch.jit.script
def f(x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xnew = x[:2] + 0.07 * u[:2]
    xnew = torch.clamp(xnew, 0.0, 1.0)
    return xnew, xnew


def get_hk(k: np.ndarray) -> float:
    hk_arr = (2.0 * k + np.sin(2.0 * k)) / (4.0 * k + 1e-8)
    hk_arr[np.isnan(hk_arr)] = 1.0
    return float(np.sqrt(np.prod(hk_arr)))


def get_ck_weighted(tr: torch.Tensor, k_expanded: torch.Tensor, weights: torch.Tensor, hk: torch.Tensor) -> torch.Tensor:
    fk = torch.cos(tr[:, None, :] * k_expanded).prod(dim=-1)
    z = weights.sum() + 1e-8
    return (fk.T @ weights) / (z * hk)


def compute_sensor_lambda(u: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.sigmoid(5.0 * u[:, 2]), 0.0, 1.0)


def normalize_info_map(info_map: torch.Tensor) -> torch.Tensor:
    map_min = torch.min(info_map)
    map_max = torch.max(info_map)
    return (info_map - map_min) / (map_max - map_min + 1e-8)


def sample_info_values(points: torch.Tensor, info_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, w = info_map.shape
    ix = (points[:, 0] * (w - 1)).long().clamp(0, w - 1)
    iy = (points[:, 1] * (h - 1)).long().clamp(0, h - 1)
    return info_map[iy, ix], ix, iy


def extract_hotspots(
    info_map_norm: torch.Tensor,
    max_hotspots: int = MAX_HOTSPOTS,
    hotspot_quantile: float = HOTSPOT_QUANTILE,
    min_sep: float = HOTSPOT_MIN_SEP,
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w = info_map_norm.shape
    flat = info_map_norm.flatten()
    thr = torch.quantile(flat, hotspot_quantile)
    candidate_ids = torch.nonzero(flat >= thr, as_tuple=False).flatten()
    if candidate_ids.numel() == 0:
        candidate_ids = torch.topk(flat, k=min(max_hotspots, flat.numel())).indices
    scores_all = flat[candidate_ids]
    sorted_idx = torch.argsort(scores_all, descending=True)

    hotspots = []
    scores = []
    denom_x = max(w - 1, 1)
    denom_y = max(h - 1, 1)
    for idx in candidate_ids[sorted_idx]:
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


def choose_ordered_goals_from_recon(phik_recon: torch.Tensor, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    phik_recon_norm = normalize_info_map(phik_recon)
    hotspots, scores = extract_hotspots(phik_recon_norm)
    start_xy = x0[:2].to(dtype=torch.float32, device=hotspots.device)
    d_start = torch.norm(hotspots - start_xy.unsqueeze(0), dim=1)
    h1_score = scores - GOAL_DISTANCE_WEIGHT * d_start
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
        score_next = scores[rem] - H2H3_DISTANCE_WEIGHT * d_prev
        pick = int(rem[torch.argmax(score_next)].item())
        ordered_ids.append(pick)
        remaining.remove(pick)
        prev = pick

    ordered_goals = hotspots[torch.tensor(ordered_ids[:3], dtype=torch.long, device=hotspots.device)]
    return ordered_goals, hotspots


def rollout_states(x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    x = x0.clone()
    tr = [x]
    for step in u:
        x, _ = f(x, step[:2])
        tr.append(x)
    return torch.stack(tr)


def phik_from_map(map_flattened: torch.Tensor, sample: torch.Tensor, k_expanded: torch.Tensor) -> torch.Tensor:
    fk_vals = torch.cos(sample[:, None, :] * k_expanded).prod(dim=-1)
    return (fk_vals * map_flattened[:, None]).sum(dim=0) / (map_flattened.sum() + 1e-8)


def build_warm_seed_from_remainder(active_plan_u: torch.Tensor, horizon: int, executed_steps: int) -> torch.Tensor:
    remainder = active_plan_u[executed_steps:].detach()
    if remainder.shape[0] == 0:
        remainder = active_plan_u[-1:, :].detach()
    if remainder.shape[0] >= horizon:
        return remainder[:horizon].clone().detach()
    pad = remainder[-1:, :].repeat(horizon - remainder.shape[0], 1)
    return torch.cat([remainder, pad], dim=0).clone().detach()


def seed_controls_from_goals(x0: torch.Tensor, ordered_goals: torch.Tensor, info_map: torch.Tensor, t_horizon: int, tau: int) -> torch.Tensor:
    u = torch.zeros((t_horizon, 3), dtype=torch.float32, device=x0.device)
    g1, g2, g3 = ordered_goals[0], ordered_goals[1], ordered_goals[2]

    seg1 = min(tau, t_horizon)
    seg2 = min(40, max(t_horizon - seg1, 0))
    seg3 = max(t_horizon - seg1 - seg2, 0)

    if seg1 > 0:
        u[:seg1, :2] = ((g1 - x0) / (0.07 * seg1)).clamp(min=-1.0, max=1.0)
    if seg2 > 0:
        u[seg1:seg1 + seg2, :2] = ((g2 - g1) / (0.07 * seg2)).clamp(min=-1.0, max=1.0)
    if seg3 > 0:
        u[seg1 + seg2:, :2] = ((g3 - g2) / (0.07 * seg3)).clamp(min=-1.0, max=1.0)

    u[:, :2] += 0.004 * torch.randn_like(u[:, :2])
    return attach_sensor_channel(u, x0, info_map)


def attach_sensor_channel(u: torch.Tensor, x0: torch.Tensor, info_map: torch.Tensor) -> torch.Tensor:
    u = u.clone()
    tr = rollout_states(x0, u)[1:, :2]
    info_map_norm = normalize_info_map(info_map)
    info_vals, _, _ = sample_info_values(tr, info_map_norm)
    q_hi = torch.quantile(info_map_norm.flatten(), HIGH_INFO_QUANTILE)
    lam_target = torch.sigmoid((info_vals - q_hi) / SENSOR_TEMP).clamp(1e-4, 1.0 - 1e-4)
    u[:, 2] = torch.log(lam_target / (1.0 - lam_target)) / 5.0
    return u


def fourier_ergodic_loss(
    u: torch.Tensor,
    x0: torch.Tensor,
    phik: torch.Tensor,
    k_expanded: torch.Tensor,
    lamk: torch.Tensor,
    hk: torch.Tensor,
    info_map: torch.Tensor,
    tau: int | None = None,
    head_w: float = 1.0,
    tail_w: float = TAIL_W_OPT,
    hotspots: torch.Tensor | None = None,
    ordered_goals: torch.Tensor | None = None,
) -> torch.Tensor:
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

    coverage_term = torch.tensor(0.0, device=u.device)
    if hotspots is not None and hotspots.numel() > 0:
        dist2 = torch.sum((tr[:, None, :2] - hotspots[None, :, :]) ** 2, dim=2)
        smooth_min = -torch.logsumexp(-20.0 * dist2, dim=0) / 20.0
        hotspot_info, _, _ = sample_info_values(hotspots, info_map_norm)
        hotspot_w = hotspot_info / (torch.sum(hotspot_info) + 1e-8)
        coverage_term = COVERAGE_W * torch.sum(hotspot_w * smooth_min)

    ordered_term = torch.tensor(0.0, device=u.device)
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

    head_direct_term = torch.tensor(0.0, device=u.device)
    if ordered_goals is not None and tau is not None and t_steps > 0:
        tau_i = int(min(max(tau, 2), t_steps))
        head_seg = tr[:tau_i, :2]
        alpha = torch.linspace(1.0 / tau_i, 1.0, tau_i, device=u.device).unsqueeze(1)
        line_seg = x0.unsqueeze(0) + alpha * (ordered_goals[0] - x0).unsqueeze(0)
        head_direct_term = HEAD_DIRECT_W * torch.mean(torch.sum((head_seg - line_seg) ** 2, dim=1))

    q_hi = torch.quantile(info_map_norm.flatten(), HIGH_INFO_QUANTILE)
    target_lam = torch.sigmoid((info_values - q_hi) / SENSOR_TEMP)
    eps = 1e-6
    sensor_bce = -(2.5 * target_lam * torch.log(lam + eps) + (1.0 - target_lam) * torch.log(1.0 - lam + eps))
    sensor_bce_term = SENSOR_BCE_W * torch.mean(sensor_bce)
    sensor_off_term = SENSOR_OFF_W * torch.mean(lam * (1.0 - target_lam))
    sensor_bin_term = SENSOR_BIN_W * torch.mean(lam * (1.0 - lam))
    sensor_smooth_term = SENSOR_SMOOTH_W * torch.mean((lam[1:] - lam[:-1]).pow(2)) if lam.shape[0] > 1 else 0.0

    seam_term = torch.tensor(0.0, device=u.device)
    if tau is not None and t_steps >= 3:
        tau_i = int(min(max(tau, 1), t_steps - 2))
        seam_vel = SEAM_VEL_W * torch.sum((u[tau_i, :2] - u[tau_i - 1, :2]) ** 2)
        seam_acc = SEAM_ACC_W * torch.sum((u[tau_i + 1, :2] - 2.0 * u[tau_i, :2] + u[tau_i - 1, :2]) ** 2)
        seam_lam = SEAM_LAM_W * (lam[tau_i] - lam[tau_i - 1]).pow(2)
        seam_term = seam_vel + seam_acc + seam_lam

    return (
        ergodic_term
        + CONTROL_W * torch.mean(u[:, :2] ** 2)
        + SMOOTH_W * torch.sum((u[1:, :2] - u[:-1, :2]) ** 2)
        + CURV_W * torch.sum((u[2:, :2] - 2.0 * u[1:-1, :2] + u[:-2, :2]) ** 2)
        + BARRIER_W * torch.sum(torch.clamp_min(tr - 1.0, 0.0) ** 2 + torch.clamp_min(-tr, 0.0) ** 2)
        + path_info_term
        + speed_term
        + low_fast_term
        + coverage_term
        + ordered_term
        + head_direct_term
        + sensor_bce_term
        + sensor_off_term
        + sensor_bin_term
        + sensor_smooth_term
        + seam_term
    )


def loss_with_goal(
    u: torch.Tensor,
    x0: torch.Tensor,
    phik: torch.Tensor,
    k_expanded: torch.Tensor,
    lamk: torch.Tensor,
    hk: torch.Tensor,
    info_map: torch.Tensor,
    tau: int,
    goal: torch.Tensor,
    head_goal: torch.Tensor,
    hotspots: torch.Tensor,
    ordered_goals: torch.Tensor,
) -> torch.Tensor:
    l_erg = fourier_ergodic_loss(
        u, x0, phik, k_expanded, lamk, hk, info_map,
        tau=tau, head_w=1.0, tail_w=TAIL_W_OPT,
        hotspots=hotspots, ordered_goals=ordered_goals
    )
    x = x0.clone()
    x_head = None
    head_idx = min(max(int(tau), 1), u.shape[0])
    for i, step in enumerate(u):
        x, _ = f(x, step[:2])
        if (i + 1) == head_idx:
            x_head = x
    if x_head is None:
        x_head = x
    term = TERM_GOAL_W * torch.sum((x - goal) ** 2) + HEAD_GOAL_W * torch.sum((x_head - head_goal) ** 2)
    return l_erg + term


def optimize_objective_plan(
    x0: torch.Tensor,
    phik: torch.Tensor,
    phik_recon: torch.Tensor,
    k_expanded: torch.Tensor,
    lamk: torch.Tensor,
    hk: torch.Tensor,
    info_map: torch.Tensor,
    u_prev_seed: torch.Tensor | None,
    t_horizon: int,
    tau: int,
    max_iters: int,
    min_iters: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ordered_goals, hotspots = choose_ordered_goals_from_recon(phik_recon, x0)
    ordered_goals = ordered_goals.to(device=x0.device, dtype=torch.float32)
    hotspots = hotspots.to(device=x0.device, dtype=torch.float32)
    head_goal = ordered_goals[0]
    term_goal = ordered_goals[2]

    if u_prev_seed is None:
        head = seed_controls_from_goals(x0, ordered_goals, info_map, t_horizon, tau)
        tail = None
        opt_window = t_horizon
    else:
        u_seed = u_prev_seed.detach().clone()
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
    optimizer = torch.optim.Adam([head], lr=LR)

    def build_u() -> torch.Tensor:
        return head if tail is None else torch.cat([head, tail], dim=0)

    with torch.no_grad():
        init_loss = float(
            loss_with_goal(
                build_u(),
                x0, phik, k_expanded, lamk, hk, info_map,
                tau=tau, goal=term_goal, head_goal=head_goal,
                hotspots=hotspots, ordered_goals=ordered_goals
            ).item()
        )

    last_loss = init_loss
    plateau = 0
    iters_used = 0
    for i in range(max_iters):
        optimizer.zero_grad()
        u_now = build_u()
        loss = loss_with_goal(
            u_now, x0, phik, k_expanded, lamk, hk, info_map,
            tau=tau, goal=term_goal, head_goal=head_goal,
            hotspots=hotspots, ordered_goals=ordered_goals
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            head[:, :2].clamp_(min=-1.0, max=1.0)
            head[:, 2].clamp_(min=-2.0, max=2.0)

        curr_loss = float(loss.item())
        rel_improve = (last_loss - curr_loss) / (abs(last_loss) + 1e-8)
        plateau = plateau + 1 if rel_improve < REL_IMPROVE_TOL else 0
        last_loss = curr_loss
        iters_used = i + 1
        if iters_used >= min_iters and plateau >= PATIENCE:
            break

    u_out = build_u().detach()
    if u_out.shape[0] < t_horizon:
        pad = u_out[-1:, :].repeat(t_horizon - u_out.shape[0], 1)
        u_out = torch.cat([u_out, pad], dim=0)
    else:
        u_out = u_out[:t_horizon]

    diag = {
        "iters_used": float(iters_used),
        "init_loss": init_loss,
        "final_loss": last_loss,
        "opt_window": float(opt_window),
    }
    return u_out, diag


# -------------------------
# Baseline planners
# -------------------------
def points_to_visited_mask(points_xy: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    ix = np.clip((points_xy[:, 0] * (w - 1)).astype(int), 0, w - 1)
    iy = np.clip((points_xy[:, 1] * (h - 1)).astype(int), 0, h - 1)
    mask[iy, ix] = True
    return mask


def points_to_coarse_mask(points_xy: np.ndarray, bins: int = 20) -> np.ndarray:
    mask = np.zeros((bins, bins), dtype=bool)
    if points_xy.shape[0] == 0:
        return mask
    ix = np.clip((points_xy[:, 0] * bins).astype(int), 0, bins - 1)
    iy = np.clip((points_xy[:, 1] * bins).astype(int), 0, bins - 1)
    mask[iy, ix] = True
    return mask


def map_to_coarse(info_map: np.ndarray, bins: int = 20) -> np.ndarray:
    t = torch.from_numpy(info_map).float().unsqueeze(0).unsqueeze(0)
    coarse = torch.nn.functional.interpolate(t, size=(bins, bins), mode="area").squeeze(0).squeeze(0)
    return coarse.numpy()


def _neighbor_visited(visited: np.ndarray) -> np.ndarray:
    h, w = visited.shape
    out = np.zeros_like(visited)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            sy0 = max(0, -dy)
            sy1 = h - max(0, dy)
            sx0 = max(0, -dx)
            sx1 = w - max(0, dx)
            dy0 = max(0, dy)
            dy1 = h - max(0, -dy)
            dx0 = max(0, dx)
            dx1 = w - max(0, -dx)
            out[dy0:dy1, dx0:dx1] |= visited[sy0:sy1, sx0:sx1]
    return out


def _goal_list_to_controls(x0: torch.Tensor, goals: List[np.ndarray], t_horizon: int) -> torch.Tensor:
    u = torch.zeros((t_horizon, 3), dtype=torch.float32, device=x0.device)
    if not goals:
        return u
    seg_sizes = [max(1, t_horizon // len(goals))] * len(goals)
    seg_sizes[-1] = t_horizon - sum(seg_sizes[:-1])
    curr = x0.detach().cpu().numpy()
    st = 0
    for gi, g in enumerate(goals):
        seg = seg_sizes[gi]
        goal = np.asarray(g, dtype=np.float32)
        ctrl = np.clip((goal - curr) / (0.07 * seg), -1.0, 1.0)
        u[st:st + seg, 0] = float(ctrl[0])
        u[st:st + seg, 1] = float(ctrl[1])
        curr = goal
        st += seg
    u[:, :2] += 0.002 * torch.randn_like(u[:, :2])
    return u


def plan_frontier(
    x0: torch.Tensor,
    info_map: torch.Tensor,
    visited_counts: np.ndarray,
    t_horizon: int,
) -> torch.Tensor:
    info = normalize_info_map(info_map).cpu().numpy()
    h, w = info.shape
    visited = visited_counts > 0
    x_idx = int(np.clip(round(float(x0[0].item()) * (w - 1)), 0, w - 1))
    y_idx = int(np.clip(round(float(x0[1].item()) * (h - 1)), 0, h - 1))
    visited[y_idx, x_idx] = True

    goals: List[np.ndarray] = []
    prev_goal = np.array([x0[0].item(), x0[1].item()], dtype=np.float32)
    frontier = (~visited) & _neighbor_visited(visited)
    candidates = np.argwhere(frontier)
    if candidates.shape[0] == 0:
        candidates = np.argwhere(~visited)
    if candidates.shape[0] == 0:
        candidates = np.argwhere(np.ones_like(visited, dtype=bool))

    for _ in range(3):
        if candidates.shape[0] == 0:
            goals.append(prev_goal.copy())
            continue
        scores = []
        pts = []
        for yy, xx in candidates:
            p = np.array([xx / max(w - 1, 1), yy / max(h - 1, 1)], dtype=np.float32)
            d_start = np.linalg.norm(p - np.array([x0[0].item(), x0[1].item()], dtype=np.float32))
            d_prev = np.linalg.norm(p - prev_goal)
            frontier_bonus = 0.2 if frontier[yy, xx] else 0.0
            s = info[yy, xx] + frontier_bonus - FRONTIER_DIST_W * d_start - FRONTIER_PREV_W * d_prev
            scores.append(s)
            pts.append(p)
        best = int(np.argmax(np.asarray(scores)))
        goal = pts[best]
        goals.append(goal)
        prev_goal = goal

        gy = int(np.clip(round(goal[1] * (h - 1)), 0, h - 1))
        gx = int(np.clip(round(goal[0] * (w - 1)), 0, w - 1))
        visited[gy, gx] = True
        frontier = (~visited) & _neighbor_visited(visited)
        candidates = np.argwhere(frontier)
        if candidates.shape[0] == 0:
            candidates = np.argwhere(~visited)

    u = _goal_list_to_controls(x0, goals, t_horizon)
    return attach_sensor_channel(u, x0, info_map)


def plan_probabilistic(
    x0: torch.Tensor,
    info_map: torch.Tensor,
    visited_counts: np.ndarray,
    t_horizon: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    info = normalize_info_map(info_map).cpu().numpy()
    novelty = 1.0 / (1.0 + visited_counts.astype(np.float32))
    weights = np.power(info + 1e-6, PROB_ALPHA) * np.power(novelty, PROB_NOVELTY_BETA)
    flat_w = weights.reshape(-1)
    flat_w = flat_w / max(flat_w.sum(), 1e-8)

    n_pick = 3
    idxs = rng.choice(flat_w.size, size=n_pick, replace=False, p=flat_w)
    h, w = info.shape
    goals = []
    for idx in idxs:
        yy = int(idx // w)
        xx = int(idx % w)
        goals.append(np.array([xx / max(w - 1, 1), yy / max(h - 1, 1)], dtype=np.float32))

    # Order goals nearest-neighbor from current position.
    ordered = []
    curr = np.array([x0[0].item(), x0[1].item()], dtype=np.float32)
    remaining = goals[:]
    while remaining:
        dists = [np.linalg.norm(g - curr) for g in remaining]
        j = int(np.argmin(np.asarray(dists)))
        nxt = remaining.pop(j)
        ordered.append(nxt)
        curr = nxt

    u = _goal_list_to_controls(x0, ordered, t_horizon)
    return attach_sensor_channel(u, x0, info_map)


def plan_greedy_info(
    x0: torch.Tensor,
    info_map: torch.Tensor,
    visited_counts: np.ndarray,
    t_horizon: int,
) -> torch.Tensor:
    info = normalize_info_map(info_map)
    h, w = info.shape
    device = x0.device

    actions = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
            [0.707, 0.707], [0.707, -0.707], [-0.707, 0.707], [-0.707, -0.707],
        ],
        dtype=torch.float32,
        device=device,
    )
    u = torch.zeros((t_horizon, 3), dtype=torch.float32, device=device)

    local_visit = visited_counts.astype(np.float32).copy()
    pos = x0.clone()
    prev_a = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)
    for t in range(t_horizon):
        cand_pos = (pos.unsqueeze(0) + 0.07 * actions).clamp(0.0, 1.0)
        vals, ix, iy = sample_info_values(cand_pos, info)
        revisit = torch.from_numpy(local_visit[iy.cpu().numpy(), ix.cpu().numpy()]).to(device=device)
        inertia = torch.norm(actions - prev_a.unsqueeze(0), dim=1)
        score = vals - GREEDY_REVISIT_W * revisit - GREEDY_INERTIA_W * inertia
        best = int(torch.argmax(score).item())
        a = actions[best]
        u[t, :2] = a
        pos = cand_pos[best]
        prev_a = a
        gx = int(ix[best].item())
        gy = int(iy[best].item())
        local_visit[gy, gx] += 1.0

    return attach_sensor_channel(u, x0, info_map)


# -------------------------
# Metrics
# -------------------------
def ergodic_distance(phik: torch.Tensor, ck: torch.Tensor, lamk: torch.Tensor) -> float:
    phik_n = (phik - phik.mean()) / (phik.std() + 1e-6)
    ck_n = (ck - ck.mean()) / (ck.std() + 1e-6)
    return float(torch.sum(lamk * (phik_n - ck_n) ** 2).item())


def compute_cycle_coverage_metrics(
    points_xy: np.ndarray, info_map_norm: np.ndarray, hi_q: float
) -> Tuple[float, float, float, int, int]:
    if points_xy.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0, 0

    # 1) High-info hotspot reach score (coverage of objective-relevant regions).
    info_t = torch.from_numpy(info_map_norm).float()
    hotspots, _ = extract_hotspots(info_t, max_hotspots=MAX_HOTSPOTS, hotspot_quantile=hi_q, min_sep=HOTSPOT_MIN_SEP)
    hs = hotspots.cpu().numpy()
    dmat = np.linalg.norm(points_xy[:, None, :] - hs[None, :, :], axis=2)
    dmin = dmat.min(axis=0)
    sigma = 0.04
    hi_cov = float(np.mean(np.exp(-(dmin ** 2) / (2.0 * sigma * sigma))))
    hotspot_hits = int(np.sum(dmin < 0.06))
    hotspot_count = int(hs.shape[0])

    # 2) Path weighted info score (how informative visited points are).
    h, w = info_map_norm.shape
    ix = np.clip((points_xy[:, 0] * (w - 1)).astype(int), 0, w - 1)
    iy = np.clip((points_xy[:, 1] * (h - 1)).astype(int), 0, h - 1)
    weighted_cov = float(np.mean(info_map_norm[iy, ix]))

    # 3) Coarse global spatial coverage score.
    vis_coarse = points_to_coarse_mask(points_xy, bins=20)
    global_cov = float(vis_coarse.mean())
    return hi_cov, weighted_cov, global_cov, hotspot_hits, hotspot_count


def compute_k_index_stats(
    points_t: torch.Tensor,
    lam_t: torch.Tensor,
    phik: torch.Tensor,
    k_expanded: torch.Tensor,
    hk: torch.Tensor,
    lamk: torch.Tensor,
    k_norms: np.ndarray,
) -> Tuple[List[float], List[float]]:
    if points_t.shape[0] == 0:
        return [], []
    fk = torch.cos(points_t[:, None, :] * k_expanded).prod(dim=-1)
    ck_step = torch.abs(fk / hk.unsqueeze(0))
    spectral_w = lamk.unsqueeze(0) * ck_step
    if spectral_w.shape[1] > 0:
        spectral_w[:, 0] = 0.0  # ignore DC mode
    k_norms_t = torch.from_numpy(k_norms).to(dtype=ck_step.dtype, device=ck_step.device).unsqueeze(0)
    centroid = torch.sum(spectral_w * k_norms_t, dim=1) / (torch.sum(spectral_w, dim=1) + 1e-8)
    centroid_np = centroid.cpu().numpy()
    red = centroid_np[lam_t.cpu().numpy() >= SENSOR_ON_THRESHOLD].tolist()
    white = centroid_np[lam_t.cpu().numpy() < SENSOR_ON_THRESHOLD].tolist()
    return red, white


# -------------------------
# IO helpers
# -------------------------
def load_entropy_maps(repo_root: Path) -> torch.Tensor:
    map_dir = repo_root / "entropy_maps"
    files = sorted([p for p in map_dir.glob("*.npy") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No .npy files found in {map_dir}")
    maps = [torch.from_numpy(np.load(str(f))).float() for f in files]
    return torch.stack(maps, dim=0)


def build_map_sequence(full_maps: torch.Tensor, n_cycles: int, use_repeat: bool, repeat_prob: float) -> Tuple[torch.Tensor, List[int]]:
    base_indices = torch.randperm(full_maps.shape[0])[:n_cycles].tolist()
    base_maps = [full_maps[i] for i in base_indices]
    if not use_repeat:
        return torch.stack(base_maps, dim=0), base_indices

    maps_list = [base_maps[0]]
    idx_list = [base_indices[0]]
    for i in range(1, n_cycles):
        if random.random() < repeat_prob:
            maps_list.append(maps_list[-1].clone())
            idx_list.append(idx_list[-1])
        else:
            maps_list.append(base_maps[i])
            idx_list.append(base_indices[i])
    return torch.stack(maps_list, dim=0), idx_list


# -------------------------
# Simulation
# -------------------------
def run_method(
    method: str,
    maps: torch.Tensor,
    map_ids: List[int],
    sample_grid: torch.Tensor,
    k_expanded: torch.Tensor,
    lamk: torch.Tensor,
    hk: torch.Tensor,
    fk_vals_all: torch.Tensor,
    k_norms: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, object]:
    h, w = maps.shape[1], maps.shape[2]
    x0 = torch.tensor([0.54, 0.30], dtype=torch.float32)
    visited_counts = np.zeros((h, w), dtype=np.int32)
    last_map = None
    active_u = None

    cycle_rows = []
    erg_curve = []
    red_k_all: List[float] = []
    white_k_all: List[float] = []
    plan_times = []

    all_points = []
    all_lam = []
    all_cycle_ids = []

    for i, info_map_raw in enumerate(maps):
        info_map = normalize_info_map(info_map_raw)
        phik = phik_from_map(info_map.flatten(), sample_grid, k_expanded)
        phik_recon = torch.matmul(fk_vals_all, phik).reshape(h, w)

        t0 = time.perf_counter()
        if method == "objective_ergodic":
            u_prev_seed = None
            if active_u is not None:
                delta = float(torch.mean(torch.abs(info_map - last_map)).item()) if last_map is not None else float("inf")
                if delta <= SMALL_CHANGE_THR:
                    u_prev_seed = build_warm_seed_from_remainder(active_u, T_HORIZON, HEAD_STEPS)
                    max_iters = ITERS_WARM
                    min_iters = MIN_ITERS_WARM
                else:
                    max_iters = ITERS_FULL
                    min_iters = MIN_ITERS_FULL
            else:
                max_iters = ITERS_FULL
                min_iters = MIN_ITERS_FULL

            active_u, opt_diag = optimize_objective_plan(
                x0=x0, phik=phik, phik_recon=phik_recon,
                k_expanded=k_expanded, lamk=lamk, hk=hk, info_map=info_map,
                u_prev_seed=u_prev_seed, t_horizon=T_HORIZON, tau=HEAD_STEPS,
                max_iters=max_iters, min_iters=min_iters
            )
            u_plan = active_u
            extra = {
                "iters_used": int(opt_diag["iters_used"]),
                "init_loss": float(opt_diag["init_loss"]),
                "final_loss": float(opt_diag["final_loss"]),
            }
        elif method == "frontier":
            u_plan = plan_frontier(x0, info_map, visited_counts, T_HORIZON)
            extra = {"iters_used": 0, "init_loss": 0.0, "final_loss": 0.0}
        elif method == "probabilistic":
            u_plan = plan_probabilistic(x0, info_map, visited_counts, T_HORIZON, rng)
            extra = {"iters_used": 0, "init_loss": 0.0, "final_loss": 0.0}
        elif method == "greedy_info":
            u_plan = plan_greedy_info(x0, info_map, visited_counts, T_HORIZON)
            extra = {"iters_used": 0, "init_loss": 0.0, "final_loss": 0.0}
        else:
            raise ValueError(f"Unknown method: {method}")
        t_plan = time.perf_counter() - t0
        plan_times.append(t_plan)

        u_head = u_plan[:HEAD_STEPS]
        states_head = rollout_states(x0, u_head)
        pts_t = states_head[1:, :2]
        lam_t = compute_sensor_lambda(u_head)

        pts_np = pts_t.cpu().numpy()
        lam_np = lam_t.cpu().numpy()
        info_np = info_map.cpu().numpy()
        hi_cov, weighted_cov, global_cov, _, _ = compute_cycle_coverage_metrics(pts_np, info_np, HIGH_INFO_QUANTILE)

        for t in range(1, pts_t.shape[0] + 1):
            ck_t = get_ck_weighted(pts_t[:t], k_expanded, lam_t[:t], hk)
            erg_curve.append(ergodic_distance(phik, ck_t, lamk))

        red_k, white_k = compute_k_index_stats(pts_t, lam_t, phik, k_expanded, hk, lamk, k_norms)
        red_k_all.extend(red_k)
        white_k_all.extend(white_k)

        ix = np.clip((pts_np[:, 0] * (w - 1)).astype(int), 0, w - 1)
        iy = np.clip((pts_np[:, 1] * (h - 1)).astype(int), 0, h - 1)
        for gy, gx in zip(iy, ix):
            visited_counts[gy, gx] += 1

        all_points.append(pts_np)
        all_lam.append(lam_np)
        all_cycle_ids.extend([i] * pts_np.shape[0])

        cycle_rows.append(
            {
                "method": method,
                "cycle": i,
                "map_id": map_ids[i],
                "plan_time_sec": t_plan,
                "coverage_high_info": hi_cov,
                "coverage_weighted": weighted_cov,
                "coverage_global": global_cov,
                "sensor_on_ratio_head": float(np.mean(lam_np >= SENSOR_ON_THRESHOLD)),
                "iters_used": extra["iters_used"],
                "init_loss": extra["init_loss"],
                "final_loss": extra["final_loss"],
            }
        )

        x0 = states_head[-1].detach()
        last_map = info_map.clone()

    all_pts_np = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 2), dtype=np.float32)
    all_lam_np = np.concatenate(all_lam, axis=0) if all_lam else np.zeros((0,), dtype=np.float32)
    global_vis_coarse = points_to_coarse_mask(all_pts_np, bins=20)
    hi_map = np.mean(maps.cpu().numpy(), axis=0)
    hi_map_coarse = map_to_coarse(hi_map, bins=20)
    hi_thr = float(np.quantile(hi_map_coarse.reshape(-1), HIGH_INFO_QUANTILE))
    hi_mask_coarse = hi_map_coarse >= hi_thr
    global_cov_ratio = float(global_vis_coarse.mean())
    global_hi_cov = float((global_vis_coarse & hi_mask_coarse).sum() / max(1, int(hi_mask_coarse.sum())))

    red_mean = float(np.mean(red_k_all)) if red_k_all else 0.0
    white_mean = float(np.mean(white_k_all)) if white_k_all else 0.0
    k_change = red_mean - white_mean

    summary = {
        "method": method,
        "cycles": len(cycle_rows),
        "mean_coverage_high_info": float(np.mean([r["coverage_high_info"] for r in cycle_rows])),
        "mean_coverage_weighted": float(np.mean([r["coverage_weighted"] for r in cycle_rows])),
        "mean_coverage_global": float(np.mean([r["coverage_global"] for r in cycle_rows])),
        "global_coverage_ratio": global_cov_ratio,
        "global_high_info_coverage_ratio": global_hi_cov,
        "mean_plan_time_sec": float(np.mean(plan_times)),
        "total_plan_time_sec": float(np.sum(plan_times)),
        "final_ergodicity": float(erg_curve[-1]) if erg_curve else 0.0,
        "mean_ergodicity": float(np.mean(erg_curve)) if erg_curve else 0.0,
        "red_k_mean": red_mean,
        "white_k_mean": white_mean,
        "k_change_red_minus_white": k_change,
    }

    return {
        "summary": summary,
        "cycle_rows": cycle_rows,
        "erg_curve": erg_curve,
        "all_points": all_pts_np,
        "all_lambda": all_lam_np,
        "all_cycle_ids": np.array(all_cycle_ids, dtype=np.int32),
        "red_k": red_k_all,
        "white_k": white_k_all,
    }


# -------------------------
# Plotting
# -------------------------
def plot_coverage(summary_rows: List[Dict[str, float]], out_path: Path) -> None:
    methods = [r["method"] for r in summary_rows]
    x = np.arange(len(methods))
    bar_w = 0.24

    fig, ax = plt.subplots(figsize=(10, 4.8))
    y1 = [r["mean_coverage_high_info"] for r in summary_rows]
    y2 = [r["mean_coverage_global"] for r in summary_rows]
    y3 = [r["global_coverage_ratio"] for r in summary_rows]

    ax.bar(x - bar_w, y1, width=bar_w, color=[METHOD_COLOR[m] for m in methods], alpha=0.95, label="Mean high-info coverage (cycle)")
    ax.bar(x, y2, width=bar_w, color=[METHOD_COLOR[m] for m in methods], alpha=0.65, label="Mean global coverage (cycle)")
    ax.bar(x + bar_w, y3, width=bar_w, color=[METHOD_COLOR[m] for m in methods], alpha=0.35, label="Global coarse coverage (long horizon)")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=12, ha="right")
    ax.set_ylabel("Coverage ratio")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Coverage Performance Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_runtime(summary_rows: List[Dict[str, float]], out_path: Path) -> None:
    methods = [r["method"] for r in summary_rows]
    x = np.arange(len(methods))
    y_total = [r["total_plan_time_sec"] for r in summary_rows]
    y_mean = [r["mean_plan_time_sec"] for r in summary_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(x, y_total, color=[METHOD_COLOR[m] for m in methods], alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=12, ha="right")
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Total planning runtime")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, y_mean, color=[METHOD_COLOR[m] for m in methods], alpha=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=12, ha="right")
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Mean planning runtime per cycle")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Runtime Comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_ergodicity_curves(results_by_method: Dict[str, Dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for method in METHOD_ORDER:
        erg = np.asarray(results_by_method[method]["erg_curve"], dtype=float)
        if erg.size == 0:
            continue
        x = np.arange(1, erg.size + 1)
        ax.plot(x, erg, linewidth=2.0, color=METHOD_COLOR[method], label=METHOD_LABEL[method])
    ax.set_xlabel("Trajectory step (across all executed heads)")
    ax.set_ylabel("Ergodicity metric (lower is better)")
    ax.set_title("Ergodicity vs Trajectory Step")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_k_index_change(summary_rows: List[Dict[str, float]], out_path: Path) -> None:
    methods = [r["method"] for r in summary_rows]
    x = np.arange(len(methods))
    red = [r["red_k_mean"] for r in summary_rows]
    white = [r["white_k_mean"] for r in summary_rows]
    delta = [r["k_change_red_minus_white"] for r in summary_rows]

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    bar_w = 0.33
    ax.bar(x - bar_w / 2, red, width=bar_w, color="#EF4444", alpha=0.85, label="Red (sensor ON)")
    ax.bar(x + bar_w / 2, white, width=bar_w, color="#E5E7EB", edgecolor="#6B7280", alpha=0.95, label="White (sensor OFF)")
    for i, dv in enumerate(delta):
        ax.text(x[i], max(red[i], white[i]) + 0.08, f"Δ={dv:+.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=12, ha="right")
    ax.set_ylabel("Mean dominant |k| index magnitude")
    ax.set_title("Change of K Indexes for Red/White")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    rng = np.random.default_rng(SEED)

    repo_root = Path(__file__).resolve().parent
    out_dir = repo_root / "comparison_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_maps = load_entropy_maps(repo_root)
    maps, map_ids = build_map_sequence(full_maps, N_CYCLES, USE_REPEAT_MODE, REPEAT_PROB)
    _, h, w = maps.shape

    xs = torch.linspace(0, 1, w)
    ys = torch.linspace(0, 1, h)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    sample_grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    k1, k2 = np.meshgrid(np.arange(0, 20), np.arange(0, 20))
    k_np = np.stack([k1.ravel(), k2.ravel()], axis=-1).astype(np.float32)
    k = torch.tensor(k_np, dtype=torch.float32)
    lamk = torch.exp(-0.8 * torch.norm(k, dim=1))
    hk = torch.clamp(torch.tensor([get_hk(ki) for ki in k_np], dtype=torch.float32), min=1e-6)
    k_expanded = k.unsqueeze(0)
    fk_vals_all = torch.cos(sample_grid[:, None, :] * k_expanded).prod(dim=-1)
    k_norms = np.linalg.norm(k_np, axis=1)

    results_by_method: Dict[str, Dict[str, object]] = {}
    cycle_rows_all: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, float]] = []

    for method in METHOD_ORDER:
        print(f"\n=== Running method: {METHOD_LABEL[method]} ({method}) ===")
        result = run_method(method, maps, map_ids, sample_grid, k_expanded, lamk, hk, fk_vals_all, k_norms, rng)
        results_by_method[method] = result
        summary = result["summary"]
        summary_rows.append(summary)
        cycle_rows_all.extend(result["cycle_rows"])
        print(
            f"  mean_high_info_cov={summary['mean_coverage_high_info']:.4f}, "
            f"global_high_info_cov={summary['global_high_info_coverage_ratio']:.4f}, "
            f"final_ergodicity={summary['final_ergodicity']:.4f}, "
            f"total_runtime={summary['total_plan_time_sec']:.3f}s"
        )

    # Save tables
    summary_csv = out_dir / "method_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    cycle_csv = out_dir / "cycle_metrics.csv"
    with cycle_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cycle_rows_all[0].keys()))
        writer.writeheader()
        writer.writerows(cycle_rows_all)

    # Baseline output text block requested by user.
    baseline_txt = out_dir / "baseline_outputs.txt"
    with baseline_txt.open("w") as f:
        for method in ["frontier", "probabilistic", "greedy_info"]:
            s = results_by_method[method]["summary"]
            f.write(f"{METHOD_LABEL[method]} ({method})\n")
            f.write(
                f"  mean_coverage_high_info={s['mean_coverage_high_info']:.6f}\n"
                f"  mean_coverage_weighted={s['mean_coverage_weighted']:.6f}\n"
                f"  mean_coverage_global={s['mean_coverage_global']:.6f}\n"
                f"  global_high_info_coverage_ratio={s['global_high_info_coverage_ratio']:.6f}\n"
                f"  mean_plan_time_sec={s['mean_plan_time_sec']:.6f}\n"
                f"  total_plan_time_sec={s['total_plan_time_sec']:.6f}\n"
                f"  mean_ergodicity={s['mean_ergodicity']:.6f}\n"
                f"  final_ergodicity={s['final_ergodicity']:.6f}\n"
                f"  red_k_mean={s['red_k_mean']:.6f}\n"
                f"  white_k_mean={s['white_k_mean']:.6f}\n"
                f"  k_change_red_minus_white={s['k_change_red_minus_white']:.6f}\n\n"
            )

    # Figures by requested categories
    plot_coverage(summary_rows, out_dir / "fig_coverage_performance.png")
    plot_runtime(summary_rows, out_dir / "fig_runtime_comparison.png")
    plot_ergodicity_curves(results_by_method, out_dir / "fig_ergodicity_vs_step.png")
    plot_k_index_change(summary_rows, out_dir / "fig_k_index_red_white.png")

    # Print compact table
    print("\n=== Summary ===")
    for row in summary_rows:
        print(
            f"{row['method']:>18} | hi_cov={row['mean_coverage_high_info']:.4f} | "
            f"glob_hi_cov={row['global_high_info_coverage_ratio']:.4f} | "
            f"runtime={row['total_plan_time_sec']:.2f}s | "
            f"final_erg={row['final_ergodicity']:.4f}"
        )
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
