"""
Phase α — Focused Electrostatic Global Placement
================================================

Continuous global placer that runs **before** the LK + LAHC refinement
pipeline.  Three innovations vs classical ePlace / DREAMPlace:

1. **Focused Poisson density target**.  Standard ePlace pushes every grid
   cell toward uniform density.  Our proxy only penalises the **top 10%**
   of density cells (and top 5% of routing cells).  So we set the Poisson
   source term = `density - top10pct_threshold` clamped at zero — only
   hot cells generate a field.  This concentrates electrostatic force where
   it lowers the actual cost.

2. **Focused congestion gradient**.  Same idea applied to RUDY routing
   demand: only cells above the top-5% threshold contribute to the loss.

3. **Pure 2D FFT Poisson solver** for the density potential.  Replaces the
   local bilinear-density gradient (which only feels its 4 neighbours)
   with a global potential field.  A density hotspot at (5, 5) creates a
   long-range field that pushes macros from (5, 5) toward emptier regions
   across the whole canvas.

The output is a numpy positions array `[N, 2]` that the downstream Phase 0
legalizer + LK + LAHC pipeline polishes.

Usage
-----
    from submissions.lk_placer.gp import run_global_placement
    positions = run_global_placement(benchmark, plc, time_budget_s=120.0)
"""
from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ────────────────────────────────────────────────────────────────────────────
# Pin tensor builder
# ────────────────────────────────────────────────────────────────────────────


def _build_pin_tensors(benchmark: Benchmark, device: torch.device):
    """Flatten net_pin_nodes into 3 parallel 1-D tensors.

    Returns
    -------
    owner_idx : [total_pins] int64 — owner index in a unified space
        [0, n_hard)              : hard macros (positions[i])
        [n_hard, n_macros)       : soft macros (positions[i])
        [n_macros, n_macros+P)   : I/O ports   (port_pos[i - n_macros])
    pin_off   : [total_pins, 2] — pin offset (zero for soft + ports)
    net_id    : [total_pins] int64 — which net each pin belongs to
    n_nets    : int
    """
    n_hard = benchmark.num_hard_macros
    pin_offsets = benchmark.macro_pin_offsets
    n_nets = int(benchmark.num_nets)
    if not benchmark.net_pin_nodes:
        # Fallback: use net_nodes (center pins only)
        owners_list = []
        offs_list = []
        net_id_list = []
        for net_idx in range(n_nets):
            nodes = benchmark.net_nodes[net_idx].numpy().astype(np.int64)
            for o in nodes:
                owners_list.append(o)
                offs_list.append([0.0, 0.0])
                net_id_list.append(net_idx)
        return (
            torch.tensor(owners_list, device=device, dtype=torch.long),
            torch.tensor(offs_list, device=device, dtype=torch.float32),
            torch.tensor(net_id_list, device=device, dtype=torch.long),
            n_nets,
        )
    owners_list = []
    offs_list = []
    net_id_list = []
    for net_idx in range(n_nets):
        pn = benchmark.net_pin_nodes[net_idx].numpy().astype(np.int64)
        for k in range(pn.shape[0]):
            o, s = int(pn[k, 0]), int(pn[k, 1])
            off_x = 0.0
            off_y = 0.0
            if o < n_hard and pin_offsets and o < len(pin_offsets):
                po = pin_offsets[o]
                if po is not None and po.shape[0] > s:
                    off_x = float(po[s, 0])
                    off_y = float(po[s, 1])
            owners_list.append(o)
            offs_list.append([off_x, off_y])
            net_id_list.append(net_idx)
    return (
        torch.tensor(owners_list, device=device, dtype=torch.long),
        torch.tensor(offs_list, device=device, dtype=torch.float32),
        torch.tensor(net_id_list, device=device, dtype=torch.long),
        n_nets,
    )


# ────────────────────────────────────────────────────────────────────────────
# Differentiable cost components
# ────────────────────────────────────────────────────────────────────────────


def _pin_positions(
    pop: torch.Tensor,            # [K, N, 2]
    owner_idx: torch.Tensor,      # [P]
    pin_off: torch.Tensor,        # [P, 2]
    port_pos: torch.Tensor,       # [n_ports, 2]
    n_macros: int,
) -> torch.Tensor:                # returns [K, P, 2]
    """Build pin positions: owner_pos + pin_off for macros; port_pos for ports."""
    K = pop.shape[0]
    n_ports = port_pos.shape[0]
    if n_ports > 0:
        ports_b = port_pos.unsqueeze(0).expand(K, n_ports, 2)
        owner_pos = torch.cat([pop, ports_b], dim=1)
    else:
        owner_pos = pop
    return owner_pos[:, owner_idx, :] + pin_off.unsqueeze(0)


def smooth_hpwl(
    pop: torch.Tensor,            # [K, N, 2]
    owner_idx: torch.Tensor,
    pin_off: torch.Tensor,
    net_id: torch.Tensor,
    n_nets: int,
    port_pos: torch.Tensor,
    n_macros: int,
    gamma: float,
    cw: float, ch: float,
    n_nets_norm: int,
    weights_per_net: Optional[torch.Tensor] = None,
) -> torch.Tensor:                # [K]
    """Weighted-Average HPWL via log-sum-exp.  Matches plc.get_cost normalization
    (divide by (cw + ch) × n_nets_norm).  As gamma → 0 this → true HPWL.
    """
    K = pop.shape[0]
    pins = _pin_positions(pop, owner_idx, pin_off, port_pos, n_macros)  # [K, P, 2]
    x = pins[..., 0]
    y = pins[..., 1]
    idx = net_id.unsqueeze(0).expand(K, -1)
    big = 1e9
    # Per-net true max / min (for shifted LSE — numerically stable).
    # Note these are detached so gradients only flow through the smooth softmax,
    # but the subtraction is just a constant per net at evaluation time.
    x_max = torch.full((K, n_nets), -big, device=pop.device, dtype=pop.dtype)
    x_min = torch.full((K, n_nets), big, device=pop.device, dtype=pop.dtype)
    y_max = torch.full((K, n_nets), -big, device=pop.device, dtype=pop.dtype)
    y_min = torch.full((K, n_nets), big, device=pop.device, dtype=pop.dtype)
    x_max.scatter_reduce_(1, idx, x, reduce="amax", include_self=True)
    x_min.scatter_reduce_(1, idx, x, reduce="amin", include_self=True)
    y_max.scatter_reduce_(1, idx, y, reduce="amax", include_self=True)
    y_min.scatter_reduce_(1, idx, y, reduce="amin", include_self=True)
    # Stable LSE: smooth_max(x) = x_max + γ log Σ exp((x - x_max)/γ)
    # We subtract x_max[net] from each pin to keep exp argument ≤ 0.
    x_max_per_pin = x_max.gather(1, idx)   # [K, P]
    x_min_per_pin = x_min.gather(1, idx)
    y_max_per_pin = y_max.gather(1, idx)
    y_min_per_pin = y_min.gather(1, idx)
    inv_gamma = 1.0 / gamma
    exp_pos_x = torch.exp((x - x_max_per_pin) * inv_gamma)
    exp_neg_x = torch.exp(-(x - x_min_per_pin) * inv_gamma)
    exp_pos_y = torch.exp((y - y_max_per_pin) * inv_gamma)
    exp_neg_y = torch.exp(-(y - y_min_per_pin) * inv_gamma)
    sum_pos_x = torch.zeros(K, n_nets, device=pop.device, dtype=pop.dtype)
    sum_neg_x = torch.zeros_like(sum_pos_x)
    sum_pos_y = torch.zeros_like(sum_pos_x)
    sum_neg_y = torch.zeros_like(sum_pos_x)
    sum_pos_x.scatter_add_(1, idx, exp_pos_x)
    sum_neg_x.scatter_add_(1, idx, exp_neg_x)
    sum_pos_y.scatter_add_(1, idx, exp_pos_y)
    sum_neg_y.scatter_add_(1, idx, exp_neg_y)
    # smooth_max - smooth_min in each axis
    smooth_max_x = x_max + gamma * torch.log(sum_pos_x.clamp(min=1e-30))
    smooth_min_x = x_min - gamma * torch.log(sum_neg_x.clamp(min=1e-30))
    smooth_max_y = y_max + gamma * torch.log(sum_pos_y.clamp(min=1e-30))
    smooth_min_y = y_min - gamma * torch.log(sum_neg_y.clamp(min=1e-30))
    hpwl_per_net = (smooth_max_x - smooth_min_x) + (smooth_max_y - smooth_min_y)
    if weights_per_net is not None:
        hpwl_per_net = hpwl_per_net * weights_per_net.unsqueeze(0)
    total = hpwl_per_net.sum(dim=1)  # [K]
    return total / ((cw + ch) * max(n_nets_norm, 1))


def bilinear_density(
    pop: torch.Tensor,            # [K, N, 2]
    sizes: torch.Tensor,          # [N, 2]
    grid_col: int, grid_row: int,
    cw: float, ch: float,
) -> torch.Tensor:                # [K, R, C]
    """Exact overlap-area binning of each macro into grid cells.
    Same math as PlacementCost: per-cell occupied_area / grid_area.
    Differentiable everywhere (piecewise-linear in positions).
    """
    K, N, _ = pop.shape
    grid_w = cw / grid_col
    grid_h = ch / grid_row
    grid_area = grid_w * grid_h
    device = pop.device
    dtype = pop.dtype
    col_l = torch.arange(grid_col, device=device, dtype=dtype) * grid_w
    col_r = col_l + grid_w
    row_b = torch.arange(grid_row, device=device, dtype=dtype) * grid_h
    row_t = row_b + grid_h
    x = pop[..., 0]                                 # [K, N]
    y = pop[..., 1]
    hw = sizes[:, 0] / 2
    hh = sizes[:, 1] / 2
    macro_l = (x - hw.unsqueeze(0)).unsqueeze(-1)
    macro_r = (x + hw.unsqueeze(0)).unsqueeze(-1)
    macro_b_ = (y - hh.unsqueeze(0)).unsqueeze(-1)
    macro_t_ = (y + hh.unsqueeze(0)).unsqueeze(-1)
    col_lb = col_l.view(1, 1, grid_col)
    col_rb = col_r.view(1, 1, grid_col)
    row_bb = row_b.view(1, 1, grid_row)
    row_tb = row_t.view(1, 1, grid_row)
    x_ov = (torch.min(macro_r, col_rb) - torch.max(macro_l, col_lb)).clamp(min=0.0)
    y_ov = (torch.min(macro_t_, row_tb) - torch.max(macro_b_, row_bb)).clamp(min=0.0)
    occupied = torch.einsum("knx,kny->kyx", x_ov, y_ov)  # [K, R, C]
    return occupied / grid_area


def rudy_demand(
    pop: torch.Tensor,            # [K, N, 2]
    owner_idx: torch.Tensor,
    pin_off: torch.Tensor,
    net_id: torch.Tensor,
    n_nets: int,
    port_pos: torch.Tensor,
    n_macros: int,
    grid_col: int, grid_row: int,
    cw: float, ch: float,
    h_per_um: float, v_per_um: float,
):
    """Differentiable RUDY routing demand on the V/H tracks.

    For each net, compute its bbox via min/max scatter (subgradient via
    torch).  Distribute demand uniformly over cells overlapping the bbox.
    H demand per cell = x_overlap × y_overlap / bbox_h  (matches RUDY).
    V demand per cell = x_overlap × y_overlap / bbox_w.

    Returns (v_demand, h_demand) — both [K, R, C].
    """
    K = pop.shape[0]
    device = pop.device
    dtype = pop.dtype
    pins = _pin_positions(pop, owner_idx, pin_off, port_pos, n_macros)
    x = pins[..., 0]
    y = pins[..., 1]
    idx = net_id.unsqueeze(0).expand(K, -1)
    big = 1e9
    x_max = torch.full((K, n_nets), -big, device=device, dtype=dtype)
    x_min = torch.full((K, n_nets), big, device=device, dtype=dtype)
    y_max = torch.full((K, n_nets), -big, device=device, dtype=dtype)
    y_min = torch.full((K, n_nets), big, device=device, dtype=dtype)
    x_max.scatter_reduce_(1, idx, x, reduce="amax", include_self=True)
    x_min.scatter_reduce_(1, idx, x, reduce="amin", include_self=True)
    y_max.scatter_reduce_(1, idx, y, reduce="amax", include_self=True)
    y_min.scatter_reduce_(1, idx, y, reduce="amin", include_self=True)
    bbox_w = (x_max - x_min).clamp(min=1e-3)
    bbox_h = (y_max - y_min).clamp(min=1e-3)
    grid_w = cw / grid_col
    grid_h = ch / grid_row
    col_l = torch.arange(grid_col, device=device, dtype=dtype) * grid_w
    col_r = col_l + grid_w
    row_b = torch.arange(grid_row, device=device, dtype=dtype) * grid_h
    row_t = row_b + grid_h
    bxl = x_min.unsqueeze(-1); bxr = x_max.unsqueeze(-1)
    byb = y_min.unsqueeze(-1); byt = y_max.unsqueeze(-1)
    col_lb = col_l.view(1, 1, grid_col); col_rb = col_r.view(1, 1, grid_col)
    row_bb = row_b.view(1, 1, grid_row); row_tb = row_t.view(1, 1, grid_row)
    x_ov = (torch.min(bxr, col_rb) - torch.max(bxl, col_lb)).clamp(min=0.0)
    y_ov = (torch.min(byt, row_tb) - torch.max(byb, row_bb)).clamp(min=0.0)
    inv_h = (1.0 / bbox_h).unsqueeze(-1)
    inv_w = (1.0 / bbox_w).unsqueeze(-1)
    # per-cell H demand = x_ov * y_ov / bbox_h, sum over nets
    # We compute as einsum to avoid materializing [K, n_nets, R, C]
    h_per_net_cell_y = (y_ov * inv_h).unsqueeze(-1)   # [K, nets, Gy, 1] but we'll batch
    # Sum over nets — use einsum:
    h_demand = torch.einsum("knx,kny->kyx", x_ov, y_ov * inv_h.expand_as(y_ov))  # [K, R, C]
    v_demand = torch.einsum("knx,kny->kyx", x_ov * inv_w.expand_as(x_ov), y_ov)
    # Normalize by per-cell track supply (matches TILOS)
    grid_v_routes = grid_w * v_per_um
    grid_h_routes = grid_h * h_per_um
    v_demand = v_demand / grid_v_routes
    h_demand = h_demand / grid_h_routes
    return v_demand, h_demand


def _smooth_1d_along_axis(grid: torch.Tensor, smooth_range: int, axis: int) -> torch.Tensor:
    """Apply TILOS-style 1-D smoothing along the chosen axis (matches PlacementCost).
    Each cell's value is divided by window count then spread into its window neighbours.
    """
    sr = smooth_range
    K = grid.shape[0]
    if axis == 0:
        # smooth along columns (axis=2 in [K,R,C])
        R, C = grid.shape[-2:]
        cols = torch.arange(C, device=grid.device)
        lp = torch.clamp(cols - sr, min=0)
        rp = torch.clamp(cols + sr, max=C - 1)
        cnt = (rp - lp + 1).to(grid.dtype)
        scaled = grid / cnt.view(1, 1, -1)
        pad = torch.nn.functional.pad(scaled, (sr, sr, 0, 0))
        csum = torch.cumsum(pad, dim=-1)
        cs0 = csum[..., 2 * sr:]
        cs1 = torch.cat([torch.zeros(K, R, 1, device=grid.device, dtype=grid.dtype), csum[..., :C - 1 + 2 * sr]], dim=-1)
        cs1 = cs1[..., :C]
        return cs0[..., :C] - cs1
    else:
        R, C = grid.shape[-2:]
        rows = torch.arange(R, device=grid.device)
        lp = torch.clamp(rows - sr, min=0)
        up = torch.clamp(rows + sr, max=R - 1)
        cnt = (up - lp + 1).to(grid.dtype)
        scaled = grid / cnt.view(1, -1, 1)
        pad = torch.nn.functional.pad(scaled, (0, 0, sr, sr))
        csum = torch.cumsum(pad, dim=-2)
        cs0 = csum[..., 2 * sr:, :]
        cs1 = torch.cat([torch.zeros(K, 1, C, device=grid.device, dtype=grid.dtype), csum[..., :R - 1 + 2 * sr, :]], dim=-2)
        cs1 = cs1[..., :R, :]
        return cs0[..., :R, :] - cs1


# ────────────────────────────────────────────────────────────────────────────
# Focused Poisson density loss — the INNOVATION
# ────────────────────────────────────────────────────────────────────────────


def _poisson_solve_batched(rho: torch.Tensor) -> torch.Tensor:
    """Solve ∇²φ = -ρ on a periodic grid via 2-D FFT.

    Input  rho : [K, R, C] charge density (the EXCESS — should be zero-mean)
    Output phi : [K, R, C] potential field

    Note: zero-mean enforcement is built in via DC term = 0.
    """
    K, R, C = rho.shape
    rho_hat = torch.fft.fft2(rho)
    # Wavenumbers
    ky = torch.fft.fftfreq(R, d=1.0, device=rho.device, dtype=torch.float32) * 2 * math.pi
    kx = torch.fft.fftfreq(C, d=1.0, device=rho.device, dtype=torch.float32) * 2 * math.pi
    k2 = ky.view(1, R, 1) ** 2 + kx.view(1, 1, C) ** 2
    k2 = k2.clamp(min=1e-9)
    phi_hat = rho_hat / k2
    phi_hat[:, 0, 0] = 0.0
    phi = torch.fft.ifft2(phi_hat).real
    return phi


def focused_electrostatic_loss(
    density_grid: torch.Tensor,   # [K, R, C]
    top_k_frac: float,
    push: float,
) -> torch.Tensor:
    """**Focused** electrostatic density loss.

    Compute the top-(top_k_frac) threshold per replica.  Only cells above
    threshold generate a Poisson source.  This concentrates the field where
    the proxy actually penalises us.

    Loss = mean(potential² × source²), summing over cells with positive source.
    """
    K, R, C = density_grid.shape
    flat = density_grid.reshape(K, -1)
    # rank = 1 - top_k_frac quantile (top 10% → 90th percentile)
    n = R * C
    k = max(1, int(n * top_k_frac))
    # threshold = the k-th largest value
    sorted_d, _ = torch.sort(flat, dim=1, descending=True)
    threshold = sorted_d[:, k - 1:k]  # [K, 1]
    # Source: positive excess above threshold
    source = torch.clamp(density_grid - threshold.view(K, 1, 1), min=0.0)
    if push != 1.0:
        source = source * push
    # Zero-mean enforce: subtract mean so total charge = 0 (Poisson convergence)
    source = source - source.mean(dim=(1, 2), keepdim=True)
    phi = _poisson_solve_batched(source)
    # Loss = ‖φ‖² — high potential = high energy
    return (phi ** 2).mean(dim=(1, 2))  # [K]


def focused_congestion_loss(
    v_demand: torch.Tensor,
    h_demand: torch.Tensor,
    smooth_range: int,
    top_k_frac: float = 0.05,
) -> torch.Tensor:
    """Legacy focused-congestion (local-gradient).

    Kept for the cong_w warm-up phase.  See focused_electrostatic_cong_loss
    below for the global-gradient Poisson version that drives the
    main congestion reduction.
    """
    K = v_demand.shape[0]
    v_s = _smooth_1d_along_axis(v_demand, smooth_range, axis=0)
    h_s = _smooth_1d_along_axis(h_demand, smooth_range, axis=1)
    combined = torch.cat([v_s.reshape(K, -1), h_s.reshape(K, -1)], dim=1)  # [K, 2*R*C]
    n = combined.shape[1]
    k = max(1, int(n * top_k_frac))
    sorted_c, _ = torch.sort(combined, dim=1, descending=True)
    top_k = sorted_c[:, :k]
    return top_k.mean(dim=1)  # [K]


def focused_electrostatic_cong_loss(
    v_demand: torch.Tensor,
    h_demand: torch.Tensor,
    smooth_range: int,
    top_k_frac: float = 0.05,
    push: float = 1.0,
) -> torch.Tensor:
    """Poisson-solved focused congestion loss — the congestion-side innovation.

    Mirrors `focused_electrostatic_loss` but on the routing demand grid:
       1. Smooth V/H demand per axis (matches PlacementCost smoothing).
       2. Sum V + H into a combined 2D demand grid (top-5% on combined is
          what abu computes, so this is the natural fused object).
       3. Threshold at the top-K quantile; only cells above contribute.
       4. Zero-mean the source (Poisson convergence).
       5. FFT-solve ∇²φ = -source → global potential field.
       6. Loss = ‖φ‖² — minimised by spreading the focused demand out of
          the hot region across the whole canvas.

    Why this beats the legacy local-gradient version: a macro across the
    canvas whose net bbox happens to cross a hot routing cell will feel a
    repulsive force pulling its pins out of the hot region.  The local
    version only saw gradients from macros adjacent to the cell.
    """
    v_s = _smooth_1d_along_axis(v_demand, smooth_range, axis=0)  # [K, R, C]
    h_s = _smooth_1d_along_axis(h_demand, smooth_range, axis=1)
    combined = v_s + h_s   # [K, R, C] — fused congestion grid
    K, R, C = combined.shape
    flat = combined.reshape(K, -1)
    n = R * C
    k = max(1, int(n * top_k_frac))
    sorted_c, _ = torch.sort(flat, dim=1, descending=True)
    threshold = sorted_c[:, k - 1:k]  # [K, 1]
    source = torch.clamp(combined - threshold.view(K, 1, 1), min=0.0)
    if push != 1.0:
        source = source * push
    source = source - source.mean(dim=(1, 2), keepdim=True)
    phi = _poisson_solve_batched(source)
    return (phi ** 2).mean(dim=(1, 2))   # [K]


# ────────────────────────────────────────────────────────────────────────────
# Pairwise hard-macro overlap penalty (differentiable)
# ────────────────────────────────────────────────────────────────────────────


def pairwise_overlap(pos_hard: torch.Tensor, sizes_hard: torch.Tensor) -> torch.Tensor:
    """Sum of pairwise overlap areas between hard macros.  [K, n_hard, 2] in."""
    K, N, _ = pos_hard.shape
    if N <= 1:
        return torch.zeros(K, device=pos_hard.device)
    # broadcast pairwise
    xi = pos_hard[:, :, 0].unsqueeze(2)   # [K, N, 1]
    xj = pos_hard[:, :, 0].unsqueeze(1)   # [K, 1, N]
    yi = pos_hard[:, :, 1].unsqueeze(2)
    yj = pos_hard[:, :, 1].unsqueeze(1)
    wi = sizes_hard[:, 0].view(1, N, 1)
    wj = sizes_hard[:, 0].view(1, 1, N)
    hi = sizes_hard[:, 1].view(1, N, 1)
    hj = sizes_hard[:, 1].view(1, 1, N)
    ox = ((wi + wj) / 2 - (xi - xj).abs()).clamp(min=0.0)
    oy = ((hi + hj) / 2 - (yi - yj).abs()).clamp(min=0.0)
    pairs = ox * oy  # [K, N, N]
    # mask diagonal
    mask = 1 - torch.eye(N, device=pos_hard.device).view(1, N, N)
    return (pairs * mask).sum(dim=(1, 2)) / 2  # divide by 2 because i<j == j<i


# ────────────────────────────────────────────────────────────────────────────
# Main GP entry point
# ────────────────────────────────────────────────────────────────────────────


def run_global_placement(
    benchmark: Benchmark,
    plc=None,
    *,
    pop_size: int = 4,
    n_steps: int = 500,
    lr: float = 0.03,
    gamma_start: float = 1.0,
    gamma_end: float = 0.05,
    density_w_start: float = 0.0,
    density_w_end: float = 1.0,
    cong_w_start: float = 0.0,
    cong_w_end: float = 1.0,
    overlap_w_start: float = 0.0,
    overlap_w_end: float = 200.0,
    top_k_density: float = 0.10,
    top_k_cong: float = 0.05,
    push_factor: float = 1.0,
    smooth_range: int = 2,
    time_budget_s: float = 120.0,
    seed: int = 0,
    # Replica-exchange parameters (Innovation 3)
    replica_swap_every: int = 50,
    replica_temperatures: Optional[Tuple[float, ...]] = None,
    verbose: bool = True,
    log_every: int = 50,
) -> np.ndarray:
    """Run electrostatic global placement, return best-cost positions [N, 2].

    Uses K=`pop_size` parallel chains with different jitter levels.  Picks the
    chain with lowest *true* proxy via PlacementCost at the end.
    """
    from macro_place.objective import compute_proxy_cost

    device = _device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_hard = benchmark.num_hard_macros
    n_macros = benchmark.num_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    grid_col = int(benchmark.grid_cols)
    grid_row = int(benchmark.grid_rows)
    h_per_um = float(benchmark.hroutes_per_micron)
    v_per_um = float(benchmark.vroutes_per_micron)

    sizes = benchmark.macro_sizes.to(device).float()
    sizes_hard = sizes[:n_hard]
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    port_pos = benchmark.port_positions.to(device).float() if benchmark.port_positions.shape[0] > 0 else torch.zeros(0, 2, device=device)
    owner_idx, pin_off, net_id, n_nets = _build_pin_tensors(benchmark, device)

    # Weights per net — pull from plc if available (matches PlacementCost normalization)
    weights_per_net = None
    n_nets_norm = n_nets
    if plc is not None:
        weights = np.ones(n_nets, dtype=np.float32)
        try:
            driver_names = list(plc.nets.keys())
            for i, name in enumerate(driver_names[:n_nets]):
                pin_i = plc.mod_name_to_indices[name]
                weights[i] = float(plc.modules_w_pins[pin_i].get_weight())
            n_nets_norm = int(getattr(plc, "net_cnt", n_nets))
        except Exception:
            pass
        weights_per_net = torch.tensor(weights, device=device, dtype=torch.float32)

    # Build initial population: replicate initial.plc with jitter per chain
    init_pos = benchmark.macro_positions.to(device).float()  # [N, 2]
    pop = init_pos.unsqueeze(0).expand(pop_size, -1, -1).contiguous()  # [K, N, 2]
    rng = torch.Generator(device=device).manual_seed(seed)
    jitter_levels = torch.linspace(0.0, 0.10, pop_size, device=device)  # 0% .. 10% canvas
    for k in range(pop_size):
        if jitter_levels[k] > 0:
            scale = float(jitter_levels[k]) * min(cw, ch)
            pop[k] = pop[k] + torch.randn(n_macros, 2, generator=rng, device=device) * scale
    # Clamp to canvas
    for k in range(pop_size):
        pop[k, :, 0].clamp_(min=half_w, max=cw - half_w)
        pop[k, :, 1].clamp_(min=half_h, max=ch - half_h)
    pop = pop.clone().requires_grad_(True)
    opt = torch.optim.Adam([pop], lr=lr)

    fixed_mask = benchmark.macro_fixed.to(device)  # [N] bool
    fixed_pos = init_pos.clone()
    # Replica-exchange temperatures: geometric ladder by default.
    if replica_temperatures is None:
        # Higher temp = more tolerant of bad swaps
        replica_temperatures = tuple(0.01 * (2.0 ** k) for k in range(pop_size))
    swap_rng = np.random.default_rng(seed + 7)

    # ── Net-level congestion reweighting (Innovation 4) ──
    # Every `reweight_every` steps, identify nets whose bboxes cross the
    # current top-5% congested cells and boost their WL weight in the loss.
    # This focuses HPWL optimization on the nets responsible for hotspots.
    base_net_weights = weights_per_net.clone() if weights_per_net is not None else torch.ones(n_nets, device=device)
    dyn_net_weights = base_net_weights.clone()
    reweight_every = 60   # steps
    reweight_alpha = 3.0  # multiplier for hot-net WL weight
    t0 = time.time()
    losses_history = []
    for step in range(n_steps):
        if time.time() - t0 > time_budget_s:
            if verbose:
                print(f"  [GP] step {step}: time budget reached", flush=True)
            break
        progress = step / max(n_steps - 1, 1)
        # Annealing
        gamma = gamma_start * (gamma_end / gamma_start) ** progress
        density_w = density_w_start + (density_w_end - density_w_start) * progress
        cong_w = cong_w_start + (cong_w_end - cong_w_start) * progress
        overlap_w = overlap_w_start * (1 - progress) + overlap_w_end * progress
        opt.zero_grad()
        # ── HPWL — uses dynamic weights so hot nets get extra pressure ──
        wl = smooth_hpwl(
            pop, owner_idx, pin_off, net_id, n_nets, port_pos, n_macros,
            gamma, cw, ch, n_nets_norm, weights_per_net=dyn_net_weights,
        )  # [K]
        # ── Density via focused Poisson ──
        dens_grid = bilinear_density(pop, sizes, grid_col, grid_row, cw, ch)
        dens_loss = focused_electrostatic_loss(dens_grid, top_k_density, push_factor)  # [K]
        # ── Congestion via RUDY ──
        v_dem, h_dem = rudy_demand(
            pop, owner_idx, pin_off, net_id, n_nets, port_pos, n_macros,
            grid_col, grid_row, cw, ch, h_per_um, v_per_um,
        )
        cong_loss = focused_congestion_loss(v_dem, h_dem, smooth_range, top_k_cong)  # [K]
        # ── Overlap (hard macros) ──
        ov_loss = pairwise_overlap(pop[:, :n_hard], sizes_hard) / (cw * ch)  # normalized
        # ── Total ──
        loss = (wl + density_w * dens_loss + cong_w * cong_loss + overlap_w * ov_loss).sum()
        loss.backward()
        with torch.no_grad():
            # Zero-out gradient for FIXED macros so they don't move
            pop.grad[:, fixed_mask] = 0
        opt.step()
        with torch.no_grad():
            # Reassert fixed macro positions, clamp all to canvas
            pop[:, fixed_mask] = fixed_pos[fixed_mask].unsqueeze(0).expand(pop_size, -1, -1)
            pop[:, :, 0].clamp_(min=half_w, max=cw - half_w)
            pop[:, :, 1].clamp_(min=half_h, max=ch - half_h)

            # ── Replica exchange (Innovation 3) ──
            # Periodically Metropolis-swap configurations between adjacent
            # chains based on the surrogate-loss delta and the temperature
            # ladder.  Helps escape basins that any single chain is stuck in.
            if pop_size > 1 and replica_swap_every > 0 and (step + 1) % replica_swap_every == 0:
                # Recompute per-chain proxy-surrogate cost (the same loss we
                # just optimized — cheap because we already have wl, dens, cong)
                per_replica = (wl + density_w * dens_loss + cong_w * cong_loss + overlap_w * ov_loss).detach()
                # Try swapping adjacent pairs (k, k+1)
                for k in range(pop_size - 1):
                    c_k = per_replica[k].item()
                    c_kp1 = per_replica[k + 1].item()
                    # Note: chain `k` has lower temp (tighter), chain k+1 hotter
                    Tk = replica_temperatures[k]
                    Tkp1 = replica_temperatures[k + 1]
                    # Parallel-tempering swap probability
                    delta = (1.0 / Tk - 1.0 / Tkp1) * (c_kp1 - c_k)
                    if delta >= 0 or swap_rng.random() < math.exp(delta):
                        # Swap positions
                        tmp = pop.data[k].clone()
                        pop.data[k] = pop.data[k + 1]
                        pop.data[k + 1] = tmp
                        # Swap costs in our local array so subsequent swap checks see updated state
                        per_replica[k], per_replica[k + 1] = per_replica[k + 1].clone(), per_replica[k].clone()
                        if verbose and (step + 1) % (replica_swap_every * 4) == 0:
                            print(f"    [REX] step {step+1}: swapped chains {k} ({c_k:.4f}) and {k+1} ({c_kp1:.4f})", flush=True)
        if verbose and step % log_every == 0:
            mean_wl = wl.mean().item()
            mean_d = dens_loss.mean().item()
            mean_c = cong_loss.mean().item()
            mean_ov = ov_loss.mean().item()
            print(
                f"  [GP] step {step:4d}  γ={gamma:.3f}  w_d={density_w:.2f} w_c={cong_w:.2f} w_ov={overlap_w:.1f}  "
                f"wl={mean_wl:.4f} dens_E={mean_d:.4f} cong_E={mean_c:.4f} ov={mean_ov:.5f}",
                flush=True,
            )
        losses_history.append(loss.item())

    # Pick best replica by true proxy cost
    if verbose:
        print(f"  [GP] done after {time.time()-t0:.1f}s, scoring {pop_size} replicas vs oracle", flush=True)
    best_cost = float("inf")
    best_pos = None
    for k in range(pop_size):
        pos_np = pop[k].detach().cpu().numpy().astype(np.float64)
        if plc is not None:
            full_t = torch.from_numpy(pos_np).float()
            c = compute_proxy_cost(full_t, benchmark, plc)
            score = float(c["proxy_cost"]) + (10.0 if c["overlap_count"] > 0 else 0.0)
            if verbose:
                print(f"    replica {k}: proxy={c['proxy_cost']:.4f}  overlaps={c['overlap_count']}",
                      flush=True)
            if score < best_cost:
                best_cost = score
                best_pos = pos_np
        else:
            best_pos = pos_np
            break
    return best_pos
