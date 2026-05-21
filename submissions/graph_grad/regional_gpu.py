"""
Phase 4 (GPU variant) — K parallel chains for hierarchical regional polish.

Each chain starts from the same post-LK positions but uses an independent RNG,
diverging into different basins as it walks the region sweep (R=3 → 5 → 7).
A single batched proxy_cost evaluates all K candidates per iteration in one
kernel call; the chain with the lowest final proxy is returned.

Re-uses gp.py building blocks (bilinear_density, rudy_demand,
macro_routing_demand, _smooth_1d_along_axis) and a torch-scatter HPWL.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark

# Load gp.py dynamically so regional_gpu.py can be importlib-loaded as a
# standalone module (matching how placer.py loads gp.py).
import importlib.util as _ilu
from pathlib import Path as _P
_gp_spec = _ilu.spec_from_file_location(
    "lk_placer_gp", str(_P(__file__).resolve().parent / "gp.py")
)
_gp = _ilu.module_from_spec(_gp_spec)
_gp_spec.loader.exec_module(_gp)
_build_pin_tensors = _gp._build_pin_tensors
_device = _gp._device
_smooth_1d_along_axis = _gp._smooth_1d_along_axis
bilinear_density = _gp.bilinear_density
macro_routing_demand = _gp.macro_routing_demand
rudy_demand = _gp.rudy_demand


# ────────────────────────────────────────────────────────────────────────────
# Batched proxy cost (bit-exact-ish vs PlacementCost)
# ────────────────────────────────────────────────────────────────────────────


def _exact_hpwl(
    pop: torch.Tensor,            # [K, N, 2]
    owner_idx: torch.Tensor,
    pin_off: torch.Tensor,
    net_id: torch.Tensor,
    n_nets: int,
    port_pos: torch.Tensor,
    n_macros: int,
    cw: float, ch: float,
    n_nets_norm: int,
    weights_per_net: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact HPWL per chain (no LSE smoothing — used for accept/reject, not gradient)."""
    K = pop.shape[0]
    n_ports = port_pos.shape[0]
    if n_ports > 0:
        ports_b = port_pos.unsqueeze(0).expand(K, n_ports, 2)
        owner_pos = torch.cat([pop, ports_b], dim=1)
    else:
        owner_pos = pop
    pins = owner_pos[:, owner_idx, :] + pin_off.unsqueeze(0)
    x = pins[..., 0]
    y = pins[..., 1]
    idx = net_id.unsqueeze(0).expand(K, -1)
    big = 1e9
    device = pop.device
    dtype = pop.dtype
    x_max = torch.full((K, n_nets), -big, device=device, dtype=dtype)
    x_min = torch.full((K, n_nets), big, device=device, dtype=dtype)
    y_max = torch.full((K, n_nets), -big, device=device, dtype=dtype)
    y_min = torch.full((K, n_nets), big, device=device, dtype=dtype)
    x_max.scatter_reduce_(1, idx, x, reduce="amax", include_self=True)
    x_min.scatter_reduce_(1, idx, x, reduce="amin", include_self=True)
    y_max.scatter_reduce_(1, idx, y, reduce="amax", include_self=True)
    y_min.scatter_reduce_(1, idx, y, reduce="amin", include_self=True)
    hpwl = (x_max - x_min) + (y_max - y_min)
    if weights_per_net is not None:
        hpwl = hpwl * weights_per_net.unsqueeze(0)
    return hpwl.sum(dim=1) / ((cw + ch) * max(n_nets_norm, 1))


def batched_proxy_cost(
    pop: torch.Tensor,            # [K, N, 2]
    sizes: torch.Tensor,          # [N, 2]
    owner_idx: torch.Tensor, pin_off: torch.Tensor,
    net_id: torch.Tensor, n_nets: int,
    port_pos: torch.Tensor,
    n_macros: int, n_hard: int,
    grid_col: int, grid_row: int,
    cw: float, ch: float,
    h_per_um: float, v_per_um: float,
    h_alloc: float, v_alloc: float,
    smooth_range: int,
    weights_per_net: Optional[torch.Tensor],
    n_nets_norm: int,
) -> Dict[str, torch.Tensor]:
    """Bit-exact-ish proxy cost on K parallel position batches.

    Matches PlacementCost.get_cost / get_density_cost / get_congestion_cost
    (used in FastEvaluator.proxy_cost), modulo two small approximations:
      1. density top-10% mean is over ALL cells (incl. zeros) rather than only
         nonzero cells — equivalent whenever n_nonzero ≥ 0.1·n_cells.
      2. macro_routing_demand uses a smooth row/col presence ramp rather than
         the proxy's hard step with partial-cell correction — matches exactly
         in the 50/50-partial case and is in the right direction elsewhere.
    """
    K = pop.shape[0]
    # Wirelength
    wl = _exact_hpwl(
        pop, owner_idx, pin_off, net_id, n_nets,
        port_pos, n_macros, cw, ch, n_nets_norm, weights_per_net,
    )
    # Density: 0.5 · top-10% mean of (density_grid / grid_area)
    dens_grid = bilinear_density(pop, sizes, grid_col, grid_row, cw, ch)  # [K, R, C]
    flat = dens_grid.reshape(K, -1)
    n_cells = flat.shape[1]
    k_top = max(1, int(n_cells * 0.1))
    sorted_d, _ = torch.sort(flat, dim=1, descending=True)
    dens_cost = 0.5 * sorted_d[:, :k_top].mean(dim=1)
    # Congestion: top-5% mean of concat(v_smoothed + v_macro, h_smoothed + h_macro)
    v_pin, h_pin = rudy_demand(
        pop, owner_idx, pin_off, net_id, n_nets,
        port_pos, n_macros, grid_col, grid_row, cw, ch,
        h_per_um, v_per_um,
    )
    v_macro, h_macro = macro_routing_demand(
        pop[:, :n_hard], sizes[:n_hard],
        grid_col, grid_row, cw, ch,
        h_alloc, v_alloc, h_per_um, v_per_um,
    )
    v_s = _smooth_1d_along_axis(v_pin, smooth_range, axis=0) + v_macro
    h_s = _smooth_1d_along_axis(h_pin, smooth_range, axis=1) + h_macro
    combined = torch.cat([v_s.reshape(K, -1), h_s.reshape(K, -1)], dim=1)
    n_cong = combined.shape[1]
    k_cong = max(1, int(n_cong * 0.05))
    sorted_c, _ = torch.sort(combined, dim=1, descending=True)
    cong_cost = sorted_c[:, :k_cong].mean(dim=1)
    proxy = wl + 0.5 * dens_cost + 0.5 * cong_cost
    return {
        "proxy_cost": proxy,
        "wirelength_cost": wl,
        "density_cost": dens_cost,
        "congestion_cost": cong_cost,
    }


# ────────────────────────────────────────────────────────────────────────────
# Legality: pairwise hard-macro overlap detection per chain
# ────────────────────────────────────────────────────────────────────────────


def _any_hard_overlap(
    pos_hard: torch.Tensor,         # [K, n_hard, 2]
    sizes_hard: torch.Tensor,       # [n_hard, 2]
    thresh: float = 1e-6,
) -> torch.Tensor:                  # [K] bool
    """True for chains where any pair of hard macros overlaps above `thresh`."""
    K, N, _ = pos_hard.shape
    if N <= 1:
        return torch.zeros(K, dtype=torch.bool, device=pos_hard.device)
    xi = pos_hard[:, :, 0].unsqueeze(2)
    xj = pos_hard[:, :, 0].unsqueeze(1)
    yi = pos_hard[:, :, 1].unsqueeze(2)
    yj = pos_hard[:, :, 1].unsqueeze(1)
    wi = sizes_hard[:, 0].view(1, N, 1)
    wj = sizes_hard[:, 0].view(1, 1, N)
    hi = sizes_hard[:, 1].view(1, N, 1)
    hj = sizes_hard[:, 1].view(1, 1, N)
    ox = ((wi + wj) / 2 - (xi - xj).abs()).clamp(min=0.0)
    oy = ((hi + hj) / 2 - (yi - yj).abs()).clamp(min=0.0)
    pairs = ox * oy
    mask = 1 - torch.eye(N, device=pos_hard.device).view(1, N, N)
    return ((pairs * mask) > thresh).any(dim=(1, 2))


# ────────────────────────────────────────────────────────────────────────────
# Multi-start regional polish
# ────────────────────────────────────────────────────────────────────────────


def _sample_per_chain_macro_in_region(
    pop_cpu: np.ndarray,            # [K, N, 2]
    movable: np.ndarray,            # [N] bool
    n_hard: int,
    x0: float, y0: float, x1: float, y1: float,
    rngs: List[np.random.Generator],
    require_hard: bool = False,
    require_soft: bool = False,
) -> List[int]:
    """For each chain, sample one movable macro whose center is in the region.

    Returns a list of length K — entries are -1 if no eligible macro exists
    in that chain's region.  Done in numpy + Python because per-chain
    rejection sampling is awkward to vectorize and the cost is K small lookups.
    """
    K = pop_cpu.shape[0]
    out: List[int] = []
    for k in range(K):
        xs = pop_cpu[k, :, 0]
        ys = pop_cpu[k, :, 1]
        in_rect = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1) & movable
        if require_hard:
            in_rect[n_hard:] = False
        elif require_soft:
            in_rect[:n_hard] = False
        idx = np.where(in_rect)[0]
        if idx.size == 0:
            out.append(-1)
        else:
            out.append(int(idx[rngs[k].integers(0, idx.size)]))
    return out


def regional_polish_gpu(
    benchmark: Benchmark,
    initial_positions: np.ndarray,
    plc,
    *,
    n_chains: int = 8,
    region_grids: Tuple[int, ...] = (3, 5, 7),
    list_len: int = 60,
    move_radius_frac: float = 0.12,
    soft_move_radius_frac: float = 0.06,
    soft_centroid_prob: float = 0.0,   # disabled in GPU variant for simplicity
    swap_prob: float = 0.30,
    soft_prob: float = 0.40,
    min_macros_per_region: int = 3,
    time_budget_s: float = 300.0,
    seed: int = 0,
    verbose: bool = True,
    rerank_with_true_proxy_cb=None,    # callable(pos_np) -> float
) -> Tuple[np.ndarray, Dict[str, float]]:
    """K-chain parallel regional polish on GPU (multi-start).

    Returns the positions of the best-cost chain (numpy [N, 2]) and a stats
    dict.  Legality is enforced per-chain via pairwise hard-macro overlap —
    illegal candidate moves are rejected before cost evaluation.
    """
    device = _device()
    n_hard = benchmark.num_hard_macros
    n_macros = benchmark.num_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    grid_col = int(benchmark.grid_cols)
    grid_row = int(benchmark.grid_rows)
    h_per_um = float(benchmark.hroutes_per_micron)
    v_per_um = float(benchmark.vroutes_per_micron)
    smooth_range = 2
    h_alloc = 0.0
    v_alloc = 0.0
    if plc is not None:
        try:
            ha, va = plc.get_macro_routing_allocation()
            h_alloc = float(ha)
            v_alloc = float(va)
        except Exception:
            h_alloc = float(getattr(plc, "hrouting_alloc", 0.0))
            v_alloc = float(getattr(plc, "vrouting_alloc", 0.0))
        try:
            smooth_range = int(plc.get_congestion_smooth_range())
        except Exception:
            smooth_range = int(getattr(plc, "smooth_range", 2))

    sizes = benchmark.macro_sizes.to(device).float()
    sizes_hard = sizes[:n_hard]
    half = sizes / 2.0
    half_np = half.cpu().numpy()
    movable_np = benchmark.get_movable_mask().cpu().numpy().astype(bool)
    port_pos = (
        benchmark.port_positions.to(device).float()
        if benchmark.port_positions.shape[0] > 0
        else torch.zeros(0, 2, device=device)
    )
    owner_idx, pin_off, net_id, n_nets = _build_pin_tensors(benchmark, device)

    n_nets_norm = n_nets
    weights_per_net = None
    if plc is not None:
        weights = np.ones(n_nets, dtype=np.float32)
        try:
            driver_names = list(plc.nets.keys())
            for i, name in enumerate(driver_names[:n_nets]):
                pi = plc.mod_name_to_indices[name]
                weights[i] = float(plc.modules_w_pins[pi].get_weight())
            n_nets_norm = int(getattr(plc, "net_cnt", n_nets))
        except Exception:
            pass
        if n_nets_norm <= 0:
            n_nets_norm = max(n_nets, 1)
        weights_per_net = torch.tensor(weights, device=device, dtype=torch.float32)

    init_pos_t = torch.from_numpy(initial_positions).to(device).float()
    pop = init_pos_t.unsqueeze(0).expand(n_chains, n_macros, 2).contiguous().clone()
    fixed_mask_t = benchmark.macro_fixed.to(device).bool()  # [N]
    fixed_pos = init_pos_t.clone()
    rngs = [np.random.default_rng(seed + 1000 + k) for k in range(n_chains)]

    def cost_fn(p):
        return batched_proxy_cost(
            p, sizes,
            owner_idx, pin_off, net_id, n_nets,
            port_pos, n_macros, n_hard,
            grid_col, grid_row, cw, ch,
            h_per_um, v_per_um,
            h_alloc, v_alloc,
            smooth_range,
            weights_per_net, n_nets_norm,
        )

    with torch.no_grad():
        cur_costs = cost_fn(pop)["proxy_cost"]  # [K]
    best_costs = cur_costs.clone()
    best_pop = pop.clone()
    histories = cur_costs.unsqueeze(1).expand(n_chains, list_len).clone()
    t0 = time.time()
    total_iters = 0
    total_accepted = 0

    chain_arange = torch.arange(n_chains, device=device)
    soft_count = n_macros - n_hard

    for sweep_idx, R in enumerate(region_grids):
        if time.time() - t0 >= time_budget_s:
            break
        rw = cw / R
        rh = ch / R
        regions = [(ri, ci) for ri in range(R) for ci in range(R)]
        # Visit regions in a fixed shuffle (seeded) — all chains see the same order
        np.random.default_rng(seed + sweep_idx).shuffle(regions)
        sweeps_left = len(region_grids) - sweep_idx
        time_left = time_budget_s - (time.time() - t0)
        sweep_budget = time_left / sweeps_left
        per_region_budget = max(2.0, sweep_budget / len(regions))

        for (ri, ci) in regions:
            if time.time() - t0 >= time_budget_s:
                break
            x0, x1 = ci * rw, (ci + 1) * rw
            y0, y1 = ri * rh, (ri + 1) * rh

            t_region = time.time()
            region_iters = 0
            region_accepted = 0
            while time.time() - t_region < per_region_budget:
                # Move-type selection: all chains do the same TYPE this iter (cheap to vectorize),
                # but each chain picks its own macro and offsets.
                r_global = float(rngs[0].random())
                pop_cpu = pop.detach().cpu().numpy()
                if r_global < swap_prob:
                    move_type = "swap"
                elif r_global < swap_prob + soft_prob and soft_count > 0:
                    move_type = "soft"
                else:
                    move_type = "slide"

                new_pop = pop.clone()
                legal_proposed = torch.ones(n_chains, dtype=torch.bool, device=device)

                if move_type == "swap":
                    i_choices = _sample_per_chain_macro_in_region(
                        pop_cpu, movable_np, n_hard, x0, y0, x1, y1, rngs, require_hard=True,
                    )
                    j_choices = _sample_per_chain_macro_in_region(
                        pop_cpu, movable_np, n_hard, x0, y0, x1, y1, rngs, require_hard=True,
                    )
                    for k in range(n_chains):
                        i = i_choices[k]; j = j_choices[k]
                        if i < 0 or j < 0 or i == j:
                            legal_proposed[k] = False
                            continue
                        # Swap positions
                        pi = new_pop[k, i].clone()
                        new_pop[k, i] = new_pop[k, j]
                        new_pop[k, j] = pi
                elif move_type == "soft":
                    choices = _sample_per_chain_macro_in_region(
                        pop_cpu, movable_np, n_hard, x0, y0, x1, y1, rngs, require_soft=True,
                    )
                    rx = soft_move_radius_frac * cw
                    ry = soft_move_radius_frac * ch
                    for k in range(n_chains):
                        i = choices[k]
                        if i < 0:
                            legal_proposed[k] = False
                            continue
                        dx = float(rngs[k].uniform(-rx, rx))
                        dy = float(rngs[k].uniform(-ry, ry))
                        hx = float(half_np[i, 0]); hy = float(half_np[i, 1])
                        nx = max(hx, min(cw - hx, float(pop_cpu[k, i, 0]) + dx))
                        ny = max(hy, min(ch - hy, float(pop_cpu[k, i, 1]) + dy))
                        new_pop[k, i, 0] = float(nx)
                        new_pop[k, i, 1] = float(ny)
                else:  # slide
                    choices = _sample_per_chain_macro_in_region(
                        pop_cpu, movable_np, n_hard, x0, y0, x1, y1, rngs, require_hard=True,
                    )
                    rx = move_radius_frac * cw
                    ry = move_radius_frac * ch
                    for k in range(n_chains):
                        i = choices[k]
                        if i < 0:
                            legal_proposed[k] = False
                            continue
                        dx = float(rngs[k].uniform(-rx, rx))
                        dy = float(rngs[k].uniform(-ry, ry))
                        hx = float(half_np[i, 0]); hy = float(half_np[i, 1])
                        nx = max(hx, min(cw - hx, float(pop_cpu[k, i, 0]) + dx))
                        ny = max(hy, min(ch - hy, float(pop_cpu[k, i, 1]) + dy))
                        new_pop[k, i, 0] = float(nx)
                        new_pop[k, i, 1] = float(ny)

                # Reassert fixed macro positions
                new_pop[:, fixed_mask_t] = fixed_pos[fixed_mask_t].unsqueeze(0).expand(n_chains, -1, -1)
                # Legality: no hard macro overlap above threshold
                with torch.no_grad():
                    overlap_bad = _any_hard_overlap(new_pop[:, :n_hard], sizes_hard)
                legal_proposed = legal_proposed & (~overlap_bad)
                # If nothing legal, skip eval entirely
                if not bool(legal_proposed.any()):
                    region_iters += 1
                    total_iters += 1
                    continue
                with torch.no_grad():
                    new_costs = cost_fn(new_pop)["proxy_cost"]  # [K]
                idx_h = region_iters % list_len
                hist_thresh = histories[:, idx_h]
                accept = (new_costs < cur_costs) | (new_costs < hist_thresh)
                accept = accept & legal_proposed
                # Commit accepted chains
                if bool(accept.any()):
                    new_pop_sel = new_pop.clone()
                    pop = torch.where(accept.view(n_chains, 1, 1), new_pop_sel, pop)
                    cur_costs = torch.where(accept, new_costs, cur_costs)
                    histories[:, idx_h] = torch.where(accept, new_costs, histories[:, idx_h])
                    # Update bests
                    better = cur_costs < best_costs
                    best_costs = torch.where(better, cur_costs, best_costs)
                    best_pop = torch.where(better.view(n_chains, 1, 1), pop, best_pop)
                    region_accepted += int(accept.sum().item())
                    total_accepted += int(accept.sum().item())
                region_iters += 1
                total_iters += 1

            if verbose and region_iters > 0:
                # Log only the worst-region or every-Nth would be cleaner; skip per-region noise.
                pass

        if verbose:
            with torch.no_grad():
                best_now = float(best_costs.min().item())
                mean_now = float(cur_costs.mean().item())
            print(
                f"  [REGIONAL-GPU] sweep {sweep_idx+1}/{len(region_grids)} R={R}  "
                f"chains={n_chains}  iters={total_iters}  accepted={total_accepted}  "
                f"mean_cur={mean_now:.4f}  best={best_now:.4f}",
                flush=True,
            )

    # Re-rank the K chains by the TRUE proxy (CPU FastEvaluator) before picking
    # the winner — protects against the GPU surrogate (RUDY pin congestion) being
    # biased relative to the proxy's Steiner-tree routing.
    chain_true_costs = []
    if rerank_with_true_proxy_cb is not None:
        for k in range(n_chains):
            pos_k = best_pop[k].detach().cpu().numpy().astype(np.float64)
            chain_true_costs.append(float(rerank_with_true_proxy_cb(pos_k)))
        k_best = int(np.argmin(chain_true_costs))
        true_best = chain_true_costs[k_best]
    else:
        k_best = int(torch.argmin(best_costs).item())
        true_best = float(best_costs[k_best].item())
        chain_true_costs = best_costs.detach().cpu().numpy().tolist()
    best_positions_np = best_pop[k_best].detach().cpu().numpy().astype(np.float64)
    return best_positions_np, {
        "proxy_cost": true_best,
        "iters": total_iters,
        "accepted": total_accepted,
        "best_chain": k_best,
        "all_chain_costs_gpu": best_costs.detach().cpu().numpy().tolist(),
        "all_chain_costs_true": chain_true_costs,
    }
