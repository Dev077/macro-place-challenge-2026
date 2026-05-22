"""
GraphGradPlacer — Unified best-of-three pipeline.

Pipeline (target ~800s/benchmark):
  α₁  Focused-Poisson electrostatic GP                (lk_placer/gp.py)        60s
  α₂  koral soft-only TILOS-faithful Adam (K=24)      (inline below)          220s
  α₃  Top-K oracle ranking + anchor safety net        (inline below)           15s
  1   FastEvaluator build (bit-exact incremental)     (lk_placer/placer.py)    10s
  2   True-cost numerical subgradient                 (lk_placer/placer.py)    60s
  3   Lin-Kernighan k-opt swap passes                 (lk_placer/placer.py)   150s
  4   Hierarchical regional LAHC (3×3 → 5×5)          (lk_placer/placer.py)   130s
  5   Global LAHC polish                              (lk_placer/placer.py)   130s

Class name `GraphGradPlacer` is preserved so the evaluation harness picks it up.

Every phase oracle-verifies via `compute_proxy_cost` and only commits zero-overlap
results; the legalized initial.plc anchor is the floor — if no phase beats it,
the anchor is returned.
"""
from __future__ import annotations

import importlib.util
import math
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark


# ────────────────────────────────────────────────────────────────────────────
# Device + plc loading
# ────────────────────────────────────────────────────────────────────────────


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_plc(name: str):
    """Best-effort plc loader for IBM and NG45 benchmarks."""
    from macro_place.loader import load_benchmark, load_benchmark_from_dir

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    ng45 = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
        "ariane133": "ariane133",
        "ariane136": "ariane136",
        "nvdla": "nvdla",
        "mempool_tile": "mempool_tile",
    }
    d = ng45.get(name)
    if d:
        base = (
            Path("external/MacroPlacement/Flows/NanGate45")
            / d
            / "netlist"
            / "output_CT_Grouping"
        )
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(
                str(base / "netlist.pb.txt"), str(base / "initial.plc")
            )
            return plc
    return None


# ────────────────────────────────────────────────────────────────────────────
# Pin tensors (for soft-only Adam phase)
# ────────────────────────────────────────────────────────────────────────────


def _build_pin_tensors(benchmark: Benchmark, device: torch.device):
    n_hard = benchmark.num_hard_macros
    owner_list, off_list, net_list = [], [], []
    nid = 0
    for pins in benchmark.net_pin_nodes:
        if pins.shape[0] < 2:
            continue
        for owner_v, slot_v in pins.tolist():
            o = int(owner_v)
            s = int(slot_v)
            if o < n_hard:
                pin_off_t = benchmark.macro_pin_offsets[o]
                if pin_off_t.shape[0] > s:
                    ox, oy = pin_off_t[s].tolist()
                else:
                    ox, oy = 0.0, 0.0
            else:
                ox, oy = 0.0, 0.0
            owner_list.append(o)
            off_list.append([ox, oy])
            net_list.append(nid)
        nid += 1
    owner_idx = torch.tensor(owner_list, dtype=torch.long, device=device)
    pin_off = torch.tensor(off_list, dtype=torch.float32, device=device)
    net_id = torch.tensor(net_list, dtype=torch.long, device=device)
    return owner_idx, pin_off, net_id, nid


# ────────────────────────────────────────────────────────────────────────────
# Legalization (vectorized push-apart + spiral fallback)
# ────────────────────────────────────────────────────────────────────────────


def _legalize_spread(
    pos: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    cw: float,
    ch: float,
    gap: float = 0.005,
    max_iters: int = 400,
) -> Tuple[np.ndarray, bool]:
    n = pos.shape[0]
    out = pos.copy()
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0
    bmargin = max(gap, 1e-3)
    for k in range(n):
        if not movable[k]:
            out[k] = pos[k]
    out[:, 0] = np.clip(out[:, 0], half_w + bmargin, cw - half_w - bmargin)
    out[:, 1] = np.clip(out[:, 1], half_h + bmargin, ch - half_h - bmargin)
    safety = max(gap * 0.5, 1e-5)
    buffer = gap
    for _ in range(max_iters):
        dx = out[:, 0:1] - out[:, 0:1].T
        dy = out[:, 1:2] - out[:, 1:2].T
        absdx = np.abs(dx)
        absdy = np.abs(dy)
        real_ox = sep_x - absdx
        real_oy = sep_y - absdy
        overlap = (real_ox + safety > 0) & (real_oy + safety > 0)
        np.fill_diagonal(overlap, False)
        if not overlap.any():
            break
        sign_dx = np.where(absdx < 1e-9, 1.0, np.sign(dx))
        sign_dy = np.where(absdy < 1e-9, 1.0, np.sign(dy))
        push_axis_x = overlap & (real_ox <= real_oy)
        push_axis_y = overlap & ~push_axis_x
        contrib_x = np.where(push_axis_x, sign_dx * (real_ox * 0.5 + buffer), 0.0)
        contrib_y = np.where(push_axis_y, sign_dy * (real_oy * 0.5 + buffer), 0.0)
        out[movable, 0] += contrib_x.sum(axis=1)[movable]
        out[movable, 1] += contrib_y.sum(axis=1)[movable]
        out[~movable] = pos[~movable]
        out[:, 0] = np.clip(out[:, 0], half_w + bmargin, cw - half_w - bmargin)
        out[:, 1] = np.clip(out[:, 1], half_h + bmargin, ch - half_h - bmargin)
    out_f32 = out.astype(np.float32).astype(np.float64)
    absdx32 = np.abs(out_f32[:, 0:1] - out_f32[:, 0:1].T)
    absdy32 = np.abs(out_f32[:, 1:2] - out_f32[:, 1:2].T)
    ov32 = (sep_x - absdx32 > 0) & (sep_y - absdy32 > 0)
    np.fill_diagonal(ov32, False)
    bounds_ok = bool(
        ((out_f32[:, 0] - half_w) >= 0).all()
        and ((out_f32[:, 0] + half_w) <= cw).all()
        and ((out_f32[:, 1] - half_h) >= 0).all()
        and ((out_f32[:, 1] + half_h) <= ch).all()
    )
    return out, (not bool(ov32.any())) and bounds_ok


def _legalize_spiral(
    pos: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    cw: float,
    ch: float,
    gap: float = 0.005,
) -> np.ndarray:
    n = pos.shape[0]
    out = pos.copy()
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0
    bmargin = max(gap, 1e-3)
    areas = sizes[:, 0] * sizes[:, 1]
    placed = np.zeros(n, dtype=bool)
    for k in range(n):
        if not movable[k]:
            placed[k] = True

    def hits(idx, cx, cy):
        if not placed.any():
            return False
        d_x = np.abs(cx - out[:, 0])
        d_y = np.abs(cy - out[:, 1])
        m = (d_x < sep_x[idx] + gap) & (d_y < sep_y[idx] + gap) & placed
        m[idx] = False
        return bool(m.any())

    for idx in np.argsort(-areas):
        if not movable[idx]:
            continue
        cx = float(np.clip(pos[idx, 0], half_w[idx] + bmargin, cw - half_w[idx] - bmargin))
        cy = float(np.clip(pos[idx, 1], half_h[idx] + bmargin, ch - half_h[idx] - bmargin))
        if not hits(idx, cx, cy):
            out[idx, 0] = cx
            out[idx, 1] = cy
            placed[idx] = True
            continue
        step = max(0.02, min(float(sizes[idx, 0]), float(sizes[idx, 1])) * 0.1)
        best_p = (cx, cy)
        best_d = float("inf")
        for r in range(1, 400):
            found = False
            for dxs in range(-r, r + 1):
                for dys in range(-r, r + 1):
                    if abs(dxs) != r and abs(dys) != r:
                        continue
                    nx = float(np.clip(cx + dxs * step, half_w[idx] + bmargin, cw - half_w[idx] - bmargin))
                    ny = float(np.clip(cy + dys * step, half_h[idx] + bmargin, ch - half_h[idx] - bmargin))
                    if hits(idx, nx, ny):
                        continue
                    d = (nx - cx) ** 2 + (ny - cy) ** 2
                    if d < best_d:
                        best_d = d
                        best_p = (nx, ny)
                        found = True
            if found:
                break
        out[idx, 0] = best_p[0]
        out[idx, 1] = best_p[1]
        placed[idx] = True
    return out


def _legalize(
    pos: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    cw: float,
    ch: float,
    gap: float = 0.005,
) -> np.ndarray:
    out, ok = _legalize_spread(pos, sizes, movable, cw, ch, gap=gap)
    if ok:
        return out
    return _legalize_spiral(out, sizes, movable, cw, ch, gap=gap)


# ────────────────────────────────────────────────────────────────────────────
# Differentiable surrogate (TILOS-faithful)
# ────────────────────────────────────────────────────────────────────────────


def _lse_max_per_net(
    x: torch.Tensor, net_id: torch.Tensor, gamma: float, n_nets: int
) -> torch.Tensor:
    K, E = x.shape
    idx = net_id.unsqueeze(0).expand(K, E)
    max_pn = torch.full((K, n_nets), float("-inf"), device=x.device, dtype=x.dtype)
    max_pn.scatter_reduce_(1, idx, x, reduce="amax", include_self=True)
    max_pn = torch.where(torch.isfinite(max_pn), max_pn, torch.zeros_like(max_pn))
    shifted = x - max_pn.gather(1, idx)
    exp_v = torch.exp(shifted / gamma)
    sum_exp = torch.zeros(K, n_nets, device=x.device, dtype=x.dtype)
    sum_exp.scatter_add_(1, idx, exp_v)
    return gamma * torch.log(sum_exp.clamp(min=1e-12)) + max_pn


def wahpwl(
    pop: torch.Tensor,
    owner_idx: torch.Tensor,
    pin_off: torch.Tensor,
    net_id: torch.Tensor,
    n_nets: int,
    port_pos: torch.Tensor,
    n_hard: int,
    n_macros: int,
    gamma: float,
) -> torch.Tensor:
    K = pop.shape[0]
    n_ports = port_pos.shape[0]
    if n_ports > 0:
        ports_b = port_pos.unsqueeze(0).expand(K, n_ports, 2)
        owner_pos = torch.cat([pop, ports_b], dim=1)
    else:
        owner_pos = pop
    pin_owner = owner_pos[:, owner_idx, :]
    pin_pos = pin_owner + pin_off.unsqueeze(0)
    x_pins = pin_pos[..., 0]
    y_pins = pin_pos[..., 1]
    lse_max_x = _lse_max_per_net(x_pins, net_id, gamma, n_nets)
    lse_min_x = -_lse_max_per_net(-x_pins, net_id, gamma, n_nets)
    lse_max_y = _lse_max_per_net(y_pins, net_id, gamma, n_nets)
    lse_min_y = -_lse_max_per_net(-y_pins, net_id, gamma, n_nets)
    hpwl = (lse_max_x - lse_min_x) + (lse_max_y - lse_min_y)
    return hpwl.sum(dim=1)


def tilos_density_loss(
    pop: torch.Tensor,
    sizes: torch.Tensor,
    grid_col: int,
    grid_row: int,
    cw: float,
    ch: float,
) -> torch.Tensor:
    K, N, _ = pop.shape
    grid_w = cw / grid_col
    grid_h = ch / grid_row
    grid_area = grid_w * grid_h
    total_cells = grid_col * grid_row
    k_top = max(1, int(total_cells * 0.1))
    col_idx = torch.arange(grid_col, device=pop.device, dtype=pop.dtype)
    row_idx = torch.arange(grid_row, device=pop.device, dtype=pop.dtype)
    col_l = col_idx * grid_w
    col_r = col_l + grid_w
    row_b = row_idx * grid_h
    row_t = row_b + grid_h
    x = pop[..., 0]
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
    occupied = torch.einsum("knx,kny->kyx", x_ov, y_ov)
    density = occupied / grid_area
    cells = density.reshape(K, -1)
    sorted_d, _ = torch.sort(cells, dim=1, descending=True)
    top_sum = sorted_d[:, :k_top].sum(dim=1)
    return 0.5 * (top_sum / k_top)


def tilos_wl_normalized(
    pop: torch.Tensor,
    owner_idx: torch.Tensor,
    pin_off: torch.Tensor,
    net_id: torch.Tensor,
    n_nets: int,
    port_pos: torch.Tensor,
    n_hard: int,
    n_macros: int,
    gamma: float,
    cw: float,
    ch: float,
) -> torch.Tensor:
    raw = wahpwl(pop, owner_idx, pin_off, net_id, n_nets, port_pos, n_hard, n_macros, gamma)
    return raw / ((cw + ch) * max(n_nets, 1))


def tilos_rudy_normalized(
    pop: torch.Tensor,
    owner_idx: torch.Tensor,
    pin_off: torch.Tensor,
    net_id: torch.Tensor,
    n_nets: int,
    port_pos: torch.Tensor,
    n_hard: int,
    n_macros: int,
    grid_col: int,
    grid_row: int,
    cw: float,
    ch: float,
    h_routes_per_um: float,
    v_routes_per_um: float,
    smooth_range: int = 2,
    k_frac: float = 0.05,
) -> torch.Tensor:
    K = pop.shape[0]
    device = pop.device
    n_ports = port_pos.shape[0]
    if n_ports > 0:
        ports_b = port_pos.unsqueeze(0).expand(K, n_ports, 2)
        owner_pos = torch.cat([pop, ports_b], dim=1)
    else:
        owner_pos = pop
    pin_pos = owner_pos[:, owner_idx, :] + pin_off.unsqueeze(0)
    x = pin_pos[..., 0]
    y = pin_pos[..., 1]
    idx = net_id.unsqueeze(0).expand(K, -1)
    big = 1e9
    x_max = torch.full((K, n_nets), -big, device=device, dtype=pop.dtype)
    x_min = torch.full((K, n_nets), big, device=device, dtype=pop.dtype)
    y_max = torch.full((K, n_nets), -big, device=device, dtype=pop.dtype)
    y_min = torch.full((K, n_nets), big, device=device, dtype=pop.dtype)
    x_max.scatter_reduce_(1, idx, x, reduce="amax", include_self=True)
    x_min.scatter_reduce_(1, idx, x, reduce="amin", include_self=True)
    y_max.scatter_reduce_(1, idx, y, reduce="amax", include_self=True)
    y_min.scatter_reduce_(1, idx, y, reduce="amin", include_self=True)
    bbox_w = (x_max - x_min).clamp(min=1e-3)
    bbox_h = (y_max - y_min).clamp(min=1e-3)

    grid_w = cw / grid_col
    grid_h = ch / grid_row
    col_idx = torch.arange(grid_col, device=device, dtype=pop.dtype)
    row_idx = torch.arange(grid_row, device=device, dtype=pop.dtype)
    col_l = col_idx * grid_w
    col_r = col_l + grid_w
    row_b = row_idx * grid_h
    row_t = row_b + grid_h
    bxl = x_min.unsqueeze(-1); bxr = x_max.unsqueeze(-1)
    byb = y_min.unsqueeze(-1); byt = y_max.unsqueeze(-1)
    col_lb = col_l.view(1, 1, grid_col); col_rb = col_r.view(1, 1, grid_col)
    row_bb = row_b.view(1, 1, grid_row); row_tb = row_t.view(1, 1, grid_row)
    x_ov = (torch.min(bxr, col_rb) - torch.max(bxl, col_lb)).clamp(min=0.0)
    y_ov = (torch.min(byt, row_tb) - torch.max(byb, row_bb)).clamp(min=0.0)
    inv_h = (1.0 / bbox_h).unsqueeze(-1)
    inv_w = (1.0 / bbox_w).unsqueeze(-1)
    h_per_n_x = inv_h * x_ov
    v_per_n_x = inv_w * x_ov
    h_dem = torch.einsum("knx,kny->kyx", h_per_n_x, y_ov)
    v_dem = torch.einsum("knx,kny->kyx", v_per_n_x, y_ov)

    grid_v_routes = grid_w * v_routes_per_um
    grid_h_routes = grid_h * h_routes_per_um
    v_dem = v_dem / max(grid_v_routes, 1e-9)
    h_dem = h_dem / max(grid_h_routes, 1e-9)

    if smooth_range > 0:
        ksz = 2 * smooth_range + 1
        v_dem_r = v_dem.unsqueeze(1)
        h_dem_r = h_dem.unsqueeze(1)
        kx = torch.ones(1, 1, 1, ksz, device=device, dtype=pop.dtype) / ksz
        ky = torch.ones(1, 1, ksz, 1, device=device, dtype=pop.dtype) / ksz
        v_dem = torch.nn.functional.conv2d(v_dem_r, kx, padding=(0, smooth_range)).squeeze(1)
        h_dem = torch.nn.functional.conv2d(h_dem_r, ky, padding=(smooth_range, 0)).squeeze(1)

    flat = torch.cat([v_dem.reshape(K, -1), h_dem.reshape(K, -1)], dim=1)
    total = flat.shape[1]
    k_top = max(1, int(total * k_frac))
    sorted_c, _ = torch.sort(flat, dim=1, descending=True)
    return sorted_c[:, :k_top].mean(dim=1)


def _clip_soft_to_canvas(
    pos_full: np.ndarray, benchmark: Benchmark, n_hard: int, cw: float, ch: float
) -> np.ndarray:
    all_sizes = benchmark.macro_sizes.numpy()
    margin = 1e-3
    for i in range(n_hard, benchmark.num_macros):
        hw = float(all_sizes[i, 0]) / 2.0
        hh = float(all_sizes[i, 1]) / 2.0
        pos_full[i, 0] = max(hw + margin, min(cw - hw - margin, pos_full[i, 0]))
        pos_full[i, 1] = max(hh + margin, min(ch - hh - margin, pos_full[i, 1]))
    return pos_full


# ────────────────────────────────────────────────────────────────────────────
# Dynamic import of lk_placer modules (gp.py + placer.py)
# ────────────────────────────────────────────────────────────────────────────


def _lk_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_lk_gp():
    spec = importlib.util.spec_from_file_location("_koral_lk_gp", str(_lk_dir() / "gp.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_lk_placer():
    spec = importlib.util.spec_from_file_location("_koral_lk_placer", str(_lk_dir() / "placer2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ────────────────────────────────────────────────────────────────────────────
# Main placer
# ────────────────────────────────────────────────────────────────────────────


class GraphGradPlacer:
    """Unified pipeline: focused-Poisson GP → koral soft-only Adam →
    FastEvaluator-driven LK + regional + global LAHC."""

    def __init__(
        self,
        seed: int = 42,
        time_budget_s: float = 3300.0,
        verbose: bool = True,
        # Phase α₁ (focused-Poisson GP)
        gp_pop_size: int = 8,
        gp_steps: int = 5000,
        gp_budget_s: float = 300.0,
        # Phase α₂ (koral soft-only Adam)
        run_soft_adam: bool = True,
        soft_K: int = 16,
        soft_steps: int = 2500,
        soft_lr: float = 0.01,
        soft_budget_s: float = 220.0,
        # Phase 2 — α₂ true-cost subgradient
        run_subgrad: bool = True,
        subgrad_budget_s: float = 100.0,
        # Phase 3 — LK k-opt
        run_lk: bool = True,
        lk_passes: int = 2,
        lk_neighbors: int = 24,
        lk_chain_depth: int = 4,
        lk_budget_s: float = 150.0,
        # Phase 4 — Hierarchical regional LAHC
        run_regional: bool = True,
        regional_grid_sizes: Tuple[int, ...] = (3, 5, 7),
        regional_min_macros: int = 30,
        regional_budget_s: float = 200.0,
        # Phase 5 — Global LAHC
        run_lahc: bool = True,
        lahc_list_len: int = 100,
        lahc_min_budget_s: float = 60.0,
    ):
        self.seed = seed
        self.time_budget_s = time_budget_s
        self.verbose = verbose
        self.gp_pop_size = gp_pop_size
        self.gp_steps = gp_steps
        self.gp_budget_s = gp_budget_s
        self.run_soft_adam = run_soft_adam
        self.soft_K = soft_K
        self.soft_steps = soft_steps
        self.soft_lr = soft_lr
        self.soft_budget_s = soft_budget_s
        self.run_subgrad = run_subgrad
        self.subgrad_budget_s = subgrad_budget_s
        self.run_lk = run_lk
        self.lk_passes = lk_passes
        self.lk_neighbors = lk_neighbors
        self.lk_chain_depth = lk_chain_depth
        self.lk_budget_s = lk_budget_s
        self.run_regional = run_regional
        self.regional_grid_sizes = tuple(regional_grid_sizes)
        self.regional_min_macros = regional_min_macros
        self.regional_budget_s = regional_budget_s
        self.run_lahc = run_lahc
        self.lahc_list_len = lahc_list_len
        self.lahc_min_budget_s = lahc_min_budget_s

    def _log(self, msg: str):
        if self.verbose:
            print(f"[graph_grad] {msg}", flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        from macro_place.objective import compute_proxy_cost

        t0 = time.time()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        n_hard = benchmark.num_hard_macros
        n_macros = benchmark.num_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes_np_hard = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
        movable_hard = benchmark.get_movable_mask()[:n_hard].cpu().numpy().astype(bool)

        plc = _load_plc(benchmark.name)
        if plc is None:
            self._log(f"WARNING: plc=None for {benchmark.name}; oracle phases skipped")

        # Preserve original positions so we can rebuild the anchor at any point.
        orig_positions = benchmark.macro_positions.cpu().numpy().astype(np.float64).copy()

        # Per-benchmark K adaptation
        if n_macros < 100:
            soft_K = 16
        else:
            soft_K = self.soft_K
        soft_steps = self.soft_steps
        if n_hard > 500:
            soft_steps = min(soft_steps, 2000)

        # Phase 0: legalize initial.plc → anchor floor
        init_hard = orig_positions[:n_hard].copy()
        anchor_hard = _legalize(init_hard, sizes_np_hard, movable_hard, cw, ch, gap=0.005)
        anchor_full = orig_positions.copy()
        anchor_full[:n_hard] = anchor_hard
        anchor_full = _clip_soft_to_canvas(anchor_full, benchmark, n_hard, cw, ch)

        def _oracle(pos_np: np.ndarray) -> Tuple[float, int]:
            if plc is None:
                return float("inf"), 0
            try:
                c = compute_proxy_cost(torch.from_numpy(pos_np).float(), benchmark, plc)
                return float(c["proxy_cost"]), int(c["overlap_count"])
            except Exception:
                return float("inf"), -1

        anchor_cost, anchor_ov = _oracle(anchor_full)
        self._log(
            f"setup: name={benchmark.name} n_hard={n_hard} n_macros={n_macros} "
            f"budget={self.time_budget_s:.0f}s  anchor_proxy={anchor_cost:.4f} overlaps={anchor_ov}"
        )

        best_pos = anchor_full.copy() if anchor_ov == 0 else None
        best_true = anchor_cost if anchor_ov == 0 else float("inf")

        def _commit(pos_np: np.ndarray, label: str):
            nonlocal best_pos, best_true
            c, ov = _oracle(pos_np)
            self._log(f"  [{label}] oracle proxy={c:.4f} overlaps={ov}")
            if ov == 0 and c < best_true:
                best_true = c
                best_pos = pos_np.copy()
                self._log(f"  [{label}] new best={best_true:.4f}")
                return True
            return False

        # ── Phase α₁: focused-Poisson electrostatic GP ──
        # Start GP from anchor (the legalized initial.plc).
        benchmark.macro_positions = torch.from_numpy(anchor_full).float()
        if time.time() - t0 < self.time_budget_s * 0.95:
            try:
                gp_mod = _load_lk_gp()
                budget = min(self.gp_budget_s, self.time_budget_s - (time.time() - t0) - 30.0)
                self._log(
                    f"Phase α₁: focused-Poisson GP (pop={self.gp_pop_size}, "
                    f"steps={self.gp_steps}, budget={budget:.0f}s)"
                )
                gp_positions = gp_mod.run_global_placement(
                    benchmark, plc,
                    pop_size=self.gp_pop_size,
                    n_steps=self.gp_steps,
                    time_budget_s=max(10.0, budget),
                    seed=self.seed,
                    verbose=self.verbose,
                )
                # Legalize hard macros after GP (it can move them slightly into overlap)
                gp_hard = gp_positions[:n_hard].astype(np.float64)
                gp_hard_leg = _legalize(gp_hard, sizes_np_hard, movable_hard, cw, ch, gap=0.005)
                gp_full = gp_positions.astype(np.float64).copy()
                gp_full[:n_hard] = gp_hard_leg
                gp_full = _clip_soft_to_canvas(gp_full, benchmark, n_hard, cw, ch)
                _commit(gp_full, "α₁ gp")
                benchmark.macro_positions = torch.from_numpy(gp_full).float()
            except Exception as e:
                self._log(f"Phase α₁: skipped due to exception: {e}")

        # ── Phase α₂: koral soft-only Adam (K=24, hard locked) ──
        if self.run_soft_adam and time.time() - t0 < self.time_budget_s * 0.85:
            budget = min(self.soft_budget_s, self.time_budget_s - (time.time() - t0) - 60.0)
            self._log(
                f"Phase α₂: soft-only Adam (K={soft_K}, steps={soft_steps}, "
                f"budget={budget:.0f}s)"
            )
            try:
                pos_full, surr_cost = self._run_soft_adam(
                    benchmark, K=soft_K, n_steps=soft_steps,
                    budget_s=budget, plc=plc,
                )
                if pos_full is not None:
                    _commit(pos_full, "α₂ soft-Adam")
            except Exception as e:
                self._log(f"Phase α₂: exception: {e}")

        # ── Phase 1: FastEvaluator on best so far ──
        if best_pos is None:
            # Nothing valid yet — fall back to anchor (it's always legal but might
            # have overlap_count > 0 only if the input plc was broken)
            best_pos = anchor_full.copy()
            best_true = anchor_cost
        benchmark.macro_positions = torch.from_numpy(best_pos).float()

        try:
            lk_mod = _load_lk_placer()
        except Exception as e:
            self._log(f"Could not import lk_placer: {e}; returning best so far ({best_true:.4f})")
            return torch.from_numpy(best_pos).float()

        self._log("Phase 1: building FastEvaluator")
        try:
            ev = lk_mod.FastEvaluator(benchmark, plc)
        except Exception as e:
            self._log(f"FastEvaluator build failed: {e}; returning best so far ({best_true:.4f})")
            return torch.from_numpy(best_pos).float()

        c_fast = ev.proxy_cost()
        self._log(
            f"  fast baseline: proxy={c_fast['proxy_cost']:.4f} "
            f"wl={c_fast['wirelength_cost']:.4f} d={c_fast['density_cost']:.4f} "
            f"c={c_fast['congestion_cost']:.4f}"
        )
        # Drift check: fast vs oracle should agree to ~1e-3.
        drift = abs(c_fast["proxy_cost"] - best_true)
        if drift > 5e-3:
            self._log(f"  WARNING: FastEvaluator/oracle drift={drift:.4f}; downstream phases may misrank")

        # ── Phase 2: true-cost numerical subgradient ──
        if self.run_subgrad and time.time() - t0 < self.time_budget_s - self.lahc_min_budget_s:
            budget = min(self.subgrad_budget_s, self.time_budget_s - (time.time() - t0) - self.lahc_min_budget_s)
            self._log(f"Phase 2: true-cost subgradient (budget={budget:.0f}s)")
            try:
                lk_mod.true_cost_subgradient(ev, time_budget_s=budget, seed=self.seed, verbose=self.verbose)
                _commit(ev.positions.copy(), "subgrad")
            except Exception as e:
                self._log(f"Phase 2: exception: {e}")
            if best_pos is not None:
                ev.restore(best_pos)

        # ── Phase 3: LK k-opt ──
        if self.run_lk and time.time() - t0 < self.time_budget_s - self.lahc_min_budget_s:
            lk_deadline = time.time() + min(self.lk_budget_s,
                                            self.time_budget_s - (time.time() - t0) - self.lahc_min_budget_s)
            for p in range(self.lk_passes):
                if time.time() >= lk_deadline:
                    self._log(f"Phase 3 pass {p}: LK budget exhausted")
                    break
                self._log(f"Phase 3 pass {p}: LK k-opt")
                try:
                    order = lk_mod._macro_priority(ev)
                    cur_cost, n_acc = lk_mod.lk_swap_pass(
                        ev, order,
                        chain_depth=self.lk_chain_depth,
                        n_neighbors_per_macro=self.lk_neighbors,
                        log_every=max(1, len(order) // 6) if self.verbose else None,
                    )
                    self._log(f"  pass {p}: fast={cur_cost:.4f} accepted={n_acc}")
                    _commit(ev.positions.copy(), f"LK p{p}")
                except Exception as e:
                    self._log(f"Phase 3 pass {p}: exception: {e}")
                    break
            if best_pos is not None:
                ev.restore(best_pos)

        # ── Phase 4: Hierarchical regional LAHC ──
        if (
            self.run_regional
            and n_hard >= self.regional_min_macros
            and time.time() - t0 < self.time_budget_s - self.lahc_min_budget_s
        ):
            budget = min(self.regional_budget_s,
                         self.time_budget_s - (time.time() - t0) - self.lahc_min_budget_s)
            self._log(
                f"Phase 4: regional LAHC grids={self.regional_grid_sizes} budget={budget:.0f}s"
            )
            try:
                lk_mod.regional_polish(
                    ev,
                    region_grids=self.regional_grid_sizes,
                    time_budget_s=budget,
                    seed=self.seed,
                    verbose=self.verbose,
                )
                _commit(ev.positions.copy(), "regional")
            except Exception as e:
                self._log(f"Phase 4: exception: {e}")
            if best_pos is not None:
                ev.restore(best_pos)

        # ── Phase 5: Global LAHC polish ──
        if self.run_lahc:
            remaining = max(self.lahc_min_budget_s, self.time_budget_s - (time.time() - t0))
            self._log(f"Phase 5: global LAHC polish (budget={remaining:.0f}s)")
            try:
                lk_mod.lahc_polish(
                    ev,
                    list_len=self.lahc_list_len,
                    time_budget_s=remaining,
                    seed=self.seed,
                    verbose=self.verbose,
                )
                _commit(ev.positions.copy(), "LAHC")
            except Exception as e:
                self._log(f"Phase 5: exception: {e}")

        # ── Final safety net: never regress below the anchor floor ──
        if anchor_ov == 0 and anchor_cost < best_true:
            self._log(f"safety net: returning anchor ({anchor_cost:.4f}) < best ({best_true:.4f})")
            best_pos = anchor_full.copy()
            best_true = anchor_cost

        self._log(f"DONE  best_proxy={best_true:.4f}  time={time.time()-t0:.1f}s")
        return torch.from_numpy(best_pos).float()

    # ────────────────────────────────────────────────────────────────────
    # Phase α₂ — koral soft-only Adam (the proven sub-anchor winner)
    # ────────────────────────────────────────────────────────────────────

    def _run_soft_adam(
        self,
        benchmark: Benchmark,
        K: int,
        n_steps: int,
        budget_s: float,
        plc,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Run K-batched Adam on the TILOS-faithful surrogate with hard macros
        locked at their current positions in `benchmark.macro_positions`.

        Returns (best_full_positions_or_None, best_surrogate_cost).
        """
        from macro_place.objective import compute_proxy_cost

        t_start = time.time()
        device = _device()
        n_hard = benchmark.num_hard_macros
        n_macros = benchmark.num_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes_all = benchmark.macro_sizes.to(device).float()
        owner_idx, pin_off, net_id, n_nets = _build_pin_tensors(benchmark, device)
        port_pos = benchmark.port_positions.to(device).float()
        half_w_t = sizes_all[:, 0] / 2
        half_h_t = sizes_all[:, 1] / 2
        grid_col = int(benchmark.grid_cols)
        grid_row = int(benchmark.grid_rows)
        h_per_um = float(benchmark.hroutes_per_micron)
        v_per_um = float(benchmark.vroutes_per_micron)

        hard_lock_t = benchmark.macro_positions[:n_hard].to(device).float()
        soft_anchor = benchmark.macro_positions[n_hard:].to(device).float()

        pop = torch.zeros(K, n_macros, 2, device=device, dtype=torch.float32)
        pop[:, :n_hard] = hard_lock_t
        pop[:, n_hard:] = soft_anchor + torch.randn(K, n_macros - n_hard, 2, device=device) * 0.1
        pop.requires_grad_(True)
        opt = torch.optim.Adam([pop], lr=self.soft_lr)

        log_every = max(n_steps // 6, 1)
        for step in range(n_steps):
            if time.time() - t_start > budget_s:
                if self.verbose:
                    self._log(f"  soft-Adam: budget reached at step {step}")
                break
            opt.zero_grad()
            progress = step / max(n_steps, 1)
            gamma = 1.0 * (0.05 ** progress)
            wl_n = tilos_wl_normalized(
                pop, owner_idx, pin_off, net_id, n_nets, port_pos,
                n_hard, n_macros, gamma, cw, ch,
            )
            dens = tilos_density_loss(pop, sizes_all, grid_col, grid_row, cw, ch)
            cong = tilos_rudy_normalized(
                pop, owner_idx, pin_off, net_id, n_nets, port_pos,
                n_hard, n_macros, grid_col, grid_row, cw, ch,
                h_per_um, v_per_um, smooth_range=2, k_frac=0.05,
            )
            proxy_surr = wl_n + 0.5 * dens + 0.5 * cong
            loss = proxy_surr.sum()
            loss.backward()
            with torch.no_grad():
                pop.grad[:, :n_hard].zero_()
            opt.step()
            with torch.no_grad():
                pop[:, n_hard:, 0].clamp_(min=half_w_t[n_hard:], max=cw - half_w_t[n_hard:])
                pop[:, n_hard:, 1].clamp_(min=half_h_t[n_hard:], max=ch - half_h_t[n_hard:])
                pop[:, :n_hard] = hard_lock_t
            if self.verbose and step % log_every == 0:
                self._log(
                    f"  soft-Adam step {step}: wl={wl_n.mean().item():.4f} "
                    f"d={dens.mean().item():.4f} c={cong.mean().item():.4f} "
                    f"surr={proxy_surr.mean().item():.4f}"
                )

        # Rank candidates by surrogate; oracle-evaluate top-8.
        with torch.no_grad():
            wl_n = tilos_wl_normalized(
                pop, owner_idx, pin_off, net_id, n_nets, port_pos,
                n_hard, n_macros, 0.05, cw, ch,
            )
            dens = tilos_density_loss(pop, sizes_all, grid_col, grid_row, cw, ch)
            cong = tilos_rudy_normalized(
                pop, owner_idx, pin_off, net_id, n_nets, port_pos,
                n_hard, n_macros, grid_col, grid_row, cw, ch,
                h_per_um, v_per_um, smooth_range=2, k_frac=0.05,
            )
            surr = wl_n + 0.5 * dens + 0.5 * cong
        k_eval = min(K, 8)
        top_idx = torch.topk(-surr, k=k_eval).indices.tolist()

        best_full = None
        best_cost = float("inf")
        if plc is not None:
            for k in top_idx:
                pos_full = pop[k].detach().cpu().numpy().astype(np.float64)
                pos_full = _clip_soft_to_canvas(pos_full, benchmark, n_hard, cw, ch)
                try:
                    full_t = torch.from_numpy(pos_full).float()
                    c = compute_proxy_cost(full_t, benchmark, plc)
                    if c["overlap_count"] == 0 and c["proxy_cost"] < best_cost:
                        best_cost = float(c["proxy_cost"])
                        best_full = pos_full
                except Exception:
                    continue
        else:
            # No oracle: return the surrogate-best candidate
            k = int(torch.argmin(surr).item())
            best_full = pop[k].detach().cpu().numpy().astype(np.float64)
            best_full = _clip_soft_to_canvas(best_full, benchmark, n_hard, cw, ch)
            best_cost = float(surr[k].item())

        return best_full, best_cost
