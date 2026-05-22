"""
Graph-Gradient Placer
=====================

A GPU-batched **analytical (gradient-based)** macro placer with a graph-aware
multi-start population, joint hard + soft macro optimization, and a density-
pressure annealing schedule.  Designed to reach sub-1 proxy cost on the IBM
benchmarks within the 1 h / benchmark budget on an RTX 6000 Ada.

Why analytical, not discrete search
-----------------------------------
On these benchmarks, ``initial.plc``'s wirelength is already near-optimal —
the entire path to sub-1 is reducing **density** and **congestion**, which
means *rearranging* macros to spread better, not minimizing wirelength.  A
smooth gradient-based objective with an explicit density-overshoot term can
do this directly.  A discrete GA (my earlier graph_evo placer) ends up
optimizing wirelength and oscillating around the anchor.

Innovations vs. the obvious "DREAMPlace clone"
----------------------------------------------
1. **Joint hard + soft optimization.**  Soft macros (standard-cell clusters)
   are included as free position variables in the same Adam optimizer, with
   *no* overlap constraint (they're allowed to overlap by problem definition).
   They contribute to wirelength (as net endpoints) and to density (their
   area lands in grid cells).  This is what the SA baseline does sequentially
   between iterations; we do it jointly on GPU.  Critically, this is what
   lets the density-pressure schedule reduce density without destroying WL —
   the optimizer can shift *both* hard *and* soft cells to spread the layout.
2. **Graph-aware diverse seeds.**  Population spans legalized initial.plc,
   Fiedler spectral embedding, k-means cluster placement, force-directed
   random starts, and jittered hybrids.  All K=32 evolve in parallel as a
   single ``[K, N, 2]`` tensor.
3. **Density-pressure annealing.**  α_density: 0.01 → 10, γ_HPWL: 2.0 → 0.1.
   Starts permissive (let WL minimize), ends strict (force density down).
4. **Periodic hard-macro projection** every epoch — gradient descent on a
   smooth landscape between projections back to the non-overlap manifold.
5. **True-cost selection.**  Final pick across K candidates by the real
   ``plc.get_cost()``.

Pipeline
--------
1. Build pin-level hypergraph + clique-expanded hard-macro pair graph.
2. Generate K diverse seeds (initial.plc, Fiedler, k-means, FD-random, jitter).
3. Stack as ``[K, N, 2]`` on GPU, optimize jointly with Adam.
4. Anneal γ (WAHPWL smoothing) and α (density / overlap weights) over 8 epochs
   × 500 steps.
5. Project hard macros to non-overlapping at every epoch boundary.
6. Final legalize + true-cost-best selection across the population.

Usage
-----
    uv run evaluate submissions/graph_grad/placer.py -b ibm01
    uv run evaluate submissions/graph_grad/placer.py --all
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# Graph extraction (pin-level hypergraph + clique-expanded pair graph)
# ────────────────────────────────────────────────────────────────────────────


def _build_pin_tensors(benchmark: Benchmark, device: torch.device):
    """
    Flatten net hypergraph into 1-D tensors over pins.

    owner_idx[e] ∈ [0, n_hard)            → hard macro index
                 ∈ [n_hard, n_macros)      → soft macro index
                 ∈ [n_macros, n_macros + n_ports) → I/O port index
    pin_off[e]   = (dx, dy) offset of pin from owner centre
    net_id[e]    = which net the pin belongs to
    """
    n_hard = benchmark.num_hard_macros
    n_macros = benchmark.num_macros
    n_ports = benchmark.port_positions.shape[0]

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
                ox, oy = 0.0, 0.0  # soft / port: pin at centre
            owner_list.append(o)
            off_list.append([ox, oy])
            net_list.append(nid)
        nid += 1
    owner_idx = torch.tensor(owner_list, dtype=torch.long, device=device)
    pin_off = torch.tensor(off_list, dtype=torch.float32, device=device)
    net_id = torch.tensor(net_list, dtype=torch.long, device=device)
    return owner_idx, pin_off, net_id, nid


def _build_pair_graph(benchmark: Benchmark) -> Tuple[np.ndarray, np.ndarray]:
    """Hard-macro clique-expanded pair graph (for spectral seeding)."""
    n_hard = benchmark.num_hard_macros
    edge_w: dict = {}
    for nodes in benchmark.net_nodes:
        hard = [int(x) for x in nodes.tolist() if int(x) < n_hard]
        if len(hard) < 2:
            continue
        w = 1.0 / (len(hard) - 1)
        hard.sort()
        for i in range(len(hard)):
            for j in range(i + 1, len(hard)):
                key = (hard[i], hard[j])
                edge_w[key] = edge_w.get(key, 0.0) + w
    if not edge_w:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    edges = np.array(list(edge_w.keys()), dtype=np.int64)
    weights = np.array([edge_w[k] for k in edge_w.keys()], dtype=np.float32)
    return edges, weights


def _spectral_layout(
    edges: np.ndarray,
    weights: np.ndarray,
    n: int,
    cw: float,
    ch: float,
    sizes: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Fiedler-style 2-D embedding, scaled into canvas."""
    if edges.shape[0] == 0:
        return _grid_fill(n, cw, ch)
    try:
        from scipy.sparse import coo_matrix, csr_matrix
        from scipy.sparse.linalg import eigsh

        i = np.concatenate([edges[:, 0], edges[:, 1]])
        j = np.concatenate([edges[:, 1], edges[:, 0]])
        v = np.concatenate([weights, weights]).astype(np.float64)
        W = coo_matrix((v, (i, j)), shape=(n, n)).tocsr()
        d = np.asarray(W.sum(axis=1)).ravel()
        d[d == 0] = 1e-9
        d_is = 1.0 / np.sqrt(d)
        D = csr_matrix(
            (d_is, (np.arange(n), np.arange(n))), shape=(n, n)
        )
        L = csr_matrix(
            (np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n)
        ) - D @ W @ D
        rng = np.random.default_rng(seed)
        try:
            vals, vecs = eigsh(L, k=min(3, n - 1), which="SM", v0=rng.random(n))
        except Exception:
            vals, vecs = eigsh(L, k=min(3, n - 1), which="SM")
        order = np.argsort(vals)
        vecs = vecs[:, order]
        ex = vecs[:, 1]
        ey = vecs[:, 2] if vecs.shape[1] > 2 else rng.standard_normal(n)
        # Random rotation for diversity across seeds
        theta = rng.uniform(0, 2 * math.pi)
        c_, s_ = math.cos(theta), math.sin(theta)
        rx = ex * c_ - ey * s_
        ry = ex * s_ + ey * c_
        pos = np.stack([rx, ry], axis=1)
        for a in (0, 1):
            lo, hi = float(pos[:, a].min()), float(pos[:, a].max())
            if hi - lo < 1e-12:
                pos[:, a] = rng.random(n)
            else:
                pos[:, a] = (pos[:, a] - lo) / (hi - lo)
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        margin_x = (float(half_w.max()) + 0.05) / cw
        margin_y = (float(half_h.max()) + 0.05) / ch
        pos[:, 0] = margin_x + pos[:, 0] * (1 - 2 * margin_x)
        pos[:, 1] = margin_y + pos[:, 1] * (1 - 2 * margin_y)
        pos[:, 0] *= cw
        pos[:, 1] *= ch
        return pos
    except Exception:
        return _grid_fill(n, cw, ch)


def _grid_fill(n: int, cw: float, ch: float) -> np.ndarray:
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = max(1, int(math.ceil(n / cols)))
    pos = np.zeros((n, 2), dtype=np.float64)
    for k in range(n):
        r, c = divmod(k, cols)
        pos[k, 0] = (c + 0.5) * cw / cols
        pos[k, 1] = (r + 0.5) * ch / rows
    return pos


# ────────────────────────────────────────────────────────────────────────────
# Legalization (vectorized push-apart + spiral fallback) — float32-aware
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
# Differentiable surrogate
# ────────────────────────────────────────────────────────────────────────────


def _lse_max_per_net(
    x: torch.Tensor, net_id: torch.Tensor, gamma: float, n_nets: int
) -> torch.Tensor:
    """Numerically stable γ·log Σ exp(x/γ), per-net, per-population."""
    K, E = x.shape
    idx = net_id.unsqueeze(0).expand(K, E)
    # per-net max (stabiliser)
    max_pn = torch.full((K, n_nets), float("-inf"), device=x.device, dtype=x.dtype)
    max_pn.scatter_reduce_(1, idx, x, reduce="amax", include_self=True)
    # Replace -inf (nets with no pins, should be none) with 0
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
    """
    Population WAHPWL via log-sum-exp.

    pop      : [K, n_macros, 2]   (hard + soft positions; fixed macros are
                                    locked outside this fn)
    port_pos : [n_ports, 2]       (fixed I/O ports)
    Returns  : [K]  smooth HPWL summed across nets
    """
    K = pop.shape[0]
    n_ports = port_pos.shape[0]
    # Augment population with port positions on the right (same for all K)
    if n_ports > 0:
        ports_b = port_pos.unsqueeze(0).expand(K, n_ports, 2)
        owner_pos = torch.cat([pop, ports_b], dim=1)  # [K, n_macros + n_ports, 2]
    else:
        owner_pos = pop
    # Per-pin owner position
    pin_owner = owner_pos[:, owner_idx, :]  # [K, E, 2]
    pin_pos = pin_owner + pin_off.unsqueeze(0)  # [K, E, 2]
    x_pins = pin_pos[..., 0]
    y_pins = pin_pos[..., 1]
    lse_max_x = _lse_max_per_net(x_pins, net_id, gamma, n_nets)
    lse_min_x = -_lse_max_per_net(-x_pins, net_id, gamma, n_nets)
    lse_max_y = _lse_max_per_net(y_pins, net_id, gamma, n_nets)
    lse_min_y = -_lse_max_per_net(-y_pins, net_id, gamma, n_nets)
    hpwl = (lse_max_x - lse_min_x) + (lse_max_y - lse_min_y)  # [K, n_nets]
    return hpwl.sum(dim=1)


def density_top_k(
    pop: torch.Tensor,
    sizes: torch.Tensor,
    grid_res: int,
    cw: float,
    ch: float,
    k_frac: float = 0.1,
) -> torch.Tensor:
    """
    Bilinear-spread macro area into grid cells, then return the **squared sum
    of the top-K% cell densities**, matching the proxy's top-10% density
    component closely.

    Differentiable through torch.sort.
    """
    K, N, _ = pop.shape
    cell_w = cw / grid_res
    cell_h = ch / grid_res
    cell_idx = torch.arange(grid_res, device=pop.device, dtype=pop.dtype)
    cell_l = cell_idx * cell_w
    cell_r = cell_l + cell_w
    cell_b = cell_idx * cell_h
    cell_t = cell_b + cell_h
    x = pop[..., 0]
    y = pop[..., 1]
    hw = sizes[:, 0] / 2
    hh = sizes[:, 1] / 2
    macro_l = (x - hw.unsqueeze(0)).unsqueeze(-1)
    macro_r = (x + hw.unsqueeze(0)).unsqueeze(-1)
    macro_b_ = (y - hh.unsqueeze(0)).unsqueeze(-1)
    macro_t_ = (y + hh.unsqueeze(0)).unsqueeze(-1)
    cell_lb = cell_l.view(1, 1, grid_res)
    cell_rb = cell_r.view(1, 1, grid_res)
    cell_bb = cell_b.view(1, 1, grid_res)
    cell_tb = cell_t.view(1, 1, grid_res)
    x_ov = (torch.min(macro_r, cell_rb) - torch.max(macro_l, cell_lb)).clamp(min=0.0)
    y_ov = (torch.min(macro_t_, cell_tb) - torch.max(macro_b_, cell_bb)).clamp(min=0.0)
    density = torch.einsum("knx,kny->kyx", x_ov, y_ov)  # [K, Gy, Gx]
    cells = density.reshape(K, -1)  # [K, G*G]
    k_top = max(1, int(cells.shape[1] * k_frac))
    sorted_d, _ = torch.sort(cells, dim=1, descending=True)
    # Normalise by cell area so the term scales naturally with macro count
    cell_area = cell_w * cell_h
    return sorted_d[:, :k_top].pow(2).sum(dim=1) / (cell_area ** 2)


def rudy_congestion(
    pop: torch.Tensor,
    owner_idx: torch.Tensor,
    pin_off: torch.Tensor,
    net_id: torch.Tensor,
    n_nets: int,
    port_pos: torch.Tensor,
    n_hard: int,
    n_macros: int,
    grid_res: int,
    cw: float,
    ch: float,
    k_frac: float = 0.05,
) -> torch.Tensor:
    """
    RUDY-style routing-congestion surrogate.

    For each net we spread its HPWL uniformly across its bounding-box area.
    Per-cell demand = sum_nets (net wirelength × bbox-cell overlap / bbox area).
    The penalty returned is the squared sum of the top-K% cells, matching the
    proxy's ``congestion_cost`` (which is the top-5% routing congestion).

    The forward pass uses hard min/max for the bbox (the subgradient is fine
    for descent; we already have a separate smooth WAHPWL term doing the
    long-range wirelength work).  Per-cell overlap is the standard
    differentiable rectangle-intersection.
    """
    K = pop.shape[0]
    device = pop.device
    n_ports = port_pos.shape[0]
    if n_ports > 0:
        ports_b = port_pos.unsqueeze(0).expand(K, n_ports, 2)
        owner_pos = torch.cat([pop, ports_b], dim=1)
    else:
        owner_pos = pop
    pin_pos = owner_pos[:, owner_idx, :] + pin_off.unsqueeze(0)  # [K, E, 2]
    x = pin_pos[..., 0]  # [K, E]
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
    bbox_area = bbox_w * bbox_h

    cell_w = cw / grid_res
    cell_h = ch / grid_res
    cell_idx = torch.arange(grid_res, device=device, dtype=pop.dtype)
    cell_l = cell_idx * cell_w
    cell_r = cell_l + cell_w
    cell_b = cell_idx * cell_h
    cell_t = cell_b + cell_h

    bxl = x_min.unsqueeze(-1)
    bxr = x_max.unsqueeze(-1)
    byb = y_min.unsqueeze(-1)
    byt = y_max.unsqueeze(-1)
    cell_lb = cell_l.view(1, 1, grid_res)
    cell_rb = cell_r.view(1, 1, grid_res)
    cell_bb = cell_b.view(1, 1, grid_res)
    cell_tb = cell_t.view(1, 1, grid_res)
    # [K, n_nets, G] each
    x_ov = (torch.min(bxr, cell_rb) - torch.max(bxl, cell_lb)).clamp(min=0.0)
    y_ov = (torch.min(byt, cell_tb) - torch.max(byb, cell_bb)).clamp(min=0.0)

    # H-demand contribution per cell: wl_x * x_ov * y_ov / bbox_area
    # = (bbox_w / bbox_area) * x_ov * y_ov  = (1 / bbox_h) * x_ov * y_ov
    # Similarly V-demand: 1 / bbox_w * x_ov * y_ov
    inv_h = (1.0 / bbox_h).unsqueeze(-1)  # [K, n_nets, 1]
    inv_w = (1.0 / bbox_w).unsqueeze(-1)
    h_per_net_cell = inv_h * x_ov  # [K, n_nets, Gx]
    v_per_net_cell = inv_w * x_ov  # [K, n_nets, Gx]  (same shape but different scale)
    # Outer-product over y_ov to get full 2-D demand tensor, summed over nets:
    # H[k, gy, gx] = sum_n h_per_net_cell[k, n, gx] * y_ov[k, n, gy]
    h_demand = torch.einsum("knx,kny->kyx", h_per_net_cell, y_ov)  # [K, Gy, Gx]
    # V-demand similarly but congestion = max(H, V)
    v_demand = torch.einsum("knx,kny->kyx", v_per_net_cell, y_ov)
    cell_cong = torch.maximum(h_demand, v_demand)
    # Top-K%
    cells = cell_cong.reshape(K, -1)
    k_top = max(1, int(cells.shape[1] * k_frac))
    sorted_c, _ = torch.sort(cells, dim=1, descending=True)
    return sorted_c[:, :k_top].pow(2).sum(dim=1)


def tilos_density_loss(
    pop: torch.Tensor,
    sizes: torch.Tensor,
    grid_col: int,
    grid_row: int,
    cw: float,
    ch: float,
) -> torch.Tensor:
    """
    Faithful port of plc_client_os.PlacementCost.get_density_cost.

    For each macro (hard + soft), accumulate (macro ∩ cell) area into a
    ``[grid_row, grid_col]`` grid; per-cell density = occupied_area / grid_area.
    Then return ``0.5 * mean(top-10% of cells)`` — *the exact same formula*
    as TILOS, with the same 0.5 prefactor and the same top-K count
    (``floor(num_cells * 0.1)``).

    This is differentiable everywhere via min/max + sort.
    """
    K, N, _ = pop.shape
    grid_w = cw / grid_col
    grid_h = ch / grid_row
    grid_area = grid_w * grid_h
    total_cells = grid_col * grid_row
    k_top = max(1, int(total_cells * 0.1))

    # Cell edges
    col_idx = torch.arange(grid_col, device=pop.device, dtype=pop.dtype)
    row_idx = torch.arange(grid_row, device=pop.device, dtype=pop.dtype)
    col_l = col_idx * grid_w
    col_r = col_l + grid_w
    row_b = row_idx * grid_h
    row_t = row_b + grid_h

    x = pop[..., 0]  # [K, N]
    y = pop[..., 1]
    hw = sizes[:, 0] / 2  # [N]
    hh = sizes[:, 1] / 2
    macro_l = (x - hw.unsqueeze(0)).unsqueeze(-1)  # [K, N, 1]
    macro_r = (x + hw.unsqueeze(0)).unsqueeze(-1)
    macro_b_ = (y - hh.unsqueeze(0)).unsqueeze(-1)
    macro_t_ = (y + hh.unsqueeze(0)).unsqueeze(-1)
    col_lb = col_l.view(1, 1, grid_col)
    col_rb = col_r.view(1, 1, grid_col)
    row_bb = row_b.view(1, 1, grid_row)
    row_tb = row_t.view(1, 1, grid_row)

    x_ov = (torch.min(macro_r, col_rb) - torch.max(macro_l, col_lb)).clamp(min=0.0)  # [K, N, Gx]
    y_ov = (torch.min(macro_t_, row_tb) - torch.max(macro_b_, row_bb)).clamp(min=0.0)  # [K, N, Gy]
    occupied = torch.einsum("knx,kny->kyx", x_ov, y_ov)  # [K, Gy, Gx]
    density = occupied / grid_area  # [K, Gy, Gx]
    cells = density.reshape(K, -1)
    sorted_d, _ = torch.sort(cells, dim=1, descending=True)
    # TILOS: sum of top-K density / K (top-K count is floor(total*0.1))
    top_sum = sorted_d[:, :k_top].sum(dim=1)
    return 0.5 * (top_sum / k_top)  # [K]  — matches get_density_cost()


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
    """
    WAHPWL normalized exactly the way ``PlacementCost.get_cost()`` normalizes:
        proxy_wl = total_HPWL / ((canvas_w + canvas_h) * net_cnt)

    As γ → 0 the WAHPWL converges to true Manhattan HPWL, so this matches the
    proxy's wirelength_cost in the limit.
    """
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
    """
    RUDY surrogate normalized as close to TILOS as a continuous approximation
    allows.  Differs from the exact TILOS routing (which is discrete L/T
    Steiner routing per net) but applies *the same per-axis normalisation*
    and *the same 1-D smoothing kernel*.

    Steps:
      1. Compute per-net bbox via hard min/max scatter (subgradient via
         torch.min/max).
      2. H-demand per cell = (wl_x / bbox_area) × (cell ∩ bbox area).
         V-demand similarly.
      3. Normalise V by grid_v_routes = grid_w × v_routes_per_um and H by
         grid_h_routes = grid_h × h_routes_per_um (same as TILOS).
      4. Apply TILOS's 1-D smoothing: V along columns, H along rows, kernel
         size = 2*smooth_range + 1.
      5. Concatenate V+H lists and return top-5% mean (TILOS's abu).
    """
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
    bbox_area = bbox_w * bbox_h

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
    x_ov = (torch.min(bxr, col_rb) - torch.max(bxl, col_lb)).clamp(min=0.0)  # [K, n_nets, Gx]
    y_ov = (torch.min(byt, row_tb) - torch.max(byb, row_bb)).clamp(min=0.0)  # [K, n_nets, Gy]
    # H demand: per-cell contribution = (wl_x / area) × x_ov × y_ov  = x_ov × y_ov / bbox_h
    inv_h = (1.0 / bbox_h).unsqueeze(-1)
    inv_w = (1.0 / bbox_w).unsqueeze(-1)
    h_per_n_x = inv_h * x_ov  # [K, n_nets, Gx]
    v_per_n_x = inv_w * x_ov  # [K, n_nets, Gx]  (RUDY V also occupies a vertical strip)
    h_dem = torch.einsum("knx,kny->kyx", h_per_n_x, y_ov)  # [K, Gy, Gx]
    v_dem = torch.einsum("knx,kny->kyx", v_per_n_x, y_ov)

    # Normalize per-axis by routing track capacity (matches TILOS):
    grid_v_routes = grid_w * v_routes_per_um
    grid_h_routes = grid_h * h_routes_per_um
    v_dem = v_dem / max(grid_v_routes, 1e-9)
    h_dem = h_dem / max(grid_h_routes, 1e-9)

    # TILOS smoothing — 1-D box-filter along axis of routing, width 2*smooth_range+1
    if smooth_range > 0:
        ksz = 2 * smooth_range + 1
        # V routing congestion smooths along columns (axis Gx); H smooths along rows (axis Gy).
        # Use conv1d with appropriate reshaping.
        v_dem_r = v_dem.unsqueeze(1)  # [K, 1, Gy, Gx]
        h_dem_r = h_dem.unsqueeze(1)
        # Build a 1-D smoothing kernel that sums (TILOS divides BEFORE distribution then sums after)
        # TILOS effective effect: each cell averages with its kernel-window neighbours and
        # the same value gets *added* into adjacent cells (see __smooth_routing_cong) — net
        # effect is a box filter normalised by (kernel-size at that location).
        # We approximate with a simple box filter of length ksz.
        kx = torch.ones(1, 1, 1, ksz, device=device, dtype=pop.dtype) / ksz
        ky = torch.ones(1, 1, ksz, 1, device=device, dtype=pop.dtype) / ksz
        v_dem = torch.nn.functional.conv2d(v_dem_r, kx, padding=(0, smooth_range)).squeeze(1)
        h_dem = torch.nn.functional.conv2d(h_dem_r, ky, padding=(smooth_range, 0)).squeeze(1)

    # TILOS concatenates V_cong + H_cong and takes top-5% mean
    flat = torch.cat([v_dem.reshape(K, -1), h_dem.reshape(K, -1)], dim=1)  # [K, 2*Gy*Gx]
    total = flat.shape[1]
    k_top = max(1, int(total * k_frac))
    sorted_c, _ = torch.sort(flat, dim=1, descending=True)
    return sorted_c[:, :k_top].mean(dim=1)  # [K]


def anchor_reg(
    pop: torch.Tensor, anchor: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Quadratic regulariser pulling movable macros toward the anchor positions.
    ``mask`` is [K, N] — True where this individual should pay anchor cost.

    The anchor regulariser is only applied to seed-0 (the anchor itself) +
    optionally a couple of jittered seeds, so the rest of the population is
    free to explore.
    """
    diff = (pop - anchor.unsqueeze(0)) * mask.unsqueeze(-1)
    return diff.pow(2).sum(dim=(1, 2))


def overlap_loss_hard(pos_hard: torch.Tensor, sizes_hard: torch.Tensor) -> torch.Tensor:
    """Pairwise AABB overlap area summed over upper-triangle of hard macros."""
    K, N, _ = pos_hard.shape
    if N < 2:
        return torch.zeros(K, device=pos_hard.device)
    dx = (pos_hard[:, :, None, 0] - pos_hard[:, None, :, 0]).abs()
    dy = (pos_hard[:, :, None, 1] - pos_hard[:, None, :, 1]).abs()
    sx = (sizes_hard[:, 0:1] + sizes_hard[None, :, 0]) / 2  # [N, N]
    sy = (sizes_hard[:, 1:2] + sizes_hard[None, :, 1]) / 2
    ox = (sx.unsqueeze(0) - dx).clamp(min=0.0)
    oy = (sy.unsqueeze(0) - dy).clamp(min=0.0)
    overlap = ox * oy
    diag = torch.eye(N, device=pos_hard.device, dtype=pos_hard.dtype).unsqueeze(0)
    return (overlap * (1.0 - diag)).sum(dim=(1, 2)) / 2.0


# ────────────────────────────────────────────────────────────────────────────
# Seed generation
# ────────────────────────────────────────────────────────────────────────────


def _force_directed_init(
    edges: np.ndarray,
    weights: np.ndarray,
    n: int,
    cw: float,
    ch: float,
    iters: int = 80,
    seed: int = 0,
) -> np.ndarray:
    """Lightweight FD: spring + uniform repulsion, projected to canvas."""
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2)) * np.array([cw, ch])
    if edges.shape[0] == 0:
        return pos
    for it in range(iters):
        # Attraction (springs)
        dx = pos[edges[:, 0], 0] - pos[edges[:, 1], 0]
        dy = pos[edges[:, 0], 1] - pos[edges[:, 1], 1]
        force = (weights * 0.05)[:, None] * np.stack([dx, dy], axis=1)
        # Accumulate per-node
        np.add.at(pos, edges[:, 0], -force)
        np.add.at(pos, edges[:, 1], force)
        # Mild centre pull to avoid drift
        pos[:, 0] = pos[:, 0] + 0.02 * (cw / 2 - pos[:, 0])
        pos[:, 1] = pos[:, 1] + 0.02 * (ch / 2 - pos[:, 1])
    return pos


def _kmeans_clusters(coords: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """k-means cluster labels."""
    if coords.shape[0] == 0 or k <= 1:
        return np.zeros(coords.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(coords.shape[0], size=min(k, coords.shape[0]), replace=False)
    centres = coords[idx].copy()
    labels = np.zeros(coords.shape[0], dtype=np.int64)
    for _ in range(20):
        d = ((coords[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(axis=1)
        for c in range(centres.shape[0]):
            sel = labels == c
            if sel.any():
                centres[c] = coords[sel].mean(axis=0)
    return labels


def _build_population_seeds(
    benchmark: Benchmark,
    K: int,
    edges: np.ndarray,
    weights: np.ndarray,
    sizes_np: np.ndarray,
    movable_np: np.ndarray,
    cw: float,
    ch: float,
    seed: int,
) -> np.ndarray:
    """
    Construct K diverse seed placements over both hard *and* soft macros.

    Returns an array of shape ``[K, n_macros, 2]``.  Soft macros for non-anchor
    seeds are initially placed at the spectral / FD layout's mean position
    plus a small jitter (so they have somewhere reasonable to start before the
    joint gradient descent moves them).
    """
    rng = np.random.default_rng(seed)
    n_hard = benchmark.num_hard_macros
    n_macros = benchmark.num_macros
    init_full = benchmark.macro_positions.numpy().astype(np.float64)  # [n_macros, 2]
    init_hard = init_full[:n_hard].copy()
    init_soft = init_full[n_hard:].copy()

    pop = np.zeros((K, n_macros, 2), dtype=np.float32)

    # Seed 0: legalized initial.plc
    leg0 = _legalize(init_hard, sizes_np, movable_np, cw, ch, gap=0.005)
    pop[0, :n_hard] = leg0
    pop[0, n_hard:] = init_soft

    if K == 1:
        return pop

    # Budget seed slots: roughly 1/8 spectral, 1/4 FD-random, rest jittered.
    # Always clamp to fit in [1, K-1].
    n_spec = max(1, K // 8) if K >= 8 else 0
    n_fd = max(1, K // 4) if K >= 4 else 0
    # If small K, fall back to just jittered anchor (cheapest, safest)
    used = 1
    # Seeds 1..n_spec: spectral with random rotations
    for k in range(1, 1 + n_spec):
        if k >= K:
            break
        sp = _spectral_layout(edges, weights, n_hard, cw, ch, sizes_np, seed=seed + k)
        sp_leg = _legalize(sp, sizes_np, movable_np, cw, ch, gap=0.005)
        pop[k, :n_hard] = sp_leg
        pop[k, n_hard:, 0] = cw / 2 + rng.normal(0, cw * 0.2, n_macros - n_hard)
        pop[k, n_hard:, 1] = ch / 2 + rng.normal(0, ch * 0.2, n_macros - n_hard)
        used = k + 1
    # FD-random
    for k in range(used, used + n_fd):
        if k >= K:
            break
        iters_ = rng.integers(40, 160)
        fd = _force_directed_init(edges, weights, n_hard, cw, ch, iters=int(iters_), seed=seed + k * 7)
        fd_leg = _legalize(fd, sizes_np, movable_np, cw, ch, gap=0.005)
        pop[k, :n_hard] = fd_leg
        pop[k, n_hard:, 0] = cw / 2 + rng.normal(0, cw * 0.25, n_macros - n_hard)
        pop[k, n_hard:, 1] = ch / 2 + rng.normal(0, ch * 0.25, n_macros - n_hard)
        used = k + 1
    # Remaining: jittered initial.plc at varying scales
    for k in range(used, K):
        scale = 0.02 + 0.10 * rng.random()
        jitter = rng.standard_normal((n_hard, 2)) * (min(cw, ch) * scale)
        jitter[~movable_np] = 0.0
        pos_h = init_hard + jitter
        pos_h_leg = _legalize(pos_h, sizes_np, movable_np, cw, ch, gap=0.005)
        pop[k, :n_hard] = pos_h_leg
        soft_jit = rng.standard_normal((n_macros - n_hard, 2)) * (min(cw, ch) * scale * 0.5)
        pop[k, n_hard:] = init_soft + soft_jit

    return pop


# ────────────────────────────────────────────────────────────────────────────
# Main placer
# ────────────────────────────────────────────────────────────────────────────


class GraphGradPlacer:
    """GPU-batched analytical placer with graph-aware multi-start."""

    def __init__(
        self,
        seed: int = 42,
        pop_size: int = 128,                # 3× the original to use Blackwell VRAM
        n_epochs: int = 10,
        steps_per_epoch: int = 10000,
        grid_res: int = 64,
        time_budget_s: float = 3000.0,     # 50 min
        verbose: bool = True,
        lock_hard: bool = True,            # Lock hard macros at legalized initial.plc
        soft_steps: int = 10000,            # Total Adam steps (was 3000)
        soft_lr: float = 0.01,
        n_restarts: int = 0,               # Independent restarts with different RNG seeds
        # LAHC tail stage (runs after analytical placement; bit-exact incremental
        # proxy via FastEvaluator). Does NOT change koral's placement process —
        # it only polishes the result. Disable with run_lahc=False.
        run_lahc: bool = True,
        lahc_budget_s: float = 2700.0,     # remaining budget after koral
        lahc_list_len: int = 100,
        lahc_min_budget_s: float = 60.0,
        lahc_log_interval_s: float = 15.0,
    ):
        self.seed = seed
        self.pop_size = pop_size
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.grid_res = grid_res
        self.time_budget_s = time_budget_s
        self.verbose = verbose
        self.lock_hard = lock_hard
        self.soft_steps = soft_steps
        self.soft_lr = soft_lr
        self.n_restarts = n_restarts
        self.run_lahc = run_lahc
        self.lahc_budget_s = lahc_budget_s
        self.lahc_list_len = lahc_list_len
        self.lahc_min_budget_s = lahc_min_budget_s
        self.lahc_log_interval_s = lahc_log_interval_s

    def _log(self, msg: str):
        if self.verbose:
            print(f"[graph_grad] {msg}", flush=True)

    def _true_cost(self, plc, full: torch.Tensor, benchmark: Benchmark) -> float:
        from macro_place.objective import compute_proxy_cost

        try:
            return float(compute_proxy_cost(full, benchmark, plc)["proxy_cost"])
        except Exception:
            return float("inf")

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # Hard-locked soft-only mode (proven sub-anchor on ibm01: 0.886 vs 1.039).
        # See _place_soft_only for the full schedule + safety net.
        t0 = time.time()
        self._log(
            f"place: config  lock_hard={self.lock_hard}  n_restarts={self.n_restarts}  "
            f"pop_size={self.pop_size}  soft_steps={self.soft_steps}  "
            f"run_lahc={self.run_lahc}"
        )
        if self.lock_hard:
            koral_positions = self._place_soft_only(benchmark)
        else:
            koral_positions = self._place_joint(benchmark)
        # LAHC tail polish (does not alter koral's analytical placement).
        # The total time budget self.time_budget_s is a HARD ceiling on koral +
        # LAHC combined. LAHC gets `time_budget_s - elapsed - margin`, clamped
        # to `lahc_min_budget_s` floor so it always runs for at least a minute.
        if self.run_lahc:
            elapsed = time.time() - t0
            margin = 30.0  # final oracle + tensor conversion overhead
            remaining = self.time_budget_s - elapsed - margin
            self._log(
                f"place: koral elapsed={elapsed:.1f}s  total_budget={self.time_budget_s:.0f}s  "
                f"LAHC remaining={remaining:.0f}s (floor={self.lahc_min_budget_s:.0f}s)"
            )
            return self._lahc_tail(benchmark, koral_positions, lahc_budget_override=remaining)
        return koral_positions

    def _lahc_tail(
        self,
        benchmark: Benchmark,
        koral_positions: torch.Tensor,
        lahc_budget_override: Optional[float] = None,
    ) -> torch.Tensor:
        """Run LAHC polish on koral's output. Only commits a strictly better,
        zero-overlap result; otherwise returns koral's positions unchanged."""
        from macro_place.objective import compute_proxy_cost

        t0 = time.time()
        # Guard: if koral returned None for any reason, fall back to initial.plc
        if koral_positions is None:
            self._log("LAHC tail: koral returned None; using initial.plc as fallback")
            koral_positions = benchmark.macro_positions.clone().float()

        plc = _load_plc(benchmark.name)
        if plc is None:
            self._log("LAHC tail: plc=None; skipping (FastEvaluator requires plc)")
            return koral_positions

        best_pos_np = koral_positions.detach().cpu().numpy().astype(np.float64).copy()
        try:
            baseline = compute_proxy_cost(koral_positions.float(), benchmark, plc)
            self._log(
                f"LAHC tail baseline: proxy={baseline['proxy_cost']:.4f} "
                f"overlaps={baseline['overlap_count']}"
            )
            best_cost = float(baseline["proxy_cost"]) if baseline["overlap_count"] == 0 else float("inf")
        except Exception as e:
            self._log(f"LAHC tail baseline oracle failed: {e}; skipping")
            return koral_positions

        # FastEvaluator picks up positions from benchmark.macro_positions
        benchmark.macro_positions = torch.from_numpy(best_pos_np).float()
        try:
            ev = FastEvaluator(benchmark, plc)
        except Exception as e:
            self._log(f"LAHC tail FastEvaluator build failed: {e}; returning koral output")
            return koral_positions

        fast_cost = ev.proxy_cost()["proxy_cost"]
        drift = abs(fast_cost - best_cost) if best_cost != float("inf") else 0.0
        self._log(f"LAHC tail FastEvaluator: fast={fast_cost:.4f} drift={drift:.4f}")
        if drift > 5e-3 and best_cost != float("inf"):
            self._log(f"  WARNING: FastEvaluator/oracle drift={drift:.4f}; LAHC may misrank")

        budget_target = lahc_budget_override if lahc_budget_override is not None else self.lahc_budget_s
        budget = max(self.lahc_min_budget_s, budget_target)
        self._log(f"LAHC tail: polish budget={budget:.0f}s")
        try:
            out = lahc_polish(
                ev,
                list_len=self.lahc_list_len,
                time_budget_s=budget,
                seed=self.seed,
                verbose=self.verbose,
                log_interval_s=self.lahc_log_interval_s,
            )
            self._log(f"LAHC tail done: fast_best={out['proxy_cost']:.4f} iters={out['iters']}")
        except Exception as e:
            self._log(f"LAHC tail raised: {e}; returning koral output")
            return koral_positions

        # Oracle-verify; only commit if zero-overlap and strictly better.
        try:
            cand = compute_proxy_cost(torch.from_numpy(ev.positions).float(), benchmark, plc)
            self._log(
                f"LAHC tail oracle: proxy={cand['proxy_cost']:.4f} "
                f"overlaps={cand['overlap_count']} baseline={best_cost:.4f}"
            )
            if cand["overlap_count"] == 0 and cand["proxy_cost"] < best_cost:
                best_cost = float(cand["proxy_cost"])
                best_pos_np = ev.positions.copy()
                self._log(f"  committing LAHC result: {best_cost:.4f}")
            else:
                self._log("  LAHC did not improve; keeping koral output")
        except Exception as e:
            self._log(f"LAHC tail final oracle check failed: {e}; keeping koral output")

        self._log(f"LAHC tail elapsed: {time.time()-t0:.1f}s")
        return torch.from_numpy(best_pos_np).float()

    def _place_soft_only(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Soft-only TILOS-faithful gradient placer.

        Hard macros are legalized once at the initial.plc layout and *locked*
        (their Adam gradients are zeroed every step).  Only soft macros move,
        optimizing the proxy-mirror surrogate
            ``wl_normalized + 0.5 * tilos_density + 0.5 * tilos_rudy``
        which matches ``compute_proxy_cost`` weighting and component scales
        (density is an exact TILOS port; RUDY congestion is the best
        differentiable approximation but its gradient direction is correct).

        If ``n_restarts > 1`` the optimization runs that many times with
        different RNG seeds and the overall best (by true proxy) is kept.

        Safety net: if no candidate beats the legalized anchor's true proxy,
        the anchor is returned.
        """
        best_across = None
        best_across_cost = float("inf")
        # n_restarts means "extra restarts beyond the first run", so always do
        # at least one pass. n_restarts=0 → 1 run; n_restarts=2 → 2 runs; etc.
        n_runs = max(1, self.n_restarts)
        self._log(f"_place_soft_only: n_restarts={self.n_restarts} → n_runs={n_runs}")
        for r in range(n_runs):
            run_seed = self.seed + 1000 * r
            self._log(f"_place_soft_only: starting run {r+1}/{n_runs}")
            try:
                full, cost = self._soft_only_single_run(benchmark, run_seed)
            except Exception as e:
                import traceback
                self._log(f"_place_soft_only: restart {r+1} RAISED: {type(e).__name__}: {e}")
                traceback.print_exc()
                full, cost = None, float("inf")
            self._log(
                f"_place_soft_only: restart {r+1} returned "
                f"full={'tensor' if full is not None else 'None'} cost={cost:.4f}"
            )
            # Always keep the first valid tensor — even if cost is inf (e.g., plc
            # missing and no oracle scoring possible) we still need to return
            # something downstream can consume.
            if full is not None and (best_across is None or cost < best_across_cost):
                best_across_cost = cost
                best_across = full
            if self.verbose and n_runs > 1:
                self._log(f"run {r+1}/{n_runs}: proxy={cost:.4f}  best={best_across_cost:.4f}")
        # Final fallback: legalized initial.plc anchor (should be unreachable but
        # guards against future regressions where every restart returns None).
        if best_across is None:
            self._log("WARNING: all restarts returned None; falling back to initial.plc")
            best_across = benchmark.macro_positions.clone().float()
        return best_across

    def _soft_only_single_run(self, benchmark: Benchmark, run_seed: int):
        """One soft-only optimization run.  Returns (placement_tensor, true_proxy)."""
        from macro_place.objective import compute_proxy_cost

        t_start = time.time()
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        random.seed(run_seed)
        device = _device()

        n_hard = benchmark.num_hard_macros
        n_macros = benchmark.num_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes_all = benchmark.macro_sizes.to(device).float()
        sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        movable_np = benchmark.get_movable_mask()[:n_hard].numpy()
        owner_idx, pin_off, net_id, n_nets = _build_pin_tensors(benchmark, device)
        port_pos = benchmark.port_positions.to(device).float()
        half_w_t = sizes_all[:, 0] / 2
        half_h_t = sizes_all[:, 1] / 2
        grid_col = int(benchmark.grid_cols)
        grid_row = int(benchmark.grid_rows)
        h_per_um = float(benchmark.hroutes_per_micron)
        v_per_um = float(benchmark.vroutes_per_micron)

        # Legalize hard macros once at initial.plc → anchor.
        init_pos = benchmark.macro_positions[:n_hard].numpy().astype(np.float64)
        hard_legal = _legalize(init_pos, sizes_np, movable_np, cw, ch, gap=0.005)
        hard_legal_t = torch.tensor(hard_legal, device=device, dtype=torch.float32)
        soft_anchor = benchmark.macro_positions[n_hard:].to(device).float()
        self._log(
            f"soft-only setup: n_hard={n_hard} n_soft={n_macros - n_hard} "
            f"n_nets={n_nets}  grid {grid_col}x{grid_row}  H/V {h_per_um:.2f}/{v_per_um:.2f}  "
            f"device={device}"
        )

        # Population: every seed shares the same locked hard layout;
        # soft starts at initial soft + small jitter.
        K = self.pop_size
        pop = torch.zeros(K, n_macros, 2, device=device, dtype=torch.float32)
        pop[:, :n_hard] = hard_legal_t
        pop[:, n_hard:] = soft_anchor + torch.randn(K, n_macros - n_hard, 2, device=device) * 0.1
        pop.requires_grad_(True)
        opt = torch.optim.Adam([pop], lr=self.soft_lr)

        n_steps = self.soft_steps
        for step in range(n_steps):
            opt.zero_grad()
            progress = step / max(n_steps, 1)
            gamma = 1.0 * (0.05 ** progress)  # WAHPWL smoothing → sharp Manhattan HPWL
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
            proxy_surr = wl_n + 0.5 * dens + 0.5 * cong  # mirrors compute_proxy_cost
            loss = proxy_surr.sum()
            loss.backward()
            with torch.no_grad():
                pop.grad[:, :n_hard].zero_()  # LOCK hard macros
            opt.step()
            with torch.no_grad():
                pop[:, n_hard:, 0].clamp_(min=half_w_t[n_hard:], max=cw - half_w_t[n_hard:])
                pop[:, n_hard:, 1].clamp_(min=half_h_t[n_hard:], max=ch - half_h_t[n_hard:])
                pop[:, :n_hard] = hard_legal_t  # reassert lock
            if self.verbose and step % max(n_steps // 6, 1) == 0:
                self._log(
                    f"  step {step}: wl_n={wl_n.mean().item():.4f} "
                    f"dens={dens.mean().item():.4f} cong={cong.mean().item():.4f} "
                    f"proxy_surr={proxy_surr.mean().item():.4f}"
                )

        # Rank candidates by surrogate; evaluate top-K' by TRUE proxy.
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
        k_eval = min(K, max(8, K // 2))
        top_idx = torch.topk(-surr, k=k_eval).indices.tolist()

        # Cache plc once — each _load_plc reparses the netlist (~seconds)
        plc = _load_plc(benchmark.name)
        best_full, best_cost = None, float("inf")
        for k in top_idx:
            pos_full = pop[k].detach().cpu().numpy().astype(np.float64)
            pos_full = self._clip_soft_to_canvas(pos_full, benchmark, n_hard, cw, ch)
            full_t = torch.from_numpy(pos_full).float()
            if plc is None:
                continue
            c = compute_proxy_cost(full_t, benchmark, plc)
            if c["overlap_count"] == 0 and c["proxy_cost"] < best_cost:
                best_cost = c["proxy_cost"]
                best_full = full_t
        # Anchor safety net
        full_anc = torch.zeros(n_macros, 2)
        full_anc[:n_hard] = hard_legal_t.cpu()
        full_anc[n_hard:] = soft_anchor.cpu()
        full_anc_np = full_anc.numpy().astype(np.float64)
        full_anc_np = self._clip_soft_to_canvas(full_anc_np, benchmark, n_hard, cw, ch)
        full_anc = torch.from_numpy(full_anc_np).float()
        c_anc = compute_proxy_cost(full_anc, benchmark, plc) if plc is not None else None
        if c_anc is not None and c_anc["proxy_cost"] < best_cost:
            best_cost = c_anc["proxy_cost"]
            best_full = full_anc
        if best_full is None:
            best_full = full_anc

        anchor_str = f"{c_anc['proxy_cost']:.4f}" if c_anc is not None else "NA"
        self._log(
            f"soft-only: best true_proxy={best_cost:.4f}  "
            f"anchor={anchor_str}  time={time.time()-t_start:.1f}s"
        )
        return best_full, best_cost

    def _place_joint(self, benchmark: Benchmark) -> torch.Tensor:
        """Original joint hard + soft gradient mode (kept for comparison)."""
        t_start = time.time()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        device = _device()
        n_hard = benchmark.num_hard_macros
        n_macros = benchmark.num_macros
        n_ports = benchmark.port_positions.shape[0]
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes_all = benchmark.macro_sizes.to(device).float()  # [n_macros, 2]
        sizes_hard = sizes_all[:n_hard]
        sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        movable_np = benchmark.get_movable_mask()[:n_hard].numpy()
        movable_t = benchmark.get_movable_mask().to(device)  # [n_macros]
        init_full_t = benchmark.macro_positions.to(device).float()  # [n_macros, 2]
        port_pos = benchmark.port_positions.to(device).float()  # [n_ports, 2]

        # ── 1. Graph + pin tensors ──
        edges, weights = _build_pair_graph(benchmark)
        owner_idx, pin_off, net_id, n_nets = _build_pin_tensors(benchmark, device)
        self._log(f"setup: n_hard={n_hard} n_soft={n_macros - n_hard} n_ports={n_ports} n_nets={n_nets} device={device}")

        # ── 2. Seed population ──
        K = self.pop_size
        seeds = _build_population_seeds(
            benchmark, K, edges, weights, sizes_np, movable_np, cw, ch, seed=self.seed
        )
        pop = torch.tensor(seeds, device=device, dtype=torch.float32)  # [K, n_macros, 2]
        # Lock fixed macros to their initial positions throughout
        fixed_mask = ~movable_t  # [n_macros]
        # Half-size tensors for canvas clamping
        half_w_t = sizes_all[:, 0] / 2.0
        half_h_t = sizes_all[:, 1] / 2.0

        # ── 3. Optimization ──
        pop.requires_grad_(True)
        opt = torch.optim.Adam([pop], lr=0.1)

        best_pos_full = seeds[0].copy()
        best_cost = float("inf")

        plc = _load_plc(benchmark.name)
        # Anchor for the anchor-preservation regularizer.  We split the
        # population: ~half receive strong anchor pull (these refine *near*
        # initial.plc and are likely the safest bets), the other half are
        # free to explore.  This way the population covers both "local
        # refinement of the strong anchor" and "find a new basin."
        anchor_pos_t = torch.tensor(seeds[0], device=device, dtype=torch.float32)
        anchor_mask = torch.zeros(K, n_macros, device=device, dtype=torch.float32)
        n_anchored = max(2, K // 2)
        for k in range(n_anchored):
            anchor_mask[k] = 1.0 if k == 0 else 0.5

        # TILOS-faithful grid + routing parameters
        grid_col = int(benchmark.grid_cols)
        grid_row = int(benchmark.grid_rows)
        h_per_um = float(benchmark.hroutes_per_micron)
        v_per_um = float(benchmark.vroutes_per_micron)
        smooth_rng = 2  # TILOS default for the IBM benchmarks
        self._log(
            f"TILOS surrogate: grid {grid_col}x{grid_row}  H/V tracks {h_per_um:.2f}/{v_per_um:.2f}  smooth={smooth_rng}"
        )

        for epoch in range(self.n_epochs):
            if time.time() - t_start > self.time_budget_s * 0.85:
                self._log(f"epoch {epoch}: budget exhausted, breaking")
                break
            progress = epoch / max(1, self.n_epochs - 1)
            gamma = 2.0 * (0.05) ** progress             # 2.0 → 0.1  (WAHPWL smoothing)
            alpha_ov = 1.0 * (5000.0) ** progress         # 1 → 5000 (overlap pressure ramps hard)
            alpha_anchor = 30.0 * (0.05) ** progress      # 30 → 1.5 (slow release)
            lr = 0.05 * (0.02) ** progress                # 0.05 → 0.001
            for pg in opt.param_groups:
                pg["lr"] = lr

            self._log(
                f"epoch {epoch+1}/{self.n_epochs}: γ={gamma:.3f} α_ov={alpha_ov:.1f} α_anc={alpha_anchor:.2f} lr={lr:.4f}"
            )

            for step in range(self.steps_per_epoch):
                opt.zero_grad()
                # The TILOS-faithful proxy surrogate:
                #   proxy_surrogate = wl_norm + 0.5 * tilos_density + 0.5 * tilos_rudy
                # WL is normalized exactly like get_cost() (raw_HPWL / ((cw+ch)*n_nets)).
                # Density is the exact TILOS top-10% mean × 0.5.
                # Congestion is the differentiable RUDY approximation with TILOS's
                # H/V track normalization and 1-D smoothing kernel.
                wl_n = tilos_wl_normalized(
                    pop, owner_idx, pin_off, net_id, n_nets, port_pos,
                    n_hard, n_macros, gamma, cw, ch,
                )
                dens = tilos_density_loss(pop, sizes_all, grid_col, grid_row, cw, ch)
                cong = tilos_rudy_normalized(
                    pop, owner_idx, pin_off, net_id, n_nets, port_pos,
                    n_hard, n_macros, grid_col, grid_row, cw, ch,
                    h_per_um, v_per_um, smooth_range=smooth_rng, k_frac=0.05,
                )
                ov = overlap_loss_hard(pop[:, :n_hard], sizes_hard)
                anch = anchor_reg(pop, anchor_pos_t, anchor_mask)
                # proxy surrogate (mirrors compute_proxy_cost weights exactly)
                proxy_surr = wl_n + 0.5 * dens + 0.5 * cong
                loss = (proxy_surr + alpha_ov * ov + alpha_anchor * anch).sum()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    pop[..., 0].clamp_(min=half_w_t.unsqueeze(0), max=(cw - half_w_t).unsqueeze(0))
                    pop[..., 1].clamp_(min=half_h_t.unsqueeze(0), max=(ch - half_h_t).unsqueeze(0))
                    if fixed_mask.any():
                        pop[:, fixed_mask] = init_full_t[fixed_mask]

            # Mid-epoch legalization (only halfway through training, on top-K candidates
            # by surrogate cost — cheap because legalization scales with N²).
            if epoch == self.n_epochs // 2 - 1 and epoch < self.n_epochs - 1:
                with torch.no_grad():
                    surr_eval = (
                        wahpwl(pop, owner_idx, pin_off, net_id, n_nets, port_pos,
                               n_hard, n_macros, max(gamma, 0.2))
                        + 100.0 * overlap_loss_hard(pop[:, :n_hard], sizes_hard)
                    )
                    k_proj = min(K, max(4, K // 4))
                    top_idx = torch.topk(-surr_eval, k=k_proj).indices.tolist()
                pop_cpu = pop.detach().cpu().numpy()
                for k in top_idx:
                    hard_k = pop_cpu[k, :n_hard].astype(np.float64)
                    hard_leg = _legalize(hard_k, sizes_np, movable_np, cw, ch, gap=0.005)
                    pop_cpu[k, :n_hard] = hard_leg.astype(np.float32)
                pop.data = torch.tensor(pop_cpu, device=device, dtype=torch.float32)
                self._log(f"  mid-epoch legalize on top {k_proj} candidates")

            # True-cost gate at the end of each epoch on the surrogate-best (cheap).
            if plc is not None and time.time() - t_start < self.time_budget_s * 0.85:
                with torch.no_grad():
                    wl_eval = wahpwl(
                        pop, owner_idx, pin_off, net_id, n_nets, port_pos,
                        n_hard, n_macros, max(gamma, 0.1),
                    )
                    ov_eval = overlap_loss_hard(pop[:, :n_hard], sizes_hard)
                    surr = wl_eval + 100.0 * ov_eval
                k_best = int(torch.argmin(surr).item())
                pos_full = pop[k_best].detach().cpu().numpy().astype(np.float64)
                pos_full[:n_hard] = _legalize(
                    pos_full[:n_hard], sizes_np, movable_np, cw, ch, gap=0.005
                )
                pos_full = self._clip_soft_to_canvas(pos_full, benchmark, n_hard, cw, ch)
                full_t = torch.from_numpy(pos_full).float()
                tc = self._true_cost(plc, full_t, benchmark)
                self._log(f"  epoch-end true_proxy={tc:.4f}  best_so_far={best_cost:.4f}")
                if tc < best_cost:
                    best_cost = tc
                    best_pos_full = pos_full

        # ── 4. Final sweep: rank by surrogate, then legalize + true-cost on top-K' ──
        if plc is not None:
            with torch.no_grad():
                wl_eval = wahpwl(
                    pop, owner_idx, pin_off, net_id, n_nets, port_pos,
                    n_hard, n_macros, 0.1,
                )
                ov_eval = overlap_loss_hard(pop[:, :n_hard], sizes_hard)
                surr = wl_eval + 100.0 * ov_eval
            k_final = min(K, max(8, K // 2))
            top_idx = torch.topk(-surr, k=k_final).indices.tolist()
            self._log(f"final sweep over top {k_final} of {K} candidates")
            for k in top_idx:
                if time.time() - t_start > self.time_budget_s:
                    self._log("budget out; stopping final sweep")
                    break
                pos_full = pop[k].detach().cpu().numpy().astype(np.float64)
                pos_full[:n_hard] = _legalize(
                    pos_full[:n_hard], sizes_np, movable_np, cw, ch, gap=0.005
                )
                pos_full = self._clip_soft_to_canvas(pos_full, benchmark, n_hard, cw, ch)
                full_t = torch.from_numpy(pos_full).float()
                tc = self._true_cost(plc, full_t, benchmark)
                if tc < best_cost:
                    best_cost = tc
                    best_pos_full = pos_full
                    self._log(f"  new best: candidate {k} true_proxy={tc:.4f}")
        else:
            # No plc: pick by surrogate
            with torch.no_grad():
                wl_eval = wahpwl(pop, owner_idx, pin_off, net_id, n_nets, port_pos, n_hard, n_macros, 0.1)
                ov_eval = overlap_loss_hard(pop[:, :n_hard], sizes_hard)
                surr = wl_eval + 1000.0 * ov_eval
                k_best = int(torch.argmin(surr).item())
                best_pos_full = pop[k_best].detach().cpu().numpy().astype(np.float64)
                best_pos_full[:n_hard] = _legalize(
                    best_pos_full[:n_hard], sizes_np, movable_np, cw, ch, gap=0.005
                )
                best_pos_full = self._clip_soft_to_canvas(best_pos_full, benchmark, n_hard, cw, ch)

        # ── 5. Compare against the anchor (safety net) ──
        anchor_pos = np.zeros((n_macros, 2), dtype=np.float64)
        anchor_pos[:n_hard] = _legalize(
            benchmark.macro_positions[:n_hard].numpy().astype(np.float64),
            sizes_np, movable_np, cw, ch, gap=0.005,
        )
        anchor_pos[n_hard:] = benchmark.macro_positions[n_hard:].numpy().astype(np.float64)
        anchor_pos = self._clip_soft_to_canvas(anchor_pos, benchmark, n_hard, cw, ch)
        if plc is not None:
            anchor_cost_true = self._true_cost(plc, torch.from_numpy(anchor_pos).float(), benchmark)
            self._log(f"anchor true_proxy={anchor_cost_true:.4f}   best_grad={best_cost:.4f}")
            if anchor_cost_true < best_cost:
                best_pos_full = anchor_pos
                best_cost = anchor_cost_true

        # ── 6. Assemble final tensor ──
        full = torch.from_numpy(best_pos_full).float()
        return full

    @staticmethod
    def _clip_soft_to_canvas(
        pos_full: np.ndarray, benchmark: Benchmark, n_hard: int, cw: float, ch: float
    ) -> np.ndarray:
        """Clip soft macros inside the canvas (initial.plc occasionally puts a
        few of them slightly outside)."""
        all_sizes = benchmark.macro_sizes.numpy()
        margin = 1e-3
        for i in range(n_hard, benchmark.num_macros):
            hw = float(all_sizes[i, 0]) / 2.0
            hh = float(all_sizes[i, 1]) / 2.0
            pos_full[i, 0] = max(hw + margin, min(cw - hw - margin, pos_full[i, 0]))
            pos_full[i, 1] = max(hh + margin, min(ch - hh - margin, pos_full[i, 1]))
        return pos_full
# ────────────────────────────────────────────────────────────────────────────
# Phase 1 — FastEvaluator (bit-exact mirror of PlacementCost)
# ────────────────────────────────────────────────────────────────────────────


class FastEvaluator:
    """NumPy reimplementation of PlacementCost.get_cost / get_density_cost /
    get_congestion_cost with incremental update support.

    Validated bit-exact against PlacementCost on ibm01 (and others); a single
    move_macro() call is ~2 ms (vs ~4000 ms for the oracle).
    """

    def __init__(self, benchmark: Benchmark, plc):
        self.benchmark = benchmark
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.grid_col = int(benchmark.grid_cols)
        self.grid_row = int(benchmark.grid_rows)
        self.gw = self.cw / self.grid_col
        self.gh = self.ch / self.grid_row
        self.grid_area = self.gw * self.gh
        self.h_per_um = float(benchmark.hroutes_per_micron)
        self.v_per_um = float(benchmark.vroutes_per_micron)
        self.grid_v_routes = self.gw * self.v_per_um
        self.grid_h_routes = self.gh * self.h_per_um
        # Routing allocation + smoothing range come from PlacementCost.
        self.h_alloc = 0.0
        self.v_alloc = 0.0
        self.smooth_range = 2
        if plc is not None:
            try:
                self.h_alloc, self.v_alloc = plc.get_macro_routing_allocation()
            except Exception:
                self.h_alloc = getattr(plc, "hrouting_alloc", 0.0)
                self.v_alloc = getattr(plc, "vrouting_alloc", 0.0)
            try:
                self.smooth_range = int(plc.get_congestion_smooth_range())
            except Exception:
                self.smooth_range = int(getattr(plc, "smooth_range", 2))
        self.n_hard = benchmark.num_hard_macros
        self.n_macros = benchmark.num_macros
        self.n_soft = self.n_macros - self.n_hard
        self.n_nets = int(benchmark.num_nets)
        self.n_ports = int(benchmark.port_positions.shape[0])
        # WL normalization uses plc.net_cnt (counts every driver pin, not just nets with sinks)
        self.wl_norm_n_nets = int(getattr(plc, "net_cnt", self.n_nets)) if plc is not None else self.n_nets
        if self.wl_norm_n_nets <= 0:
            self.wl_norm_n_nets = max(self.n_nets, 1)
        # State arrays
        self.positions = benchmark.macro_positions.detach().cpu().numpy().astype(np.float64)
        self.sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)
        self.half = self.sizes / 2.0
        self.port_pos = benchmark.port_positions.detach().cpu().numpy().astype(np.float64) if self.n_ports else np.zeros((0, 2))
        self.movable = benchmark.get_movable_mask().detach().cpu().numpy().astype(bool)
        # Per-net tables
        self._build_net_pin_tables(benchmark)
        self._net_xmin = np.zeros(self.n_nets, dtype=np.float64)
        self._net_ymin = np.zeros(self.n_nets, dtype=np.float64)
        self._net_xmax = np.zeros(self.n_nets, dtype=np.float64)
        self._net_ymax = np.zeros(self.n_nets, dtype=np.float64)
        self._net_weight = np.ones(self.n_nets, dtype=np.float64)
        if plc is not None:
            self._fetch_net_weights(plc)
        self._owner_to_nets: Dict[int, List[int]] = {}
        for n in range(self.n_nets):
            for o in self.net_owner[n]:
                self._owner_to_nets.setdefault(int(o), []).append(n)
        # Grids
        self.density_grid = np.zeros((self.grid_row, self.grid_col), dtype=np.float64)
        self.h_pin_cong = np.zeros((self.grid_row, self.grid_col), dtype=np.float64)
        self.v_pin_cong = np.zeros((self.grid_row, self.grid_col), dtype=np.float64)
        self.h_macro_cong = np.zeros((self.grid_row, self.grid_col), dtype=np.float64)
        self.v_macro_cong = np.zeros((self.grid_row, self.grid_col), dtype=np.float64)
        self._init_caches()

    def _build_net_pin_tables(self, benchmark: Benchmark):
        pin_offsets = benchmark.macro_pin_offsets
        npn = benchmark.net_pin_nodes
        self.net_owner: List[np.ndarray] = []
        self.net_offx: List[np.ndarray] = []
        self.net_offy: List[np.ndarray] = []
        if not npn:
            for n in range(self.n_nets):
                nodes = benchmark.net_nodes[n].cpu().numpy().astype(np.int64) if benchmark.net_nodes else np.zeros(0, dtype=np.int64)
                self.net_owner.append(nodes)
                self.net_offx.append(np.zeros(nodes.shape[0]))
                self.net_offy.append(np.zeros(nodes.shape[0]))
            return
        for n in range(self.n_nets):
            pn = npn[n].cpu().numpy().astype(np.int64)
            if pn.size == 0:
                self.net_owner.append(np.zeros(0, dtype=np.int64))
                self.net_offx.append(np.zeros(0))
                self.net_offy.append(np.zeros(0))
                continue
            owners = pn[:, 0]
            slots = pn[:, 1]
            offx = np.zeros(owners.shape[0])
            offy = np.zeros(owners.shape[0])
            for k in range(owners.shape[0]):
                o, s = int(owners[k]), int(slots[k])
                if o < self.n_hard and pin_offsets and o < len(pin_offsets):
                    po = pin_offsets[o]
                    if po is not None and po.shape[0] > s:
                        offx[k] = float(po[s, 0])
                        offy[k] = float(po[s, 1])
            self.net_owner.append(owners)
            self.net_offx.append(offx)
            self.net_offy.append(offy)

    def _fetch_net_weights(self, plc):
        try:
            driver_names = list(plc.nets.keys())
            for n in range(min(self.n_nets, len(driver_names))):
                pi = plc.mod_name_to_indices[driver_names[n]]
                self._net_weight[n] = float(plc.modules_w_pins[pi].get_weight())
        except Exception:
            pass

    def _pin_x(self, owners, offx):
        out = np.empty(owners.shape[0], dtype=np.float64)
        m = owners < self.n_macros
        out[m] = self.positions[owners[m], 0] + offx[m]
        if (~m).any():
            p_idx = owners[~m] - self.n_macros
            out[~m] = self.port_pos[p_idx, 0] + offx[~m]
        return out

    def _pin_y(self, owners, offy):
        out = np.empty(owners.shape[0], dtype=np.float64)
        m = owners < self.n_macros
        out[m] = self.positions[owners[m], 1] + offy[m]
        if (~m).any():
            p_idx = owners[~m] - self.n_macros
            out[~m] = self.port_pos[p_idx, 1] + offy[~m]
        return out

    def _net_bbox(self, n):
        owners = self.net_owner[n]
        if owners.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        xs = self._pin_x(owners, self.net_offx[n])
        ys = self._pin_y(owners, self.net_offy[n])
        return xs.min(), ys.min(), xs.max(), ys.max()

    def _grid_cell(self, x, y):
        c = int(math.floor(x / self.gw))
        r = int(math.floor(y / self.gh))
        return max(0, min(self.grid_row - 1, r)), max(0, min(self.grid_col - 1, c))

    def _add_macro_density(self, macro_idx, sign=+1):
        x, y = self.positions[macro_idx]
        w, h = self.sizes[macro_idx]
        x_min, x_max = x - w / 2, x + w / 2
        y_min, y_max = y - h / 2, y + h / 2
        ur_r, ur_c = self._grid_cell(x_max, y_max)
        bl_r, bl_c = self._grid_cell(x_min, y_min)
        for r in range(bl_r, ur_r + 1):
            gy0 = r * self.gh
            gy1 = (r + 1) * self.gh
            dy = min(y_max, gy1) - max(y_min, gy0)
            if dy <= 0:
                continue
            for c in range(bl_c, ur_c + 1):
                gx0 = c * self.gw
                gx1 = (c + 1) * self.gw
                dx = min(x_max, gx1) - max(x_min, gx0)
                if dx <= 0:
                    continue
                self.density_grid[r, c] += sign * dx * dy

    def _add_macro_route(self, macro_idx, sign=+1):
        x, y = self.positions[macro_idx]
        w, h = self.sizes[macro_idx]
        x_min, x_max = x - w / 2, x + w / 2
        y_min, y_max = y - h / 2, y + h / 2
        ur_r, ur_c = self._grid_cell(x_max, y_max)
        bl_r, bl_c = self._grid_cell(x_min, y_min)
        partial_v = False
        partial_h = False
        eps = 1e-5
        for r in range(bl_r, ur_r + 1):
            gy0 = r * self.gh
            gy1 = (r + 1) * self.gh
            dy = min(y_max, gy1) - max(y_min, gy0)
            if dy <= 0:
                continue
            for c in range(bl_c, ur_c + 1):
                gx0 = c * self.gw
                gx1 = (c + 1) * self.gw
                dx = min(x_max, gx1) - max(x_min, gx0)
                if dx <= 0:
                    continue
                self.v_macro_cong[r, c] += sign * dx * self.v_alloc
                self.h_macro_cong[r, c] += sign * dy * self.h_alloc
                if ur_r != bl_r and (r == bl_r or r == ur_r) and abs(dy - self.gh) > eps:
                    partial_v = True
                if ur_c != bl_c and (c == bl_c or c == ur_c) and abs(dx - self.gw) > eps:
                    partial_h = True
        if partial_v:
            r = ur_r
            for c in range(bl_c, ur_c + 1):
                gx0, gx1 = c * self.gw, (c + 1) * self.gw
                dx = min(x_max, gx1) - max(x_min, gx0)
                if dx > 0:
                    self.v_macro_cong[r, c] -= sign * dx * self.v_alloc
        if partial_h:
            c = ur_c
            for r in range(bl_r, ur_r + 1):
                gy0, gy1 = r * self.gh, (r + 1) * self.gh
                dy = min(y_max, gy1) - max(y_min, gy0)
                if dy > 0:
                    self.h_macro_cong[r, c] -= sign * dy * self.h_alloc

    def _route_pin_cong(self, net_idx, sign=+1):
        owners = self.net_owner[net_idx]
        if owners.size == 0:
            return
        xs = self._pin_x(owners, self.net_offx[net_idx])
        ys = self._pin_y(owners, self.net_offy[net_idx])
        cells = []
        cells_set = set()
        for i in range(owners.shape[0]):
            r, c = self._grid_cell(xs[i], ys[i])
            cells.append((r, c))
            cells_set.add((r, c))
        if len(cells_set) <= 1:
            return
        src = cells[0]
        w = self._net_weight[net_idx]
        if len(cells_set) == 2:
            self._two_pin(src, list(cells_set), w, sign)
        elif len(cells_set) == 3:
            self._three_pin(list(cells_set), w, sign)
        else:
            for n in cells_set:
                if n == src:
                    continue
                self._two_pin(src, [src, n], w, sign)

    def _two_pin(self, src, two, w, sign):
        sink = two[1] if two[0] == src else two[0]
        r_min, r_max = min(src[0], sink[0]), max(src[0], sink[0])
        c_min, c_max = min(src[1], sink[1]), max(src[1], sink[1])
        if c_max > c_min:
            self.h_pin_cong[src[0], c_min:c_max] += sign * w
        if r_max > r_min:
            self.v_pin_cong[r_min:r_max, sink[1]] += sign * w

    def _three_pin(self, cells, w, sign):
        cs = sorted(cells, key=lambda x: (x[1], x[0]))
        (y1, x1), (y2, x2), (y3, x3) = cs
        if x1 < x2 < x3 and min(y1, y3) < y2 and max(y1, y3) > y2:
            self._l(cs, w, sign)
        elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
            if x2 > x1:
                self.h_pin_cong[y1, x1:x2] += sign * w
            r_lo, r_hi = y1, max(y2, y3)
            if r_hi > r_lo:
                self.v_pin_cong[r_lo:r_hi, x2] += sign * w
        elif y2 == y3:
            if x2 > x1:
                self.h_pin_cong[y1, x1:x2] += sign * w
            if x3 > x2:
                self.h_pin_cong[y2, x2:x3] += sign * w
            r_lo, r_hi = min(y1, y2), max(y1, y2)
            if r_hi > r_lo:
                self.v_pin_cong[r_lo:r_hi, x2] += sign * w
        else:
            self._t(cs, w, sign)

    def _l(self, cs, w, sign):
        (y1, x1), (y2, x2), (y3, x3) = cs
        if x2 > x1:
            self.h_pin_cong[y1, x1:x2] += sign * w
        if x3 > x2:
            self.h_pin_cong[y2, x2:x3] += sign * w
        r_lo, r_hi = min(y1, y2), max(y1, y2)
        if r_hi > r_lo:
            self.v_pin_cong[r_lo:r_hi, x2] += sign * w
        r_lo, r_hi = min(y2, y3), max(y2, y3)
        if r_hi > r_lo:
            self.v_pin_cong[r_lo:r_hi, x3] += sign * w

    def _t(self, cs, w, sign):
        cs2 = sorted(cs)
        (y1, x1), (y2, x2), (y3, x3) = cs2
        xmin = min(x1, x2, x3)
        xmax = max(x1, x2, x3)
        if xmax > xmin:
            self.h_pin_cong[y2, xmin:xmax] += sign * w
        r_lo, r_hi = min(y1, y2), max(y1, y2)
        if r_hi > r_lo:
            self.v_pin_cong[r_lo:r_hi, x1] += sign * w
        r_lo, r_hi = min(y2, y3), max(y2, y3)
        if r_hi > r_lo:
            self.v_pin_cong[r_lo:r_hi, x3] += sign * w

    def _init_caches(self):
        self.density_grid[...] = 0
        self.h_pin_cong[...] = 0
        self.v_pin_cong[...] = 0
        self.h_macro_cong[...] = 0
        self.v_macro_cong[...] = 0
        for m in range(self.n_macros):
            self._add_macro_density(m, +1)
        for m in range(self.n_hard):
            self._add_macro_route(m, +1)
        for n in range(self.n_nets):
            x0, y0, x1, y1 = self._net_bbox(n)
            self._net_xmin[n] = x0
            self._net_ymin[n] = y0
            self._net_xmax[n] = x1
            self._net_ymax[n] = y1
            self._route_pin_cong(n, +1)

    def _density_cost(self):
        gc = (self.density_grid / self.grid_area).ravel()
        nz = gc[gc > 0]
        if nz.size == 0:
            return 0.0
        N = gc.size
        if N < 10:
            return 0.5 * float(nz.mean())
        cnt = math.floor(N * 0.1)
        if cnt == 0:
            return 0.5 * float(nz.max())
        sd = np.sort(nz)[::-1]
        take = min(cnt, sd.size)
        return 0.5 * float(sd[:take].sum() / cnt)

    def _smooth(self, grid, axis):
        sr = self.smooth_range
        R, C = grid.shape
        if axis == 0:
            cols = np.arange(C)
            lp = np.maximum(0, cols - sr)
            rp = np.minimum(C - 1, cols + sr)
            cnt = (rp - lp + 1).astype(np.float64)
            scaled = grid / cnt[np.newaxis, :]
            pad = np.pad(scaled, ((0, 0), (sr, sr)), mode="constant")
            cs = np.cumsum(pad, axis=1)
            cs0 = cs[:, 2 * sr:]
            cs1 = np.concatenate([np.zeros((R, 1)), cs[:, :C - 1 + 2 * sr]], axis=1)[:, :C]
            return cs0[:, :C] - cs1
        else:
            rows = np.arange(R)
            lp = np.maximum(0, rows - sr)
            up = np.minimum(R - 1, rows + sr)
            cnt = (up - lp + 1).astype(np.float64)
            scaled = grid / cnt[:, np.newaxis]
            pad = np.pad(scaled, ((sr, sr), (0, 0)), mode="constant")
            cs = np.cumsum(pad, axis=0)
            cs0 = cs[2 * sr:, :]
            cs1 = np.concatenate([np.zeros((1, C)), cs[:R - 1 + 2 * sr, :]], axis=0)[:R, :]
            return cs0[:R, :] - cs1

    def _congestion_cost(self):
        v = self.v_pin_cong / self.grid_v_routes
        h = self.h_pin_cong / self.grid_h_routes
        vm = self.v_macro_cong / self.grid_v_routes
        hm = self.h_macro_cong / self.grid_h_routes
        v_s = self._smooth(v, axis=0)
        h_s = self._smooth(h, axis=1)
        combined = np.concatenate([(v_s + vm).ravel(), (h_s + hm).ravel()])
        xs = np.sort(combined)[::-1]
        cnt = math.floor(xs.size * 0.05)
        if cnt == 0:
            return float(xs.max()) if xs.size else 0.0
        return float(xs[:cnt].mean())

    def _wirelength_cost(self):
        hpwl = (self._net_xmax - self._net_xmin) + (self._net_ymax - self._net_ymin)
        return float(np.sum(hpwl * self._net_weight)) / ((self.cw + self.ch) * self.wl_norm_n_nets)

    def proxy_cost(self):
        wl = self._wirelength_cost()
        d = self._density_cost()
        c = self._congestion_cost()
        return {
            "proxy_cost": wl + 0.5 * d + 0.5 * c,
            "wirelength_cost": wl,
            "density_cost": d,
            "congestion_cost": c,
        }

    def move_macro(self, macro_idx, new_x, new_y, is_hard=True):
        if is_hard:
            self._add_macro_route(macro_idx, -1)
        self._add_macro_density(macro_idx, -1)
        nets = self._owner_to_nets.get(macro_idx, ())
        for n in nets:
            self._route_pin_cong(n, -1)
        self.positions[macro_idx, 0] = new_x
        self.positions[macro_idx, 1] = new_y
        self._add_macro_density(macro_idx, +1)
        if is_hard:
            self._add_macro_route(macro_idx, +1)
        for n in nets:
            x0, y0, x1, y1 = self._net_bbox(n)
            self._net_xmin[n] = x0
            self._net_ymin[n] = y0
            self._net_xmax[n] = x1
            self._net_ymax[n] = y1
            self._route_pin_cong(n, +1)

    def swap_macros(self, i, j):
        xi, yi = self.positions[i]
        xj, yj = self.positions[j]
        self.move_macro(i, xj, yj, is_hard=(i < self.n_hard))
        self.move_macro(j, xi, yi, is_hard=(j < self.n_hard))

    def snapshot(self):
        return self.positions.copy()

    def restore(self, positions):
        if np.array_equal(positions, self.positions):
            return
        self.positions[:] = positions
        self._init_caches()



def _swap_legal(ev: FastEvaluator, i: int, j: int) -> bool:
    pi = ev.positions[i].copy()
    pj = ev.positions[j].copy()
    hi = ev.half[i]
    hj = ev.half[j]
    if pj[0] - hi[0] < 0 or pj[0] + hi[0] > ev.cw:
        return False
    if pj[1] - hi[1] < 0 or pj[1] + hi[1] > ev.ch:
        return False
    if pi[0] - hj[0] < 0 or pi[0] + hj[0] > ev.cw:
        return False
    if pi[1] - hj[1] < 0 or pi[1] + hj[1] > ev.ch:
        return False
    for k in range(ev.n_hard):
        if k == i or k == j:
            continue
        pk = ev.positions[k]
        sk = ev.sizes[k]
        ox = (ev.sizes[i, 0] + sk[0]) / 2 - abs(pj[0] - pk[0])
        oy = (ev.sizes[i, 1] + sk[1]) / 2 - abs(pj[1] - pk[1])
        if ox > 0 and oy > 0:
            return False
        ox = (ev.sizes[j, 0] + sk[0]) / 2 - abs(pi[0] - pk[0])
        oy = (ev.sizes[j, 1] + sk[1]) / 2 - abs(pi[1] - pk[1])
        if ox > 0 and oy > 0:
            return False
    return True


def _slide_legal(ev: FastEvaluator, i: int, nx: float, ny: float) -> bool:
    half = ev.half[i]
    if nx - half[0] < 0 or nx + half[0] > ev.cw:
        return False
    if ny - half[1] < 0 or ny + half[1] > ev.ch:
        return False
    for k in range(ev.n_hard):
        if k == i:
            continue
        pk = ev.positions[k]
        sk = ev.sizes[k]
        ox = (ev.sizes[i, 0] + sk[0]) / 2 - abs(nx - pk[0])
        oy = (ev.sizes[i, 1] + sk[1]) / 2 - abs(ny - pk[1])
        if ox > 0 and oy > 0:
            return False
    return True
# ────────────────────────────────────────────────────────────────────────────
# Phase 2.5 — Direct congestion-attack (true grid)
# ────────────────────────────────────────────────────────────────────────────


def _smoothed_congestion_grid(ev: FastEvaluator) -> np.ndarray:
    """Compute the FastEvaluator's smoothed combined V+H congestion grid.

    This is the SAME math that produces the top-5% mean in `_congestion_cost`,
    exposed for the direct-attack phase that wants to know WHICH cells are hot.
    """
    v = ev.v_pin_cong / ev.grid_v_routes
    h = ev.h_pin_cong / ev.grid_h_routes
    vm = ev.v_macro_cong / ev.grid_v_routes
    hm = ev.h_macro_cong / ev.grid_h_routes
    v_s = ev._smooth(v, axis=0)
    h_s = ev._smooth(h, axis=1)
    return (v_s + vm) + (h_s + hm)
# ────────────────────────────────────────────────────────────────────────────
# Phase 3 — LAHC polish (centroid-biased soft + hard swap/slide)
# ────────────────────────────────────────────────────────────────────────────


def _soft_centroid_target(ev: FastEvaluator, soft_global_idx: int):
    nets = ev._owner_to_nets.get(soft_global_idx, ())
    if not nets:
        return None
    total_w = 0.0
    sum_x = 0.0
    sum_y = 0.0
    for n in nets:
        owners = ev.net_owner[n]
        if owners.size < 2:
            continue
        xs = ev._pin_x(owners, ev.net_offx[n])
        ys = ev._pin_y(owners, ev.net_offy[n])
        own_pos = np.where(owners == soft_global_idx)[0]
        if own_pos.size == 0:
            continue
        i = int(own_pos[0])
        k = owners.size
        px = (xs.sum() - xs[i]) / (k - 1)
        py = (ys.sum() - ys[i]) / (k - 1)
        w = ev._net_weight[n] / (k - 1)
        sum_x += w * px
        sum_y += w * py
        total_w += w
    if total_w <= 0:
        return None
    return float(sum_x / total_w), float(sum_y / total_w)


def lahc_polish(
    ev: FastEvaluator,
    list_len: int = 100,
    time_budget_s: float = 600.0,
    move_radius_frac: float = 0.06,
    soft_move_radius_frac: float = 0.03,
    soft_centroid_prob: float = 0.50,
    swap_prob: float = 0.30,
    soft_prob: float = 0.40,
    decongest_prob: float = 0.0,         # disabled: didn't outperform random LAHC + LK
    n_swap_neighbors: int = 12,
    n_decongest_top_cells: int = 16,
    decongest_refresh_every: int = 100,  # recompute hot-cell list every N iters
    seed: int = 0,
    verbose: bool = True,
    log_interval_s: float = 20.0,
):
    rng = np.random.default_rng(seed)
    cur_cost = ev.proxy_cost()["proxy_cost"]
    best_cost = cur_cost
    best_pos = ev.positions.copy()
    history = [cur_cost] * list_len
    t0 = time.time()
    last_log = t0
    it = 0
    # Hot-cell cache for decongest proposals
    hot_cells: List[Tuple[int, int, float]] = []   # (row, col, heat)
    hot_macros: List[int] = []                       # macros contributing to hot cells
    last_hot_refresh = -1
    while time.time() - t0 < time_budget_s:
        if verbose and time.time() - last_log >= log_interval_s:
            print(f"  [LAHC] t={time.time()-t0:.0f}s it={it} cur={cur_cost:.4f} best={best_cost:.4f}", flush=True)
            last_log = time.time()
        # Periodically refresh the hot-cell list and the macros contributing to them
        if it - last_hot_refresh >= decongest_refresh_every:
            cong = _smoothed_congestion_grid(ev)
            flat = cong.ravel()
            n_top = min(n_decongest_top_cells, max(1, int(flat.size * 0.05)))
            top_idx = np.argpartition(-flat, n_top - 1)[:n_top]
            hot_cells = [((int(idx) // ev.grid_col), (int(idx) % ev.grid_col), float(flat[idx])) for idx in top_idx]
            hot_macro_scores: Dict[int, float] = {}
            for net_idx in range(ev.n_nets):
                ymin_c, xmin_c = ev._grid_cell(ev._net_xmin[net_idx], ev._net_ymin[net_idx])
                ymax_c, xmax_c = ev._grid_cell(ev._net_xmax[net_idx], ev._net_ymax[net_idx])
                stress = 0.0
                for (r_h, c_h, h_val) in hot_cells:
                    if ymin_c <= r_h <= ymax_c and xmin_c <= c_h <= xmax_c:
                        stress += h_val
                if stress > 0:
                    for o in ev.net_owner[net_idx]:
                        o = int(o)
                        if o < ev.n_hard and ev.movable[o]:
                            hot_macro_scores[o] = hot_macro_scores.get(o, 0.0) + stress
            if hot_macro_scores:
                # Top-50 hottest hard macros
                hot_macros = [m for m, _ in sorted(hot_macro_scores.items(), key=lambda x: -x[1])[:50]]
            else:
                hot_macros = []
            last_hot_refresh = it
        r = rng.random()
        do_swap = r < swap_prob
        do_soft = (r >= swap_prob) and (r < swap_prob + soft_prob) and (ev.n_soft > 0)
        do_decongest = (
            (r >= swap_prob + soft_prob)
            and (r < swap_prob + soft_prob + decongest_prob)
            and (len(hot_macros) > 0)
        )
        if do_swap:
            i = int(rng.integers(0, ev.n_hard))
            if not ev.movable[i]:
                it += 1
                continue
            d = np.linalg.norm(ev.positions[:ev.n_hard] - ev.positions[i], axis=1)
            d[i] = np.inf
            cands = np.argsort(d)[:n_swap_neighbors]
            j = int(cands[int(rng.integers(0, cands.size))])
            if not ev.movable[j] or not _swap_legal(ev, i, j):
                it += 1
                continue
            ev.swap_macros(i, j)
            cand = ev.proxy_cost()["proxy_cost"]
            idx_h = it % list_len
            if cand < cur_cost or cand < history[idx_h]:
                cur_cost = cand
                history[idx_h] = cand
                if cand < best_cost:
                    best_cost = cand
                    best_pos = ev.positions.copy()
            else:
                ev.swap_macros(i, j)
        elif do_soft:
            i_soft = int(rng.integers(0, ev.n_soft))
            i = ev.n_hard + i_soft
            if not ev.movable[i]:
                it += 1
                continue
            ox, oy = ev.positions[i]
            use_cent = rng.random() < soft_centroid_prob
            if use_cent:
                tgt = _soft_centroid_target(ev, i)
                if tgt is None:
                    it += 1
                    continue
                tx, ty = tgt
                f = float(rng.uniform(0.05, 0.5))
                nx = ox + f * (tx - ox)
                ny = oy + f * (ty - oy)
            else:
                rx = soft_move_radius_frac * ev.cw
                ry = soft_move_radius_frac * ev.ch
                nx = ox + float(rng.uniform(-rx, rx))
                ny = oy + float(rng.uniform(-ry, ry))
            nx = max(ev.half[i, 0], min(ev.cw - ev.half[i, 0], nx))
            ny = max(ev.half[i, 1], min(ev.ch - ev.half[i, 1], ny))
            ev.move_macro(i, nx, ny, is_hard=False)
            cand = ev.proxy_cost()["proxy_cost"]
            idx_h = it % list_len
            if cand < cur_cost or cand < history[idx_h]:
                cur_cost = cand
                history[idx_h] = cand
                if cand < best_cost:
                    best_cost = cand
                    best_pos = ev.positions.copy()
            else:
                ev.move_macro(i, ox, oy, is_hard=False)
        elif do_decongest:
            # Pick a hard macro contributing to a hot cell; propose moving it
            # AWAY from the hot cell centroid (with small random jitter so LAHC
            # can explore around the bias direction).
            i = int(rng.choice(hot_macros))
            if not ev.movable[i]:
                it += 1
                continue
            # Hot centroid (heat-weighted)
            heat_sum = sum(h for _, _, h in hot_cells)
            if heat_sum <= 0:
                it += 1
                continue
            hot_cx = sum((c + 0.5) * ev.gw * h for _, c, h in hot_cells) / heat_sum
            hot_cy = sum((r + 0.5) * ev.gh * h for r, _, h in hot_cells) / heat_sum
            ox, oy = ev.positions[i]
            # Direction AWAY from hot centroid
            dx_dir = ox - hot_cx
            dy_dir = oy - hot_cy
            norm = math.sqrt(dx_dir * dx_dir + dy_dir * dy_dir) + 1e-9
            dx_dir /= norm
            dy_dir /= norm
            step = move_radius_frac * 0.5 * (ev.cw + ev.ch) * float(rng.uniform(0.3, 1.0))
            # Add some lateral noise so we don't always move along the same line
            jitter_x = float(rng.uniform(-0.3, 0.3)) * step
            jitter_y = float(rng.uniform(-0.3, 0.3)) * step
            nx = ox + dx_dir * step + jitter_x
            ny = oy + dy_dir * step + jitter_y
            nx = max(ev.half[i, 0], min(ev.cw - ev.half[i, 0], nx))
            ny = max(ev.half[i, 1], min(ev.ch - ev.half[i, 1], ny))
            if not _slide_legal(ev, i, nx, ny):
                it += 1
                continue
            ev.move_macro(i, nx, ny, is_hard=True)
            cand = ev.proxy_cost()["proxy_cost"]
            idx_h = it % list_len
            if cand < cur_cost or cand < history[idx_h]:
                cur_cost = cand
                history[idx_h] = cand
                if cand < best_cost:
                    best_cost = cand
                    best_pos = ev.positions.copy()
            else:
                ev.move_macro(i, ox, oy, is_hard=True)
        else:
            i = int(rng.integers(0, ev.n_hard))
            if not ev.movable[i]:
                it += 1
                continue
            rx = move_radius_frac * ev.cw
            ry = move_radius_frac * ev.ch
            ox, oy = ev.positions[i]
            nx = max(ev.half[i, 0], min(ev.cw - ev.half[i, 0], ox + float(rng.uniform(-rx, rx))))
            ny = max(ev.half[i, 1], min(ev.ch - ev.half[i, 1], oy + float(rng.uniform(-ry, ry))))
            if not _slide_legal(ev, i, nx, ny):
                it += 1
                continue
            ev.move_macro(i, nx, ny, is_hard=True)
            cand = ev.proxy_cost()["proxy_cost"]
            idx_h = it % list_len
            if cand < cur_cost or cand < history[idx_h]:
                cur_cost = cand
                history[idx_h] = cand
                if cand < best_cost:
                    best_cost = cand
                    best_pos = ev.positions.copy()
            else:
                ev.move_macro(i, ox, oy, is_hard=True)
        it += 1
    if not np.array_equal(ev.positions, best_pos):
        ev.restore(best_pos)
    return {"proxy_cost": best_cost, "iters": it}
