"""
Graph-Gradient Placer — ePlace variant (F)
==========================================

Same A+C1+E+D pipeline as placer.py, but with TILOS top-10% density swapped
out for **ePlace-style electrostatic density**: each macro is treated as a
2-D charge distribution, the Poisson equation nabla^2 phi = -rho is solved
via FFT, and the loss is the electrostatic potential energy U = (1/2) sum(rho * phi).

The autograd gradient of U w.r.t. macro positions naturally produces smooth
global "repulsive" forces that uniformly spread macros across the canvas —
no top-K artifacts, no hot-spot focus, no separate uniform-density term
needed.  This is the core mechanism behind DREAMPlace, Xplace, ePlace2,
RePlAce (the analytical placers that dominate the leaderboard).

Differences from placer.py:
  - alpha_dens=0     (TILOS top-10% density disabled in loss; kept as metric)
  - alpha_eplace=0.5 (ePlace electrostatic density active)
  - alpha_unif=0     (D disabled — F's spreading force covers its role)

If F doesn't help, set alpha_eplace=0 and alpha_dens=0.5 to recover the
original A+C1+E+D pipeline.

Original docstring follows.

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
from typing import List, Tuple

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


def _grid_seeded_hard(
    sizes: np.ndarray,  # [n_hard, 2]
    cw: float,
    ch: float,
    seed: int = 0,
    jitter_frac: float = 0.08,
) -> np.ndarray:
    """
    Distribute hard macros uniformly across the canvas on a size-aware grid.

    Anti-ring starting basin: large macros take the central cells (room to
    breathe), smaller macros take the peripheral cells.  Subsequent gradient
    descent + legalization then refine within this basin.  Differs from
    _grid_fill in three ways: aspect-ratio-aware cols/rows, area-sorted macro
    assignment to center-distance-sorted cells, and RNG jitter so multiple
    grid seeds are not duplicates.
    """
    rng = np.random.default_rng(seed)
    n_hard = sizes.shape[0]
    if n_hard == 0:
        return np.zeros((0, 2), dtype=np.float64)
    # Aspect-aware grid dimensions
    aspect = cw / max(ch, 1e-6)
    cols = max(1, int(math.ceil(math.sqrt(n_hard * aspect))))
    rows = max(1, int(math.ceil(n_hard / cols)))
    cell_w = cw / cols
    cell_h = ch / rows

    # Enumerate cell centers
    cell_positions = []
    for r in range(rows):
        for c in range(cols):
            cell_positions.append(((c + 0.5) * cell_w, (r + 0.5) * cell_h))
    cell_positions = np.array(cell_positions[: rows * cols], dtype=np.float64)
    # Trim to n_hard, ordered by distance from canvas centre (central cells first)
    canvas_centre = np.array([cw / 2.0, ch / 2.0])
    cell_d = np.linalg.norm(cell_positions - canvas_centre, axis=1)
    cell_order = np.argsort(cell_d)[:n_hard]
    cell_positions = cell_positions[cell_order]

    # Assign macros to cells: largest-area macros to closest-to-centre cells
    areas = sizes[:, 0] * sizes[:, 1]
    macro_order = np.argsort(-areas)
    out = np.zeros((n_hard, 2), dtype=np.float64)
    for slot, m_idx in enumerate(macro_order):
        out[m_idx] = cell_positions[slot]

    # Jitter (per-seed diversity); scale by cell size
    out += rng.standard_normal((n_hard, 2)) * np.array([cell_w, cell_h]) * jitter_frac

    # Clamp inside canvas with macro-half margin
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    margin = 0.005
    out[:, 0] = np.clip(out[:, 0], half_w + margin, cw - half_w - margin)
    out[:, 1] = np.clip(out[:, 1], half_h + margin, ch - half_h - margin)
    return out


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
    preserve_centroid: bool = True,
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
    # C1: snapshot movable centroid AFTER initial clamp; we will pull the
    # movable cloud back toward this reference each iteration to counter the
    # outward bias from canvas-wall pushes (root cause of the ring pattern).
    if movable.any():
        ref_centroid = out[movable].mean(axis=0)
    else:
        ref_centroid = np.zeros(2)
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
        # C1: re-center movable cloud BEFORE clamping so corrections aren't
        # eaten by the wall. Damp by 0.5 to avoid oscillation against the push.
        if preserve_centroid and movable.any():
            new_centroid = out[movable].mean(axis=0)
            drift = new_centroid - ref_centroid
            out[movable] -= 0.5 * drift
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


def eplace_density_loss(
    pop: torch.Tensor,
    sizes: torch.Tensor,
    grid_col: int,
    grid_row: int,
    cw: float,
    ch: float,
) -> torch.Tensor:
    """
    F: ePlace electrostatic-density loss.

    Treats each macro as a 2-D charge distribution (bilinear-spread onto the
    grid), subtracts the uniform target so net charge is zero (charge
    neutrality, required for solvability of the Poisson equation), then
    solves nabla^2 phi = -rho via FFT.  The loss is the electrostatic
    potential energy U = (1/2) sum_grid(rho * phi) — minimised when the
    density field is uniform.

    The autograd gradient of U w.r.t. macro positions naturally produces
    smooth global "repulsive" forces that spread macros across the entire
    canvas in one shot — no top-K artifacts, no hot-spot focus, no need for
    a separate uniform-density term.  This is the core mechanism behind
    DREAMPlace, Xplace, ePlace2, RePlAce — the analytical placers that
    dominate the leaderboard.

    Boundary conditions: 2x zero-padded FFT approximates free (open) BC,
    matching how DREAMPlace handles canvas boundaries.

    Returns [K].
    """
    K, N, _ = pop.shape
    device = pop.device
    grid_w = cw / grid_col
    grid_h = ch / grid_row
    grid_area = grid_w * grid_h

    # ── 1. Bilinear-spread macros onto density grid (differentiable) ──
    col_idx = torch.arange(grid_col, device=device, dtype=pop.dtype)
    row_idx = torch.arange(grid_row, device=device, dtype=pop.dtype)
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
    occupied = torch.einsum("knx,kny->kyx", x_ov, y_ov)  # [K, Gy, Gx]  area units
    rho = occupied / grid_area  # [K, Gy, Gx]  in [0, ~1]

    # ── 2. Charge neutrality: rho - target so sum(rho) = 0 ──
    total_macro_area = (sizes[:, 0] * sizes[:, 1]).sum()
    canvas_area = cw * ch
    target = total_macro_area / canvas_area  # scalar
    rho_centered = rho - target  # [K, Gy, Gx]

    # ── 3. Zero-pad 2x for free boundary conditions ──
    Gy, Gx = grid_row, grid_col
    Gy2, Gx2 = Gy * 2, Gx * 2
    rho_padded = torch.nn.functional.pad(rho_centered, (0, Gx, 0, Gy))  # [K, Gy2, Gx2]

    # ── 4. Solve Poisson nabla^2 phi = -rho via FFT ──
    rho_hat = torch.fft.rfft2(rho_padded)  # [K, Gy2, Gx2//2+1]  complex

    # Wavenumbers (cycles/length × 2π)
    kx = 2.0 * math.pi * torch.fft.rfftfreq(Gx2, d=grid_w).to(device=device, dtype=pop.dtype)  # [Gx2//2+1]
    ky = 2.0 * math.pi * torch.fft.fftfreq(Gy2, d=grid_h).to(device=device, dtype=pop.dtype)   # [Gy2]
    k2 = (kx.view(1, 1, -1) ** 2 + ky.view(1, -1, 1) ** 2).clamp(min=1e-12)  # [1, Gy2, Gx2//2+1]

    phi_hat = rho_hat / k2  # solve in frequency space
    phi_padded = torch.fft.irfft2(phi_hat, s=(Gy2, Gx2))  # [K, Gy2, Gx2]
    phi = phi_padded[:, :Gy, :Gx]  # crop back to original size

    # ── 5. Electrostatic potential energy U = (1/2) sum(rho * phi) ──
    energy = 0.5 * (rho_centered * phi).sum(dim=(1, 2))  # [K]

    # Normalize so the loss is O(1) regardless of benchmark size.
    return energy / canvas_area


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


def uniform_density_loss(
    pop: torch.Tensor,
    sizes: torch.Tensor,
    grid_col: int,
    grid_row: int,
    cw: float,
    ch: float,
) -> torch.Tensor:
    """
    D: Mean-squared deviation of per-cell density from the uniform target.

    target = total_macro_area / canvas_area
    loss   = mean over cells of (cell_density - target)^2

    tilos_density_loss only penalises the top-10% hot spots; that's a
    one-sided pressure that's silent about under-filled regions.  This term
    is symmetric: it equally punishes "too empty" (the donut hole at the
    centre of the ring-pattern benchmarks) and "too full" cells.  Together
    they drive the layout toward uniform spread.

    Returns [K].
    """
    K, N, _ = pop.shape
    grid_w = cw / grid_col
    grid_h = ch / grid_row
    grid_area = grid_w * grid_h

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
    occupied = torch.einsum("knx,kny->kyx", x_ov, y_ov)  # [K, Gy, Gx]
    density = occupied / grid_area  # [K, Gy, Gx]

    total_macro_area = (sizes[:, 0] * sizes[:, 1]).sum()
    canvas_area = cw * ch
    target = total_macro_area / canvas_area  # scalar

    return ((density - target) ** 2).mean(dim=(1, 2))  # [K]


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


def compute_cell_congestion(
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
) -> torch.Tensor:
    """
    Per-cell congestion field [K, Gy, Gx] = max(H_smoothed, V_smoothed).

    Same internal computation as tilos_rudy_normalized through the smoothing
    step, but returns the full field instead of the top-K%-mean scalar.
    Used by hard_congestion_pull (A) — the field is treated as a *target*
    (detach() before use) so the pull doesn't chase its own tail.
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
    return torch.maximum(v_dem, h_dem)  # [K, Gy, Gx]


def hard_congestion_pull(
    pop: torch.Tensor,
    sizes: torch.Tensor,
    cell_cong_target: torch.Tensor,  # [K, Gy, Gx]  — MUST be detached
    n_hard: int,
    grid_col: int,
    grid_row: int,
    cw: float,
    ch: float,
) -> torch.Tensor:
    """
    A: Attractive pull of hard macros toward high-congestion cells.

    Hard macros are physical obstacles: dropping one into a congestion hot
    spot displaces routing demand around it, breaking up the hot zone.  The
    raw RUDY loss only attacks congestion *indirectly* by shrinking net
    bboxes; this term gives a direct "go plug the hole" gradient.

    loss = -sum_over_cells(hard_macro_occupied_area × cell_cong_target)
    The minus sign means minimising the loss *maximises* overlap of hard
    macros with hot cells.  Normalized by (total_hard_area × mean_cong) so
    the per-K value is roughly O(1) — independent of benchmark size.

    cell_cong_target MUST be detached or the gradient will chase itself
    (the field depends on positions through bbox/RUDY, so leaving it
    attached creates an unstable feedback loop).
    """
    K = pop.shape[0]
    grid_w = cw / grid_col
    grid_h = ch / grid_row

    col_idx = torch.arange(grid_col, device=pop.device, dtype=pop.dtype)
    row_idx = torch.arange(grid_row, device=pop.device, dtype=pop.dtype)
    col_l = col_idx * grid_w
    col_r = col_l + grid_w
    row_b = row_idx * grid_h
    row_t = row_b + grid_h

    pos_h = pop[:, :n_hard]
    sizes_h = sizes[:n_hard]
    x = pos_h[..., 0]
    y = pos_h[..., 1]
    hw = sizes_h[:, 0] / 2
    hh = sizes_h[:, 1] / 2
    macro_l = (x - hw.unsqueeze(0)).unsqueeze(-1)
    macro_r = (x + hw.unsqueeze(0)).unsqueeze(-1)
    macro_b_ = (y - hh.unsqueeze(0)).unsqueeze(-1)
    macro_t_ = (y + hh.unsqueeze(0)).unsqueeze(-1)
    col_lb = col_l.view(1, 1, grid_col); col_rb = col_r.view(1, 1, grid_col)
    row_bb = row_b.view(1, 1, grid_row); row_tb = row_t.view(1, 1, grid_row)

    x_ov = (torch.min(macro_r, col_rb) - torch.max(macro_l, col_lb)).clamp(min=0.0)
    y_ov = (torch.min(macro_t_, row_tb) - torch.max(macro_b_, row_bb)).clamp(min=0.0)
    # Aggregate per-cell occupied by all hard macros: [K, Gy, Gx]
    occupied = torch.einsum("knx,kny->kyx", x_ov, y_ov)

    # Raw pull magnitude: high when occupied aligns with hot cells.
    # We minimize -pull, so this is a negative number.
    raw_pull = -(occupied * cell_cong_target).sum(dim=(1, 2))  # [K]

    # Normalize so the term is O(1) regardless of benchmark size.
    total_hard_area = (sizes_h[:, 0] * sizes_h[:, 1]).sum().clamp(min=1e-6)
    mean_cong = cell_cong_target.mean(dim=(1, 2)).clamp(min=1e-6)  # [K]
    return raw_pull / (total_hard_area * mean_cong)  # [K]


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

    # Budget seed slots: 1/8 spectral, 1/4 FD-random, 1/8 grid, rest jittered.
    # Always clamp to fit in [1, K-1].
    n_spec = max(1, K // 8) if K >= 8 else 0
    n_fd = max(1, K // 4) if K >= 4 else 0
    n_grid = max(1, K // 8) if K >= 8 else 0
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
    # E: grid-distributed seeds — anti-ring starting basin.  Hard macros placed
    # on an aspect-aware uniform grid sized to canvas, largest macros central,
    # smallest peripheral.  Multiple seeds differ by RNG jitter.
    for k in range(used, used + n_grid):
        if k >= K:
            break
        grid_hard = _grid_seeded_hard(sizes_np, cw, ch, seed=seed + 13 * k)
        grid_leg = _legalize(grid_hard, sizes_np, movable_np, cw, ch, gap=0.005)
        pop[k, :n_hard] = grid_leg
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


def _reset_adam_for_losers(opt, pop, losers, n_hard):
    """Zero Adam moments for soft slots of the given candidate indices.

    Without this, PBT respawns inherit stale momentum from the loser's prior
    trajectory and drift toward wrong basins. `step` counter is left alone
    since it's a scalar shared across the whole tensor.
    """
    if not losers:
        return
    state = opt.state.get(pop, None)
    if not state or 'exp_avg' not in state:
        return
    idx = torch.as_tensor(list(losers), device=pop.device, dtype=torch.long)
    state['exp_avg'][idx, n_hard:].zero_()
    state['exp_avg_sq'][idx, n_hard:].zero_()


# ────────────────────────────────────────────────────────────────────────────
# Main placer
# ────────────────────────────────────────────────────────────────────────────


class GraphGradPlacer:
    """GPU-batched analytical placer with graph-aware multi-start."""

    def __init__(
        self,
        seed: int = 42,
        pop_size: int = 96,                # 3× the original to use Blackwell VRAM
        n_epochs: int = 8,
        steps_per_epoch: int = 500,
        grid_res: int = 32,
        time_budget_s: float = 3000.0,     # 50 min
        verbose: bool = False,
        lock_hard: bool = True,            # Lock hard macros at legalized initial.plc
        soft_steps: int = 5000,            # Total Adam steps (was 3000)
        soft_lr: float = 0.01,
        n_restarts: int = 1,               # Independent restarts with different RNG seeds
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
        if self.lock_hard:
            return self._place_soft_only(benchmark)
        return self._place_joint(benchmark)

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
        for r in range(self.n_restarts):
            run_seed = self.seed + 1000 * r
            full, cost = self._soft_only_single_run(benchmark, run_seed)
            if cost < best_across_cost:
                best_across_cost = cost
                best_across = full
            if self.verbose and self.n_restarts > 1:
                self._log(f"restart {r+1}/{self.n_restarts}: proxy={cost:.4f}  best={best_across_cost:.4f}")
        return best_across

    def _soft_only_single_run(self, benchmark: Benchmark, run_seed: int):
        """One soft-only optimization run with hard-release final phase.

        Phase A (steps [0, release_step)): Hard macros locked at legalized
        initial.plc.  96-way Adam on soft positions only (anchor + 0.1 jitter).

        Phase B (steps [release_step, n_steps)): Hard macros unlocked, added
        to the loss with a strong anchor-pull regularizer and a heavy
        differentiable overlap penalty.  Periodic _legalize sweeps every 200
        steps keep accumulated overlap small.  Final _legalize always runs
        before scoring.

        Returns (placement_tensor, true_proxy).
        """
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
        sizes_hard_t = sizes_all[:n_hard]
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

        # Legalize hard macros once at initial.plc → anchor for safety net.
        init_pos = benchmark.macro_positions[:n_hard].numpy().astype(np.float64)
        hard_legal = _legalize(init_pos, sizes_np, movable_np, cw, ch, gap=0.005)
        hard_legal_t = torch.tensor(hard_legal, device=device, dtype=torch.float32)
        soft_anchor = benchmark.macro_positions[n_hard:].to(device).float()
        n_soft = n_macros - n_hard

        self._log(
            f"soft-only setup: n_hard={n_hard} n_soft={n_soft} "
            f"n_nets={n_nets}  grid {grid_col}x{grid_row}  H/V {h_per_um:.2f}/{v_per_um:.2f}  "
            f"device={device}"
        )

        # E: per-candidate hard layouts.  Slot 0 = legalized initial.plc
        # (preserved for safety net). Slots 1..n_grid = grid-distributed
        # anti-ring starts. Remaining = jittered initial.plc.  Each candidate's
        # hard layout is also its own anchor for the Phase B pull.
        K = self.pop_size
        rng_np = np.random.default_rng(run_seed)
        n_grid = max(1, K // 4) if K >= 4 else 0  # 25% of population on grid seeds
        hard_layouts = np.zeros((K, n_hard, 2), dtype=np.float64)
        hard_layouts[0] = hard_legal
        for k in range(1, min(1 + n_grid, K)):
            grid_h = _grid_seeded_hard(sizes_np, cw, ch, seed=run_seed + 17 * k)
            hard_layouts[k] = _legalize(grid_h, sizes_np, movable_np, cw, ch, gap=0.005)
        for k in range(1 + n_grid, K):
            jit_scale = 0.02 + 0.10 * rng_np.random()
            jitter = rng_np.standard_normal((n_hard, 2)) * (min(cw, ch) * jit_scale)
            jitter[~movable_np] = 0.0
            hard_layouts[k] = _legalize(
                hard_legal + jitter, sizes_np, movable_np, cw, ch, gap=0.005,
            )
        hard_layouts_t = torch.tensor(hard_layouts, device=device, dtype=torch.float32)  # [K, n_hard, 2]

        pop = torch.zeros(K, n_macros, 2, device=device, dtype=torch.float32)
        pop[:, :n_hard] = hard_layouts_t
        pop[:, n_hard:] = soft_anchor + torch.randn(K, n_soft, 2, device=device) * 0.1
        pop.requires_grad_(True)
        opt = torch.optim.Adam([pop], lr=self.soft_lr)

        # Joint-release config: unlock hard for last 20% of training.
        # Hard gets pulled toward its PER-CANDIDATE anchor (so grid-seeded
        # candidates aren't dragged back to initial.plc), plus a heavy
        # differentiable overlap penalty.  Periodic _legalize keeps
        # accumulated overlap small.
        release_step = int(0.80 * self.soft_steps)
        canvas_norm_t = torch.tensor([cw, ch], device=device, dtype=torch.float32).view(1, 1, 2)
        hard_anchor_b = hard_layouts_t  # [K, n_hard, 2] — per-candidate anchor
        alpha_anchor = 50.0   # normalized squared drift × 50; drift ~10% canvas → ~0.5 loss
        alpha_ov = 1000.0     # normalized overlap-area × 1000
        alpha_dens = 0.0      # TILOS top-10% density — disabled with ePlace active (F replaces it)
        alpha_eplace = 0.5    # F: ePlace electrostatic density (smooth global spreading)
        alpha_unif = 0.0      # D: uniform-density MSE — disabled (redundant with F)
        alpha_pull = 1.0      # A: congestion-gradient pull on hard macros (Phase B only)
        cong_refresh_every = 50  # A: how often to refresh the detached cong target
        relegalize_every = 200

        cell_cong_target = None  # A: lazily initialised inside Phase B
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
            unif = uniform_density_loss(pop, sizes_all, grid_col, grid_row, cw, ch)  # D
            eplace = eplace_density_loss(pop, sizes_all, grid_col, grid_row, cw, ch)  # F
            proxy_surr = (
                wl_n
                + alpha_dens * dens
                + alpha_eplace * eplace
                + 0.5 * cong
                + alpha_unif * unif
            )  # F replaces 0.5*dens; D + dens both keep weight 0 by default
            in_joint = step >= release_step
            if in_joint:
                hard_diff_n = (pop[:, :n_hard] - hard_anchor_b) / canvas_norm_t
                anchor_pen = hard_diff_n.pow(2).mean(dim=(1, 2))  # [K]
                ov_pen = overlap_loss_hard(pop[:, :n_hard], sizes_hard_t) / (cw * ch)  # [K]
                # A: refresh detached cell-congestion target every N steps,
                # then add the pull term that attracts hard macros into hot cells.
                rel_step = step - release_step
                if cell_cong_target is None or rel_step % cong_refresh_every == 0:
                    with torch.no_grad():
                        cell_cong_target = compute_cell_congestion(
                            pop, owner_idx, pin_off, net_id, n_nets, port_pos,
                            n_hard, n_macros, grid_col, grid_row, cw, ch,
                            h_per_um, v_per_um, smooth_range=2,
                        ).detach()
                pull = hard_congestion_pull(
                    pop, sizes_all, cell_cong_target,
                    n_hard, grid_col, grid_row, cw, ch,
                )
                loss = (
                    proxy_surr
                    + alpha_anchor * anchor_pen
                    + alpha_ov * ov_pen
                    + alpha_pull * pull
                ).sum()
            else:
                loss = proxy_surr.sum()
            loss.backward()
            with torch.no_grad():
                if not in_joint:
                    pop.grad[:, :n_hard].zero_()  # LOCK hard macros (Phase A)
            opt.step()
            with torch.no_grad():
                # Always clamp softs to canvas
                pop[:, n_hard:, 0].clamp_(min=half_w_t[n_hard:], max=cw - half_w_t[n_hard:])
                pop[:, n_hard:, 1].clamp_(min=half_h_t[n_hard:], max=ch - half_h_t[n_hard:])
                if not in_joint:
                    pop[:, :n_hard] = hard_layouts_t  # reassert per-candidate lock
                else:
                    # Clamp hard to canvas during joint phase
                    pop[:, :n_hard, 0].clamp_(min=half_w_t[:n_hard], max=cw - half_w_t[:n_hard])
                    pop[:, :n_hard, 1].clamp_(min=half_h_t[:n_hard], max=ch - half_h_t[:n_hard])
                    # Periodic re-legalize during joint phase to prevent overlap accumulation
                    rel_step = step - release_step
                    if rel_step > 0 and rel_step % relegalize_every == 0:
                        pop_cpu = pop[:, :n_hard].detach().cpu().numpy().astype(np.float64)
                        for k in range(K):
                            pop_cpu[k] = _legalize(pop_cpu[k], sizes_np, movable_np, cw, ch, gap=0.005)
                        pop.data[:, :n_hard] = torch.tensor(pop_cpu, device=device, dtype=torch.float32)
            if self.verbose and step % max(n_steps // 6, 1) == 0:
                tag = "JOINT" if in_joint else "soft"
                self._log(
                    f"  step {step} [{tag}]: wl_n={wl_n.mean().item():.4f} "
                    f"dens={dens.mean().item():.4f} eplace={eplace.mean().item():.4f} "
                    f"cong={cong.mean().item():.4f} "
                    f"proxy_surr={proxy_surr.mean().item():.4f}"
                )

        # Final re-legalize for all candidates (guarantees zero hard overlap).
        with torch.no_grad():
            pop_cpu = pop[:, :n_hard].detach().cpu().numpy().astype(np.float64)
            for k in range(K):
                pop_cpu[k] = _legalize(pop_cpu[k], sizes_np, movable_np, cw, ch, gap=0.005)
            pop.data[:, :n_hard] = torch.tensor(pop_cpu, device=device, dtype=torch.float32)

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
        true_costs: List[float] = []
        for k in top_idx:
            pos_full = pop[k].detach().cpu().numpy().astype(np.float64)
            pos_full = self._clip_soft_to_canvas(pos_full, benchmark, n_hard, cw, ch)
            full_t = torch.from_numpy(pos_full).float()
            if plc is None:
                continue
            c = compute_proxy_cost(full_t, benchmark, plc)
            if c["overlap_count"] == 0:
                true_costs.append(float(c["proxy_cost"]))
                if c["proxy_cost"] < best_cost:
                    best_cost = c["proxy_cost"]
                    best_full = full_t
        if true_costs:
            print(
                f"[koral release] top_idx true-cost min={min(true_costs):.4f} "
                f"max={max(true_costs):.4f} span={max(true_costs)-min(true_costs):.4f} n={len(true_costs)}",
                flush=True,
            )
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
