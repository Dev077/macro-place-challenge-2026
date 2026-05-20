"""
Graph-Evolutionary Placer
=========================

A GPU-parallel evolutionary algorithm anchored on the hand-crafted initial.plc
layout, using graph-aware operators (spectral cluster moves, geographic
crossover, hypergraph HPWL surrogate) and a final SA polish.

Pipeline
--------
1.  Build a pin-level net hypergraph + a hard-macro pair graph (clique-expanded
    nets, weight = 1/(k-1)).
2.  Minimum-displacement legalization of the initial.plc placement -> elite seed.
3.  Initial population on GPU (or CPU fallback):
      - elite (legalized initial.plc)
      - small Gaussian perturbations of elite
      - one spectral (Fiedler) seed, scaled into the canvas
      - one force-directed seed from random
      - random uniform seeds
    All seeds are legalized.
4.  Population evolution with a vectorized surrogate
    (pin-level HPWL + overlap penalty + coarse density variance), GA operators:
      - tournament selection
      - geographic (axis-cut) crossover preserving connected regions
      - cluster-aware Gaussian mutation (Fiedler community shifts)
      - single-macro Gaussian mutation
      - repair via projected legalization
    The true plc.get_cost() is only invoked on elites at the end of each epoch
    so we never optimise a misleading surrogate.
5.  Numpy SA polish around the best individual (Will-seed style, micro moves).
6.  Optional plc.optimize_stdcells for soft macros (Tier 2), gated by remaining
    runtime budget.

Usage
-----
    uv run evaluate submissions/graph_evo/placer.py -b ibm01
    uv run evaluate submissions/graph_evo/placer.py --all
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark


# ────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_plc(name: str):
    """Best-effort plc loader for IBM and NG45 benchmarks (mirrors Will-seed)."""
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
# Hypergraph + pair-graph extraction
# ────────────────────────────────────────────────────────────────────────────


def _build_pin_tensors(benchmark: Benchmark, device: torch.device):
    """
    Flatten the pin-level hypergraph into 1-D tensors so we can compute the
    full Manhattan HPWL per net for a *population* of placements with a single
    scatter pass.

    Returns
    -------
    owner_idx : [E] long   - flat owner index per pin instance
                              (hard: [0, num_hard); soft: [num_hard, num_macros);
                               port: [num_macros, num_macros + num_ports))
    pin_off   : [E, 2] f32 - pin offset (zero for soft / port)
    net_id    : [E] long   - which net this pin belongs to
    n_nets    : int        - number of valid nets (those with >= 2 pins)
    pin_count : [n_nets]   - pins per net (after dedup)
    """
    n_hard = benchmark.num_hard_macros
    n_macros = benchmark.num_macros
    n_ports = benchmark.port_positions.shape[0]

    owner_list, off_list, net_list = [], [], []
    pin_count_list = []
    nid = 0
    for pins in benchmark.net_pin_nodes:
        if pins.shape[0] < 2:
            continue
        # pins[:,0] = owner index (already in our flat scheme),
        # pins[:,1] = slot within owner's pin list
        owners = pins[:, 0].tolist()
        slots = pins[:, 1].tolist()
        for o, s in zip(owners, slots):
            if o < n_hard:
                pin_off_list = benchmark.macro_pin_offsets[o]
                if pin_off_list.shape[0] > 0:
                    ox, oy = pin_off_list[s].tolist()
                else:
                    ox, oy = 0.0, 0.0
            else:
                # soft macro pins live at the macro centre; ports are points
                ox, oy = 0.0, 0.0
            owner_list.append(o)
            off_list.append([ox, oy])
            net_list.append(nid)
        pin_count_list.append(len(owners))
        nid += 1

    owner_idx = torch.tensor(owner_list, dtype=torch.long, device=device)
    pin_off = torch.tensor(off_list, dtype=torch.float32, device=device)
    net_id = torch.tensor(net_list, dtype=torch.long, device=device)
    pin_count = torch.tensor(pin_count_list, dtype=torch.long, device=device)

    return owner_idx, pin_off, net_id, nid, pin_count, n_hard, n_macros, n_ports


def _build_pair_graph(benchmark: Benchmark) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hard-macro clique-expanded graph: each net contributes pair edges (i,j) with
    weight 1/(|net|-1) for every pair of *hard* macros in the net.

    Returns (edges [E,2] int64, weights [E] float32).
    """
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


def _spectral_seed(
    edges: np.ndarray,
    weights: np.ndarray,
    n_hard: int,
    canvas_w: float,
    canvas_h: float,
    sizes: np.ndarray,
) -> np.ndarray:
    """
    Fiedler-style 2-D embedding of the hard-macro pair graph.  Returns
    [n_hard, 2] positions inside the canvas.  Falls back to a uniform grid if
    the graph is degenerate.
    """
    pos = np.zeros((n_hard, 2), dtype=np.float64)
    if edges.shape[0] == 0:
        return _grid_fill(n_hard, canvas_w, canvas_h, sizes)
    try:
        from scipy.sparse import coo_matrix, csr_matrix
        from scipy.sparse.linalg import eigsh

        i = np.concatenate([edges[:, 0], edges[:, 1]])
        j = np.concatenate([edges[:, 1], edges[:, 0]])
        v = np.concatenate([weights, weights]).astype(np.float64)
        W = coo_matrix((v, (i, j)), shape=(n_hard, n_hard)).tocsr()
        d = np.asarray(W.sum(axis=1)).ravel()
        d[d == 0] = 1e-9
        d_inv_sqrt = 1.0 / np.sqrt(d)
        D = csr_matrix(
            (d_inv_sqrt, (np.arange(n_hard), np.arange(n_hard))),
            shape=(n_hard, n_hard),
        )
        # Normalized Laplacian = I - D^-1/2 W D^-1/2
        L = csr_matrix(
            (np.ones(n_hard), (np.arange(n_hard), np.arange(n_hard))),
            shape=(n_hard, n_hard),
        ) - D @ W @ D
        # Smallest 3 eigenvectors (skip trivial 1st)
        vals, vecs = eigsh(L, k=min(3, n_hard - 1), which="SM")
        order = np.argsort(vals)
        vecs = vecs[:, order]
        # Skip eigenvector 0 (constant); use #1 and #2 for x/y
        ex = vecs[:, 1]
        ey = vecs[:, 2] if vecs.shape[1] > 2 else np.random.randn(n_hard)
        # Normalize to [0,1]
        for v_, axis in ((ex, 0), (ey, 1)):
            lo, hi = float(np.min(v_)), float(np.max(v_))
            if hi - lo < 1e-12:
                pos[:, axis] = np.random.rand(n_hard)
            else:
                pos[:, axis] = (v_ - lo) / (hi - lo)
        # Scale into canvas leaving margin for macro halves
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        margin_x = (float(np.max(half_w)) + 0.05) / canvas_w
        margin_y = (float(np.max(half_h)) + 0.05) / canvas_h
        pos[:, 0] = margin_x + pos[:, 0] * (1 - 2 * margin_x)
        pos[:, 1] = margin_y + pos[:, 1] * (1 - 2 * margin_y)
        pos[:, 0] *= canvas_w
        pos[:, 1] *= canvas_h
    except Exception:
        return _grid_fill(n_hard, canvas_w, canvas_h, sizes)
    return pos


def _grid_fill(
    n_hard: int, canvas_w: float, canvas_h: float, sizes: np.ndarray
) -> np.ndarray:
    """Uniform grid fallback."""
    cols = max(1, int(math.ceil(math.sqrt(n_hard))))
    rows = max(1, int(math.ceil(n_hard / cols)))
    pos = np.zeros((n_hard, 2), dtype=np.float64)
    for k in range(n_hard):
        r, c = divmod(k, cols)
        pos[k, 0] = (c + 0.5) * canvas_w / cols
        pos[k, 1] = (r + 0.5) * canvas_h / rows
    return pos


# ────────────────────────────────────────────────────────────────────────────
# Legalization (minimum displacement)
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
    """
    Vectorized "push-apart" legalization.

    Iteratively resolves overlapping pairs by shifting along their *smaller*-
    overlap axis by exactly the overlap amount, so the resulting placement
    stays as close as possible to the input.  Far cheaper than spiral search
    when the seed is mostly legal (e.g. the hand-crafted initial.plc).

    Returns (positions, fully_legal_in_float32).
    """
    n = pos.shape[0]
    out = pos.copy()
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0
    # Boundary margin must survive float32 conversion (ULP ~1e-5 at 100μm).
    bmargin = max(gap, 1e-3)

    # Lock fixed positions
    for k in range(n):
        if not movable[k]:
            out[k] = pos[k]

    # Clamp every macro into canvas first
    out[:, 0] = np.clip(out[:, 0], half_w + bmargin, cw - half_w - bmargin)
    out[:, 1] = np.clip(out[:, 1], half_h + bmargin, ch - half_h - bmargin)

    # Push pairs that are touching or within `safety` of touching so that
    # float32 conversion cannot recreate an overlap.
    safety = max(gap * 0.5, 1e-5)
    buffer = gap

    for it in range(max_iters):
        dx = out[:, 0:1] - out[:, 0:1].T  # signed
        dy = out[:, 1:2] - out[:, 1:2].T
        absdx = np.abs(dx)
        absdy = np.abs(dy)
        real_ox = sep_x - absdx
        real_oy = sep_y - absdy
        # "near-overlap": treat as overlap any pair within `safety` of touching
        overlap = (real_ox + safety > 0) & (real_oy + safety > 0)
        np.fill_diagonal(overlap, False)
        if not overlap.any():
            break

        push_x = np.zeros(n)
        push_y = np.zeros(n)
        # Deterministic tie-break for coincident macros (rare)
        sign_dx = np.where(absdx < 1e-9, 1.0, np.sign(dx))
        sign_dy = np.where(absdy < 1e-9, 1.0, np.sign(dy))
        push_axis_x = overlap & (real_ox <= real_oy)
        push_axis_y = overlap & ~push_axis_x
        # Push enough to leave dx >= sep_x + 2*buffer
        contrib_x = np.where(push_axis_x, sign_dx * (real_ox * 0.5 + buffer), 0.0)
        contrib_y = np.where(push_axis_y, sign_dy * (real_oy * 0.5 + buffer), 0.0)
        push_x += contrib_x.sum(axis=1)
        push_y += contrib_y.sum(axis=1)
        out[movable, 0] += push_x[movable]
        out[movable, 1] += push_y[movable]
        out[~movable] = pos[~movable]
        out[:, 0] = np.clip(out[:, 0], half_w + bmargin, cw - half_w - bmargin)
        out[:, 1] = np.clip(out[:, 1], half_h + bmargin, ch - half_h - bmargin)

    # Verify in float32 (the precision the evaluator/validator actually use).
    out_f32 = out.astype(np.float32).astype(np.float64)
    absdx32 = np.abs(out_f32[:, 0:1] - out_f32[:, 0:1].T)
    absdy32 = np.abs(out_f32[:, 1:2] - out_f32[:, 1:2].T)
    overlap32 = (sep_x - absdx32 > 0) & (sep_y - absdy32 > 0)
    np.fill_diagonal(overlap32, False)
    bounds_ok = bool(
        ((out_f32[:, 0] - half_w) >= 0).all()
        and ((out_f32[:, 0] + half_w) <= cw).all()
        and ((out_f32[:, 1] - half_h) >= 0).all()
        and ((out_f32[:, 1] + half_h) <= ch).all()
    )
    return out, (not bool(overlap32.any())) and bounds_ok


def _legalize_spiral(
    pos: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    cw: float,
    ch: float,
    gap: float = 0.005,
) -> np.ndarray:
    """
    Spiral-search fallback when spread can't fully resolve overlaps.

    Processes macros in order of decreasing area, placing each at the nearest
    overlap-free location on a fine spiral around its desired centre.
    """
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

    def overlaps_with_placed(idx, cx, cy):
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
        if not overlaps_with_placed(idx, cx, cy):
            out[idx, 0] = cx
            out[idx, 1] = cy
            placed[idx] = True
            continue
        # Adaptive step: based on smallest side of this macro
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
                    if overlaps_with_placed(idx, nx, ny):
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


def _legalize_min_disp(
    pos: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    cw: float,
    ch: float,
    gap: float = 0.005,
) -> np.ndarray:
    """
    Hybrid: try gentle spread first (cheap, preserves seed); fall back to
    spiral search if any pair remains overlapping.
    """
    spread, ok = _legalize_spread(pos, sizes, movable, cw, ch, gap=gap)
    if ok:
        return spread
    return _legalize_spiral(spread, sizes, movable, cw, ch, gap=gap)


# ────────────────────────────────────────────────────────────────────────────
# Surrogate cost (population-vectorized, GPU friendly)
# ────────────────────────────────────────────────────────────────────────────


class Surrogate:
    """
    Pin-level HPWL + soft overlap penalty + coarse density variance.

    Builds owner tables once per benchmark and evaluates a population of hard
    macro placements in a single vectorized scatter pass.  Soft macro positions
    and port positions are held fixed during the surrogate (matching the SA
    baseline's behaviour where soft cells are only re-optimised between epochs).
    """

    def __init__(self, benchmark: Benchmark, device: torch.device):
        self.device = device
        self.n_hard = benchmark.num_hard_macros
        self.n_macros = benchmark.num_macros
        self.n_ports = benchmark.port_positions.shape[0]
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)

        (
            self.owner_idx,
            self.pin_off,
            self.net_id,
            self.n_nets,
            self.pin_count,
            n_hard,
            n_macros,
            n_ports,
        ) = _build_pin_tensors(benchmark, device)

        # Fixed (non-hard) anchor positions: soft macros + ports
        anchors = torch.zeros(n_ports + (n_macros - n_hard), 2, device=device)
        if n_macros - n_hard > 0:
            anchors[: n_macros - n_hard] = benchmark.macro_positions[n_hard:n_macros].to(device)
        if n_ports > 0:
            anchors[n_macros - n_hard :] = benchmark.port_positions.to(device)
        # full owner table: [0..n_hard) movable hard, [n_hard..n_macros) soft,
        # [n_macros..n_macros+n_ports) ports
        self.anchor_offset = n_hard
        # We will assemble [P, total_owners, 2] per evaluation.
        self.anchors = anchors  # [n_macros - n_hard + n_ports, 2]

        self.sizes = benchmark.macro_sizes[:n_hard].to(device)  # [n_hard, 2]
        self.half = self.sizes / 2.0

        # Density grid (coarse 16x16)
        self.dg = 16
        self.cell_w = self.cw / self.dg
        self.cell_h = self.ch / self.dg
        self.area = self.sizes[:, 0] * self.sizes[:, 1]  # [n_hard]
        self.total_area = float(self.area.sum().item())
        self.canvas_area = self.cw * self.ch
        self.target_per_cell = self.total_area / (self.dg * self.dg)

        # Normalisation: use a rough scale ~ canvas perimeter
        self._wl_norm = 1.0 / (self.cw + self.ch) if (self.cw + self.ch) > 0 else 1.0

        # Pre-build "owner_is_hard" mask & indexing
        self.owner_is_hard = self.owner_idx < n_hard  # [E]
        # offset into anchor table for non-hard owners
        self.anchor_idx_for_nonhard = (self.owner_idx - n_hard).clamp(min=0)

    def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
        """
        pop  : [P, n_hard, 2]
        returns surrogate cost [P]
        """
        P = pop.shape[0]
        n_hard = self.n_hard
        device = self.device

        # Build per-pin owner positions: hard from pop, others from anchors
        # pin_owner_pos: [P, E, 2]
        E = self.owner_idx.shape[0]
        if E > 0:
            hard_pos = pop  # [P, n_hard, 2]
            # gather for hard owners
            owner_safe_hard = self.owner_idx.clamp(max=n_hard - 1)
            hard_part = hard_pos[:, owner_safe_hard, :]  # [P, E, 2]
            # broadcast anchor positions for non-hard owners
            if self.anchors.shape[0] > 0:
                anchor_part = self.anchors[self.anchor_idx_for_nonhard]  # [E, 2]
                anchor_part = anchor_part.unsqueeze(0).expand(P, E, 2)
            else:
                anchor_part = torch.zeros_like(hard_part)
            is_hard = self.owner_is_hard.view(1, E, 1)
            owner_pos = torch.where(is_hard, hard_part, anchor_part)
            pin_pos = owner_pos + self.pin_off.unsqueeze(0)  # [P, E, 2]
        else:
            pin_pos = torch.zeros(P, 0, 2, device=device)

        # Scatter min/max per net per population
        if E > 0 and self.n_nets > 0:
            n_nets = self.n_nets
            big = 1e9
            xmin = torch.full((P, n_nets), big, device=device)
            ymin = torch.full((P, n_nets), big, device=device)
            xmax = torch.full((P, n_nets), -big, device=device)
            ymax = torch.full((P, n_nets), -big, device=device)
            idx = self.net_id.unsqueeze(0).expand(P, E)
            xmin.scatter_reduce_(1, idx, pin_pos[..., 0], reduce="amin", include_self=True)
            ymin.scatter_reduce_(1, idx, pin_pos[..., 1], reduce="amin", include_self=True)
            xmax.scatter_reduce_(1, idx, pin_pos[..., 0], reduce="amax", include_self=True)
            ymax.scatter_reduce_(1, idx, pin_pos[..., 1], reduce="amax", include_self=True)
            hpwl = (xmax - xmin) + (ymax - ymin)  # [P, n_nets]
            wl = hpwl.sum(dim=1) * self._wl_norm  # [P]
        else:
            wl = torch.zeros(P, device=device)

        # Overlap penalty: pairwise (hard only)
        # Use the analytic AABB intersection area: positive if axes overlap.
        # For P populations with N up to ~540, P*N^2 ~ 32 * 290k = 9.3M which is fine on GPU.
        N = n_hard
        x = pop[:, :, 0]
        y = pop[:, :, 1]
        w = self.sizes[:, 0]
        h = self.sizes[:, 1]
        # dx, dy : [P, N, N]
        dx = (x.unsqueeze(2) - x.unsqueeze(1)).abs()
        dy = (y.unsqueeze(2) - y.unsqueeze(1)).abs()
        sx = (w.unsqueeze(0) + w.unsqueeze(0).transpose(0, 1)).unsqueeze(0) / 2.0
        sy = (h.unsqueeze(0) + h.unsqueeze(0).transpose(0, 1)).unsqueeze(0) / 2.0
        ox = (sx - dx).clamp(min=0.0)
        oy = (sy - dy).clamp(min=0.0)
        ov = ox * oy  # [P, N, N]
        # zero out diagonal
        diag = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        ov = ov.masked_fill(diag, 0.0)
        # Only count upper triangle to avoid double-counting
        triu = torch.ones(N, N, device=device, dtype=torch.bool).triu(diagonal=1).unsqueeze(0)
        ov = ov * triu.float()
        overlap_area = ov.sum(dim=(1, 2))  # [P]
        # large penalty per μm² of overlap
        ov_pen = overlap_area * 1000.0

        # Coarse density variance (encourage spreading)
        gx = ((x / self.cell_w).clamp(0, self.dg - 1e-3)).long()
        gy = ((y / self.cell_h).clamp(0, self.dg - 1e-3)).long()
        flat = gy * self.dg + gx  # [P, N]
        bucket = torch.zeros(P, self.dg * self.dg, device=device)
        area = self.area.unsqueeze(0).expand(P, N)
        bucket.scatter_add_(1, flat, area)
        # Density variance scaled to be in same units as WL
        density_var = ((bucket - self.target_per_cell) ** 2).mean(dim=1) / (
            self.target_per_cell ** 2 + 1e-12
        )
        density_pen = density_var * 0.05  # tuned weight

        return wl + ov_pen + density_pen


# ────────────────────────────────────────────────────────────────────────────
# GA Operators
# ────────────────────────────────────────────────────────────────────────────


def _project_to_canvas(pop: torch.Tensor, half: torch.Tensor, cw: float, ch: float):
    """Clamp positions so macros fit fully within the canvas (in-place)."""
    pop[..., 0].clamp_(min=half[:, 0], max=cw - half[:, 0])
    pop[..., 1].clamp_(min=half[:, 1], max=ch - half[:, 1])


def _tournament_select(costs: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Tournament selection returning a permutation of parent indices."""
    P = costs.shape[0]
    parents = []
    for _ in range(P):
        cand = torch.randint(0, P, (k,), device=costs.device)
        winner = cand[torch.argmin(costs[cand])]
        parents.append(int(winner.item()))
    return torch.tensor(parents, device=costs.device, dtype=torch.long)


def _crossover_geo(pop: torch.Tensor, parents: torch.Tensor, cw: float, ch: float) -> torch.Tensor:
    """
    Geographic crossover.  For each child:
      - pick axis (x or y) and a cut position
      - take parent A's macros on one side, parent B's on the other
    """
    P, N, _ = pop.shape
    pa = parents
    pb = parents[torch.randperm(P, device=pop.device)]
    children = pop[pa].clone()
    for i in range(P):
        if random.random() < 0.85:
            axis = random.randint(0, 1)
            cut = random.uniform(0.2, 0.8) * (cw if axis == 0 else ch)
            mask = pop[pa[i], :, axis] > cut
            children[i, mask] = pop[pb[i], mask]
    return children


def _mutate(
    pop: torch.Tensor,
    sizes: torch.Tensor,
    cluster_assign: torch.Tensor,
    n_clusters: int,
    cw: float,
    ch: float,
    half: torch.Tensor,
    sigma: float,
):
    """
    Combined mutation: per-macro Gaussian + occasional cluster shift.

    Cluster shift moves an entire spectral community by a random vector, which
    is the "graph-aware" mutation - macros that are tightly connected travel
    together.
    """
    P, N, _ = pop.shape
    device = pop.device
    # per-macro Gaussian
    noise = torch.randn(P, N, 2, device=device) * sigma
    # only some macros mutate per individual
    mask = (torch.rand(P, N, device=device) < 0.15).unsqueeze(-1)
    pop = pop + noise * mask
    # cluster shifts on a few individuals
    if n_clusters > 1:
        for i in range(P):
            if random.random() < 0.5:
                c = random.randint(0, n_clusters - 1)
                sel = cluster_assign == c
                if sel.any():
                    sh = torch.randn(2, device=device) * sigma * 2.0
                    pop[i, sel] = pop[i, sel] + sh
    _project_to_canvas(pop, half, cw, ch)
    return pop


def _kmeans_clusters(coords: np.ndarray, k: int) -> np.ndarray:
    """Lightweight k-means for community labels from spectral embedding."""
    if coords.shape[0] == 0 or k <= 1:
        return np.zeros(coords.shape[0], dtype=np.int64)
    rng = np.random.default_rng(0)
    idx = rng.choice(coords.shape[0], size=min(k, coords.shape[0]), replace=False)
    centres = coords[idx].copy()
    labels = np.zeros(coords.shape[0], dtype=np.int64)
    for _ in range(15):
        d = ((coords[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(axis=1)
        for c in range(centres.shape[0]):
            sel = labels == c
            if sel.any():
                centres[c] = coords[sel].mean(axis=0)
    return labels


# ────────────────────────────────────────────────────────────────────────────
# Final SA polish (numpy, Will-seed-style, micro moves only)
# ────────────────────────────────────────────────────────────────────────────


def _sa_polish(
    pos: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    cw: float,
    ch: float,
    iters: int = 4000,
    seed: int = 0,
):
    if edges.shape[0] == 0 or iters <= 0:
        return pos
    rng = random.Random(seed)
    n = pos.shape[0]
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    movable_idx = np.where(movable)[0]
    if len(movable_idx) == 0:
        return pos
    pos = pos.copy()
    neigh: List[List[int]] = [[] for _ in range(n)]
    for a, b in edges:
        neigh[a].append(int(b))
        neigh[b].append(int(a))

    def wl():
        dx = np.abs(pos[edges[:, 0], 0] - pos[edges[:, 1], 0])
        dy = np.abs(pos[edges[:, 0], 1] - pos[edges[:, 1], 1])
        return float((weights * (dx + dy)).sum())

    def overlaps(i):
        dx = np.abs(pos[i, 0] - pos[:, 0])
        dy = np.abs(pos[i, 1] - pos[:, 1])
        m = (dx < sep_x[i] + 0.05) & (dy < sep_y[i] + 0.05)
        m[i] = False
        return bool(m.any())

    cur = wl()
    best_pos = pos.copy()
    best = cur
    T0 = max(cw, ch) * 0.05
    Tend = max(cw, ch) * 0.0005
    for step in range(iters):
        T = T0 * (Tend / T0) ** (step / iters)
        i = int(rng.choice(movable_idx))
        ox, oy = pos[i, 0], pos[i, 1]
        s = T * 0.6
        pos[i, 0] = np.clip(ox + rng.gauss(0, s), half_w[i], cw - half_w[i])
        pos[i, 1] = np.clip(oy + rng.gauss(0, s), half_h[i], ch - half_h[i])
        if overlaps(i):
            pos[i, 0], pos[i, 1] = ox, oy
            continue
        new = wl()
        d = new - cur
        if d < 0 or rng.random() < math.exp(-d / max(T, 1e-12)):
            cur = new
            if cur < best:
                best = cur
                best_pos = pos.copy()
        else:
            pos[i, 0], pos[i, 1] = ox, oy
    return best_pos


# ────────────────────────────────────────────────────────────────────────────
# Placer
# ────────────────────────────────────────────────────────────────────────────


class GraphEvoPlacer:
    """Graph-aware GPU evolutionary placer anchored at initial.plc."""

    def __init__(
        self,
        seed: int = 42,
        time_budget_s: float = 2400.0,  # 40 min/bench (under 1h cap)
        pop_size: int = 16,
        n_clusters: int = 8,
        legal_gap: float = 0.005,
        soft_fd: bool = False,
        ga_radius_frac: float = 0.005,  # mutation sigma as a fraction of min canvas dim
        verbose: bool = False,
    ):
        self.seed = seed
        self.time_budget_s = time_budget_s
        self.pop_size = pop_size
        self.n_clusters = n_clusters
        self.legal_gap = legal_gap
        self.soft_fd = soft_fd
        self.ga_radius_frac = ga_radius_frac
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[graph_evo] {msg}", flush=True)

    # ----- helpers -----
    def _true_proxy(self, plc, full: torch.Tensor, benchmark: Benchmark) -> float:
        """Wrap compute_proxy_cost to return just the scalar proxy."""
        from macro_place.objective import compute_proxy_cost

        try:
            return float(compute_proxy_cost(full, benchmark, plc)["proxy_cost"])
        except Exception:
            return float("inf")

    def _edge_wl(self, pos: np.ndarray, edges: np.ndarray, weights: np.ndarray) -> float:
        if edges.shape[0] == 0:
            return 0.0
        dx = np.abs(pos[edges[:, 0], 0] - pos[edges[:, 1], 0])
        dy = np.abs(pos[edges[:, 0], 1] - pos[edges[:, 1], 1])
        return float((weights * (dx + dy)).sum())

    # ----- main -----
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_start = time.time()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        device = _device()
        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        movable_np = benchmark.get_movable_mask()[:n_hard].numpy()

        # ── 1. Build graph + pin tensors ──
        edges, weights = _build_pair_graph(benchmark)
        surr = Surrogate(benchmark, device)

        # Legalized initial.plc seed (elite anchor).  This alone gives ~1.04
        # on ibm01 and is the floor that the GA must beat.
        t_leg = time.time()
        init_pos = benchmark.macro_positions[:n_hard].numpy().astype(np.float64)
        elite_np = _legalize_min_disp(init_pos, sizes_np, movable_np, cw, ch, gap=self.legal_gap)
        self._log(f"legalize_anchor: {time.time()-t_leg:.1f}s  device={device}  n_hard={n_hard}")

        # Spectral seed (one diverse individual)
        spec_np = _spectral_seed(edges, weights, n_hard, cw, ch, sizes_np)
        spec_np = _legalize_min_disp(spec_np, sizes_np, movable_np, cw, ch, gap=self.legal_gap)

        # Cluster labels for graph-aware mutations (Fiedler communities)
        cluster_labels = _kmeans_clusters(spec_np, self.n_clusters)
        cluster_assign = torch.tensor(cluster_labels, device=device, dtype=torch.long)

        # ── 2. Build population ──
        # All individuals are tight perturbations of the elite so we never
        # drift far from the strong hand-crafted layout.  We seat the spectral
        # seed as one diverse individual to give crossover something to mix.
        P = self.pop_size
        sigma_init = min(cw, ch) * self.ga_radius_frac
        pop = torch.zeros(P, n_hard, 2, device=device, dtype=torch.float32)
        pop[0] = torch.from_numpy(elite_np).float().to(device)
        pop[1] = torch.from_numpy(spec_np).float().to(device)
        for i in range(2, P):
            jitter = np.random.randn(n_hard, 2) * sigma_init
            jitter[~movable_np] = 0.0
            pop[i] = torch.from_numpy(elite_np + jitter).float().to(device)
        _project_to_canvas(pop, surr.half, cw, ch)
        half_t = surr.half

        # ── 3. Evolution under surrogate, gated by TRUE cost ──
        # Plan: spend ~50% of budget on GA, then ~40% on SA, ~10% on final
        # legalize/eval.
        evo_budget = self.time_budget_s * 0.50
        sigma_start = sigma_init
        sigma_end = sigma_init * 0.1
        # Track the true-cost best across GA epochs
        plc = _load_plc(benchmark.name)
        elite_full = benchmark.macro_positions.clone()
        elite_full[:n_hard] = torch.tensor(elite_np, dtype=torch.float32)
        best_true = self._true_proxy(plc, elite_full, benchmark) if plc is not None else float("inf")
        best_pos = elite_np.copy()
        self._log(f"anchor true_proxy={best_true:.4f}")

        gen = 0
        epoch_gens = 10  # gens between true-cost gates
        while time.time() - t_start < evo_budget:
            frac = min(1.0, (time.time() - t_start) / max(evo_budget, 1.0))
            sigma = sigma_start * ((sigma_end / sigma_start) ** frac)

            costs = surr.evaluate(pop)
            parents = _tournament_select(costs, k=3)
            children = _crossover_geo(pop, parents, cw, ch)
            _project_to_canvas(children, half_t, cw, ch)
            children = _mutate(children, surr.sizes, cluster_assign, self.n_clusters, cw, ch, half_t, sigma)

            # Elitism: top 2 from pop replace worst 2 children
            top2 = torch.topk(-costs, k=2).indices
            child_costs = surr.evaluate(children)
            worst2 = torch.topk(child_costs, k=2).indices
            for s, d in zip(top2.tolist(), worst2.tolist()):
                children[d] = pop[s]
            pop = children
            gen += 1

            # Periodic legalization repair of best surrogate individual + TRUE-cost gate
            if gen % epoch_gens == 0:
                idx = int(torch.argmin(surr.evaluate(pop)).item())
                cand_np = pop[idx].detach().cpu().numpy().astype(np.float64)
                cand_np = _legalize_min_disp(cand_np, sizes_np, movable_np, cw, ch, gap=self.legal_gap)
                pop[idx] = torch.from_numpy(cand_np).float().to(device)
                if plc is not None and time.time() - t_start < evo_budget * 0.9:
                    full = benchmark.macro_positions.clone()
                    full[:n_hard] = torch.tensor(cand_np, dtype=torch.float32)
                    tc = self._true_proxy(plc, full, benchmark)
                    self._log(f"epoch gen={gen} sigma={sigma:.4f} true={tc:.4f} best={best_true:.4f}")
                    if tc < best_true:
                        best_true = tc
                        best_pos = cand_np

        # ── 4. SA polish on the true-cost-best individual ──
        # Use the hard-macro pair-graph HPWL as surrogate.  Compared to GA this
        # local search is much more targeted and respects the elite anchor.
        remaining = self.time_budget_s - (time.time() - t_start)
        if remaining > 30 and edges.shape[0] > 0:
            iters = int(min(80000, max(2000, remaining * 800)))
            polished = _sa_polish(
                best_pos, edges, weights, movable_np, sizes_np, cw, ch, iters=iters, seed=self.seed
            )
            polished = _legalize_min_disp(polished, sizes_np, movable_np, cw, ch, gap=self.legal_gap)
            if plc is not None:
                full = benchmark.macro_positions.clone()
                full[:n_hard] = torch.tensor(polished, dtype=torch.float32)
                tc = self._true_proxy(plc, full, benchmark)
                self._log(f"SA polish true={tc:.4f} best={best_true:.4f}")
                if tc < best_true:
                    best_true = tc
                    best_pos = polished

        # ── 5. Final legalize + assemble ──
        final_pos = _legalize_min_disp(best_pos, sizes_np, movable_np, cw, ch, gap=self.legal_gap)
        full = benchmark.macro_positions.clone()
        full[:n_hard] = torch.tensor(final_pos, dtype=torch.float32)

        # Clip soft macros / ports inside canvas (initial.plc sometimes places
        # them slightly outside, which fails validate_placement).  Use a tiny
        # margin to survive float32 rounding.
        soft_margin = np.float32(1e-3)
        all_sizes = benchmark.macro_sizes.numpy()
        all_hw = (all_sizes[:, 0] / 2.0).astype(np.float32)
        all_hh = (all_sizes[:, 1] / 2.0).astype(np.float32)
        cw_f = np.float32(cw)
        ch_f = np.float32(ch)
        # Only adjust SOFT macros and unchanged-fixed ones; never override hard movables
        soft_slice = slice(n_hard, benchmark.num_macros)
        full[soft_slice, 0] = torch.clamp(
            full[soft_slice, 0],
            min=torch.from_numpy(all_hw[soft_slice] + soft_margin),
            max=torch.from_numpy(cw_f - all_hw[soft_slice] - soft_margin),
        )
        full[soft_slice, 1] = torch.clamp(
            full[soft_slice, 1],
            min=torch.from_numpy(all_hh[soft_slice] + soft_margin),
            max=torch.from_numpy(ch_f - all_hh[soft_slice] - soft_margin),
        )

        if self.soft_fd and plc is not None:
            from macro_place.objective import _set_placement

            _set_placement(plc, full, benchmark)
            cs = max(cw, ch)
            try:
                plc.optimize_stdcells(
                    use_current_loc=False,
                    move_stdcells=True,
                    move_macros=False,
                    log_scale_conns=False,
                    use_sizes=False,
                    io_factor=1.0,
                    num_steps=[60, 60, 60],
                    max_move_distance=[cs / 100] * 3,
                    attract_factor=[100, 1.0e-3, 1.0e-5],
                    repel_factor=[0, 1.0e6, 1.0e7],
                )
                for i, midx in enumerate(benchmark.soft_macro_indices):
                    x, y = plc.modules_w_pins[midx].get_pos()
                    full[n_hard + i, 0] = float(x)
                    full[n_hard + i, 1] = float(y)
            except Exception:
                pass

        return full
