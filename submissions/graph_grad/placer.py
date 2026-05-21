"""
LKPlacer — Five-phase macro placer with electrostatic GP front-end.

Phase α  Focused Electrostatic Global Placement  (gp.run_global_placement)
Phase 0  Legalize hard macros
Phase 1  Build FastEvaluator (bit-exact mirror of PlacementCost)
Phase 2  Lin-Kernighan k-opt + grid sweep
Phase 3  LAHC polish (true cost via fast evaluator, mixed hard/soft moves
         including partner-centroid biased proposals)
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
# Phase 0 — Legalization
# ────────────────────────────────────────────────────────────────────────────


def _overlap_pair(p1, s1, p2, s2):
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    ox = (s1[0] + s2[0]) / 2 - dx
    oy = (s1[1] + s2[1]) / 2 - dy
    if ox > 0 and oy > 0:
        return ox, oy
    return 0.0, 0.0


def _has_overlap(i, pos, sizes):
    n = pos.shape[0]
    for j in range(n):
        if i == j:
            continue
        ox, oy = _overlap_pair(pos[i], sizes[i], pos[j], sizes[j])
        if ox > 0 and oy > 0:
            return True
    return False


def _spiral_search(i, pos, sizes, movable, cw, ch, gap):
    base = pos[i].copy()
    half = sizes[i] / 2
    step = min(cw, ch) * 0.01
    for r in range(1, 200):
        for ang in range(0, 360, 15):
            t = math.radians(ang)
            cand = base + np.array([math.cos(t), math.sin(t)]) * (step * r)
            cand[0] = max(half[0], min(cw - half[0], cand[0]))
            cand[1] = max(half[1], min(ch - half[1], cand[1]))
            ok = True
            for j in range(pos.shape[0]):
                if i == j:
                    continue
                ox, oy = _overlap_pair(cand, sizes[i], pos[j], sizes[j])
                if ox > 0 and oy > 0:
                    ok = False
                    break
            if ok:
                return cand
    return base


def _legalize(
    positions: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    gap: float = 0.02,
    max_passes: int = 80,
) -> np.ndarray:
    """Greedy push-apart legalizer with spiral fallback for stragglers."""
    n = positions.shape[0]
    pos = positions.copy()
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    for _ in range(max_passes):
        any_ov = False
        order = np.argsort(pos[:, 0])
        for ii in range(n):
            i = order[ii]
            for jj in range(ii + 1, n):
                j = order[jj]
                if pos[j, 0] - pos[i, 0] > (sizes[i, 0] + sizes[j, 0]) / 2 + gap:
                    break
                ox, oy = _overlap_pair(pos[i], sizes[i], pos[j], sizes[j])
                if ox <= 0 or oy <= 0:
                    continue
                any_ov = True
                if ox < oy:
                    push = ox + gap
                    sign = 1.0 if pos[j, 0] >= pos[i, 0] else -1.0
                    if movable[i] and movable[j]:
                        pos[i, 0] -= sign * push * 0.5
                        pos[j, 0] += sign * push * 0.5
                    elif movable[j]:
                        pos[j, 0] += sign * push
                    elif movable[i]:
                        pos[i, 0] -= sign * push
                else:
                    push = oy + gap
                    sign = 1.0 if pos[j, 1] >= pos[i, 1] else -1.0
                    if movable[i] and movable[j]:
                        pos[i, 1] -= sign * push * 0.5
                        pos[j, 1] += sign * push * 0.5
                    elif movable[j]:
                        pos[j, 1] += sign * push
                    elif movable[i]:
                        pos[i, 1] -= sign * push
        np.clip(pos[:, 0], half_w, canvas_w - half_w, out=pos[:, 0])
        np.clip(pos[:, 1], half_h, canvas_h - half_h, out=pos[:, 1])
        if not any_ov:
            break
    for i in range(n):
        if not movable[i]:
            continue
        if _has_overlap(i, pos, sizes):
            pos[i] = _spiral_search(i, pos, sizes, movable, canvas_w, canvas_h, gap)
    return pos


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


# ────────────────────────────────────────────────────────────────────────────
# Phase α₂ — Stochastic true-cost subgradient
# ────────────────────────────────────────────────────────────────────────────


def true_cost_subgradient(
    ev: FastEvaluator,
    time_budget_s: float = 60.0,
    eps_frac: float = 0.01,
    lr_frac: float = 0.005,
    momentum: float = 0.9,
    seed: int = 0,
    verbose: bool = True,
):
    """Adam-style stochastic gradient descent on the EXACT proxy cost.

    Innovation 2.  We compute the per-macro gradient numerically using
    finite differences on the bit-exact FastEvaluator (no surrogate gap):

        ∂proxy/∂x_i ≈ ( proxy(x_i + ε) - proxy(x_i - ε) ) / (2 ε)

    For each randomly selected macro we do 4 incremental evaluations
    (±ε in x, ±ε in y); each is ~0.5 ms.  So one stochastic update per
    macro is ~3 ms.  Updates are clamped to keep hard macros non-overlapping
    and inside the canvas.

    This phase polishes Phase α₁'s output by descending the actual cost,
    not a smoothed surrogate — closing the surrogate-truth gap that
    typically limits analytical GP convergence.
    """
    rng = np.random.default_rng(seed)
    eps_x = eps_frac * ev.cw * 0.1   # small perturbation: 0.1% canvas
    eps_y = eps_frac * ev.ch * 0.1
    lr_x = lr_frac * ev.cw           # step size: 0.5% canvas per update
    lr_y = lr_frac * ev.ch
    cur_cost = ev.proxy_cost()["proxy_cost"]
    best_cost = cur_cost
    best_pos = ev.positions.copy()
    # Per-macro momentum buffers
    mom = np.zeros_like(ev.positions)
    t0 = time.time()
    last_log = t0
    accepted = 0
    n_iters = 0
    while time.time() - t0 < time_budget_s:
        i = int(rng.integers(0, ev.n_macros))
        if not ev.movable[i]:
            n_iters += 1
            continue
        is_hard = i < ev.n_hard
        ox, oy = ev.positions[i]
        # Estimate ∂/∂x via central difference
        ev.move_macro(i, ox + eps_x, oy, is_hard=is_hard)
        c_xp = ev.proxy_cost()["proxy_cost"]
        ev.move_macro(i, ox - eps_x, oy, is_hard=is_hard)
        c_xm = ev.proxy_cost()["proxy_cost"]
        gx = (c_xp - c_xm) / (2.0 * eps_x)
        # ∂/∂y
        ev.move_macro(i, ox, oy + eps_y, is_hard=is_hard)
        c_yp = ev.proxy_cost()["proxy_cost"]
        ev.move_macro(i, ox, oy - eps_y, is_hard=is_hard)
        c_ym = ev.proxy_cost()["proxy_cost"]
        gy = (c_yp - c_ym) / (2.0 * eps_y)
        # Restore current
        ev.move_macro(i, ox, oy, is_hard=is_hard)
        # Momentum update + step
        mom[i, 0] = momentum * mom[i, 0] - lr_x * gx
        mom[i, 1] = momentum * mom[i, 1] - lr_y * gy
        # Clamp step magnitude to a single grid cell at most (avoid huge jumps)
        max_step = 1.5 * ev.gw
        sx = float(np.clip(mom[i, 0], -max_step, max_step))
        sy = float(np.clip(mom[i, 1], -max_step, max_step))
        nx = ox + sx
        ny = oy + sy
        nx = max(ev.half[i, 0], min(ev.cw - ev.half[i, 0], nx))
        ny = max(ev.half[i, 1], min(ev.ch - ev.half[i, 1], ny))
        # For hard macros, only commit if no overlap with neighbors
        if is_hard and not _slide_legal(ev, i, nx, ny):
            n_iters += 1
            continue
        ev.move_macro(i, nx, ny, is_hard=is_hard)
        new_cost = ev.proxy_cost()["proxy_cost"]
        if new_cost < cur_cost:
            cur_cost = new_cost
            accepted += 1
            if new_cost < best_cost:
                best_cost = new_cost
                best_pos = ev.positions.copy()
        else:
            # Revert (subgradient sometimes overshoots; treat as a hill-climbing oracle)
            ev.move_macro(i, ox, oy, is_hard=is_hard)
            mom[i] *= 0.5  # damp the momentum after rejection
        n_iters += 1
        if verbose and time.time() - last_log > 20.0:
            print(f"  [α₂] t={time.time()-t0:.0f}s it={n_iters} accepted={accepted} cur={cur_cost:.4f} best={best_cost:.4f}", flush=True)
            last_log = time.time()
    if not np.array_equal(ev.positions, best_pos):
        ev.restore(best_pos)
    return {"proxy_cost": best_cost, "iters": n_iters, "accepted": accepted}


# ────────────────────────────────────────────────────────────────────────────
# Phase 2 — Lin-Kernighan k-opt swaps + grid sweeps
# ────────────────────────────────────────────────────────────────────────────


def _macro_priority(ev: FastEvaluator) -> List[int]:
    cong = ev.v_pin_cong + ev.h_pin_cong
    score = np.zeros(ev.n_hard, dtype=np.float64)
    for m in range(ev.n_hard):
        if not ev.movable[m]:
            score[m] = -np.inf
            continue
        nets = ev._owner_to_nets.get(m, ())
        s = 0.0
        for n in nets:
            r1, c1 = ev._grid_cell(ev._net_xmin[n], ev._net_ymin[n])
            r2, c2 = ev._grid_cell(ev._net_xmax[n], ev._net_ymax[n])
            s += float(cong[r1:r2 + 1, c1:c2 + 1].sum())
        score[m] = s
    return list(np.argsort(-score))


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


def _slide_candidates(ev: FastEvaluator, i: int, n_steps: int = 5, radius_frac: float = 0.08):
    half = ev.half[i]
    cx, cy = ev.positions[i]
    rx = radius_frac * ev.cw
    ry = radius_frac * ev.ch
    out = []
    for dx in np.linspace(-rx, rx, n_steps):
        for dy in np.linspace(-ry, ry, n_steps):
            if dx == 0 and dy == 0:
                continue
            nx = max(half[0], min(ev.cw - half[0], cx + dx))
            ny = max(half[1], min(ev.ch - half[1], cy + dy))
            out.append((nx, ny))
    return out


def lk_swap_pass(
    ev: FastEvaluator,
    macros: List[int],
    chain_depth: int = 4,
    n_neighbors_per_macro: int = 24,
    log_every: Optional[int] = None,
):
    cur_cost = ev.proxy_cost()["proxy_cost"]
    accepted = 0
    for step, i in enumerate(macros):
        if not ev.movable[i]:
            continue
        d = np.linalg.norm(ev.positions[:ev.n_hard] - ev.positions[i], axis=1)
        d[i] = np.inf
        nbrs = np.argsort(d)[:n_neighbors_per_macro]
        best_gain = 0.0
        best_move = None
        for j in nbrs:
            j = int(j)
            if not ev.movable[j] or not _swap_legal(ev, i, j):
                continue
            ev.swap_macros(i, j)
            c = ev.proxy_cost()["proxy_cost"]
            ev.swap_macros(i, j)  # incremental undo
            gain = cur_cost - c
            if gain > best_gain:
                best_gain = gain
                best_move = ("swap", j, c)
        for (nx, ny) in _slide_candidates(ev, i, n_steps=5, radius_frac=0.08):
            if not _slide_legal(ev, i, nx, ny):
                continue
            ox, oy = ev.positions[i]
            ev.move_macro(i, nx, ny, is_hard=True)
            c = ev.proxy_cost()["proxy_cost"]
            ev.move_macro(i, ox, oy, is_hard=True)
            gain = cur_cost - c
            if gain > best_gain:
                best_gain = gain
                best_move = ("slide", (nx, ny), c)
        if best_move is None:
            continue
        if best_move[0] == "swap":
            _, j, new_c = best_move
            ev.swap_macros(i, j)
            cur_cost = new_c
            accepted += 1
            cur_node = j
            for _ in range(chain_depth - 1):
                d2 = np.linalg.norm(ev.positions[:ev.n_hard] - ev.positions[cur_node], axis=1)
                d2[cur_node] = np.inf
                nb = np.argsort(d2)[:n_neighbors_per_macro]
                lbest_gain = 0.0
                lbest = None
                for k in nb:
                    k = int(k)
                    if k == cur_node or not ev.movable[k] or not _swap_legal(ev, cur_node, k):
                        continue
                    ev.swap_macros(cur_node, k)
                    c = ev.proxy_cost()["proxy_cost"]
                    ev.swap_macros(cur_node, k)
                    g = cur_cost - c
                    if g > lbest_gain:
                        lbest_gain = g
                        lbest = (k, c)
                if lbest is None:
                    break
                k, c = lbest
                ev.swap_macros(cur_node, k)
                cur_cost = c
                accepted += 1
                cur_node = k
        else:
            _, (nx, ny), new_c = best_move
            ev.move_macro(i, nx, ny, is_hard=True)
            cur_cost = new_c
            accepted += 1
        if log_every and (step + 1) % log_every == 0:
            print(f"  [LK] step {step+1}/{len(macros)} cost={cur_cost:.4f} (accepted {accepted})", flush=True)
    return cur_cost, accepted


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


def direct_congestion_attack(
    ev: FastEvaluator,
    n_passes: int = 3,
    time_budget_s: float = 60.0,
    sweep_steps: int = 5,
    sweep_radius_frac: float = 0.06,
    n_top_cells: int = 24,
    verbose: bool = True,
):
    """Bit-exact congestion attack using FastEvaluator.

    Each pass:
      1. Compute the current smoothed congestion grid.
      2. Identify the top-`n_top_cells` hottest cells (these dominate the
         top-5% mean that defines congestion cost).
      3. For each hot cell, find every hard macro whose incident-net bbox
         touches the cell.  Score macros by sum of touching-cell heat.
      4. Process macros in priority order: 5×5 grid-sweep around current
         position, accept best move that strictly improves the proxy.
    Each move is evaluated against the bit-exact proxy.
    """
    cur_cost = ev.proxy_cost()["proxy_cost"]
    start_cost = cur_cost
    best_cost = cur_cost
    best_pos = ev.positions.copy()
    t0 = time.time()
    accepted_total = 0
    for pass_idx in range(n_passes):
        if time.time() - t0 > time_budget_s:
            break
        cong = _smoothed_congestion_grid(ev)
        flat = cong.ravel()
        n = flat.size
        k_top = min(n_top_cells, max(1, int(n * 0.05)))
        top_idx = np.argpartition(-flat, k_top - 1)[:k_top]
        # Hot cells as (row, col) with their heat
        hot_cells = [((int(idx) // ev.grid_col), (int(idx) % ev.grid_col), float(flat[idx])) for idx in top_idx]

        # Find candidate hard macros via net bboxes
        macro_priority: Dict[int, float] = {}
        for net_idx in range(ev.n_nets):
            ymin_cell, xmin_cell = ev._grid_cell(ev._net_xmin[net_idx], ev._net_ymin[net_idx])
            ymax_cell, xmax_cell = ev._grid_cell(ev._net_xmax[net_idx], ev._net_ymax[net_idx])
            net_score = 0.0
            for (r, c, h) in hot_cells:
                if ymin_cell <= r <= ymax_cell and xmin_cell <= c <= xmax_cell:
                    net_score += h
            if net_score <= 0:
                continue
            for owner in ev.net_owner[net_idx]:
                owner = int(owner)
                if owner < ev.n_hard and ev.movable[owner]:
                    macro_priority[owner] = macro_priority.get(owner, 0.0) + net_score
        if not macro_priority:
            break
        macros_sorted = sorted(macro_priority.items(), key=lambda x: -x[1])

        pass_accepted = 0
        for m, _score in macros_sorted:
            if time.time() - t0 > time_budget_s:
                break
            is_hard = m < ev.n_hard
            ox, oy = ev.positions[m]
            rx = sweep_radius_frac * ev.cw
            ry = sweep_radius_frac * ev.ch
            best_local = cur_cost
            best_xy = (ox, oy)
            for dx in np.linspace(-rx, rx, sweep_steps):
                for dy in np.linspace(-ry, ry, sweep_steps):
                    nx = max(ev.half[m, 0], min(ev.cw - ev.half[m, 0], ox + dx))
                    ny = max(ev.half[m, 1], min(ev.ch - ev.half[m, 1], oy + dy))
                    if is_hard and not _slide_legal(ev, m, nx, ny):
                        continue
                    ev.move_macro(m, nx, ny, is_hard=is_hard)
                    c_new = ev.proxy_cost()["proxy_cost"]
                    if c_new < best_local:
                        best_local = c_new
                        best_xy = (nx, ny)
                    # Restore for next candidate
                    ev.move_macro(m, ox, oy, is_hard=is_hard)
            if best_local < cur_cost:
                ev.move_macro(m, best_xy[0], best_xy[1], is_hard=is_hard)
                cur_cost = best_local
                pass_accepted += 1
                accepted_total += 1
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_pos = ev.positions.copy()
        if verbose:
            print(f"  [CONG-ATTACK] pass {pass_idx+1}/{n_passes}  hot cells={len(hot_cells)}  candidates={len(macros_sorted)}  accepted={pass_accepted}  cur={cur_cost:.4f}", flush=True)
        if pass_accepted == 0:
            break  # no improvements found, stop early
    if not np.array_equal(ev.positions, best_pos):
        ev.restore(best_pos)
    return {"proxy_cost": best_cost, "improvement": start_cost - best_cost, "accepted": accepted_total}


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
        if verbose and time.time() - last_log > 20.0:
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


# ────────────────────────────────────────────────────────────────────────────
# LKPlacer orchestrator
# ────────────────────────────────────────────────────────────────────────────


def _load_plc(name: str):
    from macro_place.loader import load_benchmark, load_benchmark_from_dir
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    ng45 = {
        "ariane133": "ariane133", "ariane136": "ariane136",
        "nvdla": "nvdla", "mempool_tile": "mempool_tile",
    }
    d = ng45.get(name.replace("_ng45", ""))
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None


class LKPlacer:
    """Five-phase macro placer.  See module docstring."""

    def __init__(
        self,
        seed: int = 42,
        time_budget_s: float = 3000.0,
        # Phase α₁ (electrostatic GP)
        run_gp: bool = True,
        gp_pop_size: int = 4,
        gp_steps: int = 500,
        gp_budget_s: float = 90.0,
        # Phase α₂ (true-cost subgradient)
        run_alpha2: bool = True,
        alpha2_budget_s: float = 60.0,
        # Phase 2 LK
        lk_passes: int = 3,
        lk_neighbors: int = 24,
        lk_chain_depth: int = 4,
        # Phase 2.5 direct congestion attack
        run_cong_attack: bool = False,  # disabled: greedy moves trap LAHC in tighter basin
        cong_attack_passes: int = 3,
        cong_attack_budget_s: float = 60.0,
        # Phase 3 LAHC
        lahc_list_len: int = 100,
        verbose: bool = True,
    ):
        self.seed = seed
        self.time_budget_s = time_budget_s
        self.run_gp = run_gp
        self.gp_pop_size = gp_pop_size
        self.gp_steps = gp_steps
        self.gp_budget_s = gp_budget_s
        self.run_alpha2 = run_alpha2
        self.alpha2_budget_s = alpha2_budget_s
        self.lk_passes = lk_passes
        self.lk_neighbors = lk_neighbors
        self.lk_chain_depth = lk_chain_depth
        self.run_cong_attack = run_cong_attack
        self.cong_attack_passes = cong_attack_passes
        self.cong_attack_budget_s = cong_attack_budget_s
        self.lahc_list_len = lahc_list_len
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[lk_placer] {msg}", flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        from macro_place.objective import compute_proxy_cost

        t0 = time.time()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        plc = _load_plc(benchmark.name)

        # ── Phase α — Focused Electrostatic GP ──
        if self.run_gp:
            try:
                # Import inside to avoid hard dependency at module import time
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "lk_placer_gp",
                    str(Path(__file__).resolve().parent / "gp.py"),
                )
                gp_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gp_mod)
                self._log(f"Phase α: focused electrostatic global placement (pop={self.gp_pop_size}, steps={self.gp_steps}, budget={self.gp_budget_s:.0f}s)")
                gp_positions = gp_mod.run_global_placement(
                    benchmark, plc,
                    pop_size=self.gp_pop_size,
                    n_steps=self.gp_steps,
                    time_budget_s=self.gp_budget_s,
                    seed=self.seed,
                    verbose=self.verbose,
                )
                benchmark.macro_positions = torch.from_numpy(gp_positions).float()
            except Exception as e:
                self._log(f"Phase α: SKIPPED due to exception: {e}")

        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes_np = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
        mov_np = benchmark.get_movable_mask()[:n_hard].cpu().numpy().astype(bool)
        init = benchmark.macro_positions[:n_hard].cpu().numpy().astype(np.float64)

        # ── Phase 0 — legalize ──
        self._log("Phase 0: legalizing hard macros")
        hard_legal = _legalize(init, sizes_np, mov_np, cw, ch)
        pos_full = benchmark.macro_positions.numpy().astype(np.float64).copy()
        pos_full[:n_hard] = hard_legal
        benchmark.macro_positions = torch.from_numpy(pos_full).float()

        # ── Phase 1 — FastEvaluator ──
        self._log("Phase 1: building FastEvaluator")
        ev = FastEvaluator(benchmark, plc)
        c0 = ev.proxy_cost()
        self._log(f"  fast baseline: proxy={c0['proxy_cost']:.4f} wl={c0['wirelength_cost']:.4f} den={c0['density_cost']:.4f} cong={c0['congestion_cost']:.4f}")
        true_c = compute_proxy_cost(torch.from_numpy(ev.positions).float(), benchmark, plc)
        self._log(f"  oracle: {true_c['proxy_cost']:.4f}  overlaps={true_c['overlap_count']}")
        best_true = float(true_c["proxy_cost"]) if true_c["overlap_count"] == 0 else float("inf")
        best_pos = ev.positions.copy() if true_c["overlap_count"] == 0 else None

        # ── Phase α₂ — Stochastic true-cost subgradient ──
        if self.run_alpha2:
            self._log(f"Phase α₂: stochastic true-cost subgradient (budget={self.alpha2_budget_s:.0f}s)")
            out = true_cost_subgradient(
                ev,
                time_budget_s=self.alpha2_budget_s,
                seed=self.seed,
                verbose=self.verbose,
            )
            self._log(f"  α₂: best fast={out['proxy_cost']:.4f}  iters={out['iters']}  accepted={out['accepted']}")
            tc = compute_proxy_cost(torch.from_numpy(ev.positions).float(), benchmark, plc)
            self._log(f"  oracle: {tc['proxy_cost']:.4f}  overlaps={tc['overlap_count']}")
            if tc["overlap_count"] == 0 and tc["proxy_cost"] < best_true:
                best_true = float(tc["proxy_cost"])
                best_pos = ev.positions.copy()
            elif best_pos is not None:
                ev.restore(best_pos)

        # ── Phase 2 — LK ──
        # Reserve at least 30% of total budget (or 60s, whichever is more) for LAHC,
        # not a cap on LK itself.  LK converges naturally after 2-3 passes anyway.
        min_lahc_s = max(60.0, self.time_budget_s * 0.30)
        for p in range(self.lk_passes):
            if time.time() - t0 > self.time_budget_s - min_lahc_s:
                self._log(f"Phase 2 pass {p}: reserving {min_lahc_s:.0f}s for LAHC, skipping further passes")
                break
            self._log(f"Phase 2 pass {p}: macro priority queue")
            order = _macro_priority(ev)
            cur_cost, n_acc = lk_swap_pass(
                ev, order,
                chain_depth=self.lk_chain_depth,
                n_neighbors_per_macro=self.lk_neighbors,
                log_every=max(1, len(order) // 6),
            )
            self._log(f"  pass {p}: fast proxy={cur_cost:.4f} accepted={n_acc}")
            tc = compute_proxy_cost(torch.from_numpy(ev.positions).float(), benchmark, plc)
            self._log(f"  true oracle: {tc['proxy_cost']:.4f}  overlaps={tc['overlap_count']}")
            if tc["overlap_count"] == 0 and tc["proxy_cost"] < best_true:
                best_true = float(tc["proxy_cost"])
                best_pos = ev.positions.copy()

        # ── Phase 2.5 — Direct congestion attack ──
        if self.run_cong_attack and best_pos is not None:
            ev.restore(best_pos)
            self._log(f"Phase 2.5: direct congestion attack (budget={self.cong_attack_budget_s:.0f}s)")
            out = direct_congestion_attack(
                ev,
                n_passes=self.cong_attack_passes,
                time_budget_s=self.cong_attack_budget_s,
                verbose=self.verbose,
            )
            self._log(f"  cong-attack: improvement={out['improvement']:+.4f} accepted={out['accepted']}")
            tc = compute_proxy_cost(torch.from_numpy(ev.positions).float(), benchmark, plc)
            self._log(f"  oracle: {tc['proxy_cost']:.4f}  overlaps={tc['overlap_count']}")
            if tc["overlap_count"] == 0 and tc["proxy_cost"] < best_true:
                best_true = float(tc["proxy_cost"])
                best_pos = ev.positions.copy()
            elif best_pos is not None:
                ev.restore(best_pos)

        # ── Phase 3 — LAHC ──
        if best_pos is not None:
            ev.restore(best_pos)
        remaining = max(60.0, self.time_budget_s - (time.time() - t0))
        self._log(f"Phase 3: LAHC polish, budget={remaining:.0f}s")
        out = lahc_polish(
            ev,
            list_len=self.lahc_list_len,
            time_budget_s=remaining,
            seed=self.seed,
            verbose=self.verbose,
        )
        self._log(f"  LAHC: best={out['proxy_cost']:.4f}  iters={out['iters']}")
        tc = compute_proxy_cost(torch.from_numpy(ev.positions).float(), benchmark, plc)
        if tc["overlap_count"] == 0 and tc["proxy_cost"] < best_true:
            best_true = float(tc["proxy_cost"])
            best_pos = ev.positions.copy()

        self._log(f"DONE  best_true={best_true:.4f}  time={time.time()-t0:.1f}s")
        return torch.from_numpy(best_pos if best_pos is not None else ev.positions).float()
