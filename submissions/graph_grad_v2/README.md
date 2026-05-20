# Graph-Gradient Placer V2 (Phase A: orientation + soft centroid init)

Built on top of `submissions/graph_grad/`. V1 is left untouched as a baseline.

## What's new vs. v1

### 1. Soft-macro centroid initialization (proxy benefit)

For non-anchor seeds (spectral / FD / etc.), V1 placed soft macros at canvas
center + jitter — the gradient had to spend many steps just dragging them to
sensible positions.  V2 initializes each soft macro at the **centroid of its
connected hard macros** (computed from `benchmark.net_pin_nodes`).

Expected impact: **~0.02-0.05** average proxy improvement, faster convergence.

### 2. Klein-4 orientation optimization (proxy + Tier 2 benefit)

Hard macros support orientation flips per the contest rules (`N`, `FN`, `FS`,
`S`).  The pin offsets within each macro flip; macro center and dimensions
stay the same.  Most placers never use this — DREAMPlace doesn't, RePlAce
doesn't, the SA baseline doesn't.

V2 adds a polish pass at the end of `place()`:
- For each hard macro, try the four orientations, pick the one minimizing
  total HPWL across the nets that macro participates in.
- Iterate 3 passes until convergence.
- **Mutate plc pin offsets** so `compute_proxy_cost` reflects the new
  orientations (proxy benefit on Tier 1).
- **Save `orientations_out/<benchmark_name>/orientations.pt`** sidecar for
  Tier 2 OpenROAD flow.

Expected impact:
- **Proxy**: ~0.01-0.04
- **Tier 2**: significantly better routability via improved pin access for
  the OpenROAD router.  This is the differentiator vs. proxy-only-tuned
  submissions.

## Architecture (otherwise unchanged from v1)

1. Build hard+soft graph + pin-level hypergraph tensors.
2. Calibrate RUDY congestion against TILOS at the anchor.
3. Time-budget-driven adaptive restarts (up to `time_budget_s * 0.95`).
4. Each restart: K=512 diverse hard layouts (Fiedler / FD / jittered initial),
   each locked at its own legalized hard; soft optimized jointly via the
   TILOS-faithful proxy-mirror loss `wl + 0.5·dens + 0.5·cong`.
5. **NEW: Phase-A orientation polish** at the end.
6. Safety net: legalized `initial.plc` anchor is always in the candidate pool.

## Usage

```bash
python macro_place/evaluate.py submissions/graph_grad_v2/placer.py -b ibm01
python macro_place/evaluate.py submissions/graph_grad_v2/placer.py --all
```

## Tunables added

| Param | Default | Notes |
|---|---|---|
| `optimize_orientation` | `True` | Run Klein-4 polish pass at the end. |
| `orientation_passes` | `3` | Max iterations through all macros. |

## What V2 doesn't add yet (Phase B)

Hard-macro local swap search via the TILOS-faithful surrogate.  This is the
next step if Phase A alone doesn't land in top-7 territory.
