# Graph-Gradient Placer

A GPU-batched **analytical** macro placer with a graph-aware multi-start
population, **joint hard + soft co-optimization**, and a density-pressure
annealing schedule.  Designed to reach sub-1 proxy cost on the IBM benchmarks.

## Why analytical, not discrete search

`initial.plc` has near-optimal wirelength on every IBM benchmark — the entire
path to sub-1 is reducing **density** and **congestion**, which means
*rearranging* macros to spread better.  A smooth gradient-based objective
with an explicit density-overshoot term does this directly; a discrete GA
(see `../graph_evo/`) tends to optimize wirelength and stall.

## Three innovations

1. **Joint hard + soft optimization.**  Soft macros (standard-cell clusters)
   are free position variables in the *same* Adam optimizer, with no overlap
   constraint (allowed to overlap by problem definition).  They contribute to
   wirelength as net endpoints and to density via their area in grid cells.
   The SA baseline does this sequentially between hard-macro moves; we do it
   jointly on GPU, which is what lets the density-pressure schedule reduce
   density without destroying wirelength.

2. **Graph-based diverse seeds.**  Population spans:
   - legalized `initial.plc` (anchor),
   - Fiedler / spectral embedding of the clique-expanded pair graph (with
     random rotations),
   - force-directed random starts (varying iter counts),
   - jittered `initial.plc` at varying scales.

   All K=32 evolve in parallel as one `[K, N, 2]` tensor on the GPU.

3. **Density-pressure annealing.**
   - γ (WAHPWL smoothing) :  2.0 → 0.1
   - α_density            :  0.01 → 10
   - α_overlap            :  1 → 2000
   - learning rate        :  0.1 → 0.005

   Starts permissive (let wirelength drop while exploring) and ends strict
   (force density and overlap both down).

## Differentiable surrogate (PyTorch autograd, all GPU)

| Term | Formula | Mirrors proxy component |
|---|---|---|
| WAHPWL | `γ · (LSE(x/γ) - LSE(-x/γ))` per net, per axis | `wirelength_cost` (smooth max - min ≈ HPWL as γ → 0) |
| Density top-K | Bilinear macro spread into 32×32 grid, then sum-of-squared top-10% cells | `density_cost` (proxy uses top-10% density) |
| **RUDY congestion** | Per-net HPWL spread uniformly over its bbox; per-cell demand `max(H, V)`; sum-of-squared top-5% cells | `congestion_cost` (proxy uses top-5% with smoothing) |
| Soft overlap | Sum of pairwise AABB intersection area (hard-only) | Hard constraint via gradient + final legalize |
| Anchor reg. | `‖pop[k] - initial.plc‖²` on seeds 0–1, annealed away over epochs | Preserves the strong hand-crafted start while allowing exploration |
| Canvas | `clamp_` after each Adam step | Hard constraint |

## Pipeline

1. Build pin-level hypergraph + clique-expanded pair graph.
2. Generate K=32 diverse seeds (legalize all hard-macro parts).
3. Run **8 epochs × 500 Adam steps** with annealing schedule.
4. Mid-epoch (halfway): re-legalize hard macros on top-K' candidates by surrogate.
5. Each epoch-end: cheap true-cost gate on the surrogate-best.
6. Final sweep: legalize + true-cost on top-K' (K/2) candidates; pick best.
7. Safety net: compare final best against legalized `initial.plc` anchor; keep whichever the true `plc.get_cost()` says is better.

## Usage

```bash
# Single benchmark
uv run evaluate submissions/graph_grad/placer.py -b ibm01

# All 17 IBM benchmarks
uv run evaluate submissions/graph_grad/placer.py --all

# NG45 (Tier 2)
uv run evaluate submissions/graph_grad/placer.py --ng45
```

## Tunables (`GraphGradPlacer.__init__`)

| Param              | Default | Notes |
|--------------------|---------|-------|
| `seed`             | 42      | RNG seed. |
| `pop_size`         | 32      | GPU population size.  RTX 6000 has memory for 64+. |
| `n_epochs`         | 8       | Annealing schedule has 8 steps. |
| `steps_per_epoch`  | 500     | Adam steps per epoch.  4000 total ≈ DREAMPlace-class convergence. |
| `grid_res`         | 32      | Density grid resolution (matches proxy's grid). |
| `time_budget_s`    | 3000    | 50 min/bench cap.  Leaves headroom under the 1 h limit. |
| `verbose`          | False   | Print per-epoch surrogate + true-cost progress. |

## Expected behaviour

CPU smoke-tests (small pop, few epochs) confirm:
- The gradient surrogate decreases monotonically.
- The anchor-regularization keeps anchored seeds near `initial.plc` (epoch-1
  true cost: 1.0385 vs anchor 1.0380 on ibm01 — within 0.05%).
- The safety net falls back to the anchor whenever the gradient result is
  worse, so the placer is **never worse than `legalized initial.plc`**.

For sub-1, the placer needs full GPU runs (8 epochs × 500 steps × K=32) so:
1. The anneal schedule has enough Adam steps to land in a low-density / low-
   congestion basin (not just slightly perturb the anchor).
2. The K=32 unanchored / weakly-anchored seeds get a real chance to find
   *better* basins than `initial.plc` (which is the only path to below-1
   on the harder ibm benchmarks).

**Tuning knobs that matter for sub-1**:
- Increase `steps_per_epoch` if the surrogate keeps moving at epoch end
  (more time to converge).
- Increase `pop_size` for diversity if many seeds converge to the same basin.
- Lower `alpha_anchor` start (e.g. `30 → 0.5`) if the anchored seeds aren't
  exploring enough; raise it if they're drifting and final cost increases.
- The relative weights of `α_dens` / `α_cong` may need per-benchmark tuning
  — congestion-bound benchmarks (large nets) benefit from higher
  `α_cong`; density-bound (high utilisation, ibm17/ibm18) benefit from
  higher `α_dens`.

## How this is "graphical evolutionary"

- **Graph**: clique-expanded pair graph powers spectral seeding and the
  pin-level hypergraph drives the differentiable WAHPWL.
- **Evolutionary**: K=32 diverse seeds evolve in parallel; selection at end-
  of-epoch and end-of-run by true cost; safety-net is an explicit fitness
  comparison against the anchor.
- **Gradient**: replaces discrete mutation with continuous Adam descent on
  the smooth differentiable surrogate, which is what makes sub-1 reachable.

## Running on the contest hardware

The placer auto-selects `cuda` when available.  On the RTX 6000 Ada:
- ~2-5 min per benchmark expected
- Total for 17 IBM benchmarks: ~1-1.5 h
- VRAM usage: ~2-4 GB at K=32 (plenty of headroom)
