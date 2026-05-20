# Graph-Evolutionary Macro Placer

GPU-accelerated genetic algorithm for hard-macro placement, anchored on the
hand-crafted `initial.plc` layout and using graph-aware operators
(clique-expanded hypergraph HPWL, spectral seeding, Fiedler-cluster mutations).

## Pipeline

1. **Hypergraph + pair-graph extraction.**  Build a flat pin-level tensor
   layout (one row per pin endpoint, with owner-index and offset) so we can
   compute full Manhattan HPWL across a *population* of placements in a single
   `scatter_reduce_` pass.  Independently build a clique-expanded hard-macro
   pair graph (`weight = 1/(|net|-1)`) for spectral seeding and the SA polish.

2. **Minimum-displacement legalization.**  Two-stage:
   - `_legalize_spread`: vectorized push-apart on the pairwise overlap matrix.
     Pushes the *smaller-overlap axis* of each strictly-overlapping pair by
     exactly the overlap amount plus a tiny buffer.  Verifies legality in
     **float32** (the precision the contest validator uses), not just float64.
   - `_legalize_spiral`: fallback when spread can't converge.  Places macros
     in area-descending order on a fine spiral around their desired centres.

3. **Population seeding.**  Pop = 16:
   - `pop[0]` = legalized `initial.plc` (elite anchor — the hand-crafted layout
     is competitive on its own once legalized).
   - `pop[1]` = legalized Fiedler / spectral embedding of the pair graph.
   - `pop[2..P]` = tight Gaussian perturbations of the elite.

4. **GA loop (≈50% of time budget).**
   - Vectorized surrogate `Surrogate.evaluate(pop)`: pin-level HPWL +
     pairwise-overlap area + coarse density-cell variance, all on GPU.
   - Tournament selection, geographic (axis-cut) crossover, per-macro Gaussian
     mutation, occasional **Fiedler-cluster shifts** (graph-aware
     diversification — moves an entire spectral community together).
   - True-cost gate: every 10 generations, legalize the surrogate-best and
     compute the real `plc.get_cost()` — only update the tracked best when the
     true proxy improves.  Prevents the GA from optimising a misleading
     surrogate.

5. **SA polish (≈40% of time budget).**  Numpy-only simulated annealing on the
   pair-graph edge HPWL.  Single-macro Gaussian moves with overlap rejection.
   The final result is only adopted if the true cost actually drops.

6. **Final legalize + soft-macro boundary clip.**  Re-legalize; clip any soft
   macro that initial.plc placed slightly outside the canvas back inside (some
   benchmarks have float-precision boundary violations in the seed data).

7. **Tier-2 hook (optional).**  `soft_fd=True` runs `plc.optimize_stdcells` on
   soft macros after hard placement is locked.  Disabled by default because
   the FD step is slow (>20 min on ibm17 with `num_steps=[40,40,40]`).
   Enable for NG45 / Tier-2 runs where the hand-crafted soft seed is absent.

## Tunable parameters (`GraphEvoPlacer.__init__`)

| Param              | Default | Notes                                                                 |
|--------------------|---------|-----------------------------------------------------------------------|
| `seed`             | 42      | RNG seed for reproducibility.                                         |
| `time_budget_s`    | 2400    | Per-benchmark cap (40 min — leaves headroom under the 1 h contest limit). |
| `pop_size`         | 16      | GPU population.  Increase to 32+ on the RTX 6000 (plenty of memory). |
| `n_clusters`       | 8       | Fiedler communities for cluster-shift mutations.                      |
| `legal_gap`        | 0.005   | Min macro separation (μm).  Larger values survive float32 rounding more safely; smaller values preserve initial.plc density better. |
| `ga_radius_frac`   | 0.005   | Mutation σ as a fraction of `min(canvas_w, canvas_h)`.  Tight is good — anchor preservation is the main win. |
| `soft_fd`          | False   | Enable for NG45 / Tier-2 only.                                        |
| `verbose`          | False   | Print per-epoch surrogate + true-cost progress.                       |

## Usage

```bash
# Single benchmark
uv run evaluate submissions/graph_evo/placer.py -b ibm01

# All 17 IBM benchmarks
uv run evaluate submissions/graph_evo/placer.py --all

# NG45 (Tier 2 sanity check) — consider tweaking the placer to set soft_fd=True
uv run evaluate submissions/graph_evo/placer.py --ng45
```

## Where to look for improvements

The current surrogate captures HPWL but only weakly captures the true density
(top-10%) and congestion (top-5%) cost components.  Future work:
- Replace the `density_var` term with a **top-K density** penalty.
- Add a vectorized RUDY estimate for congestion.
- Tune `ga_radius_frac` per-benchmark based on canvas density.
- Use an island model that occasionally re-seeds from `_spectral_seed` to
  escape the initial.plc local minimum on benchmarks where the hand-crafted
  layout is poor.
