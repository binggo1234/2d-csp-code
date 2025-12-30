# Reproducibility notes

- Global randomness is controlled by the per-run `seed` passed into the runner.
- The runner calls `inrp.repro.set_global_seed(seed)` which sets:
  - `random` seed
  - `numpy.random` seed

To reproduce a specific table/figure in the paper, use the same:
- `SEED0` (starting seed)
- `N_SEEDS` (number of runs)
- configuration parameters (board size, TRIM, TOOL_D, etc.)

A configuration snapshot is dumped to `outputs/<case>/config_dump.json`.
