# Module-to-paper mapping (notes)

These notes help reviewers locate the implementation corresponding to the paper narrative.

- `src/inrp/packer.py`
  - rectangle placement heuristics (baseline A/B, and shared-edge-aware variant)
  - implements feasibility under `TRIM` and part inflation (`trim`, `gap`, optional kerf proxy)

- `src/inrp/routing.py`
  - toolpath graph construction and stroke sequencing
  - lift/air-move estimation and simple sequencing heuristics

- `src/inrp/metrics.py`
  - utilization, shared-edge length, cut length (kerf/lead-in/out accounting), time estimate

- `src/inrp/shared_edges.py` (if present in your extended package)
  - shared-edge extraction and line snapping logic

- `src/inrp/runner.py`
  - experiment loop over seeds and variants
  - outputs CSV tables used in the paper

- `src/inrp/repro.py`
  - global seed control and configuration snapshot dumping
