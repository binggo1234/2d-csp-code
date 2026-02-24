# Module-to-paper mapping (notes)

These notes help reviewers locate the implementation corresponding to the paper narrative.

- `src/inrp/packer.py`
  - rectangle placement heuristics (baseline A/B, and shared-edge-aware variant)
  - implements feasibility under `TRIM` and part inflation (`trim`, `gap`, optional kerf proxy)
  - propagates optional per-edge attributes (`EB_L/R/B/T`) into placed parts
  - optional maxrects placement (`maxrects_bssf` / `maxrects_baf`)
  - stage-1 sequence search with lightweight LNS destroy-repair neighborhood

- `src/inrp/routing.py`
  - toolpath graph construction and stroke sequencing
  - shared-edge eligibility by side (`EB_L/R/B/T`), incl. edgeband gating and risk stats
  - lift/air-move estimation and sequencing heuristics (`ROUTE_START_POLICY`, `ROUTE_PRIORITY`)
  - adaptive tabs for slender parts (`TAB_ADAPTIVE`, `TAB_SLENDER_*`)
  - union-mode endpoint snapping + axis-collinear merge before trail decomposition
  - local-window NN + 2-opt (objective in time domain) with adaptive trail direction

- `src/inrp/metrics.py`
  - utilization, shared-edge length, cut length (kerf/lead-in/out + dual-pass accounting), time estimate
  - per-board risk outputs (`Risk_shared`, `Risk_slender`)
  - totals add `U_avg_excl_last` for industrial interpretation

- `src/inrp/shared_edges.py` (if present in your extended package)
  - shared-edge extraction and line snapping logic

- `src/inrp/runner.py`
  - experiment loop over seeds and variants
  - outputs CSV tables used in the paper

- `src/inrp/repro.py`
  - global seed control and configuration snapshot dumping
