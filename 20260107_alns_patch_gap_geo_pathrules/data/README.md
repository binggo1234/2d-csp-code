# Data

This repository ships with **synthetic (non-confidential) demo parts** in `sample_parts.csv` so reviewers can run the code end-to-end.

To reproduce the paper experiments with your real instances, replace `data/sample_parts.csv` with your dataset **using the same schema**:

- `uid` (int): unique part id
- `pid_raw` (str): part name / label
- `Fleng_mm` (float): part length (mm)
- `Fwidth_mm` (float): part width (mm)
- `EB_L, EB_R, EB_B, EB_T` (optional int/bool): edge-banding flag on Left/Right/Bottom/Top edge
- `edge_class` (optional str): coarse class (`edgeband`, `non_edgeband`, `backboard`)

If you use a different header (e.g., `Fleng`, `Fwidth`), the loader attempts to auto-detect it.
If optional edge columns are missing, they default to 0 (non-edgeband), so legacy CSVs remain valid.

Example row with edge attributes:
`1,468,259,469,259.5,,1,1,0,0,1,edgeband`


### If your parts table is an Excel (.xlsx)

Please export it to CSV (UTF-8 recommended) and place it in this folder.
Then run with `--parts_csv data/<your_file>.csv`.
