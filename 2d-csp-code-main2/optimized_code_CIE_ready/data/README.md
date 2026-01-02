# Data

This repository ships with **synthetic (non-confidential) demo parts** in `sample_parts.csv` so reviewers can run the code end-to-end.

To reproduce the paper experiments with your real instances, replace `data/sample_parts.csv` with your dataset **using the same schema**:

- `uid` (int): unique part id
- `pid_raw` (str): part name / label
- `Fleng_mm` (float): part length (mm)
- `Fwidth_mm` (float): part width (mm)

If you use a different header (e.g., `Fleng`, `Fwidth`), the loader attempts to auto-detect it.


### If your parts table is an Excel (.xlsx)

Please export it to CSV (UTF-8 recommended) and place it in this folder.
Then run with `--parts_csv data/<your_file>.csv`.
