from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Part:
    uid: int
    pid_raw: str
    # Inflated dimensions used for packing/validation.
    w: float
    h: float
    # Original dimensions (no trim/gap inflation).
    w0: float
    h0: float
    # Optional edge-banding flags by side: Left/Right/Bottom/Top.
    eb_L: int = 0
    eb_R: int = 0
    eb_B: int = 0
    eb_T: int = 0
    # Optional coarse classification: edgeband/non_edgeband/backboard.
    edge_class: str = ""


def _sniff_dialect(sample_text: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample_text, delimiters=",;\t")
    except Exception:
        class _D(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL

        return _D()


def _open_text_with_fallback(path: str) -> Tuple[object, str, csv.Dialect]:
    """Open CSV with common encoding fallbacks and sniff delimiter."""
    encodings = ["utf-8-sig", "utf-8", "gbk", "cp936", "gb18030", "latin1"]
    last_err = None
    p = Path(path)

    for enc in encodings:
        try:
            f = open(p, "r", encoding=enc, newline="")
            sample = f.read(4096)
            f.seek(0)
            dialect = _sniff_dialect(sample)
            return f, enc, dialect
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to open CSV: {path}. Last error: {last_err}")


def _pick_col(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    # Support case-insensitive and trimmed header matching.
    fn_map = {c.strip(): c for c in fieldnames}
    lower_map = {c.strip().lower(): c for c in fieldnames}

    for cand in candidates:
        if cand in fn_map:
            return fn_map[cand]
        lc = cand.lower()
        if lc in lower_map:
            return lower_map[lc]
    return None


def _to_float(x) -> float:
    if x is None:
        return 0.0
    s = str(x).strip()
    if s == "":
        return 0.0
    try:
        return float(s)
    except Exception:
        try:
            return float(s.replace(",", ""))
        except Exception:
            return 0.0


def _to_flag(x) -> int:
    if x is None:
        return 0
    s = str(x).strip().lower()
    if s in {"", "0", "false", "f", "no", "n"}:
        return 0
    if s in {"1", "true", "t", "yes", "y"}:
        return 1
    try:
        return 1 if float(s) != 0.0 else 0
    except Exception:
        return 0


def _norm_edge_class(x) -> str:
    s = str(x or "").strip().lower().replace("-", "_").replace(" ", "_")
    if s in {"edgeband", "edge_band", "banded"}:
        return "edgeband"
    if s in {"non_edgeband", "nonedgeband", "plain", "none", "normal"}:
        return "non_edgeband"
    if s in {"backboard", "back_panel", "backboard_panel"}:
        return "backboard"
    return ""


def _default_edge_flags(edge_class: str) -> Tuple[int, int, int, int]:
    if edge_class == "edgeband":
        return 1, 1, 1, 1
    # non_edgeband/backboard/unknown -> no edge-banding flags by default
    return 0, 0, 0, 0


def _has_value(row: dict, col: Optional[str]) -> bool:
    return bool(col is not None and str(row.get(col, "")).strip() != "")


def read_sample_parts(path: str, trim: float, gap: float, tool_d: float = 6.0) -> List[Part]:
    """Read parts CSV with backward-compatible optional edge attributes.

    Required dimensions:
      - Fleng_mm/Fwidth_mm, or Fleng/Fwidth

    Optional edge metadata:
      - EB_L, EB_R, EB_B, EB_T (edge-level flags)
      - edge_class / part_class (edgeband/non_edgeband/backboard)

    Notes:
      - TRIM is handled by board feasible-domain shrink in packing/validation.
      - gap inflates part dimensions for packing safety margin.
    """
    _ = trim, tool_d  # kept for API compatibility
    f, enc, dialect = _open_text_with_fallback(path)
    try:
        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            raise RuntimeError("CSV header is empty or cannot be detected.")

        fieldnames = [c.strip() for c in reader.fieldnames]

        # Required geometry columns
        col_L = _pick_col(fieldnames, ["Fleng_mm", "Fleng", "length", "L"])
        col_W = _pick_col(fieldnames, ["Fwidth_mm", "Fwidth", "width", "W"])
        if not col_L or not col_W:
            raise RuntimeError(
                f"CSV must contain Fleng/Fwidth or Fleng_mm/Fwidth_mm. Got headers={fieldnames}"
            )

        # Optional id columns
        col_pid = _pick_col(fieldnames, ["pid_raw", "Upi", "upi", "ID", "id", "part_id", "PartID"])
        col_uid = _pick_col(fieldnames, ["uid", "UID", "index", "Upi", "upi"])

        # Optional edge columns
        col_eb_l = _pick_col(fieldnames, ["EB_L", "eb_l", "EdgeBand_L", "edge_l", "edgeband_l"])
        col_eb_r = _pick_col(fieldnames, ["EB_R", "eb_r", "EdgeBand_R", "edge_r", "edgeband_r"])
        col_eb_b = _pick_col(fieldnames, ["EB_B", "eb_b", "EdgeBand_B", "edge_b", "edgeband_b"])
        col_eb_t = _pick_col(fieldnames, ["EB_T", "eb_t", "EdgeBand_T", "edge_t", "edgeband_t"])
        col_edge_class = _pick_col(fieldnames, ["edge_class", "part_class", "edgeClass", "class"])

        parts: List[Part] = []
        uid_auto = 1

        for row in reader:
            # uid
            if col_uid and _has_value(row, col_uid):
                try:
                    uid = int(float(str(row[col_uid]).strip()))
                except Exception:
                    uid = uid_auto
            else:
                uid = uid_auto
            uid_auto += 1

            # pid_raw (for traceability)
            if col_pid and _has_value(row, col_pid):
                pid_raw = str(row[col_pid]).strip()
            else:
                pid_raw = str(uid)

            L0 = _to_float(row.get(col_L))
            W0 = _to_float(row.get(col_W))
            if L0 <= 0 or W0 <= 0:
                continue

            # Base from edge_class (if any), then explicit EB_* overrides.
            edge_class = _norm_edge_class(row.get(col_edge_class)) if col_edge_class else ""
            eb_L, eb_R, eb_B, eb_T = _default_edge_flags(edge_class)
            if col_eb_l is not None:
                eb_L = _to_flag(row.get(col_eb_l))
            if col_eb_r is not None:
                eb_R = _to_flag(row.get(col_eb_r))
            if col_eb_b is not None:
                eb_B = _to_flag(row.get(col_eb_b))
            if col_eb_t is not None:
                eb_T = _to_flag(row.get(col_eb_t))

            inflate = float(gap)
            w = float(L0) + inflate
            h = float(W0) + inflate

            parts.append(
                Part(
                    uid=uid,
                    pid_raw=pid_raw,
                    w=w,
                    h=h,
                    w0=float(L0),
                    h0=float(W0),
                    eb_L=int(eb_L),
                    eb_R=int(eb_R),
                    eb_B=int(eb_B),
                    eb_T=int(eb_T),
                    edge_class=edge_class,
                )
            )

        logger.info(
            "[READ] open ok. encoding=%s delimiter='%s' parts=%d edge_cols=%s",
            enc,
            dialect.delimiter,
            len(parts),
            bool(col_eb_l or col_eb_r or col_eb_b or col_eb_t or col_edge_class),
        )
        return parts
    finally:
        try:
            f.close()
        except Exception:
            pass
