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
    w: float
    h: float
    w0: float
    h0: float


def _sniff_dialect(sample_text: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample_text, delimiters=",;\t")
    except Exception:
        class _Fallback(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL

        return _Fallback()


def _open_text_with_fallback(path: str) -> Tuple[object, str, csv.Dialect]:
    encodings = ["utf-8-sig", "utf-8", "gbk", "cp936", "gb18030", "latin1"]
    last_err = None
    p = Path(path)

    for enc in encodings:
        try:
            f = open(p, "r", encoding=enc, newline="")
            sample = f.read(4096)
            f.seek(0)
            return f, enc, _sniff_dialect(sample)
        except Exception as exc:
            last_err = exc

    raise RuntimeError(f"Failed to open CSV: {path}. Last error: {last_err}")


def _pick_col(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    exact = {c.strip(): c for c in fieldnames}
    lower = {c.strip().lower(): c for c in fieldnames}

    for cand in candidates:
        if cand in exact:
            return exact[cand]
        key = cand.lower()
        if key in lower:
            return lower[key]
    return None


def _to_float(value) -> float:
    if value is None:
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except Exception:
        try:
            return float(s.replace(",", ""))
        except Exception:
            return 0.0


def _has_value(row: dict, col: Optional[str]) -> bool:
    return bool(col is not None and str(row.get(col, "")).strip())


def read_sample_parts(path: str, trim: float, gap: float, tool_d: float = 6.0) -> List[Part]:
    """Read a parts table and inflate geometry by `gap` for packing only."""
    _ = trim, tool_d
    f, enc, dialect = _open_text_with_fallback(path)
    try:
        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            raise RuntimeError("CSV header is empty or cannot be detected.")

        fieldnames = [c.strip() for c in reader.fieldnames]
        col_l = _pick_col(fieldnames, ["Fleng_mm", "Fleng", "length", "L"])
        col_w = _pick_col(fieldnames, ["Fwidth_mm", "Fwidth", "width", "W"])
        if not col_l or not col_w:
            raise RuntimeError(
                f"CSV must contain Fleng/Fwidth or Fleng_mm/Fwidth_mm. Got headers={fieldnames}"
            )

        col_pid = _pick_col(fieldnames, ["pid_raw", "Upi", "upi", "ID", "id", "part_id", "PartID"])
        col_uid = _pick_col(fieldnames, ["uid", "UID", "index", "Upi", "upi"])

        parts: List[Part] = []
        uid_auto = 1
        inflate = max(0.0, float(gap))

        for row in reader:
            l0 = _to_float(row.get(col_l))
            w0 = _to_float(row.get(col_w))
            if l0 <= 0.0 or w0 <= 0.0:
                continue

            if col_uid and _has_value(row, col_uid):
                try:
                    uid = int(float(str(row[col_uid]).strip()))
                except Exception:
                    uid = uid_auto
            else:
                uid = uid_auto

            if col_pid and _has_value(row, col_pid):
                pid_raw = str(row[col_pid]).strip()
            else:
                pid_raw = str(uid)

            parts.append(
                Part(
                    uid=uid,
                    pid_raw=pid_raw,
                    w=l0 + inflate,
                    h=w0 + inflate,
                    w0=l0,
                    h0=w0,
                )
            )
            uid_auto += 1

        logger.info(
            "[READ] encoding=%s delimiter='%s' parts=%d",
            enc,
            dialect.delimiter,
            len(parts),
        )
        return parts
    finally:
        try:
            f.close()
        except Exception:
            pass
