from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)


@dataclass
class Part:
    uid: int
    pid_raw: str
    w: float     # inflate 后（用于排样/校验）
    h: float
    w0: float    # 原始尺寸（不含 trim/gap）
    h0: float


def _sniff_dialect(sample_text: str) -> csv.Dialect:
    """尽量自动识别分隔符（逗号/分号/制表符）。"""
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
    """
    依次尝试多种编码打开 CSV：
      utf-8-sig -> utf-8 -> gbk/cp936 -> gb18030 -> latin1(兜底)
    并用前几 KB 来 sniff 分隔符。
    """
    encodings = ["utf-8-sig", "utf-8", "gbk", "cp936", "gb18030", "latin1"]
    last_err = None
    p = Path(path)

    for enc in encodings:
        try:
            f = open(p, "r", encoding=enc, newline="")
            # 读少量内容用于 sniff，再回到开头
            sample = f.read(4096)
            f.seek(0)
            dialect = _sniff_dialect(sample)
            return f, enc, dialect
        except Exception as e:
            last_err = e

    raise RuntimeError(f"无法打开 CSV：{path}，尝试编码失败。最后错误：{last_err}")


def _pick_col(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    # 支持大小写、前后空格
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
        # 有些表格会出现类似 "1,234.5"
        try:
            return float(s.replace(",", ""))
        except Exception:
            return 0.0


def read_sample_parts(path: str, trim: float, gap: float, tool_d: float = 6.0) -> List[Part]:
    """
    读取 sample_parts.csv：
      - 兼容表头：
        1) uid, pid_raw, Fleng_mm, Fwidth_mm
        2) Fleng, Fwidth (+ 可选 Upi/ID 作为 pid_raw)
      - 自动识别编码（utf-8/gbk等）与分隔符
      - 生成 Part 列表，并计算 inflate 尺寸：
          inflate = 原始 + 2*trim + gap
        （gap=0 表示不额外留缝；若你之后要把刀缝/刀径反映到排样间距，可改这里）
    """
    f, enc, dialect = _open_text_with_fallback(path)
    try:
        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            raise RuntimeError("CSV 表头为空/无法识别。")

        fieldnames = [c.strip() for c in reader.fieldnames]

        # 识别列：长度/宽度
        col_L = _pick_col(fieldnames, ["Fleng_mm", "Fleng", "length", "L"])
        col_W = _pick_col(fieldnames, ["Fwidth_mm", "Fwidth", "width", "W"])
        if not col_L or not col_W:
            raise RuntimeError(f"sample CSV 必须包含 Fleng/Fwidth（或 Fleng_mm/Fwidth_mm），当前表头={fieldnames}")

        # 识别 pid_raw：优先 pid_raw，其次 Upi/ID/uid
        col_pid = _pick_col(fieldnames, ["pid_raw", "Upi", "upi", "ID", "id"])
        col_uid = _pick_col(fieldnames, ["uid", "UID", "index"])

        parts: List[Part] = []
        uid_auto = 1

        for row in reader:
            # uid
            if col_uid and str(row.get(col_uid, "")).strip() != "":
                try:
                    uid = int(float(str(row[col_uid]).strip()))
                except Exception:
                    uid = uid_auto
            else:
                uid = uid_auto
            uid_auto += 1

            # pid_raw（用于打印/追踪；允许重复）
            if col_pid and str(row.get(col_pid, "")).strip() != "":
                pid_raw = str(row[col_pid]).strip()
            else:
                pid_raw = str(uid)

            L0 = _to_float(row.get(col_L))
            W0 = _to_float(row.get(col_W))

            # 基本过滤：非法/为0 跳过
            if L0 <= 0 or W0 <= 0:
                continue

            # inflate：排样用（包含 trim+gap）
            inflate = 2.0 * float(trim) + float(gap)
            w = float(L0) + inflate
            h = float(W0) + inflate

            parts.append(Part(
                uid=uid,
                pid_raw=pid_raw,
                w=w, h=h,
                w0=float(L0), h0=float(W0),
            ))

        logger.info(f"[READ] open ok. encoding={enc} delimiter='{dialect.delimiter}' parts={len(parts)}")
        return parts

    finally:
        try:
            f.close()
        except Exception:
            pass
