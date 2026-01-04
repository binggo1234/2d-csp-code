# repro.py
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def set_global_seed(seed: int) -> None:
    """Set global seeds for reproducibility.

    This complements per-module Random(seed) usage. It ensures any accidental
    calls to the module-level random or numpy RNGs remain deterministic.
    """
    import random

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def dump_config(cfg: Any, out_dir: Path, *, extra: Dict[str, Any] | None = None) -> Path:
    """Dump config + environment to a JSON file for reproducibility."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_dataclass(cfg):
        cfg_dict = asdict(cfg)
    else:
        # best effort
        cfg_dict = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper() and not k.startswith('_')}

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    if extra:
        meta.update(extra)

    payload = {
        "cfg": cfg_dict,
        "meta": meta,
    }

    path = out_dir / "config_dump.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
