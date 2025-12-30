# cfg.py
from dataclasses import dataclass


@dataclass
class CFG:
    # Case
    CASE_NAME: str = "demo_case"
    SAMPLE_CSV: str = "data/sample_parts.csv"

    # Output
    OUT_ROOT: str = "outputs"

    # Board (mm)
    BOARD_W: float = 2440.0
    BOARD_H: float = 1220.0

    # Process params (mm, mm/min, s)
    TRIM: float = 5.0
    GAP: float = 0.0
    TOOL_D: float = 6.0

    # S1: kerf口径 + lead-in/out（简化实现）
    # - KERF_MODE="none"：保持原口径（几何边界长度）
    # - KERF_MODE="tool_center"：以刀具中心线口径估算切割长度：
    #     L_cut += 2*pi*(TOOL_D/2) * N_stroke
    #   解释：把每条 stroke 的“角点圆弧/偏置增量”简化为一圈圆弧长度。
    KERF_MODE: str = "tool_center"

    # lead-in/out（mm）：每条 stroke 开始/结束的引入/引出长度（简化：直线段）
    LEAD_IN: float = 10.0
    LEAD_OUT: float = 10.0

    FEED_CUT: float = 12000.0   # mm/min
    FEED_AIR: float = 30000.0   # mm/min
    T_LIFT: float = 0.8         # seconds per lift (经验值，可调)

    # Packing
    ALLOW_ROT: bool = True
    RESTARTS: int = 50

    # BaselineB: enable randomized BLF tie-breaking to diversify solutions
    BASELINEB_RAND_PLACE: bool = True
    BASELINEB_RAND_TOPK: int = 3

    # Proposed Route-2 Stage2: per-board fixed-membership re-pack restarts
    INNER_REPACK_RESTARTS: int = 30

    # Stage2 (fixed-membership) scoring weights
    # Primary objective: maximize shared-edge length.
    # Secondary: mild border contact preference.
    # Penalty: discourage too many disjoint cutting strokes (aka trails), which
    #          typically reduces lifts/air moves after routing.
    REPACK_BORDER_W: float = 0.05
    REPACK_TRAIL_PENALTY: float = 0.02

    # Experiment seeds
    SEED0: int = 1000
    N_SEEDS: int = 100

    # Reproducibility
    DUMP_CONFIG: bool = True

    # Routing / metric mode
    # "cpp" 更接近“欧拉化后最短加边”，"trail" 更保守
    CUT_MODE: str = "trail"

    # Shared-edge detection robustness (mm)
    # Coordinates within LINE_SNAP_EPS are treated as the same cutting line.
    # Overlaps shorter than MIN_SHARED_LEN are ignored as numerical artifacts.
    LINE_SNAP_EPS: float = 1e-4
    MIN_SHARED_LEN: float = 1.0
    ND_COORD: int = 6

    # P3: Tabs (bridges) — simplified manufacturing-friendly model
    TAB_ENABLE: bool = True
    TAB_PER_PART: int = 4          # fixed tabs per part
    TAB_LEN: float = 10.0          # mm
    TAB_CORNER_CLEAR: float = 20.0 # mm, avoid corners

    # Plot
    PLOT: bool = True
    PLOT_MAX_BOARDS: int = 6
