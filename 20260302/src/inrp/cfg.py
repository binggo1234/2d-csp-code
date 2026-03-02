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

    # Binary spacing constraint (CNC milling feasibility)
    # If True, any two parts must satisfy: delta_ij == 0 (touch/share) OR delta_ij >= SAFE_GAP.
    # We recommend SAFE_GAP = TOOL_D for panel furniture CNC routing.
    BINARY_SPACING: bool = True
    SAFE_GAP: float = TOOL_D    # mm, default equals TOOL_D
    TOUCH_TOL: float = 1e-6    # mm, treat distance <= TOUCH_TOL as 0

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
    BASELINE_PLACE_MODE: str = "blf"   # blf / blf_rand / maxrects_bssf / maxrects_baf

    # Proposed: Stage1 global sequence search
    STAGE1_METHOD: str = "sa"
    STAGE1_SA_ITERS: int = 2000
    STAGE1_PLACE_MODE: str = "maxrects_baf"
    STAGE1_LNS_ENABLE: bool = True
    STAGE1_LNS_PROB: float = 0.15
    STAGE1_LNS_DESTROY_FRAC: float = 0.08

    # Stage1 ALNS (optional, set STAGE1_METHOD="alns")
    STAGE1_ALNS_ITERS: int = 600
    # Fitness weights: w1 utilization dominates; w2 shared-edge ratio; w3 air-move ratio
    ALNS_W1: float = 1.0
    ALNS_W2: float = 0.25
    ALNS_W3: float = 0.10
    # Operator/temperature knobs
    ALNS_DESTROY_FRAC: float = 0.10
    ALNS_DESTROY_MIN: int = 6
    ALNS_DESTROY_MAX: int = 120
    ALNS_SEGMENT_LEN: int = 50
    ALNS_REACTION: float = 0.2
    ALNS_T0: float = 0.05
    ALNS_ALPHA: float = 0.995

    # BaselineB: enable randomized BLF tie-breaking to diversify solutions
    BASELINEB_RAND_PLACE: bool = True
    BASELINEB_RAND_TOPK: int = 3

    # Proposed Route-2 Stage2: per-board fixed-membership re-pack restarts
    INNER_REPACK_RESTARTS: int = 30

    # Stage2 optimizer (paper-friendly)
    # - "restarts": random restarts only (legacy)
    # - "sa"      : simulated annealing on insertion order + BLF-CCF placement
    INNER_REPACK_METHOD: str = "sa"

    # Simulated Annealing settings (only used if INNER_REPACK_METHOD=="sa")
    # NOTE: The objective is dominated by shared-edge length (mm), so temperature
    # is also defined in mm-units.
    INNER_REPACK_SA_ITERS: int = 600
    INNER_REPACK_SA_T0: float = 5000.0
    INNER_REPACK_SA_ALPHA: float = 0.995
    INNER_REPACK_SA_MAX_STARTS: int = 5
    # Early stop if no global improvement for this many iterations (per SA start)
    INNER_REPACK_SA_PATIENCE: int = 200
    # Optional: automatically scale down SA budget for very large industrial boards
    INNER_REPACK_SA_AUTO: bool = True
    INNER_REPACK_SA_BASE_N: int = 200
    INNER_REPACK_SA_MIN_ITERS: int = 10

    # Stage1 sequence search
    STAGE1_METHOD: str = "sa"  # "sa" or "baselineB"
    STAGE1_SA_ITERS: int = 2000

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

    # Parallel
    # - 1 : serial (default; safest on Windows)
    # - >1: parallel across seeds using ProcessPoolExecutor
    N_JOBS: int = 1

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
    SHARED_MIN_LEN_EDGEBAND: float = 50.0
    SHARED_ENABLE_EDGEBAND: bool = False
    # Limit one-shot long shared seam cuts: split shared cuts longer than this (mm)
    # and keep tiny hold-bridges to reduce sudden vacuum release.
    SHARED_MAX_CONTINUOUS_CUT: float = 2000.0
    SHARED_HOLD_BRIDGE_LEN: float = 8.0
    SHARED_REQUIRE_DUAL_ON_EDGEBAND: bool = False
    SHARED_PASS_MODE: str = "edgeband_only"  # none/global/edgeband_only
    SHARED_PASS_MULT: float = 1.0
    SHARED_RISK_W: float = 0.0
    ND_COORD: int = 6

    # Routing heuristics (manufacturing-center style)
    ROUTE_START_POLICY: str = "near_board_edge"  # none/near_board_edge
    ROUTE_PRIORITY: str = "small_first"          # none/small_first/regional_nn
    ROUTE_CCW: bool = True
    ROUTE_LOCAL_WINDOW: int = 10
    ROUTE_LOCAL_BACKTRACK: int = 2
    # Absolute threshold for strict small-part-first priority.
    ROUTE_SMALL_AREA_FIRST_M2: float = 0.08
    # Penalty (mm-equivalent) for choosing trail entry at high-degree junctions.
    ROUTE_ENTRY_JUNCTION_PENALTY_MM: float = 80.0

    # --- Path planning rules (shop-floor feedback) ---
    # Hierarchical sorting (L1 inner, L2 small->large, L3 large outlines last)
    ROUTE_HIERARCHICAL: bool = True
    ROUTE_LARGE_FRAC: float = 0.20   # last 20% (by part area) treated as Level-3
    # Lead-in / ramping (simplified geometric model)
    ROUTE_RAMP_ENABLE: bool = True
    ROUTE_RAMP_LEN: float = 5.0
    # Anti-shifting (two-pass finish for small slender parts)
    ANTI_SHIFT_ENABLE: bool = True
    ANTI_SHIFT_AREA_M2: float = 0.02
    ANTI_SHIFT_AR: float = 5.0
    ANTI_SHIFT_TWO_PASS: bool = True

    # P3: Tabs (bridges) — simplified manufacturing-friendly model
    TAB_ENABLE: bool = True
    TAB_PER_PART: int = 2          # fixed tabs per part
    TAB_LEN: float = 10.0          # mm
    TAB_CORNER_CLEAR: float = 20.0 # mm, avoid corners
    TAB_SKIP_TRIM_EDGE: bool = True
    TAB_ADAPTIVE: bool = True
    TAB_SLENDER_RATIO: float = 8.0
    TAB_SLENDER_EXTRA: int = 2
    TAB_SMALL_AREA_EXTRA: int = 1

    # Plot
    PLOT: bool = True
    PLOT_MAX_BOARDS: int = 6
    # Additional plot switches
    PLOT_TOOLPATH: bool = True
    PLOT_SHOW_IDS: bool = False
    # Visual-only shrink margin (mm) so touching parts show a visible gap in layout plots
    PLOT_VIS_MARGIN: float = 2.0
