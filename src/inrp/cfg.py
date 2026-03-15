from dataclasses import dataclass


@dataclass
class CFG:
    CASE_NAME: str = "demo_case"
    SAMPLE_CSV: str = "data/sample_parts.csv"
    OUT_ROOT: str = "outputs"

    BOARD_W: float = 2440.0
    BOARD_H: float = 1220.0

    TRIM: float = 5.0
    GAP: float = 0.0
    TOOL_D: float = 6.0

    ALLOW_ROT: bool = True

    # Nesting feasibility. If enabled, two parts may either touch or stay at
    # least SAFE_GAP apart.
    BINARY_SPACING: bool = True
    SAFE_GAP: float = 6.0
    TOUCH_TOL: float = 1e-6

    # Decoder geometry mode
    BASELINE_PLACE_MODE: str = "maxrects_baf"
    BASELINE_RESTARTS: int = 24
    SA_ITERS: int = 2500
    SA_T0: float = 0.03
    SA_ALPHA: float = 0.995
    SA_BUDGET_FRAC: float = 0.05
    LNS_ENABLE: bool = True
    LNS_PROB: float = 0.15
    LNS_DESTROY_FRAC: float = 0.08
    RAND_TOPK: int = 3
    VARIANTS: tuple = ("baseline_greedy", "rh_mcts_ref", "rh_mcts_main")

    # Receding-horizon MCTS
    UTIL_GAMMA: float = 2.0
    SCALAR_BIG_M: float = 1000.0
    STATE_SIG_ND: int = 3
    MCTS_N_SIM: int = 3
    MCTS_PUCT_C: float = 1.3
    PW_K0: int = 8
    PW_KMAX: int = 48
    PW_ALPHA: float = 0.5
    DYN_POOL_MULT: int = 2
    DYN_POOL_MIN: int = 12
    DYN_POOL_MAX: int = 48
    DYN_TAIL_SAMPLES: int = 4
    PRIOR_AREA_W: float = 0.30
    PRIOR_CAVITY_W: float = 0.35
    PRIOR_FRAGMENT_W: float = 0.20
    PRIOR_ESCAPE_W: float = 0.15
    DISABLE_GEOM_PRIOR: bool = False
    ROOT_DIRICHLET_ENABLE: bool = False
    ROOT_DIRICHLET_ALPHA: float = 0.30
    ROOT_DIRICHLET_EPS: float = 0.25
    ROOT_DIRICHLET_MAX_DEPTH: int = -1
    ROLLOUT_TOPK: int = 12
    ROLLOUT_RCL_TOPR: int = 4
    ROLLOUT_RCL_WEIGHTS: tuple = (0.60, 0.30, 0.10)
    ROLLOUT_GREEDY_FRAC: float = 0.0
    ROLLOUT_DETERMINISTIC_TAIL: int = 8
    ROLLOUT_PRIOR_W: float = 0.25
    ROLLOUT_UTIL_W: float = 0.30
    ROLLOUT_CAVITY_W: float = 0.20
    ROLLOUT_FRAGMENT_W: float = 0.15
    ROLLOUT_EXISTING_W: float = 0.10
    WARMSTART_VMAX: int = 64
    COMMIT_MIN_STEPS: int = 1
    COMMIT_MAX_STEPS: int = 3
    COMMIT_VISIT_RATIO: float = 1.8
    COMMIT_MIN_ROOT_VISITS: int = 12
    COMMIT_MIN_CHILD_VISITS: int = 3
    COMMIT_CONFIRM_PASSES: int = 2
    COMMIT_FORCE_ENABLE: bool = False
    COMMIT_TIMEOUT_FORCE: bool = False
    COMMIT_MAX_ROUND_SIMS: int = 0
    SOLVER_TIME_LIMIT_S: float = 3600.0
    POST_MCTS_RESERVE_FRAC: float = 0.05
    POST_MCTS_RESERVE_MIN_S: float = 0.0
    POST_MCTS_RESERVE_MAX_S: float = 45.0
    SA_TIME_CAP_S: float = 15.0
    ELITE_ARCHIVE_K: int = 8
    ELITE_TAIL_LEN: int = 16
    REPAIR_ELITE_TOPK: int = 4
    REPAIR_TAIL_BOARDS: int = 3
    REPAIR_BLOCKER_WINDOW: int = 40
    REPAIR_BLOCKER_TOPK: int = 6
    REPAIR_BEAM_WIDTH: int = 10
    REPAIR_NODE_LIMIT: int = 240
    REPAIR_ACTIONS_PER_STATE: int = 12
    REPAIR_PASS_TIME_SLICE_S: float = 8.0
    REGION_PROPOSAL_TOPK: int = 3
    REGION_START_BACKOFFS: tuple = (8, 16, 24)
    REGION_REBUILD_TOPK: int = 4
    REGION_REBUILD_BEAM_WIDTH: int = 12
    REGION_REBUILD_NODE_LIMIT: int = 320
    REGION_REBUILD_ACTIONS_PER_STATE: int = 14
    REGION_REBUILD_TIME_SLICE_S: float = 8.0
    REGION_DIVERSITY_BUCKET: int = 12
    POST_REPAIR_ENABLE: bool = True
    POST_REPAIR_TIME_LIMIT_S: float = 1800.0
    POST_REPAIR_WINDOWS_TOPK: int = 3
    POST_REPAIR_BLOCKER_BOARDS: int = 2
    POST_REPAIR_BLOCKER_TOPK: int = 12
    POST_REPAIR_PATTERNS_PER_MODE: int = 24
    POST_REPAIR_PATTERN_CAP: int = 300
    POST_REPAIR_PATTERN_MIN_UTIL: float = 0.85
    POST_REPAIR_PATTERN_STRONG_UTIL: float = 0.90
    POST_REPAIR_RANDOM_ORDERS: int = 32
    POST_REPAIR_RANDOM_TOPK: int = 4
    POST_REPAIR_EVICT_ENABLE: bool = True
    POST_REPAIR_EVICT_HOST_BOARDS: int = 3
    POST_REPAIR_EVICT_CANDIDATES: int = 6
    POST_REPAIR_EVICT_TOPK: int = 4
    POST_REPAIR_TAIL_STRIKE_ENABLE: bool = False
    POST_REPAIR_TAIL_STRIKE_WIDTHS: tuple = (2, 3, 4)
    POST_REPAIR_TAIL_STRIKE_TOPK: int = 4
    POST_REPAIR_TAIL_STRIKE_PATTERN_MULT: int = 2
    POST_REPAIR_DYNAMIC_EXPAND_ENABLE: bool = False
    POST_REPAIR_DYNAMIC_WINDOW_WIDTHS: tuple = (3, 5, 8, 10)
    POST_REPAIR_DYNAMIC_FAST_SOLVE_S: float = 2.0
    POST_REPAIR_INITIAL_WINDOW_WIDTH: int = 4
    POST_REPAIR_ALNS_ENABLE: bool = False
    POST_REPAIR_ALNS_ALPHA: float = 0.20
    POST_REPAIR_CP_NUM_WORKERS: int = 8
    POST_REPAIR_TARGET_DROP: int = 1
    POST_REPAIR_USE_BSSF: bool = True
    GLOBAL_PATTERN_MASTER_ENABLE: bool = False
    GLOBAL_PATTERN_INCLUDE_BASELINE: bool = True
    GLOBAL_LAYOUT_POOL_LIMIT: int = 96
    GLOBAL_PATTERN_LAYOUT_TOPK: int = 24
    GLOBAL_PATTERN_CAP: int = 6000
    GLOBAL_PATTERN_MIN_UTIL: float = 0.80
    GLOBAL_PATTERN_STRONG_UTIL: float = 0.90
    GLOBAL_PATTERN_TIME_LIMIT_S: float = 180.0
    GLOBAL_PATTERN_USE_BSSF: bool = True
    GLOBAL_PATTERN_MEMORY_ENABLE: bool = False
    GLOBAL_PATTERN_MEMORY_MIN_UTIL: float = 0.92
    GLOBAL_PATTERN_MEMORY_MASTER_MIN_UTIL: float = 0.95
    GLOBAL_PATTERN_MEMORY_LIMIT: int = 20000
    GLOBAL_PATTERN_MEMORY_TOPK: int = 4000
    GLOBAL_PATTERN_MEMORY_FOCUS_TAIL_BOARDS: int = 1
    MACRO_ACTION_ENABLE: bool = False
    MACRO_ACTION_TOPK: int = 12
    MACRO_MIN_PARTS: int = 2
    MACRO_MAX_PARTS: int = 6
    MACRO_PATTERN_TOPK: int = 4
    MACRO_ANCHOR_TOPK: int = 4
    MACRO_TAIL_TOPK: int = 3
    MACRO_SINGLE_TOPK: int = 3
    MACRO_PATTERN_UTIL_MIN: float = 0.92
    MACRO_MEMORY_UTIL_MIN: float = 0.95
    MACRO_STAGE_EARLY_RATIO: float = 0.35
    MACRO_STAGE_LATE_RATIO: float = 0.80
    MACRO_TAIL_TRIGGER_UTIL: float = 0.75
    MACRO_SPARSE_TRIGGER_UTIL: float = 0.60

    # Spill-point memetic search
    LOCAL_SEARCH_ENABLE: bool = True
    LOCAL_MAX_ITERS: int = 24
    LOCAL_FAIL_LIMIT: int = 6
    LOCAL_CRITICAL_TOPM: int = 8
    LOCAL_PULL_DELTA_1: int = 3
    LOCAL_PULL_DELTA_2: int = 5
    LOCAL_PULL_DELTA_3: int = 10
    LOCAL_EJECT_WINDOW: int = 18
    LOCAL_BLOCKER_TOPK: int = 3
    LOCAL_CLEAR_SPARSE_MAX: int = 8
    LOCAL_CLEAR_INSERT_DELTA: int = 4
    LOCAL_INSERT_MAX_POS: int = 8
    TABU_FREEZE_AFTER: int = 3
    SPILL_DROP_THRESHOLD: float = 0.12
    SPILL_SPARSE_UTIL: float = 0.55

    SEED0: int = 1000
    N_SEEDS: int = 20
    N_JOBS: int = 1

    DUMP_CONFIG: bool = True
