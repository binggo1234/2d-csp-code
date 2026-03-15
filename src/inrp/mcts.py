from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import logging
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    from ortools.sat.python import cp_model
except Exception:  # pragma: no cover - validated in integration env
    cp_model = None


def _load_cp_model():
    global cp_model
    if cp_model is None:
        try:
            from ortools.sat.python import cp_model as cp_model_mod  # type: ignore

            cp_model = cp_model_mod
        except Exception:
            cp_model = None
    return cp_model

from .packer import (
    Board,
    LocalPlacementBlueprint,
    apply_local_blueprint,
    best_local_placement,
    board_geometry_signature,
    board_used_area,
    board_utilization,
    make_empty_board,
    pack_multi_board,
)

logger = logging.getLogger(__name__)

EPS = 1e-9
MASK64 = (1 << 64) - 1
HASH_BASE = 11400714819323198485


def _bit_count(value: int) -> int:
    val = int(value)
    try:
        return int(val.bit_count())
    except Exception:
        return bin(val).count("1")


@dataclass(frozen=True)
class _BoardTreeNode:
    left: Optional["_BoardTreeNode"] = None
    right: Optional["_BoardTreeNode"] = None
    board: Optional[Board] = None


def _tree_set(node: Optional[_BoardTreeNode], lo: int, hi: int, idx: int, board: Board) -> _BoardTreeNode:
    if hi - lo == 1:
        return _BoardTreeNode(board=board)
    mid = lo + (hi - lo) // 2
    left = node.left if node is not None else None
    right = node.right if node is not None else None
    if idx < mid:
        left = _tree_set(left, lo, mid, idx, board)
    else:
        right = _tree_set(right, mid, hi, idx, board)
    return _BoardTreeNode(left=left, right=right)


def _tree_get(node: Optional[_BoardTreeNode], lo: int, hi: int, idx: int) -> Board:
    if node is None:
        raise IndexError(f"Missing board at index={idx}")
    if hi - lo == 1:
        if node.board is None:
            raise IndexError(f"Missing board at leaf index={idx}")
        return node.board
    mid = lo + (hi - lo) // 2
    if idx < mid:
        return _tree_get(node.left, lo, mid, idx)
    return _tree_get(node.right, mid, hi, idx)


def _tree_collect(node: Optional[_BoardTreeNode], lo: int, hi: int, size: int, out: List[Board]) -> None:
    if node is None or lo >= size:
        return
    if hi - lo == 1:
        if node.board is not None:
            out.append(node.board)
        return
    mid = lo + (hi - lo) // 2
    _tree_collect(node.left, lo, mid, size, out)
    _tree_collect(node.right, mid, hi, size, out)


def _tree_iter(node: Optional[_BoardTreeNode], lo: int, hi: int, size: int):
    if node is None or lo >= size:
        return
    if hi - lo == 1:
        if node.board is not None:
            yield lo, node.board
        return
    mid = lo + (hi - lo) // 2
    yield from _tree_iter(node.left, lo, mid, size)
    yield from _tree_iter(node.right, mid, hi, size)


@dataclass(frozen=True)
class PersistentBoards:
    size: int
    capacity: int
    root: Optional[_BoardTreeNode]
    _materialized: Optional[Tuple[Board, ...]] = field(default=None, init=False, repr=False, compare=False)

    @staticmethod
    def empty() -> "PersistentBoards":
        return PersistentBoards(size=0, capacity=1, root=None)

    def __len__(self) -> int:
        return int(self.size)

    def __iter__(self):
        cached = self._materialized
        if cached is not None:
            return iter(cached)
        return (board for _, board in _tree_iter(self.root, 0, self.capacity, self.size))

    def get(self, idx: int) -> Board:
        if idx < 0 or idx >= self.size:
            raise IndexError(idx)
        return _tree_get(self.root, 0, self.capacity, idx)

    def enumerate(self):
        cached = self._materialized
        if cached is not None:
            for idx, board in enumerate(cached):
                yield idx, board
            return
        yield from _tree_iter(self.root, 0, self.capacity, self.size)

    def materialize(self) -> Tuple[Board, ...]:
        cached = self._materialized
        if cached is not None:
            return cached
        out: List[Board] = []
        _tree_collect(self.root, 0, self.capacity, self.size, out)
        cached = tuple(out)
        object.__setattr__(self, "_materialized", cached)
        return cached

    def set(self, idx: int, board: Board) -> "PersistentBoards":
        if idx < 0 or idx >= self.size:
            raise IndexError(idx)
        return PersistentBoards(
            size=self.size,
            capacity=self.capacity,
            root=_tree_set(self.root, 0, self.capacity, idx, board),
        )

    def append(self, board: Board) -> "PersistentBoards":
        capacity = max(1, int(self.capacity))
        root = self.root
        while self.size >= capacity:
            root = _BoardTreeNode(left=root)
            capacity *= 2
        return PersistentBoards(
            size=self.size + 1,
            capacity=capacity,
            root=_tree_set(root, 0, capacity, self.size, board),
        )


@dataclass(frozen=True)
class LayoutSnapshot:
    boards: PersistentBoards
    sum_u_gamma: float
    layout_hash: Tuple[int, int]
    ordered_hash: int


@dataclass
class CacheStats:
    visits: int = 0
    reward_sum: float = 0.0


@dataclass
class DecodeResult:
    action: int
    snapshot: LayoutSnapshot
    board_index: int
    is_new_board: bool
    blueprint: LocalPlacementBlueprint
    lex_score: Tuple[float, ...]
    avg_u_gamma: float


@dataclass(frozen=True)
class MicroDecodeResult:
    blueprint: LocalPlacementBlueprint
    board_term_after: float
    board_hash_after: int


@dataclass(frozen=True)
class ActionGeomStats:
    area_score: float
    cavity_score: float
    fragment_score: float
    escape_score: float
    fit_count: int

    def prior_value(self, cfg) -> float:
        return (
            float(getattr(cfg, "PRIOR_AREA_W", 0.30)) * self.area_score
            + float(getattr(cfg, "PRIOR_CAVITY_W", 0.35)) * self.cavity_score
            + float(getattr(cfg, "PRIOR_FRAGMENT_W", 0.20)) * self.fragment_score
            + float(getattr(cfg, "PRIOR_ESCAPE_W", 0.15)) * self.escape_score
        )


@dataclass
class SearchNode:
    snapshot: LayoutSnapshot
    remaining_mask: int
    depth: int = 0
    parent: Optional["SearchNode"] = None
    action_from_parent: Optional[int] = None
    prefix_sequence: Tuple[int, ...] = field(default_factory=tuple)
    candidate_actions: List[int] = field(default_factory=list)
    candidate_width: int = 0
    visits: int = 0
    reward_sum: float = 0.0
    priors: Dict[int, float] = field(default_factory=dict)
    action_stats: Dict[int, ActionGeomStats] = field(default_factory=dict)
    action_meta: Dict[int, "MacroAction"] = field(default_factory=dict)
    action_rank: Dict[int, Tuple[float, ...]] = field(default_factory=dict)
    children: Dict[int, "SearchNode"] = field(default_factory=dict)
    decoded: Dict[int, Any] = field(default_factory=dict)
    root_noise_applied: bool = False

    def is_terminal(self) -> bool:
        return int(self.remaining_mask) == 0

    def q_value(self) -> float:
        return self.reward_sum / max(1, self.visits)

    def prefix_actions(self) -> List[int]:
        return list(self.prefix_sequence)


@dataclass
class SearchResult:
    boards: List[Board]
    sequence: Tuple[int, ...]
    score_global: Tuple[float, ...]
    scalar_reward: float
    meta: Dict[str, Any]


@dataclass(frozen=True)
class EliteCandidate:
    sequence: Tuple[int, ...]
    snapshot: LayoutSnapshot
    score: Tuple[float, ...]
    tail_signature: Tuple[int, ...]


@dataclass(frozen=True)
class HarvestedLayoutCandidate:
    sequence: Tuple[int, ...]
    snapshot: LayoutSnapshot
    score: Tuple[float, ...]
    board_signature: Tuple[Tuple[int, ...], ...]
    source: str


@dataclass(frozen=True)
class RegionProposal:
    start_idx: int
    span: int
    score: Tuple[float, float, float, float]
    bucket: int
    source: str


@dataclass
class RepairBeamState:
    sequence: Tuple[int, ...]
    snapshot: LayoutSnapshot
    snapshots: List[LayoutSnapshot]
    remaining_tail: Tuple[int, ...]
    ejected_items: Tuple[int, ...]
    depth: int
    priority: Tuple[float, ...]


@dataclass
class ConstructiveRepairState:
    placed_early: Tuple[int, ...]
    remaining_tail: Tuple[int, ...]
    ejected_blockers: Tuple[int, ...]
    snapshot: LayoutSnapshot
    prefix_snapshots: List[LayoutSnapshot]
    completion_sequence: Tuple[int, ...]
    completion_snapshot: LayoutSnapshot
    completion_snapshots: List[LayoutSnapshot]
    priority: Tuple[float, ...]
    depth: int


@dataclass
class RegionRebuildState:
    placed_early: Tuple[int, ...]
    remaining_region_items: Tuple[int, ...]
    parked_items: Tuple[int, ...]
    snapshot: LayoutSnapshot
    prefix_snapshots: List[LayoutSnapshot]
    completion_sequence: Tuple[int, ...]
    completion_snapshot: LayoutSnapshot
    completion_snapshots: List[LayoutSnapshot]
    priority: Tuple[float, ...]
    depth: int


@dataclass(frozen=True)
class RepairWindow:
    start_idx: int
    start_board: int
    span: int
    movable_part_ids: Tuple[int, ...]
    blocker_part_ids: Tuple[int, ...]
    fixed_prefix_sequence: Tuple[int, ...]
    base_board_count: int
    score: Tuple[float, ...]
    source: str


@dataclass(frozen=True)
class PatternCandidate:
    part_ids: Tuple[int, ...]
    sequence: Tuple[int, ...]
    used_area: float
    util: float
    waste_area: float
    place_mode: str
    source: str


@dataclass(frozen=True)
class PatternMasterResult:
    sequence: Tuple[int, ...]
    snapshot: LayoutSnapshot
    score: Tuple[float, ...]
    window_start_idx: int
    patterns_generated: int
    patterns_kept: int
    master_status: str
    boards_saved: int


@dataclass(frozen=True)
class MacroAction:
    key: int
    kind: str
    part_ids: Tuple[int, ...]
    sequence: Tuple[int, ...]
    prior: float
    rank_key: Tuple[float, ...]
    source: str
    pred_util: float = 0.0


@dataclass
class AppliedActionResult:
    key: int
    sequence: Tuple[int, ...]
    snapshot: LayoutSnapshot
    lex_score: Tuple[float, ...]
    avg_u_gamma: float


def _empty_snapshot() -> LayoutSnapshot:
    return LayoutSnapshot(boards=PersistentBoards.empty(), sum_u_gamma=0.0, layout_hash=(0, 0), ordered_hash=0)


def _part_dims_signature(part, nd: int = 3) -> Tuple[float, float]:
    return (round(float(part.w), nd), round(float(part.h), nd))


def _avg_u_gamma(snapshot: LayoutSnapshot) -> float:
    if not snapshot.boards:
        return 0.0
    return float(snapshot.sum_u_gamma) / float(len(snapshot.boards))


def _board_utils(snapshot: LayoutSnapshot) -> List[float]:
    return [board_utilization(board) for board in snapshot.boards.materialize()]


def _u_global(snapshot: LayoutSnapshot) -> float:
    boards = snapshot.boards.materialize()
    if not boards:
        return 0.0
    used_area = sum(board_used_area(board) for board in boards)
    board_area = sum(float(board.W) * float(board.H) for board in boards)
    return used_area / max(EPS, board_area)


def _u_avg_excl_tail(snapshot: LayoutSnapshot, tail_count: int = 1) -> float:
    utils = _board_utils(snapshot)
    if not utils:
        return 0.0
    keep = max(1, len(utils) - max(1, int(tail_count)))
    return sum(utils[:keep]) / float(keep)


def _u_last(snapshot: LayoutSnapshot) -> float:
    utils = _board_utils(snapshot)
    if not utils:
        return 0.0
    return float(utils[-1])


def _u_worst_excl_tail(snapshot: LayoutSnapshot, tail_count: int = 1) -> float:
    utils = _board_utils(snapshot)
    if not utils:
        return 0.0
    keep = max(1, len(utils) - max(1, int(tail_count)))
    return min(utils[:keep])


def _tail_waste_area(snapshot: LayoutSnapshot, tail_count: int = 1) -> float:
    boards = snapshot.boards.materialize()
    if not boards:
        return 0.0
    tail = boards[-min(len(boards), max(1, int(tail_count))) :]
    return sum(max(0.0, float(board.W) * float(board.H) - board_used_area(board)) for board in tail)


def _sparse_board_count(snapshot: LayoutSnapshot, sparse_u: float) -> int:
    return sum(1 for u in _board_utils(snapshot) if u < float(sparse_u))


def _global_lex_score(snapshot: LayoutSnapshot, sparse_u: float = 0.55) -> Tuple[float, ...]:
    return (
        -float(len(snapshot.boards)),
        _u_avg_excl_tail(snapshot, 1),
        _u_worst_excl_tail(snapshot, 1),
        -float(_sparse_board_count(snapshot, sparse_u)),
        _u_global(snapshot),
        _avg_u_gamma(snapshot),
    )


def _scalar_reward(snapshot: LayoutSnapshot, big_m: float) -> float:
    return -float(big_m) * float(len(snapshot.boards)) + _avg_u_gamma(snapshot)


def _compat(part, rect, allow_rot: bool) -> float:
    fits = (part.w <= rect.w + EPS and part.h <= rect.h + EPS)
    if allow_rot:
        fits = fits or (part.h <= rect.w + EPS and part.w <= rect.h + EPS)
    if not fits:
        return 0.0
    area_part = float(part.w0) * float(part.h0)
    area_rect = max(EPS, float(rect.w) * float(rect.h))
    return area_part / area_rect


def _orientations(part, allow_rot: bool) -> Tuple[Tuple[float, float], ...]:
    out = [(float(part.w), float(part.h))]
    if allow_rot and abs(float(part.w) - float(part.h)) > EPS:
        out.append((float(part.h), float(part.w)))
    return tuple(out)


def _hash64(x: int) -> int:
    return int(x) & MASK64


def _board_state_hash(board: Board, nd: int) -> int:
    cached = board._geom_hash_cache
    if cached is not None and int(board._geom_hash_cache_nd) == int(nd):
        return int(cached)
    out = _hash64(hash(board_geometry_signature(board, nd=nd)))
    board._geom_hash_cache_nd = int(nd)
    board._geom_hash_cache = out
    return out


def _part_state_hash(part, nd: int) -> int:
    return _hash64(hash(_part_dims_signature(part, nd)))


def _layout_hash_replace(layout_hash: Tuple[int, int], old_hash: int, new_hash: int) -> Tuple[int, int]:
    sum_hash, xor_hash = layout_hash
    return (_hash64(sum_hash - old_hash + new_hash), _hash64(xor_hash ^ old_hash ^ new_hash))


def _layout_hash_append(layout_hash: Tuple[int, int], board_hash: int) -> Tuple[int, int]:
    sum_hash, xor_hash = layout_hash
    return (_hash64(sum_hash + board_hash), _hash64(xor_hash ^ board_hash))


def _ordered_hash_replace(ordered_hash: int, old_hash: int, new_hash: int, pos_hash: int) -> int:
    return _hash64(ordered_hash - _hash64(old_hash * pos_hash) + _hash64(new_hash * pos_hash))


def _ordered_hash_append(ordered_hash: int, board_hash: int, pos_hash: int) -> int:
    return _hash64(ordered_hash + _hash64(board_hash * pos_hash))


class OmnipotentDecoder:
    def __init__(self, parts_by_id: Dict[int, object], cfg) -> None:
        self.parts_by_id = parts_by_id
        self.cfg = cfg
        self.gamma = float(getattr(cfg, "UTIL_GAMMA", 2.0))
        self.sig_nd = int(getattr(cfg, "STATE_SIG_ND", 3))
        self.place_mode = str(getattr(cfg, "BASELINE_PLACE_MODE", "maxrects_baf"))
        self.allow_rot = bool(getattr(cfg, "ALLOW_ROT", True))
        self.part_hash = {int(pid): _part_state_hash(part, self.sig_nd) for pid, part in parts_by_id.items()}
        self.env_signature = (
            round(float(getattr(cfg, "BOARD_W", 0.0)), self.sig_nd),
            round(float(getattr(cfg, "BOARD_H", 0.0)), self.sig_nd),
            round(float(getattr(cfg, "TRIM", 0.0)), self.sig_nd),
            round(float(getattr(cfg, "SAFE_GAP", 0.0)), self.sig_nd),
            self.place_mode,
            int(self.allow_rot),
        )
        self.pos_hash = [1]
        for _ in range(len(parts_by_id) + 2):
            self.pos_hash.append(_hash64(self.pos_hash[-1] * HASH_BASE))
        self.board_term_cache: Dict[int, float] = {}
        self.micro_cache: Dict[Tuple, Optional[MicroDecodeResult]] = {}
        self.decode_cache: Dict[Tuple, DecodeResult] = {}
        self.micro_hits = 0
        self.micro_misses = 0
        self.decode_hits = 0
        self.decode_misses = 0

    def _board_term(self, board: Board) -> float:
        key = _board_state_hash(board, self.sig_nd)
        if key in self.board_term_cache:
            return self.board_term_cache[key]
        val = board_utilization(board) ** self.gamma
        self.board_term_cache[key] = val
        return val

    def _micro_key(self, board: Board, part) -> Tuple:
        return (_board_state_hash(board, self.sig_nd), self.part_hash[int(part.uid)], self.env_signature)

    def _micro_decode(self, board: Board, part) -> Optional[MicroDecodeResult]:
        key = self._micro_key(board, part)
        if key in self.micro_cache:
            self.micro_hits += 1
            return self.micro_cache[key]
        self.micro_misses += 1
        bp = best_local_placement(board, part, self.allow_rot, place_mode=self.place_mode)
        if bp is None:
            self.micro_cache[key] = None
            return None
        board_after = apply_local_blueprint(board, part, bp)
        out = MicroDecodeResult(
            blueprint=bp,
            board_term_after=self._board_term(board_after),
            board_hash_after=_board_state_hash(board_after, self.sig_nd),
        )
        self.micro_cache[key] = out
        return out

    def decode(self, snapshot: LayoutSnapshot, action: int) -> DecodeResult:
        decode_key = (snapshot.layout_hash, snapshot.ordered_hash, len(snapshot.boards), int(action))
        cached = self.decode_cache.get(decode_key)
        if cached is not None:
            self.decode_hits += 1
            return cached
        self.decode_misses += 1
        part = self.parts_by_id[int(action)]
        boards = snapshot.boards
        n_before = len(boards)
        best: Optional[DecodeResult] = None

        for idx, board in boards.enumerate():
            micro = self._micro_decode(board, part)
            if micro is None:
                continue
            bp = micro.blueprint
            board_hash_before = _board_state_hash(board, self.sig_nd)
            board_after = apply_local_blueprint(board, part, bp)
            new_sum = snapshot.sum_u_gamma - self._board_term(board) + micro.board_term_after
            boards_after = boards.set(idx, board_after)
            snapshot_after = LayoutSnapshot(
                boards=boards_after,
                sum_u_gamma=new_sum,
                layout_hash=_layout_hash_replace(
                    snapshot.layout_hash,
                    board_hash_before,
                    micro.board_hash_after,
                ),
                ordered_hash=_ordered_hash_replace(
                    snapshot.ordered_hash,
                    board_hash_before,
                    micro.board_hash_after,
                    self.pos_hash[idx],
                ),
            )
            avg_gamma = _avg_u_gamma(snapshot_after)
            lex = (
                -float(n_before),
                avg_gamma,
                1.0,
                float(bp.cavity_ratio),
                -float(bp.fragmentation_penalty),
                -float(bp.area_fit),
                -float(idx),
            )
            cand = DecodeResult(
                action=int(action),
                snapshot=snapshot_after,
                board_index=int(idx),
                is_new_board=False,
                blueprint=bp,
                lex_score=lex,
                avg_u_gamma=avg_gamma,
            )
            if best is None or cand.lex_score > best.lex_score:
                best = cand

        empty_board = make_empty_board(
            n_before + 1,
            self.cfg.BOARD_W,
            self.cfg.BOARD_H,
            trim=float(getattr(self.cfg, "TRIM", 0.0)),
            safe_gap=float(getattr(self.cfg, "SAFE_GAP", 0.0)),
            touch_tol=float(getattr(self.cfg, "TOUCH_TOL", 1e-6)),
            place_mode=self.place_mode,
        )
        micro_new = self._micro_decode(empty_board, part)
        if micro_new is None:
            raise RuntimeError(f"Decoder cannot place part uid={part.uid} on a new board.")
        bp_new = micro_new.blueprint
        board_after_new = apply_local_blueprint(empty_board, part, bp_new)
        new_sum = snapshot.sum_u_gamma + micro_new.board_term_after
        boards_after = boards.append(board_after_new)
        snapshot_after = LayoutSnapshot(
            boards=boards_after,
            sum_u_gamma=new_sum,
            layout_hash=_layout_hash_append(snapshot.layout_hash, micro_new.board_hash_after),
            ordered_hash=_ordered_hash_append(snapshot.ordered_hash, micro_new.board_hash_after, self.pos_hash[n_before]),
        )
        avg_gamma = _avg_u_gamma(snapshot_after)
        cand_new = DecodeResult(
            action=int(action),
            snapshot=snapshot_after,
            board_index=int(n_before),
            is_new_board=True,
            blueprint=bp_new,
            lex_score=(
                -float(n_before + 1),
                avg_gamma,
                0.0,
                float(bp_new.cavity_ratio),
                -float(bp_new.fragmentation_penalty),
                -float(bp_new.area_fit),
                -float(n_before + 1),
            ),
            avg_u_gamma=avg_gamma,
        )
        if best is None or cand_new.lex_score > best.lex_score:
            best = cand_new

        if best is None:
            raise RuntimeError(f"Decoder failed to produce action={action}")
        self.decode_cache[decode_key] = best
        return best

    def replay_order(self, order: Sequence[int], *, prefix_snapshot: Optional[LayoutSnapshot] = None) -> Tuple[LayoutSnapshot, List[LayoutSnapshot]]:
        snapshot = prefix_snapshot if prefix_snapshot is not None else _empty_snapshot()
        snapshots = [snapshot]
        for action in order:
            snapshot = self.decode(snapshot, int(action)).snapshot
            snapshots.append(snapshot)
        return snapshot, snapshots


class RecedingHorizonMCTS:
    def __init__(self, parts: Sequence, cfg, seed: int) -> None:
        self.parts = list(parts)
        self.parts_by_id = {int(p.uid): p for p in parts}
        self.part_ids = tuple(sorted(self.parts_by_id))
        self.full_sequence_signature = tuple(sorted(self.part_ids))
        self.part_bit = {pid: 1 << idx for idx, pid in enumerate(self.part_ids)}
        self.full_remaining_mask = (1 << len(self.part_ids)) - 1
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.gamma = float(getattr(cfg, "UTIL_GAMMA", 2.0))
        self.big_m = float(getattr(cfg, "SCALAR_BIG_M", 1000.0))
        self.allow_rot = bool(getattr(cfg, "ALLOW_ROT", True))
        self.sims_per_step = int(getattr(cfg, "MCTS_N_SIM", 800))
        self.c_puct = float(getattr(cfg, "MCTS_PUCT_C", 1.3))
        self.k0 = int(getattr(cfg, "PW_K0", 8))
        self.kmax = int(getattr(cfg, "PW_KMAX", 48))
        self.pw_alpha = float(getattr(cfg, "PW_ALPHA", 0.5))
        self.dynamic_pool_mult = max(1, int(getattr(cfg, "DYN_POOL_MULT", 2)))
        self.dynamic_pool_min = max(1, int(getattr(cfg, "DYN_POOL_MIN", 12)))
        self.dynamic_pool_max = max(1, int(getattr(cfg, "DYN_POOL_MAX", 48)))
        self.dynamic_tail_samples = max(0, int(getattr(cfg, "DYN_TAIL_SAMPLES", 4)))
        self.rollout_topk = int(getattr(cfg, "ROLLOUT_TOPK", 12))
        self.rollout_topr = int(getattr(cfg, "ROLLOUT_RCL_TOPR", 4))
        self.rollout_rcl_weights = tuple(
            max(0.0, float(weight)) for weight in getattr(cfg, "ROLLOUT_RCL_WEIGHTS", (0.60, 0.30, 0.10))
        )
        self.rollout_greedy_frac = min(1.0, max(0.0, float(getattr(cfg, "ROLLOUT_GREEDY_FRAC", 0.0))))
        self.rollout_det_tail = max(1, int(getattr(cfg, "ROLLOUT_DETERMINISTIC_TAIL", 8)))
        self.disable_geom_prior = bool(getattr(cfg, "DISABLE_GEOM_PRIOR", False))
        self.root_dirichlet_enable = bool(getattr(cfg, "ROOT_DIRICHLET_ENABLE", False))
        self.root_dirichlet_alpha = max(1e-6, float(getattr(cfg, "ROOT_DIRICHLET_ALPHA", 0.30)))
        self.root_dirichlet_eps = min(1.0, max(0.0, float(getattr(cfg, "ROOT_DIRICHLET_EPS", 0.25))))
        self.root_dirichlet_max_depth = int(getattr(cfg, "ROOT_DIRICHLET_MAX_DEPTH", -1))
        self.warm_vmax = int(getattr(cfg, "WARMSTART_VMAX", 64))
        self.commit_ratio = float(getattr(cfg, "COMMIT_VISIT_RATIO", 1.8))
        self.commit_max_steps = int(getattr(cfg, "COMMIT_MAX_STEPS", 3))
        self.commit_min_steps = int(getattr(cfg, "COMMIT_MIN_STEPS", 1))
        self.commit_min_root_visits = max(1, int(getattr(cfg, "COMMIT_MIN_ROOT_VISITS", 1)))
        self.commit_min_child_visits = max(1, int(getattr(cfg, "COMMIT_MIN_CHILD_VISITS", 1)))
        self.commit_confirm_passes = max(1, int(getattr(cfg, "COMMIT_CONFIRM_PASSES", 1)))
        self.commit_force_enable = bool(getattr(cfg, "COMMIT_FORCE_ENABLE", False))
        self.commit_timeout_force = bool(getattr(cfg, "COMMIT_TIMEOUT_FORCE", False))
        self.commit_max_round_sims = max(0, int(getattr(cfg, "COMMIT_MAX_ROUND_SIMS", 0)))
        self.solver_time_limit_s = float(getattr(cfg, "SOLVER_TIME_LIMIT_S", 0.0))
        self.post_mcts_reserve_frac = min(0.50, max(0.0, float(getattr(cfg, "POST_MCTS_RESERVE_FRAC", 0.05))))
        self.post_mcts_reserve_min_s = max(0.0, float(getattr(cfg, "POST_MCTS_RESERVE_MIN_S", 0.0)))
        self.post_mcts_reserve_max_s = max(0.0, float(getattr(cfg, "POST_MCTS_RESERVE_MAX_S", 45.0)))
        self.sa_time_cap_s = max(0.25, float(getattr(cfg, "SA_TIME_CAP_S", 15.0)))
        self.sa_budget_frac = min(1.0, max(0.0, float(getattr(cfg, "SA_BUDGET_FRAC", 0.05))))
        self.baseline_restarts = max(0, int(getattr(cfg, "BASELINE_RESTARTS", 0)))
        self.rand_topk = max(1, int(getattr(cfg, "RAND_TOPK", 3)))
        self.sa_iters = max(0, int(getattr(cfg, "SA_ITERS", 0)))
        self.sa_t0 = max(1e-6, float(getattr(cfg, "SA_T0", 0.03)))
        self.sa_alpha = min(0.999999, max(0.50, float(getattr(cfg, "SA_ALPHA", 0.995))))
        self.lns_enable = bool(getattr(cfg, "LNS_ENABLE", True))
        self.lns_prob = min(1.0, max(0.0, float(getattr(cfg, "LNS_PROB", 0.15))))
        self.lns_destroy_frac = min(0.75, max(0.01, float(getattr(cfg, "LNS_DESTROY_FRAC", 0.08))))
        self.elite_archive_k = max(1, int(getattr(cfg, "ELITE_ARCHIVE_K", 8)))
        self.elite_tail_len = max(1, int(getattr(cfg, "ELITE_TAIL_LEN", 16)))
        self.repair_elite_topk = max(1, int(getattr(cfg, "REPAIR_ELITE_TOPK", 4)))
        self.repair_tail_boards = max(1, int(getattr(cfg, "REPAIR_TAIL_BOARDS", 3)))
        self.repair_blocker_window = max(1, int(getattr(cfg, "REPAIR_BLOCKER_WINDOW", 40)))
        self.repair_blocker_topk = max(1, int(getattr(cfg, "REPAIR_BLOCKER_TOPK", 6)))
        self.repair_beam_width = max(1, int(getattr(cfg, "REPAIR_BEAM_WIDTH", 10)))
        self.repair_node_limit = max(1, int(getattr(cfg, "REPAIR_NODE_LIMIT", 240)))
        self.repair_actions_per_state = max(1, int(getattr(cfg, "REPAIR_ACTIONS_PER_STATE", 12)))
        self.repair_pass_time_slice_s = max(0.1, float(getattr(cfg, "REPAIR_PASS_TIME_SLICE_S", 8.0)))
        self.region_proposal_topk = max(1, int(getattr(cfg, "REGION_PROPOSAL_TOPK", 3)))
        self.region_start_backoffs = tuple(
            sorted({max(0, int(x)) for x in getattr(cfg, "REGION_START_BACKOFFS", (8, 16, 24))})
        )
        self.region_rebuild_topk = max(1, int(getattr(cfg, "REGION_REBUILD_TOPK", 4)))
        self.region_rebuild_beam_width = max(1, int(getattr(cfg, "REGION_REBUILD_BEAM_WIDTH", 12)))
        self.region_rebuild_node_limit = max(1, int(getattr(cfg, "REGION_REBUILD_NODE_LIMIT", 320)))
        self.region_rebuild_actions_per_state = max(1, int(getattr(cfg, "REGION_REBUILD_ACTIONS_PER_STATE", 14)))
        self.region_rebuild_time_slice_s = max(0.1, float(getattr(cfg, "REGION_REBUILD_TIME_SLICE_S", 8.0)))
        self.region_diversity_bucket = max(1, int(getattr(cfg, "REGION_DIVERSITY_BUCKET", 12)))
        self.post_repair_enable = bool(getattr(cfg, "POST_REPAIR_ENABLE", True))
        self.post_repair_time_limit_s = max(0.1, float(getattr(cfg, "POST_REPAIR_TIME_LIMIT_S", 1800.0)))
        self.post_repair_windows_topk = max(1, int(getattr(cfg, "POST_REPAIR_WINDOWS_TOPK", 3)))
        self.post_repair_blocker_boards = max(0, int(getattr(cfg, "POST_REPAIR_BLOCKER_BOARDS", 2)))
        self.post_repair_blocker_topk = max(1, int(getattr(cfg, "POST_REPAIR_BLOCKER_TOPK", 12)))
        self.post_repair_patterns_per_mode = max(1, int(getattr(cfg, "POST_REPAIR_PATTERNS_PER_MODE", 24)))
        self.post_repair_pattern_cap = max(1, int(getattr(cfg, "POST_REPAIR_PATTERN_CAP", 300)))
        self.post_repair_pattern_min_util = min(
            1.0,
            max(0.0, float(getattr(cfg, "POST_REPAIR_PATTERN_MIN_UTIL", 0.85))),
        )
        self.post_repair_pattern_strong_util = min(
            1.0,
            max(self.post_repair_pattern_min_util, float(getattr(cfg, "POST_REPAIR_PATTERN_STRONG_UTIL", 0.90))),
        )
        self.post_repair_random_orders = max(0, int(getattr(cfg, "POST_REPAIR_RANDOM_ORDERS", 32)))
        self.post_repair_random_topk = max(2, int(getattr(cfg, "POST_REPAIR_RANDOM_TOPK", 4)))
        self.post_repair_evict_enable = bool(getattr(cfg, "POST_REPAIR_EVICT_ENABLE", True))
        self.post_repair_evict_host_boards = max(1, int(getattr(cfg, "POST_REPAIR_EVICT_HOST_BOARDS", 3)))
        self.post_repair_evict_candidates = max(1, int(getattr(cfg, "POST_REPAIR_EVICT_CANDIDATES", 6)))
        self.post_repair_evict_topk = max(1, int(getattr(cfg, "POST_REPAIR_EVICT_TOPK", 4)))
        self.post_repair_tail_strike_enable = bool(getattr(cfg, "POST_REPAIR_TAIL_STRIKE_ENABLE", False))
        self.post_repair_tail_strike_widths = tuple(
            sorted({max(2, int(x)) for x in getattr(cfg, "POST_REPAIR_TAIL_STRIKE_WIDTHS", (2, 3, 4))})
        )
        self.post_repair_tail_strike_topk = max(1, int(getattr(cfg, "POST_REPAIR_TAIL_STRIKE_TOPK", 4)))
        self.post_repair_tail_strike_pattern_mult = max(
            1, int(getattr(cfg, "POST_REPAIR_TAIL_STRIKE_PATTERN_MULT", 2))
        )
        self.post_repair_dynamic_expand_enable = bool(getattr(cfg, "POST_REPAIR_DYNAMIC_EXPAND_ENABLE", False))
        self.post_repair_dynamic_window_widths = tuple(
            sorted({max(2, int(x)) for x in getattr(cfg, "POST_REPAIR_DYNAMIC_WINDOW_WIDTHS", (3, 5, 8, 10))})
        )
        self.post_repair_initial_window_width = max(
            2, int(getattr(cfg, "POST_REPAIR_INITIAL_WINDOW_WIDTH", 4))
        )
        self.post_repair_dynamic_fast_solve_s = max(
            0.1, float(getattr(cfg, "POST_REPAIR_DYNAMIC_FAST_SOLVE_S", 2.0))
        )
        self.post_repair_alns_enable = bool(getattr(cfg, "POST_REPAIR_ALNS_ENABLE", False))
        self.post_repair_alns_alpha = min(1.0, max(1e-3, float(getattr(cfg, "POST_REPAIR_ALNS_ALPHA", 0.20))))
        self.post_repair_cp_num_workers = max(1, int(getattr(cfg, "POST_REPAIR_CP_NUM_WORKERS", 8)))
        self.post_repair_target_drop = max(0, int(getattr(cfg, "POST_REPAIR_TARGET_DROP", 1)))
        self.post_repair_use_bssf = bool(getattr(cfg, "POST_REPAIR_USE_BSSF", True))
        self.global_pattern_master_enable = bool(getattr(cfg, "GLOBAL_PATTERN_MASTER_ENABLE", False))
        self.global_pattern_include_baseline = bool(getattr(cfg, "GLOBAL_PATTERN_INCLUDE_BASELINE", True))
        self.global_layout_pool_limit = max(1, int(getattr(cfg, "GLOBAL_LAYOUT_POOL_LIMIT", 96)))
        self.global_pattern_layout_topk = max(1, int(getattr(cfg, "GLOBAL_PATTERN_LAYOUT_TOPK", 24)))
        self.global_pattern_cap = max(1, int(getattr(cfg, "GLOBAL_PATTERN_CAP", 6000)))
        self.global_pattern_min_util = min(1.0, max(0.0, float(getattr(cfg, "GLOBAL_PATTERN_MIN_UTIL", 0.80))))
        self.global_pattern_strong_util = min(
            1.0,
            max(self.global_pattern_min_util, float(getattr(cfg, "GLOBAL_PATTERN_STRONG_UTIL", 0.90))),
        )
        self.global_pattern_time_limit_s = max(0.1, float(getattr(cfg, "GLOBAL_PATTERN_TIME_LIMIT_S", 180.0)))
        self.global_pattern_use_bssf = bool(getattr(cfg, "GLOBAL_PATTERN_USE_BSSF", True))
        self.global_pattern_memory_enable = bool(getattr(cfg, "GLOBAL_PATTERN_MEMORY_ENABLE", False))
        self.global_pattern_memory_min_util = min(
            1.0, max(0.0, float(getattr(cfg, "GLOBAL_PATTERN_MEMORY_MIN_UTIL", 0.92)))
        )
        self.global_pattern_memory_master_min_util = min(
            1.0, max(0.0, float(getattr(cfg, "GLOBAL_PATTERN_MEMORY_MASTER_MIN_UTIL", 0.95)))
        )
        self.global_pattern_memory_limit = max(1, int(getattr(cfg, "GLOBAL_PATTERN_MEMORY_LIMIT", 20000)))
        self.global_pattern_memory_topk = max(1, int(getattr(cfg, "GLOBAL_PATTERN_MEMORY_TOPK", 4000)))
        self.global_pattern_memory_focus_tail_boards = max(
            1, int(getattr(cfg, "GLOBAL_PATTERN_MEMORY_FOCUS_TAIL_BOARDS", 1))
        )
        self.macro_action_enable = bool(getattr(cfg, "MACRO_ACTION_ENABLE", False))
        self.macro_action_topk = max(1, int(getattr(cfg, "MACRO_ACTION_TOPK", 12)))
        self.macro_min_parts = max(2, int(getattr(cfg, "MACRO_MIN_PARTS", 2)))
        self.macro_max_parts = max(self.macro_min_parts, int(getattr(cfg, "MACRO_MAX_PARTS", 6)))
        self.macro_pattern_topk = max(0, int(getattr(cfg, "MACRO_PATTERN_TOPK", 4)))
        self.macro_anchor_topk = max(0, int(getattr(cfg, "MACRO_ANCHOR_TOPK", 4)))
        self.macro_tail_topk = max(0, int(getattr(cfg, "MACRO_TAIL_TOPK", 3)))
        self.macro_single_topk = max(1, int(getattr(cfg, "MACRO_SINGLE_TOPK", 3)))
        self.macro_pattern_util_min = min(1.0, max(0.0, float(getattr(cfg, "MACRO_PATTERN_UTIL_MIN", 0.92))))
        self.macro_memory_util_min = min(1.0, max(0.0, float(getattr(cfg, "MACRO_MEMORY_UTIL_MIN", 0.95))))
        self.macro_stage_early_ratio = min(0.95, max(0.0, float(getattr(cfg, "MACRO_STAGE_EARLY_RATIO", 0.35))))
        self.macro_stage_late_ratio = min(
            0.99, max(self.macro_stage_early_ratio, float(getattr(cfg, "MACRO_STAGE_LATE_RATIO", 0.80)))
        )
        self.macro_tail_trigger_util = min(1.0, max(0.0, float(getattr(cfg, "MACRO_TAIL_TRIGGER_UTIL", 0.75))))
        self.macro_sparse_trigger_util = min(1.0, max(0.0, float(getattr(cfg, "MACRO_SPARSE_TRIGGER_UTIL", 0.60))))
        self.sig_nd = int(getattr(cfg, "STATE_SIG_ND", 3))
        board_ref_area = float(getattr(cfg, "BOARD_W", 1.0)) * float(getattr(cfg, "BOARD_H", 1.0))
        self.static_action_score = {
            pid: (
                (float(self.parts_by_id[pid].w0) * float(self.parts_by_id[pid].h0)) / max(EPS, board_ref_area)
                + 0.05
                * (
                    max(float(self.parts_by_id[pid].w0), float(self.parts_by_id[pid].h0))
                    / max(EPS, min(float(self.parts_by_id[pid].w0), float(self.parts_by_id[pid].h0)))
                )
            )
            for pid in self.part_ids
        }
        self.static_order = sorted(self.part_ids, key=lambda pid: (self.static_action_score[pid], pid), reverse=True)
        self.decoder = OmnipotentDecoder(self.parts_by_id, cfg)
        self.part_orientations = {
            int(pid): _orientations(self.parts_by_id[int(pid)], self.allow_rot) for pid in self.part_ids
        }
        self.coarse_fit_cache: Dict[Tuple[int, int, int], Tuple[float, float, int]] = {}
        self.macro_cache: Dict[Tuple, CacheStats] = {}
        self.rollout_cache: Dict[Tuple, Tuple[LayoutSnapshot, Tuple[int, ...]]] = {}
        self.seed_suffix_cache: Dict[Tuple, Tuple[LayoutSnapshot, Tuple[int, ...]]] = {}
        self.global_best_sequence: Optional[Tuple[int, ...]] = None
        self.global_best_snapshot: Optional[LayoutSnapshot] = None
        self.global_best_score: Optional[Tuple[float, ...]] = None
        self.elite_archive: List[EliteCandidate] = []
        self.constructive_warmstarts: List[EliteCandidate] = []
        self.harvested_layouts: List[HarvestedLayoutCandidate] = []
        self.global_pattern_memory: Dict[Tuple[int, ...], PatternCandidate] = {}
        self.post_repair_operator_scores: Dict[str, float] = {
            "tail-strike": 1.0,
            "evict": 1.0,
            "spill": 1.0,
            "region": 1.0,
            "last": 1.0,
            "generic": 1.0,
        }
        self.solve_deadline = math.inf
        self.meta = {
            "simulations": 0.0,
            "commit_rounds": 0.0,
            "commit_steps": 0.0,
            "commit_combo_rounds": 0.0,
            "commit_force_rounds": 0.0,
            "solver_timeout_hit": 0.0,
            "elite_archive_size": 0.0,
            "rollout_cache_hits": 0.0,
            "rollout_cache_misses": 0.0,
            "rollout_cache_entries": 0.0,
            "decode_cache_hits": 0.0,
            "decode_cache_misses": 0.0,
            "decode_cache_entries": 0.0,
            "micro_cache_hits": 0.0,
            "micro_cache_misses": 0.0,
            "micro_cache_entries": 0.0,
            "macro_cache_hits": 0.0,
            "macro_cache_entries": 0.0,
            "warmstart_capped": 0.0,
            "coarse_fit_cache_hits": 0.0,
            "coarse_fit_cache_misses": 0.0,
            "coarse_fit_cache_entries": 0.0,
            "repair_attempts": 0.0,
            "repair_improvements": 0.0,
            "repair_best_from_elite_rank": 0.0,
            "repair_nodes_expanded": 0.0,
            "repair_time_s": 0.0,
            "tail_boards_cleared": 0.0,
            "region_proposals_total": 0.0,
            "region_rebuild_attempts": 0.0,
            "region_rebuild_improvements": 0.0,
            "region_best_start_idx": -1.0,
            "region_best_span": 0.0,
            "region_nodes_expanded": 0.0,
            "region_time_s": 0.0,
            "tail_collapse_attempts": 0.0,
            "tail_collapse_improvements": 0.0,
            "post_repair_attempts": 0.0,
            "post_repair_improvements": 0.0,
            "post_repair_best_window_start": -1.0,
            "post_repair_patterns_generated": 0.0,
            "post_repair_patterns_kept": 0.0,
            "post_repair_master_status": "",
            "post_repair_backend": "",
            "post_repair_cp_sat_used": 0.0,
            "post_repair_windows_total": 0.0,
            "post_repair_tail_strike_windows": 0.0,
            "post_repair_time_s": 0.0,
            "post_repair_boards_saved": 0.0,
            "post_repair_dynamic_expansions": 0.0,
            "post_repair_window_normalizations": 0.0,
            "post_repair_alns_updates": 0.0,
            "warmstart_candidates_generated": 0.0,
            "warmstart_candidates_kept": 0.0,
            "warmstart_cache_states": 0.0,
            "seed_rollout_cache_hits": 0.0,
            "root_dirichlet_applications": 0.0,
            "post_mcts_reserve_s": 0.0,
            "sa_lns_attempts": 0.0,
            "sa_lns_improvements": 0.0,
            "sa_lns_accepts": 0.0,
            "sa_lns_time_s": 0.0,
            "baseline_seed_injected": 0.0,
            "layout_harvest_layouts": 0.0,
            "global_pattern_attempts": 0.0,
            "global_pattern_improvements": 0.0,
            "global_pattern_patterns_generated": 0.0,
            "global_pattern_patterns_kept": 0.0,
            "global_pattern_master_status": "",
            "global_pattern_backend": "",
            "global_pattern_boards_saved": 0.0,
            "global_pattern_memory_patterns": 0.0,
            "global_pattern_memory_selected": 0.0,
            "global_pattern_memory_injected": 0.0,
            "global_pattern_memory_relevant": 0.0,
            "global_pattern_hint_patterns": 0.0,
            "post_repair_hint_patterns": 0.0,
            "macro_actions_generated": 0.0,
            "macro_pattern_actions": 0.0,
            "macro_anchor_actions": 0.0,
            "macro_tail_actions": 0.0,
            "macro_single_actions": 0.0,
            "macro_rollout_steps": 0.0,
            "macro_memory_focus_hits": 0.0,
            "macro_memory_fallback_hits": 0.0,
            "macro_adaptive_pattern_boosts": 0.0,
            "macro_adaptive_anchor_suppressed": 0.0,
            "macro_adaptive_tail_boosts": 0.0,
            "result_source": "",
        }

    def _state_key(self, snapshot: LayoutSnapshot, remaining_mask: int) -> Tuple:
        return (snapshot.layout_hash, snapshot.ordered_hash, int(remaining_mask))

    def _iter_remaining(self, remaining_mask: int):
        mask = int(remaining_mask)
        while mask:
            lsb = mask & -mask
            yield int(self.part_ids[lsb.bit_length() - 1])
            mask ^= lsb

    def _candidate_actions_static(self, remaining_mask: int) -> List[int]:
        mask = int(remaining_mask)
        return [pid for pid in self.static_order if mask & int(self.part_bit[pid])]

    def _ordered_unique_actions(self, actions: Sequence[int]) -> List[int]:
        out: List[int] = []
        seen = set()
        for action in actions:
            action = int(action)
            if action in seen:
                continue
            seen.add(action)
            out.append(action)
        return out

    def _macro_action_key(self, sequence: Sequence[int]) -> int:
        acc = HASH_BASE & MASK64
        for pid in sequence:
            acc ^= (int(pid) + 0x9E3779B97F4A7C15) & MASK64
            acc = (acc * 1099511628211) & MASK64
        acc ^= len(tuple(sequence)) & MASK64
        key = -int(acc or 1)
        return key if key < 0 else -1

    def _placed_ratio(self, remaining_mask: int) -> float:
        total = max(1, len(self.part_ids))
        return 1.0 - float(_bit_count(int(remaining_mask))) / float(total)

    def _search_stage(self, remaining_mask: int) -> str:
        placed_ratio = self._placed_ratio(remaining_mask)
        if placed_ratio <= self.macro_stage_early_ratio:
            return "early"
        if placed_ratio >= self.macro_stage_late_ratio:
            return "late"
        return "mid"

    def _macro_stage_weights(self, remaining_mask: int) -> Dict[str, float]:
        stage = self._search_stage(remaining_mask)
        if stage == "early":
            return {
                "existing": 0.10,
                "u_global": 0.15,
                "u_avg": 0.10,
                "u_last": 0.05,
                "span": 0.10,
                "kind": 0.05,
                "pred_util": 0.45,
            }
        if stage == "late":
            return {
                "existing": 0.15,
                "u_global": 0.15,
                "u_avg": 0.15,
                "u_last": 0.20,
                "span": 0.05,
                "kind": 0.05,
                "pred_util": 0.25,
            }
        return {
            "existing": 0.15,
            "u_global": 0.20,
            "u_avg": 0.20,
            "u_last": 0.10,
            "span": 0.10,
            "kind": 0.05,
            "pred_util": 0.20,
        }

    def _hard_remaining_part_ids(self, remaining_mask: int, *, topk: int) -> Set[int]:
        remaining = self._candidate_actions_static(remaining_mask)
        if not remaining or topk <= 0:
            return set()
        ranked = sorted(
            remaining,
            key=lambda pid: (
                float(self.parts_by_id[int(pid)].w0) * float(self.parts_by_id[int(pid)].h0),
                max(float(self.parts_by_id[int(pid)].w0), float(self.parts_by_id[int(pid)].h0))
                / max(EPS, min(float(self.parts_by_id[int(pid)].w0), float(self.parts_by_id[int(pid)].h0))),
                float(self.static_action_score[int(pid)]),
                -int(pid),
            ),
            reverse=True,
        )
        return {int(pid) for pid in ranked[: max(1, min(int(topk), len(ranked)))]}

    def _tail_rects(self, snapshot: LayoutSnapshot, *, tail_boards: int = 2) -> List[object]:
        boards = snapshot.boards.materialize()
        if not boards:
            return []
        out: List[object] = []
        for board in boards[-min(len(boards), max(1, int(tail_boards))) :]:
            out.extend(board.free_rects)
        return out

    def _tail_focus_remaining_part_ids(
        self,
        snapshot: LayoutSnapshot,
        remaining_mask: int,
        *,
        tail_boards: int,
        topk: int,
    ) -> Set[int]:
        remaining = self._candidate_actions_static(remaining_mask)
        if not remaining or topk <= 0:
            return set()
        tail_rects = self._tail_rects(snapshot, tail_boards=tail_boards)
        if not tail_rects:
            return set()
        scored: List[Tuple[float, float, float, int]] = []
        for pid in remaining:
            part = self.parts_by_id[int(pid)]
            compat = max((_compat(part, rect, self.allow_rot) for rect in tail_rects), default=0.0)
            if compat <= 0.0:
                continue
            scored.append(
                (
                    float(compat),
                    float(self.static_action_score[int(pid)]),
                    float(part.w0) * float(part.h0),
                    int(pid),
                )
            )
        scored.sort(reverse=True)
        return {int(pid) for _, _, _, pid in scored[: max(1, min(int(topk), len(scored)))]}

    def _pattern_tail_fit_score(self, sequence: Sequence[int], tail_rects: Sequence[object]) -> float:
        if not sequence or not tail_rects:
            return 0.0
        fits: List[float] = []
        for pid in sequence:
            part = self.parts_by_id[int(pid)]
            compat = max((_compat(part, rect, self.allow_rot) for rect in tail_rects), default=0.0)
            fits.append(float(compat))
        if not fits:
            return 0.0
        return float(sum(fits)) / float(len(fits))

    def _macro_stage_limits(
        self,
        snapshot: LayoutSnapshot,
        remaining_mask: int,
        *,
        pattern_candidates: int = 0,
    ) -> Dict[str, int]:
        stage = self._search_stage(remaining_mask)
        if stage == "early":
            return {
                "pattern": 0,
                "anchor": min(self.macro_anchor_topk, 2),
                "tail": 0,
                "single": max(self.macro_single_topk, 4),
            }
        tail_u = _u_last(snapshot)
        sparse_count = _sparse_board_count(snapshot, self.macro_sparse_trigger_util)
        severe_tail = len(snapshot.boards) > 0 and tail_u < self.macro_tail_trigger_util
        strong_pattern_pool = int(pattern_candidates) >= max(2, self.macro_pattern_topk // 2 or 1)
        if stage == "late":
            pattern_cap = min(
                self.macro_action_topk,
                max(self.macro_pattern_topk + (4 if severe_tail or sparse_count > 0 else 3), self.macro_action_topk - 1),
            )
            tail_cap = min(self.macro_tail_topk, 1 if severe_tail else 0)
            single_cap = min(self.macro_single_topk, 1)
            anchor_cap = 0 if strong_pattern_pool or severe_tail or sparse_count > 0 else min(1, self.macro_anchor_topk)
            if pattern_cap >= max(1, self.macro_pattern_topk + 3):
                self.meta["macro_adaptive_pattern_boosts"] += 1.0
            if anchor_cap == 0:
                self.meta["macro_adaptive_anchor_suppressed"] += 1.0
            if tail_cap > 0:
                self.meta["macro_adaptive_tail_boosts"] += 1.0
            return {
                "pattern": pattern_cap,
                "anchor": anchor_cap,
                "tail": tail_cap,
                "single": single_cap,
            }
        pattern_cap = min(
            self.macro_action_topk,
            max(2, self.macro_pattern_topk + (2 if strong_pattern_pool or sparse_count > 0 else 1)),
        )
        anchor_cap = 1 if not strong_pattern_pool and sparse_count <= 0 else 0
        tail_cap = min(self.macro_tail_topk, 1 if severe_tail else 0)
        single_cap = max(1, min(self.macro_single_topk, 2))
        if pattern_cap > max(1, self.macro_pattern_topk):
            self.meta["macro_adaptive_pattern_boosts"] += 1.0
        if anchor_cap == 0:
            self.meta["macro_adaptive_anchor_suppressed"] += 1.0
        if tail_cap > 0:
            self.meta["macro_adaptive_tail_boosts"] += 1.0
        return {
            "pattern": pattern_cap,
            "anchor": anchor_cap,
            "tail": tail_cap,
            "single": single_cap,
        }

    def _apply_sequence(self, snapshot: LayoutSnapshot, sequence: Sequence[int], *, key: int) -> AppliedActionResult:
        state = snapshot
        for pid in sequence:
            state = self.decoder.decode(state, int(pid)).snapshot
        return AppliedActionResult(
            key=int(key),
            sequence=tuple(int(pid) for pid in sequence),
            snapshot=state,
            lex_score=self._terminal_score(state),
            avg_u_gamma=_avg_u_gamma(state),
        )

    def _macro_kind_bonus(self, kind: str, remaining_mask: int, snapshot_after: LayoutSnapshot) -> float:
        stage = self._search_stage(remaining_mask)
        if kind == "pattern":
            return 0.10 if stage == "early" else (0.12 if stage == "mid" else 0.18)
        if kind == "anchor":
            return 0.08 if stage == "early" else (0.02 if stage == "mid" else 0.0)
        if kind == "tail":
            tail_u = _u_last(snapshot_after)
            if stage == "late":
                return 0.02 + 0.10 * max(0.0, self.macro_tail_trigger_util - tail_u)
            return 0.04
        return 0.0

    def _select_macro_action_mix(
        self,
        actions: Sequence[MacroAction],
        *,
        stage: str,
        target_limit: int,
    ) -> List[MacroAction]:
        limit = max(1, min(int(target_limit), self.macro_action_topk))
        ordered = sorted(
            actions,
            key=lambda item: (item.rank_key, item.source, item.key),
            reverse=True,
        )
        if stage not in {"mid", "late"} or not ordered:
            return ordered[:limit]

        patterns = [item for item in ordered if item.kind == "pattern"]
        if not patterns:
            return ordered[:limit]

        selected: List[MacroAction] = []
        seen: Set[int] = set()

        def take(items: Sequence[MacroAction], cap: int) -> None:
            if cap <= 0:
                return
            for item in items:
                if len(selected) >= limit or int(item.key) in seen:
                    continue
                selected.append(item)
                seen.add(int(item.key))
                if sum(1 for picked in selected if picked in items) >= cap:
                    break

        if stage == "late":
            pattern_cap = min(len(patterns), max(2, limit - 2))
            take(patterns, pattern_cap)
            take([item for item in ordered if item.kind == "tail"], 1)
            take([item for item in ordered if item.kind == "single"], 1)
        else:
            pattern_cap = min(len(patterns), max(2, min(limit, self.macro_pattern_topk + 1)))
            take(patterns, pattern_cap)
            take([item for item in ordered if item.kind == "tail"], 1)
            take([item for item in ordered if item.kind == "single"], 2)
            take([item for item in ordered if item.kind == "anchor"], 1)

        if len(selected) < limit:
            for item in ordered:
                if int(item.key) in seen:
                    continue
                selected.append(item)
                seen.add(int(item.key))
                if len(selected) >= limit:
                    break
        return selected[:limit]

    def _macro_prior_rank(
        self,
        snapshot_before: LayoutSnapshot,
        remaining_mask: int,
        kind: str,
        sequence: Sequence[int],
        snapshot_after: LayoutSnapshot,
        pred_util: float = 0.0,
    ) -> Tuple[float, Tuple[float, ...]]:
        weights = self._macro_stage_weights(remaining_mask)
        board_delta = max(0, len(snapshot_after.boards) - len(snapshot_before.boards))
        existing_bonus = 1.0 if len(snapshot_after.boards) <= len(snapshot_before.boards) else 0.0
        span_score = float(len(tuple(sequence))) / float(max(1, self.macro_max_parts))
        kind_bonus = self._macro_kind_bonus(kind, remaining_mask, snapshot_after)
        u_global = _u_global(snapshot_after)
        u_avg = _u_avg_excl_tail(snapshot_after, 1)
        u_last = _u_last(snapshot_after)
        prior = (
            weights["existing"] * existing_bonus
            + weights["u_global"] * u_global
            + weights["u_avg"] * u_avg
            + weights["u_last"] * u_last
            + weights["span"] * span_score
            + weights["kind"] * kind_bonus
            + weights["pred_util"] * max(0.0, float(pred_util))
        )
        rank_key = (
            float(prior),
            float(pred_util),
            -float(board_delta),
            float(u_global),
            float(u_avg),
            float(u_last),
            float(span_score),
        )
        return max(1e-6, float(prior)), rank_key

    def _macro_memory_actions(
        self,
        snapshot: LayoutSnapshot,
        remaining_mask: int,
        limit: int,
    ) -> List[Tuple[str, Tuple[int, ...], float]]:
        if not self.global_pattern_memory_enable or limit <= 0:
            return []
        remaining = {int(pid) for pid in self._candidate_actions_static(remaining_mask)}
        if not remaining:
            return []
        focus = self._tail_focus_remaining_part_ids(
            snapshot,
            remaining_mask,
            tail_boards=max(1, self.global_pattern_memory_focus_tail_boards),
            topk=max(self.macro_max_parts + 2, 6),
        )
        hard_parts = self._hard_remaining_part_ids(remaining_mask, topk=max(self.macro_max_parts + 2, 6))
        tail_rects = self._tail_rects(snapshot, tail_boards=max(1, self.global_pattern_memory_focus_tail_boards))
        focused: List[Tuple[Tuple[float, ...], PatternCandidate]] = []
        backup: List[Tuple[Tuple[float, ...], PatternCandidate]] = []
        for pattern in self.global_pattern_memory.values():
            if pattern.util < max(self.macro_pattern_util_min, self.macro_memory_util_min):
                continue
            if len(pattern.sequence) < self.macro_min_parts or len(pattern.sequence) > self.macro_max_parts:
                continue
            if any(int(pid) not in remaining for pid in pattern.sequence):
                continue
            focus_hits = sum(1 for pid in pattern.sequence if int(pid) in focus)
            hard_hits = sum(1 for pid in pattern.sequence if int(pid) in hard_parts)
            tail_fit = self._pattern_tail_fit_score(pattern.sequence, tail_rects)
            rank = (
                float(focus_hits),
                float(hard_hits),
                float(tail_fit),
                float(pattern.util),
                float(pattern.used_area),
                -float(len(pattern.sequence)),
            )
            if focus and focus_hits > 0:
                focused.append((rank, pattern))
            else:
                backup.append((rank, pattern))
        focused.sort(key=lambda item: (item[0], item[1].source), reverse=True)
        backup.sort(key=lambda item: (item[0], item[1].source), reverse=True)
        ranked: List[PatternCandidate] = [pattern for _, pattern in focused[:limit]]
        if len(ranked) < limit:
            if not focused and backup:
                self.meta["macro_memory_fallback_hits"] += 1.0
            ranked.extend(pattern for _, pattern in backup[: max(0, limit - len(ranked))])
        self.meta["macro_memory_focus_hits"] += float(len(focused))
        return [
            ("pattern", tuple(int(pid) for pid in pattern.sequence), float(pattern.util))
            for pattern in ranked[:limit]
        ]

    def _anchor_macro_actions(
        self,
        snapshot: LayoutSnapshot,
        remaining_mask: int,
        limit: int,
    ) -> List[Tuple[str, Tuple[int, ...], float]]:
        if limit <= 0:
            return []
        scout = self._scout_actions(remaining_mask, max(self.macro_action_topk, self.macro_max_parts * 2))
        if not scout:
            return []
        candidates: List[Tuple[str, Tuple[int, ...], float]] = []
        seen = set()
        modes = [self.decoder.place_mode]
        if self.decoder.place_mode != "maxrects_bssf":
            modes.append("maxrects_bssf")
        for anchor in scout[: max(limit * 2, limit)]:
            followers = [int(pid) for pid in scout if int(pid) != int(anchor)]
            order = tuple([int(anchor)] + followers[: max(0, self.macro_max_parts - 1)])
            for place_mode in modes[:2]:
                pattern, _ = self._build_single_board_pattern(order, place_mode=place_mode, source=f"macro_anchor:{anchor}")
                if pattern is None or len(pattern.sequence) < self.macro_min_parts:
                    continue
                seq = tuple(int(pid) for pid in pattern.sequence[: self.macro_max_parts])
                if seq in seen:
                    continue
                seen.add(seq)
                candidates.append(("anchor", seq, float(pattern.util)))
                if len(candidates) >= limit:
                    return candidates
        return candidates

    def _tail_macro_actions(
        self,
        snapshot: LayoutSnapshot,
        remaining_mask: int,
        limit: int,
    ) -> List[Tuple[str, Tuple[int, ...], float]]:
        if limit <= 0 or len(snapshot.boards) <= 0:
            return []
        boards = snapshot.boards.materialize()
        tail_rects = [fr for board in boards[-min(2, len(boards)) :] for fr in board.free_rects]
        if not tail_rects:
            return []
        scored: List[Tuple[float, float, int]] = []
        for pid in self._candidate_actions_static(remaining_mask):
            part = self.parts_by_id[int(pid)]
            compat = max((_compat(part, fr, self.allow_rot) for fr in tail_rects), default=0.0)
            if compat <= 0.0:
                continue
            scored.append((float(compat), float(self.static_action_score[int(pid)]), int(pid)))
        scored.sort(reverse=True)
        if not scored:
            return []
        order = tuple(int(pid) for _, _, pid in scored[: max(self.macro_max_parts * 2, self.macro_tail_topk + 2)])
        candidates: List[Tuple[str, Tuple[int, ...], float]] = []
        seen = set()
        for ridx in range(min(limit, len(order))):
            rotated = order[ridx:] + order[:ridx]
            pattern, _ = self._build_single_board_pattern(
                rotated[: max(self.macro_max_parts, self.macro_min_parts)],
                place_mode=self.decoder.place_mode,
                source=f"macro_tail:{ridx}",
            )
            if pattern is None or len(pattern.sequence) < self.macro_min_parts:
                continue
            seq = tuple(int(pid) for pid in pattern.sequence[: self.macro_max_parts])
            if seq in seen:
                continue
            seen.add(seq)
            candidates.append(("tail", seq, float(pattern.util)))
        return candidates[:limit]

    def _single_macro_actions(self, remaining_mask: int, limit: int) -> List[Tuple[str, Tuple[int, ...], float]]:
        if limit <= 0:
            return []
        singles = self._scout_actions(remaining_mask, max(limit, 1))
        return [("single", (int(pid),), 0.0) for pid in singles[:limit]]

    def _single_stage_macro_actions(
        self,
        node: SearchNode,
        target_limit: int,
    ) -> Tuple[List[int], Dict[int, MacroAction]]:
        scouted = self._scout_actions(node.remaining_mask, target_limit)
        self._ensure_priors(node, scouted)
        action_meta: Dict[int, MacroAction] = {}
        ordered: List[int] = []
        for action in scouted:
            stats = node.action_stats[int(action)]
            rank_key = tuple(float(x) for x in self._action_rank_key(int(action), stats))
            node.action_rank[int(action)] = rank_key
            action_meta[int(action)] = MacroAction(
                key=int(action),
                kind="single",
                part_ids=(int(action),),
                sequence=(int(action),),
                prior=max(1e-6, float(node.priors[int(action)])),
                rank_key=rank_key,
                source="single_stage",
                pred_util=min(1.0, float(stats.area_score)),
            )
            ordered.append(int(action))
        return ordered, action_meta

    def _generate_macro_actions(
        self,
        snapshot: LayoutSnapshot,
        remaining_mask: int,
        target_limit: int,
    ) -> Tuple[List[int], Dict[int, MacroAction], Dict[int, AppliedActionResult]]:
        candidates: Dict[int, MacroAction] = {}
        decoded_map: Dict[int, AppliedActionResult] = {}
        remaining = {int(pid) for pid in self._candidate_actions_static(remaining_mask)}
        if not remaining:
            return [], candidates, decoded_map
        memory_probe = self._macro_memory_actions(
            snapshot,
            remaining_mask,
            min(self.macro_action_topk, max(self.macro_pattern_topk + 3, self.macro_action_topk)),
        )
        stage_limits = self._macro_stage_limits(
            snapshot,
            remaining_mask,
            pattern_candidates=len(memory_probe),
        )

        def register(kind: str, sequence: Sequence[int], pred_util: float, source: str) -> None:
            seq_t = tuple(int(pid) for pid in sequence if int(pid) in remaining)
            if not seq_t:
                return
            if kind != "single" and len(seq_t) < self.macro_min_parts:
                return
            seq_t = seq_t[: self.macro_max_parts] if kind != "single" else seq_t[:1]
            if len(set(seq_t)) != len(seq_t):
                return
            key = int(seq_t[0]) if len(seq_t) == 1 else self._macro_action_key(seq_t)
            decoded = self._apply_sequence(snapshot, seq_t, key=key)
            prior, rank_key = self._macro_prior_rank(
                snapshot,
                remaining_mask,
                kind,
                seq_t,
                decoded.snapshot,
                pred_util=float(pred_util),
            )
            action = MacroAction(
                key=int(key),
                kind=str(kind),
                part_ids=tuple(sorted(seq_t)),
                sequence=seq_t,
                prior=float(prior),
                rank_key=tuple(float(x) for x in rank_key),
                source=str(source),
                pred_util=float(pred_util),
            )
            prev = candidates.get(int(key))
            if prev is None or (action.rank_key, action.source) > (prev.rank_key, prev.source):
                candidates[int(key)] = action
                decoded_map[int(key)] = decoded

        for kind, sequence, pred_util in memory_probe[: stage_limits["pattern"]]:
            register(kind, sequence, pred_util, "memory")
        for kind, sequence, pred_util in self._anchor_macro_actions(snapshot, remaining_mask, stage_limits["anchor"]):
            register(kind, sequence, pred_util, "anchor")
        for kind, sequence, pred_util in self._tail_macro_actions(snapshot, remaining_mask, stage_limits["tail"]):
            register(kind, sequence, pred_util, "tail")
        for kind, sequence, pred_util in self._single_macro_actions(remaining_mask, stage_limits["single"]):
            register(kind, sequence, pred_util, "single")

        ordered = self._select_macro_action_mix(
            list(candidates.values()),
            stage=self._search_stage(remaining_mask),
            target_limit=max(1, min(target_limit, self.macro_action_topk)),
        )
        actions = [int(item.key) for item in ordered]
        self.meta["macro_actions_generated"] += float(len(actions))
        self.meta["macro_pattern_actions"] += float(sum(1 for item in ordered if item.kind == "pattern"))
        self.meta["macro_anchor_actions"] += float(sum(1 for item in ordered if item.kind == "anchor"))
        self.meta["macro_tail_actions"] += float(sum(1 for item in ordered if item.kind == "tail"))
        self.meta["macro_single_actions"] += float(sum(1 for item in ordered if item.kind == "single"))
        return actions, {int(item.key): item for item in ordered}, {int(key): decoded_map[int(key)] for key in actions}

    def _scout_actions(self, remaining_mask: int, target_limit: int) -> List[int]:
        remaining = self._candidate_actions_static(remaining_mask)
        if not remaining:
            return []
        scout_size = max(self.dynamic_pool_min, int(math.ceil(max(1, target_limit) * self.dynamic_pool_mult)))
        scout_size = min(max(1, scout_size), self.dynamic_pool_max, len(remaining))
        if self.disable_geom_prior:
            selected = list(remaining)
            self.rng.shuffle(selected)
            return selected[:scout_size]
        selected = list(remaining[:scout_size])
        tail = remaining[scout_size:]
        sample_n = min(self.dynamic_tail_samples, len(tail))
        if sample_n > 0:
            if sample_n >= len(tail):
                selected.extend(tail)
            elif sample_n == 1:
                selected.append(tail[len(tail) // 2])
            else:
                step = float(len(tail) - 1) / float(sample_n - 1)
                for i in range(sample_n):
                    idx = min(len(tail) - 1, int(round(i * step)))
                    selected.append(tail[idx])
        return self._ordered_unique_actions(selected)

    def _action_rank_key(self, action: int, stats: ActionGeomStats) -> Tuple[float, float, float, float, int, float, int]:
        if self.disable_geom_prior:
            return (0.0, 0.0, 0.0, 0.0, 0, 0.0, int(action))
        return (
            float(stats.prior_value(self.cfg)),
            float(stats.escape_score),
            float(stats.cavity_score),
            float(stats.fragment_score),
            int(stats.fit_count),
            float(self.static_action_score[int(action)]),
            int(action),
        )

    def _candidate_sort_key(self, node: SearchNode, action: int) -> Tuple[float, ...]:
        if self.disable_geom_prior:
            return (float(node.priors.get(int(action), 0.0)), -float(int(action)))
        if int(action) in node.action_rank:
            return (float(node.priors.get(int(action), 0.0)),) + tuple(float(x) for x in node.action_rank[int(action)])
        return (float(node.priors.get(int(action), 0.0)),) + self._action_rank_key(int(action), node.action_stats[int(action)])

    def _sample_dirichlet(self, size: int, alpha: float) -> List[float]:
        if size <= 0:
            return []
        draws = [self.rng.gammavariate(float(alpha), 1.0) for _ in range(int(size))]
        total = sum(draws)
        if total <= EPS:
            return [1.0 / float(size)] * int(size)
        return [float(draw) / total for draw in draws]

    def _apply_root_dirichlet_noise(self, node: SearchNode) -> None:
        if not self.root_dirichlet_enable or node.parent is not None or node.root_noise_applied:
            return
        if self.root_dirichlet_max_depth >= 0 and int(node.depth) >= self.root_dirichlet_max_depth:
            node.root_noise_applied = True
            return
        actions = [int(action) for action in node.candidate_actions if int(action) in node.priors]
        if len(actions) <= 1 or self.root_dirichlet_eps <= 0.0:
            node.root_noise_applied = True
            return

        old_total = sum(float(node.priors[int(action)]) for action in actions)
        if old_total <= EPS:
            node.root_noise_applied = True
            return

        base = [float(node.priors[int(action)]) / old_total for action in actions]
        noise = self._sample_dirichlet(len(actions), self.root_dirichlet_alpha)
        scale = old_total
        for idx, action in enumerate(actions):
            mixed = (1.0 - self.root_dirichlet_eps) * base[idx] + self.root_dirichlet_eps * noise[idx]
            node.priors[int(action)] = max(1e-6, mixed * scale)
        node.root_noise_applied = True
        self.meta["root_dirichlet_applications"] += 1.0

    def _warm_start(self, node: SearchNode) -> None:
        stats = self.macro_cache.get(self._state_key(node.snapshot, node.remaining_mask))
        if stats is None or stats.visits <= 0:
            return
        node.visits = min(int(stats.visits), self.warm_vmax)
        avg = float(stats.reward_sum) / max(1, int(stats.visits))
        node.reward_sum = avg * float(node.visits)
        self.meta["macro_cache_hits"] += 1.0
        if node.visits < int(stats.visits):
            self.meta["warmstart_capped"] += 1.0

    def _area_aspect_sequence(self) -> Tuple[int, ...]:
        ranked = sorted(
            self.part_ids,
            key=lambda pid: (
                float(self.parts_by_id[int(pid)].w0) * float(self.parts_by_id[int(pid)].h0),
                max(float(self.parts_by_id[int(pid)].w0), float(self.parts_by_id[int(pid)].h0))
                / max(EPS, min(float(self.parts_by_id[int(pid)].w0), float(self.parts_by_id[int(pid)].h0))),
                float(max(self.parts_by_id[int(pid)].w0, self.parts_by_id[int(pid)].h0)),
                -int(pid),
            ),
            reverse=True,
        )
        return tuple(int(pid) for pid in ranked)

    def _constructive_seed_specs(self) -> List[Tuple[str, str, int, int]]:
        if self.baseline_restarts <= 0:
            return []
        modes = [self.decoder.place_mode]
        if self.decoder.place_mode != "maxrects_bssf":
            modes.append("maxrects_bssf")
        modes.append("blf_rand")
        orders = ["area", "size"]
        specs: List[Tuple[str, str, int, int]] = []
        for idx in range(self.baseline_restarts):
            place_mode = modes[idx % len(modes)]
            order = orders[(idx // max(1, len(modes))) % len(orders)]
            rand_topk = self.rand_topk + (idx % 2 if place_mode == "blf_rand" else 0)
            specs.append((order, place_mode, rand_topk, idx))
        return specs

    def _pack_constructive_sequence(self, order: str, place_mode: str, rand_topk: int, restart_idx: int) -> Optional[Tuple[int, ...]]:
        parts = list(self.parts)
        if order == "area_aspect":
            parts = [self.parts_by_id[int(pid)] for pid in self._area_aspect_sequence()]
            order = "raw"
        rng = random.Random(self.rng.randint(0, 10**9) ^ (restart_idx + 1) * 1315423911)
        try:
            boards = pack_multi_board(
                parts,
                self.cfg.BOARD_W,
                self.cfg.BOARD_H,
                self.allow_rot,
                trim=float(getattr(self.cfg, "TRIM", 0.0)),
                safe_gap=float(getattr(self.cfg, "SAFE_GAP", 0.0)),
                touch_tol=float(getattr(self.cfg, "TOUCH_TOL", 1e-6)),
                order=str(order),
                place_mode=str(place_mode),
                rng=rng,
                rand_topk=max(1, int(rand_topk)),
            )
        except Exception:
            return None
        sequence = tuple(int(pp.uid) for board in boards for pp in board.placed)
        return sequence if self._is_complete_sequence(sequence) else None

    def _record_constructive_warmstart(
        self,
        sequence: Sequence[int],
        snapshot: LayoutSnapshot,
        snapshots: Sequence[LayoutSnapshot],
    ) -> None:
        seq_t = tuple(int(x) for x in sequence)
        score = self._terminal_score(snapshot)
        reward_scalar = self._reward_scalar(snapshot, 0)
        self._update_global_best(seq_t, snapshot)
        self._record_layout_candidate(seq_t, snapshot, source="constructive")

        candidate = EliteCandidate(
            sequence=seq_t,
            snapshot=snapshot,
            score=score,
            tail_signature=self._elite_tail_signature(seq_t),
        )
        for idx, existing in enumerate(self.constructive_warmstarts):
            if existing.sequence == seq_t:
                if candidate.score > existing.score:
                    self.constructive_warmstarts[idx] = candidate
                break
        else:
            self.constructive_warmstarts.append(candidate)
        self.constructive_warmstarts.sort(key=lambda item: (item.score, len(item.sequence)), reverse=True)
        self.constructive_warmstarts = self.constructive_warmstarts[: max(2, self.elite_archive_k)]

        remaining_mask = int(self.full_remaining_mask)
        for idx, snap in enumerate(snapshots):
            key = self._state_key(snap, remaining_mask)
            suffix = seq_t[idx:]
            prev = self.seed_suffix_cache.get(key)
            if prev is None or score > self._terminal_score(prev[0]):
                self.seed_suffix_cache[key] = (snapshot, suffix)
            stats = self.macro_cache.setdefault(key, CacheStats())
            stats.visits += 1
            stats.reward_sum += float(reward_scalar)
            if idx < len(seq_t):
                remaining_mask &= ~int(self.part_bit[int(seq_t[idx])])

        self.meta["warmstart_candidates_kept"] = float(len(self.constructive_warmstarts))
        self.meta["warmstart_cache_states"] = float(len(self.seed_suffix_cache))

    def _prime_constructive_warmstarts(self) -> None:
        seen = set()
        if self.global_pattern_include_baseline or self.baseline_restarts > 0:
            normalized = self._normalize_sequence_candidate(self._area_aspect_sequence())
            if normalized is not None:
                seq_t, snapshot, snapshots = normalized
                seen.add(seq_t)
                self._record_constructive_warmstart(seq_t, snapshot, snapshots)
                self.meta["baseline_seed_injected"] = 1.0
        if self.baseline_restarts <= 0:
            return
        seed_specs: List[Tuple[str, str, int, int]] = [("area_aspect", self.decoder.place_mode, self.rand_topk, -1)]
        seed_specs.extend(self._constructive_seed_specs())
        for order, place_mode, rand_topk, restart_idx in seed_specs:
            sequence = self._pack_constructive_sequence(order, place_mode, rand_topk, restart_idx)
            if sequence is None:
                continue
            self.meta["warmstart_candidates_generated"] += 1.0
            if sequence in seen:
                continue
            seen.add(sequence)
            normalized = self._normalize_sequence_candidate(sequence)
            if normalized is None:
                continue
            seq_t, snapshot, snapshots = normalized
            self._record_constructive_warmstart(seq_t, snapshot, snapshots)

    def _next_empty_board(self, snapshot: LayoutSnapshot) -> Board:
        return make_empty_board(
            len(snapshot.boards) + 1,
            self.cfg.BOARD_W,
            self.cfg.BOARD_H,
            trim=float(getattr(self.cfg, "TRIM", 0.0)),
            safe_gap=float(getattr(self.cfg, "SAFE_GAP", 0.0)),
            touch_tol=float(getattr(self.cfg, "TOUCH_TOL", 1e-6)),
            place_mode=self.decoder.place_mode,
        )

    def _coarse_board_fit(self, board: Board, action: int, part) -> Tuple[float, float, int]:
        key = (_board_state_hash(board, self.sig_nd), self.decoder.part_hash[int(action)], int(self.allow_rot))
        cached = self.coarse_fit_cache.get(key)
        if cached is not None:
            self.meta["coarse_fit_cache_hits"] += 1.0
            return cached
        self.meta["coarse_fit_cache_misses"] += 1.0
        board_area = max(EPS, float(board.W) * float(board.H))
        part_area = float(part.w0) * float(part.h0)
        best_cavity = 0.0
        best_fragment = 0.0
        fit_count = 0
        for fr in board.free_rects:
            fr_area = max(EPS, float(fr.w) * float(fr.h))
            for pw, ph in self.part_orientations[int(action)]:
                if pw > fr.w + EPS or ph > fr.h + EPS:
                    continue
                fit_count += 1
                cavity = part_area / fr_area
                rem_w = max(float(fr.w) - pw, 0.0)
                rem_h = max(float(fr.h) - ph, 0.0)
                slivers = float(rem_w < 0.35 * max(EPS, min(pw, ph))) + float(rem_h < 0.35 * max(EPS, min(pw, ph)))
                area_fit = max(fr_area - pw * ph, 0.0) / board_area
                fragment = 1.0 / (1.0 + area_fit + 0.5 * slivers)
                best_cavity = max(best_cavity, cavity)
                best_fragment = max(best_fragment, fragment)
        out = (best_cavity, best_fragment, fit_count)
        self.coarse_fit_cache[key] = out
        return out

    def _action_geom_stats(self, snapshot: LayoutSnapshot, action: int) -> ActionGeomStats:
        part = self.parts_by_id[int(action)]
        board_ref_area = float(getattr(self.cfg, "BOARD_W", 1.0)) * float(getattr(self.cfg, "BOARD_H", 1.0))
        area_score = (float(part.w0) * float(part.h0)) / max(EPS, board_ref_area)
        best_existing_cavity = 0.0
        best_existing_fragment = 0.0
        fit_count = 0
        for board in snapshot.boards:
            cavity, fragment, fit_local = self._coarse_board_fit(board, int(action), part)
            best_existing_cavity = max(best_existing_cavity, cavity)
            best_existing_fragment = max(best_existing_fragment, fragment)
            fit_count += fit_local

        new_cavity, new_fragment, fit_new = self._coarse_board_fit(self._next_empty_board(snapshot), int(action), part)
        if fit_new <= 0:
            raise RuntimeError(f"Cannot place uid={part.uid} on a fresh board during heuristic evaluation.")

        cavity_score = max(best_existing_cavity, new_cavity)
        fragment_score = max(best_existing_fragment, new_fragment)
        escape_score = new_cavity if fit_count == 0 else max(0.0, new_cavity - best_existing_cavity)
        return ActionGeomStats(
            area_score=max(1e-6, area_score),
            cavity_score=max(1e-6, cavity_score),
            fragment_score=max(1e-6, fragment_score),
            escape_score=max(0.0, escape_score),
            fit_count=int(fit_count),
        )

    def _rollout_action_score(self, decoded: DecodeResult, stats: ActionGeomStats) -> float:
        frag_score = 1.0 / (1.0 + max(0.0, float(decoded.blueprint.fragmentation_penalty)))
        existing_bonus = 1.0 if not decoded.is_new_board else 0.0
        return (
            float(getattr(self.cfg, "ROLLOUT_PRIOR_W", 0.25)) * stats.prior_value(self.cfg)
            + float(getattr(self.cfg, "ROLLOUT_UTIL_W", 0.30)) * float(decoded.avg_u_gamma)
            + float(getattr(self.cfg, "ROLLOUT_CAVITY_W", 0.20)) * float(decoded.blueprint.cavity_ratio)
            + float(getattr(self.cfg, "ROLLOUT_FRAGMENT_W", 0.15)) * frag_score
            + float(getattr(self.cfg, "ROLLOUT_EXISTING_W", 0.10)) * existing_bonus
        )

    def _rollout_rank_weights(self, count: int) -> List[float]:
        count = max(0, int(count))
        if count <= 0:
            return []

        weights = [float(weight) for weight in self.rollout_rcl_weights[:count]]
        if len(weights) < count:
            tail = weights[-1] if weights else 1.0
            while len(weights) < count:
                tail = max(1e-6, 0.5 * tail)
                weights.append(tail)

        total = sum(weights)
        if total <= EPS:
            return [float(count - idx) for idx in range(count)]
        return weights

    def _ensure_candidate_actions(self, node: SearchNode, target_limit: int) -> None:
        target_limit = max(1, int(target_limit))
        if self.macro_action_enable:
            if self._search_stage(node.remaining_mask) == "early":
                if node.candidate_actions and node.candidate_width >= target_limit:
                    return
                scouted, action_meta = self._single_stage_macro_actions(node, target_limit)
                node.action_meta = dict(action_meta)
                node.candidate_actions = sorted(
                    scouted,
                    key=lambda action: self._candidate_sort_key(node, int(action)),
                    reverse=True,
                )
                if node.parent is None and self.root_dirichlet_enable and not node.root_noise_applied:
                    self._apply_root_dirichlet_noise(node)
                    node.candidate_actions = sorted(
                        node.candidate_actions,
                        key=lambda action: self._candidate_sort_key(node, int(action)),
                        reverse=True,
                    )
                node.candidate_width = len(node.candidate_actions)
                return
            if node.candidate_actions and node.candidate_width >= target_limit:
                return
            scouted, action_meta, decoded_map = self._generate_macro_actions(node.snapshot, node.remaining_mask, target_limit)
            node.action_meta = dict(action_meta)
            node.decoded.update(decoded_map)
            for key, action in action_meta.items():
                node.priors[int(key)] = float(action.prior)
                node.action_rank[int(key)] = tuple(float(x) for x in action.rank_key)
            node.candidate_actions = sorted(
                scouted,
                key=lambda action: self._candidate_sort_key(node, int(action)),
                reverse=True,
            )
            if node.parent is None and self.root_dirichlet_enable and not node.root_noise_applied:
                self._apply_root_dirichlet_noise(node)
                node.candidate_actions = sorted(
                    node.candidate_actions,
                    key=lambda action: self._candidate_sort_key(node, int(action)),
                    reverse=True,
                )
            node.candidate_width = len(node.candidate_actions)
            return
        if node.parent is None and self.root_dirichlet_enable and not node.root_noise_applied:
            scouted = self._candidate_actions_static(node.remaining_mask)
            self._ensure_priors(node, scouted)
            node.candidate_actions = sorted(scouted, key=lambda action: self._candidate_sort_key(node, int(action)), reverse=True)
            self._apply_root_dirichlet_noise(node)
            node.candidate_actions = sorted(
                node.candidate_actions,
                key=lambda action: self._candidate_sort_key(node, int(action)),
                reverse=True,
            )
            node.candidate_width = len(node.candidate_actions)
            return
        if node.candidate_actions and node.candidate_width >= target_limit:
            return
        scouted = self._scout_actions(node.remaining_mask, target_limit)
        self._ensure_priors(node, scouted)
        if self.disable_geom_prior:
            node.candidate_actions = [int(action) for action in scouted]
        else:
            node.candidate_actions = sorted(
                scouted,
                key=lambda action: self._candidate_sort_key(node, int(action)),
                reverse=True,
            )
        node.candidate_width = len(scouted)

    def _ensure_priors(self, node: SearchNode, actions: Sequence[int]) -> None:
        for action in actions:
            if int(action) in node.priors:
                continue
            stats = self._action_geom_stats(node.snapshot, int(action))
            node.action_stats[int(action)] = stats
            if self.disable_geom_prior:
                node.priors[int(action)] = 1.0
            else:
                node.priors[int(action)] = max(1e-6, stats.prior_value(self.cfg))

    def _rank_actions_for_snapshot(self, snapshot: LayoutSnapshot, remaining_mask: int, target_limit: int) -> Tuple[List[int], Dict[int, ActionGeomStats]]:
        scouted = self._scout_actions(remaining_mask, target_limit)
        stats_map: Dict[int, ActionGeomStats] = {}
        if self.disable_geom_prior:
            for action in scouted:
                stats_map[int(action)] = self._action_geom_stats(snapshot, int(action))
            return [int(action) for action in scouted[: max(1, min(target_limit, len(scouted)))]], stats_map
        ranked: List[Tuple[Tuple[float, float, float, float, int, float, int], int]] = []
        for action in scouted:
            stats = self._action_geom_stats(snapshot, int(action))
            stats_map[int(action)] = stats
            ranked.append((self._action_rank_key(int(action), stats), int(action)))
        ranked.sort(reverse=True)
        ordered = [int(action) for _, action in ranked[: max(1, min(target_limit, len(ranked)))]]
        return ordered, stats_map

    def _pw_limit(self, node: SearchNode) -> int:
        return min(self.kmax, self.k0 + int(math.ceil((max(1, node.visits)) ** self.pw_alpha)))

    def _expand(self, node: SearchNode, action: int) -> SearchNode:
        if self.macro_action_enable:
            decoded = node.decoded.get(int(action))
            if decoded is None:
                meta = node.action_meta[int(action)]
                decoded = self._apply_sequence(node.snapshot, meta.sequence, key=int(action))
                node.decoded[int(action)] = decoded
            meta = node.action_meta[int(action)]
            remaining_next = int(node.remaining_mask)
            for pid in meta.sequence:
                remaining_next &= ~int(self.part_bit[int(pid)])
            child_prefix = tuple(node.prefix_sequence) + tuple(int(pid) for pid in meta.sequence)
            child_snapshot = decoded.snapshot
        else:
            decoded = self.decoder.decode(node.snapshot, int(action))
            node.decoded[int(action)] = decoded
            remaining_next = int(node.remaining_mask) & ~int(self.part_bit[int(action)])
            child_prefix = tuple(node.prefix_sequence) + (int(action),)
            child_snapshot = decoded.snapshot
        child = SearchNode(
            snapshot=child_snapshot,
            remaining_mask=remaining_next,
            depth=int(node.depth) + 1,
            parent=node,
            action_from_parent=int(action),
            prefix_sequence=child_prefix,
        )
        self._warm_start(child)
        node.children[int(action)] = child
        return child

    def _puct_score(self, node: SearchNode, action: int, prior_sum: float) -> float:
        child = node.children[int(action)]
        q = child.q_value()
        prior = float(node.priors.get(int(action), 0.0)) / max(EPS, prior_sum)
        u = self.c_puct * prior * math.sqrt(max(1.0, float(node.visits))) / (1.0 + float(child.visits))
        return q + u

    def _select_leaf(self, root: SearchNode) -> List[SearchNode]:
        path = [root]
        node = root
        while not node.is_terminal():
            pw_limit = self._pw_limit(node)
            self._ensure_candidate_actions(node, pw_limit)
            allowed_base = node.candidate_actions[: pw_limit]
            if not self.macro_action_enable:
                self._ensure_priors(node, allowed_base)
            if self.disable_geom_prior:
                allowed = list(allowed_base)
                self.rng.shuffle(allowed)
            else:
                allowed = sorted(allowed_base, key=lambda a: (node.priors.get(int(a), 0.0), int(a)), reverse=True)
            unexpanded = [a for a in allowed if a not in node.children]
            if unexpanded:
                child = self._expand(node, int(unexpanded[0]))
                path.append(child)
                return path
            prior_sum = sum(float(node.priors.get(int(a), 0.0)) for a in allowed)
            best_action = max(allowed, key=lambda a: self._puct_score(node, int(a), prior_sum))
            node = node.children[int(best_action)]
            path.append(node)
        return path

    def _rollout(self, snapshot: LayoutSnapshot, remaining_mask: int) -> Tuple[LayoutSnapshot, List[int]]:
        if self.macro_action_enable:
            return self._rollout_macro(snapshot, remaining_mask)
        if int(remaining_mask) == 0:
            return snapshot, []

        state = snapshot
        rem = int(remaining_mask)
        total_remaining = _bit_count(rem)
        greedy_steps = int(math.floor(float(total_remaining) * self.rollout_greedy_frac))
        seq: List[int] = []
        visited: List[Tuple[Tuple, int]] = []
        step_idx = 0
        while rem:
            seeded = self.seed_suffix_cache.get(self._state_key(state, rem))
            if seeded is not None:
                self.meta["seed_rollout_cache_hits"] += 1.0
                final_snapshot, seed_suffix = seeded
                seq.extend(list(seed_suffix))
                return final_snapshot, seq
            rem_count = _bit_count(rem)
            deterministic_tail = self.rollout_topr <= 1 or rem_count <= self.rollout_det_tail
            greedy_pick = step_idx < greedy_steps
            if deterministic_tail:
                tail_key = (state.layout_hash, state.ordered_hash, len(state.boards), rem, self.rollout_topk, 1)
                cached_tail = self.rollout_cache.get(tail_key)
                if cached_tail is not None:
                    self.meta["rollout_cache_hits"] += 1.0
                    final_snapshot, tail_seq = cached_tail
                    seq.extend(list(tail_seq))
                    if visited:
                        suffix = tuple(tail_seq)
                        for key_prev, action_prev in reversed(visited):
                            suffix = (int(action_prev),) + suffix
                            self.rollout_cache[key_prev] = (final_snapshot, suffix)
                    return final_snapshot, seq
                self.meta["rollout_cache_misses"] += 1.0

            coarse, stats_map = self._rank_actions_for_snapshot(state, rem, max(1, self.rollout_topk))
            candidates: List[Tuple[float, float, int, DecodeResult]] = []
            for action in coarse:
                decoded = self.decoder.decode(state, int(action))
                stats = stats_map[int(action)]
                candidates.append(
                    (
                        float(stats.area_score),
                        self._rollout_action_score(decoded, stats),
                        int(action),
                        decoded,
                    )
                )
            candidates.sort(key=lambda item: (item[0], item[1], item[3].lex_score, item[2]), reverse=True)
            rcl_limit = 1 if deterministic_tail or greedy_pick else max(1, self.rollout_topr)
            rcl = candidates[: max(1, min(rcl_limit, len(candidates)))]
            if rcl_limit <= 1:
                _, _, action, decoded = rcl[0]
            else:
                weights = self._rollout_rank_weights(len(rcl))
                _, _, action, decoded = self.rng.choices(rcl, weights=weights, k=1)[0]
            if len(decoded.snapshot.boards) > len(state.boards) and len(state.boards) > 0:
                self._record_snapshot_patterns(state, source="rollout_closed")
            if deterministic_tail:
                visited.append(
                    (
                        (state.layout_hash, state.ordered_hash, len(state.boards), rem, self.rollout_topk, 1),
                        int(action),
                    )
                )
            state = decoded.snapshot
            rem &= ~int(self.part_bit[int(action)])
            seq.append(int(action))
            step_idx += 1
        if visited:
            suffix: Tuple[int, ...] = tuple()
            final_snapshot = state
            for key_prev, action_prev in reversed(visited):
                suffix = (int(action_prev),) + suffix
                self.rollout_cache[key_prev] = (final_snapshot, suffix)
        else:
            final_snapshot = state
        self._record_snapshot_patterns(final_snapshot, source="rollout_final")
        return final_snapshot, seq

    def _rollout_macro(self, snapshot: LayoutSnapshot, remaining_mask: int) -> Tuple[LayoutSnapshot, List[int]]:
        if int(remaining_mask) == 0:
            return snapshot, []
        state = snapshot
        rem = int(remaining_mask)
        seq: List[int] = []
        visited: List[Tuple[Tuple, Tuple[int, ...]]] = []
        while rem:
            if self._search_stage(rem) == "early":
                actions, stats_map = self._rank_actions_for_snapshot(
                    state,
                    rem,
                    max(1, min(self.rollout_topk, self.macro_action_topk)),
                )
                if not actions:
                    break
                candidates: List[Tuple[float, float, int, DecodeResult]] = []
                greedy_pick = self.rollout_greedy_frac > 0.0 and self.rng.random() < self.rollout_greedy_frac
                for action in actions:
                    decoded = self.decoder.decode(state, int(action))
                    stats = stats_map[int(action)]
                    candidates.append(
                        (
                            float(stats.area_score),
                            self._rollout_action_score(decoded, stats),
                            int(action),
                            decoded,
                        )
                    )
                candidates.sort(key=lambda item: (item[0], item[1], item[3].lex_score, item[2]), reverse=True)
                rcl_limit = 1 if greedy_pick else max(1, self.rollout_topr)
                rcl = candidates[: max(1, min(rcl_limit, len(candidates)))]
                if rcl_limit <= 1:
                    _, _, action, decoded = rcl[0]
                else:
                    weights = self._rollout_rank_weights(len(rcl))
                    _, _, action, decoded = self.rng.choices(rcl, weights=weights, k=1)[0]
                if len(decoded.snapshot.boards) > len(state.boards) and len(state.boards) > 0:
                    self._record_snapshot_patterns(state, source="rollout_closed")
                state = decoded.snapshot
                rem &= ~int(self.part_bit[int(action)])
                seq.append(int(action))
                continue
            seeded = self.seed_suffix_cache.get(self._state_key(state, rem))
            if seeded is not None:
                self.meta["seed_rollout_cache_hits"] += 1.0
                final_snapshot, seed_suffix = seeded
                seq.extend(list(seed_suffix))
                return final_snapshot, seq
            rem_count = _bit_count(rem)
            deterministic_tail = rem_count <= self.rollout_det_tail
            cache_key = (state.layout_hash, state.ordered_hash, len(state.boards), rem, self.rollout_topk, 2)
            if deterministic_tail:
                cached_tail = self.rollout_cache.get(cache_key)
                if cached_tail is not None:
                    self.meta["rollout_cache_hits"] += 1.0
                    final_snapshot, tail_seq = cached_tail
                    seq.extend(list(tail_seq))
                    if visited:
                        suffix = tuple(tail_seq)
                        for key_prev, seq_prev in reversed(visited):
                            suffix = tuple(seq_prev) + suffix
                            self.rollout_cache[key_prev] = (final_snapshot, suffix)
                    return final_snapshot, seq
                self.meta["rollout_cache_misses"] += 1.0

            action_keys, action_meta, decoded_map = self._generate_macro_actions(
                state,
                rem,
                max(1, min(self.macro_action_topk, self.rollout_topk)),
            )
            if not action_keys:
                break
            ranked = [
                (
                    float(action_meta[int(key)].prior),
                    tuple(float(x) for x in action_meta[int(key)].rank_key),
                    int(key),
                    action_meta[int(key)],
                    decoded_map[int(key)],
                )
                for key in action_keys
            ]
            ranked.sort(reverse=True)
            if deterministic_tail:
                _, _, key, meta, decoded = ranked[0]
            else:
                limit = max(1, min(self.rollout_topr, len(ranked)))
                rcl = ranked[:limit]
                weights = self._rollout_rank_weights(len(rcl))
                _, _, key, meta, decoded = self.rng.choices(rcl, weights=weights, k=1)[0]
            if len(decoded.snapshot.boards) > len(state.boards) and len(state.boards) > 0:
                self._record_snapshot_patterns(state, source="rollout_closed")
            if deterministic_tail:
                visited.append((cache_key, tuple(int(pid) for pid in meta.sequence)))
            state = decoded.snapshot
            for pid in meta.sequence:
                rem &= ~int(self.part_bit[int(pid)])
                seq.append(int(pid))
            self.meta["macro_rollout_steps"] += 1.0
        final_snapshot = state
        if visited:
            suffix: Tuple[int, ...] = tuple()
            for key_prev, seq_prev in reversed(visited):
                suffix = tuple(seq_prev) + suffix
                self.rollout_cache[key_prev] = (final_snapshot, suffix)
        self._record_snapshot_patterns(final_snapshot, source="rollout_final")
        return final_snapshot, seq

    def _update_global_best(self, sequence: Sequence[int], snapshot: LayoutSnapshot) -> None:
        if not self._is_complete_sequence(sequence):
            return
        self._record_elite_candidate(sequence, snapshot)
        score = self._terminal_score(snapshot)
        if self.global_best_score is None or score > self.global_best_score:
            self.global_best_score = score
            self.global_best_sequence = tuple(int(x) for x in sequence)
            self.global_best_snapshot = snapshot

    def _backup(self, path: Sequence[SearchNode], reward_scalar: float) -> None:
        for node in path:
            node.visits += 1
            node.reward_sum += float(reward_scalar)
            key = self._state_key(node.snapshot, node.remaining_mask)
            stats = self.macro_cache.setdefault(key, CacheStats())
            stats.visits += 1
            stats.reward_sum += float(reward_scalar)

    def _detach_root(self, node: SearchNode) -> SearchNode:
        node.parent = None
        node.action_from_parent = None
        node.root_noise_applied = False
        return node

    def _rank_children(self, node: SearchNode) -> List[Tuple[int, SearchNode]]:
        return sorted(
            node.children.items(),
            key=lambda kv: (kv[1].visits, kv[1].reward_sum / max(1, kv[1].visits)),
            reverse=True,
        )

    def _has_commit_dominance(self, ranked: Sequence[Tuple[int, SearchNode]]) -> bool:
        if not ranked:
            return False
        if len(ranked) == 1:
            return True
        ratio12 = float(ranked[0][1].visits) / max(1.0, float(ranked[1][1].visits))
        if len(ranked) == 2:
            return ratio12 >= self.commit_ratio
        ratio13 = float(ranked[0][1].visits) / max(1.0, float(ranked[2][1].visits))
        return ratio12 >= self.commit_ratio and ratio13 >= self.commit_ratio

    def _can_commit_node(self, node: SearchNode, ranked: Sequence[Tuple[int, SearchNode]]) -> bool:
        if not ranked:
            return False
        if int(node.visits) < self.commit_min_root_visits:
            return False
        if int(ranked[0][1].visits) < self.commit_min_child_visits:
            return False
        return self._has_commit_dominance(ranked)

    def _commit_chain(self, root: SearchNode, *, force: bool = False) -> List[Tuple[int, SearchNode]]:
        ranked_root = self._rank_children(root)
        if not ranked_root:
            return []
        if not force and not self._can_commit_node(root, ranked_root):
            return []

        best_action, best_child = ranked_root[0]
        chain: List[Tuple[int, SearchNode]] = [(int(best_action), best_child)]
        if self.commit_max_steps <= 1:
            return chain

        node = best_child
        while node.children and len(chain) < self.commit_max_steps:
            ranked = self._rank_children(node)
            if not ranked:
                break
            if force:
                if not self._has_commit_dominance(ranked):
                    break
            elif not self._can_commit_node(node, ranked):
                break
            best_action, best_child = ranked[0]
            chain.append((int(best_action), best_child))
            node = best_child
        return chain[: max(1, self.commit_max_steps)]

    def _run_simulations(self, root: SearchNode, n_sims: int) -> None:
        for _ in range(max(1, n_sims)):
            path = self._select_leaf(root)
            leaf = path[-1]
            final_snapshot, rollout_seq = self._rollout(leaf.snapshot, leaf.remaining_mask)
            full_sequence = tuple(leaf.prefix_sequence) + tuple(int(x) for x in rollout_seq)
            self._record_layout_candidate(full_sequence, final_snapshot, source="simulation")
            self._update_global_best(full_sequence, final_snapshot)
            reward_scalar = self._reward_scalar(final_snapshot, 0)
            self._backup(path, reward_scalar)
            self.meta["simulations"] += 1.0

    def _round_sim_budget(self) -> int:
        return max(
            self.sims_per_step,
            self.commit_min_root_visits,
            self.commit_max_round_sims,
        )

    def _post_mcts_reserve_s(self) -> float:
        if self.solver_time_limit_s <= 0.0 or not math.isfinite(self.solver_time_limit_s):
            return 0.0
        reserve = self.solver_time_limit_s * self.post_mcts_reserve_frac
        reserve = max(reserve, self.post_mcts_reserve_min_s)
        reserve = min(reserve, self.post_mcts_reserve_max_s)
        reserve = min(reserve, max(0.0, self.solver_time_limit_s - 0.25))
        return max(0.0, reserve)

    def _elite_tail_signature(self, sequence: Sequence[int]) -> Tuple[int, ...]:
        seq_t = tuple(int(x) for x in sequence)
        tail_len = min(len(seq_t), self.elite_tail_len)
        if tail_len <= 0:
            return tuple()
        return seq_t[-tail_len:]

    def _terminal_score(self, snapshot: LayoutSnapshot) -> Tuple[float, ...]:
        return _global_lex_score(snapshot, sparse_u=float(getattr(self.cfg, "SPILL_SPARSE_UTIL", 0.55)))

    def _reward_scalar(self, snapshot: LayoutSnapshot, remaining_mask: int = 0) -> float:
        if not self.macro_action_enable:
            return _scalar_reward(snapshot, self.big_m)
        weights = self._macro_stage_weights(remaining_mask)
        util_bonus = (
            0.45 * weights["u_global"] * _u_global(snapshot)
            + 0.35 * weights["u_avg"] * _u_avg_excl_tail(snapshot, 1)
            + 0.20 * weights["u_last"] * _u_last(snapshot)
        )
        return -float(self.big_m) * float(len(snapshot.boards)) + float(util_bonus)

    def _layout_board_signature(self, snapshot: LayoutSnapshot) -> Tuple[Tuple[int, ...], ...]:
        boards = snapshot.boards.materialize()
        return tuple(
            sorted(
                tuple(sorted(int(pp.uid) for pp in board.placed))
                for board in boards
                if board.placed
            )
        )

    def _record_layout_candidate(self, sequence: Sequence[int], snapshot: LayoutSnapshot, *, source: str) -> None:
        seq_t = tuple(int(x) for x in sequence)
        if not self._is_complete_sequence(seq_t):
            return
        board_signature = self._layout_board_signature(snapshot)
        if not board_signature:
            return
        candidate = HarvestedLayoutCandidate(
            sequence=seq_t,
            snapshot=snapshot,
            score=self._terminal_score(snapshot),
            board_signature=board_signature,
            source=str(source),
        )
        for idx, existing in enumerate(self.harvested_layouts):
            if existing.board_signature != board_signature:
                continue
            if candidate.score > existing.score:
                self.harvested_layouts[idx] = candidate
            break
        else:
            self.harvested_layouts.append(candidate)
        self.harvested_layouts.sort(key=lambda item: (item.score, -len(item.board_signature), item.source), reverse=True)
        self.harvested_layouts = self.harvested_layouts[: self.global_layout_pool_limit]
        self.meta["layout_harvest_layouts"] = float(len(self.harvested_layouts))
        self._record_snapshot_patterns(snapshot, source=f"layout:{source}")

    def _record_pattern_candidate(self, pattern: PatternCandidate) -> None:
        if not self.global_pattern_memory_enable:
            return
        prev = self.global_pattern_memory.get(pattern.part_ids)
        if prev is None or (
            pattern.util,
            pattern.used_area,
            -len(pattern.sequence),
            pattern.source,
        ) > (
            prev.util,
            prev.used_area,
            -len(prev.sequence),
            prev.source,
        ):
            self.global_pattern_memory[pattern.part_ids] = pattern
        if len(self.global_pattern_memory) > self.global_pattern_memory_limit:
            ordered = sorted(
                self.global_pattern_memory.values(),
                key=lambda item: (len(item.part_ids), item.util, item.used_area, -len(item.sequence), item.source),
                reverse=True,
            )
            self.global_pattern_memory = {
                pattern.part_ids: pattern for pattern in ordered[: self.global_pattern_memory_limit]
            }
        self.meta["global_pattern_memory_patterns"] = float(len(self.global_pattern_memory))

    def _record_snapshot_patterns(self, snapshot: LayoutSnapshot, *, source: str) -> None:
        if not self.global_pattern_memory_enable:
            return
        for board in snapshot.boards.materialize():
            if not board.placed:
                continue
            util = float(board_utilization(board))
            if len(board.placed) > 1 and util < self.global_pattern_memory_min_util:
                continue
            sequence = tuple(int(pp.uid) for pp in board.placed)
            pattern = PatternCandidate(
                part_ids=tuple(sorted(sequence)),
                sequence=sequence,
                used_area=float(board_used_area(board)),
                util=util,
                waste_area=max(0.0, float(board.W) * float(board.H) - float(board_used_area(board))),
                place_mode=str(getattr(board, "place_mode", self.decoder.place_mode)),
                source=str(source),
            )
            self._record_pattern_candidate(pattern)

    def _record_elite_candidate(self, sequence: Sequence[int], snapshot: LayoutSnapshot) -> None:
        seq_t = tuple(int(x) for x in sequence)
        if not seq_t:
            return
        self._record_layout_candidate(seq_t, snapshot, source="elite")
        score = self._terminal_score(snapshot)
        tail_sig = self._elite_tail_signature(seq_t)
        candidate = EliteCandidate(sequence=seq_t, snapshot=snapshot, score=score, tail_signature=tail_sig)
        for idx, existing in enumerate(self.elite_archive):
            if existing.tail_signature != tail_sig:
                continue
            if candidate.score > existing.score:
                self.elite_archive[idx] = candidate
            self.elite_archive.sort(key=lambda item: (item.score, len(item.sequence)), reverse=True)
            self.elite_archive = self.elite_archive[: self.elite_archive_k]
            self.meta["elite_archive_size"] = float(len(self.elite_archive))
            return
        self.elite_archive.append(candidate)
        self.elite_archive.sort(key=lambda item: (item.score, len(item.sequence)), reverse=True)
        self.elite_archive = self.elite_archive[: self.elite_archive_k]
        self.meta["elite_archive_size"] = float(len(self.elite_archive))

    def _sequence_first_diff(self, seq_a: Sequence[int], seq_b: Sequence[int]) -> int:
        stop = min(len(seq_a), len(seq_b))
        for idx in range(stop):
            if int(seq_a[idx]) != int(seq_b[idx]):
                return idx
        return stop

    def _insert_positions(self, idx: int, pivot: int) -> List[int]:
        max_positions = max(4, int(getattr(self.cfg, "LOCAL_INSERT_MAX_POS", 8)))
        insert_delta = int(getattr(self.cfg, "LOCAL_CLEAR_INSERT_DELTA", 4))
        positions = {
            0,
            max(0, pivot),
            max(0, pivot - insert_delta),
            max(0, pivot // 2),
            max(0, idx - int(getattr(self.cfg, "LOCAL_PULL_DELTA_1", 3))),
            max(0, idx - int(getattr(self.cfg, "LOCAL_PULL_DELTA_2", 5))),
            max(0, idx - int(getattr(self.cfg, "LOCAL_PULL_DELTA_3", 10))),
        }
        for pos in self._position_ladder(pivot, insert_delta, max_positions):
            positions.add(pos)
        return sorted(pos for pos in positions if pos < idx)

    def _move_part(self, sequence: Sequence[int], from_idx: int, to_idx: int) -> Tuple[int, ...]:
        seq = list(int(x) for x in sequence)
        part = seq.pop(int(from_idx))
        seq.insert(int(to_idx), part)
        return tuple(seq)

    def _swap_parts(self, sequence: Sequence[int], idx_a: int, idx_b: int) -> Tuple[int, ...]:
        if idx_a == idx_b:
            return tuple(int(x) for x in sequence)
        seq = list(int(x) for x in sequence)
        seq[int(idx_a)], seq[int(idx_b)] = seq[int(idx_b)], seq[int(idx_a)]
        return tuple(seq)

    def _is_complete_sequence(self, sequence: Sequence[int]) -> bool:
        seq_t = tuple(int(x) for x in sequence)
        if len(seq_t) != len(self.full_sequence_signature):
            return False
        return tuple(sorted(seq_t)) == self.full_sequence_signature

    def _chain_signature(self, chain: Sequence[Tuple[int, SearchNode]]) -> Tuple[int, ...]:
        return tuple(int(action) for action, _ in chain)

    def _normalize_sequence_candidate(
        self,
        sequence: Sequence[int],
    ) -> Optional[Tuple[Tuple[int, ...], LayoutSnapshot, List[LayoutSnapshot]]]:
        seq_t = tuple(int(x) for x in sequence)
        if not self._is_complete_sequence(seq_t):
            return None
        try:
            snapshot, snapshots = self.decoder.replay_order(seq_t)
        except Exception:
            return None
        return seq_t, snapshot, snapshots

    def _tail_board_indices(self, snapshot: LayoutSnapshot, tail_boards: int) -> List[int]:
        boards = snapshot.boards.materialize()
        if not boards:
            return []
        n_tail = min(len(boards), max(1, int(tail_boards)))
        start = max(0, len(boards) - n_tail)
        return list(range(start, len(boards)))

    def _tail_part_ids(self, snapshot: LayoutSnapshot, sequence: Sequence[int], tail_boards: int) -> List[int]:
        boards = snapshot.boards.materialize()
        idx_map = {int(pid): i for i, pid in enumerate(sequence)}
        out: List[Tuple[int, int]] = []
        for board_idx in self._tail_board_indices(snapshot, tail_boards):
            for pp in boards[board_idx].placed:
                pid = int(pp.uid)
                out.append((idx_map.get(pid, 10**9), pid))
        out.sort()
        return [pid for _, pid in out]

    def _repair_priority(self, snapshot: LayoutSnapshot, base_board_count: int) -> Tuple[float, ...]:
        boards = snapshot.boards.materialize()
        sparse_u = float(getattr(self.cfg, "SPILL_SPARSE_UTIL", 0.55))
        cleared = max(0, int(base_board_count) - len(boards))
        sparse_count = sum(1 for board in boards if board_utilization(board) < sparse_u)
        tail = boards[-min(2, len(boards)) :] if boards else []
        tail_util_sum = sum(board_utilization(board) for board in tail)
        last_util = board_utilization(boards[-1]) if boards else 0.0
        return (
            float(cleared),
            -float(last_util),
            -float(tail_util_sum),
            -float(sparse_count),
            -float(len(boards)),
            float(_avg_u_gamma(snapshot)),
        )

    def _repair_accepts(self, best_score: Tuple[float, ...], cand_score: Tuple[float, ...]) -> bool:
        return cand_score > best_score

    def _estimate_blocker_score(self, pid: int, front_rects: Sequence, seq_pos: int) -> float:
        part = self.parts_by_id[int(pid)]
        area = float(part.w0) * float(part.h0)
        aspect = max(float(part.w0), float(part.h0)) / max(EPS, min(float(part.w0), float(part.h0)))
        compat_best = 0.0
        for fr in front_rects:
            compat_best = max(compat_best, _compat(part, fr, self.allow_rot))
        return area * (1.0 + 0.1 * aspect) + float(seq_pos) * 1e-6 + (1.0 - compat_best)

    def _blocker_ids(self, snapshot: LayoutSnapshot, sequence: Sequence[int], tail_ids: Sequence[int], tail_boards: int) -> List[int]:
        if not tail_ids:
            return []
        idx_map = {int(pid): i for i, pid in enumerate(sequence)}
        positions = [idx_map[pid] for pid in tail_ids if pid in idx_map]
        if not positions:
            return []
        lo = max(0, min(positions) - self.repair_blocker_window)
        hi = min(positions)
        boards = snapshot.boards.materialize()
        tail_indices = self._tail_board_indices(snapshot, tail_boards)
        tail_start = tail_indices[0] if tail_indices else len(boards)
        front_rects = [fr for board in boards[:tail_start] for fr in board.free_rects]
        scored: List[Tuple[float, int]] = []
        seen = set()
        for idx in range(lo, hi):
            pid = int(sequence[idx])
            if pid in seen or pid in tail_ids:
                continue
            seen.add(pid)
            scored.append((self._estimate_blocker_score(pid, front_rects, idx), pid))
        scored.sort(reverse=True)
        return [pid for _, pid in scored[: self.repair_blocker_topk]]

    def _focus_pivot(self, sequence: Sequence[int], focus_ids: Sequence[int]) -> int:
        idx_map = {int(pid): i for i, pid in enumerate(sequence)}
        positions = [idx_map[pid] for pid in focus_ids if pid in idx_map]
        if not positions:
            return 0
        return min(positions)

    def _beam_state_key(self, state: RepairBeamState) -> Tuple:
        return (state.sequence, state.remaining_tail, state.ejected_items)

    def _prune_repair_states(self, states: Sequence[RepairBeamState]) -> List[RepairBeamState]:
        ranked: Dict[Tuple, RepairBeamState] = {}
        for state in states:
            key = self._beam_state_key(state)
            prev = ranked.get(key)
            if prev is None or (state.priority, -state.depth) > (prev.priority, -prev.depth):
                ranked[key] = state
        ordered = sorted(
            ranked.values(),
            key=lambda item: (item.priority, self._terminal_score(item.snapshot), -item.depth),
            reverse=True,
        )
        return ordered[: self.repair_beam_width]

    def _constructive_state_key(self, state: ConstructiveRepairState) -> Tuple:
        return (state.placed_early, state.remaining_tail, state.ejected_blockers)

    def _prune_constructive_states(self, states: Sequence[ConstructiveRepairState]) -> List[ConstructiveRepairState]:
        ranked: Dict[Tuple, ConstructiveRepairState] = {}
        for state in states:
            key = self._constructive_state_key(state)
            prev = ranked.get(key)
            if prev is None or (state.priority, self._terminal_score(state.completion_snapshot), -state.depth) > (
                prev.priority,
                self._terminal_score(prev.completion_snapshot),
                -prev.depth,
            ):
                ranked[key] = state
        ordered = sorted(
            ranked.values(),
            key=lambda item: (item.priority, self._terminal_score(item.completion_snapshot), -item.depth),
            reverse=True,
        )
        return ordered[: self.repair_beam_width]

    def _board_first_positions(self, snapshot: LayoutSnapshot, sequence: Sequence[int]) -> List[int]:
        boards = snapshot.boards.materialize()
        idx_map = {int(pid): i for i, pid in enumerate(sequence)}
        seq_len = len(sequence)
        out: List[int] = []
        for board in boards:
            positions = [idx_map[int(pp.uid)] for pp in board.placed if int(pp.uid) in idx_map]
            out.append(min(positions) if positions else seq_len)
        return out

    def _prefix_cavity_area(self, boards: Sequence[Board], upto_board_idx: int) -> float:
        return sum(float(fr.w) * float(fr.h) for board in boards[: max(0, int(upto_board_idx))] for fr in board.free_rects)

    def _spill_suffix_used_area(self, snapshot: LayoutSnapshot) -> float:
        boards = snapshot.boards.materialize()
        spill = self._find_spill_boundary(snapshot)
        if spill is None or spill >= len(boards):
            return 0.0
        return sum(board_used_area(board) for board in boards[spill:])

    def _region_bucket(self, start_idx: int) -> int:
        return max(0, int(start_idx)) // self.region_diversity_bucket

    def _region_proposals(self, snapshot: LayoutSnapshot, sequence: Sequence[int]) -> List[RegionProposal]:
        boards = snapshot.boards.materialize()
        seq_t = tuple(int(pid) for pid in sequence)
        if len(boards) < 2 or len(seq_t) < 2:
            return []

        board_positions = self._board_first_positions(snapshot, seq_t)
        if not board_positions:
            return []

        candidate_starts: Dict[int, str] = {}

        def add_start(start_idx: int, source: str) -> None:
            start_idx = min(len(seq_t) - 1, max(0, int(start_idx)))
            if start_idx not in candidate_starts:
                candidate_starts[start_idx] = source

        if len(boards) >= 2:
            add_start(min(board_positions[-2:]), "last2")
        if len(boards) >= 3:
            add_start(min(board_positions[-3:]), "last3")
        spill = self._find_spill_boundary(snapshot)
        if spill is not None and spill < len(board_positions):
            add_start(board_positions[int(spill)], "spill")

        seed_starts = list(candidate_starts.items())
        for start_idx, source in seed_starts:
            for backoff in self.region_start_backoffs:
                if backoff <= 0:
                    continue
                add_start(max(0, start_idx - int(backoff)), f"{source}-b{int(backoff)}")

        board_area = float(boards[0].W) * float(boards[0].H)
        sparse_u = float(getattr(self.cfg, "SPILL_SPARSE_UTIL", 0.55))
        proposals: List[RegionProposal] = []
        for start_idx, source in candidate_starts.items():
            suffix_ids = set(int(pid) for pid in seq_t[start_idx:])
            covered_boards = [
                idx for idx, board in enumerate(boards) if any(int(pp.uid) in suffix_ids for pp in board.placed)
            ]
            if not covered_boards:
                continue
            start_board = min(covered_boards)
            suffix_boards = boards[start_board:]
            suffix_equiv = sum(board_used_area(board) for board in suffix_boards) / max(EPS, board_area)
            nearest_integer = max(1.0, float(round(suffix_equiv)))
            integer_gap = -abs(suffix_equiv - nearest_integer)
            sparse_count = sum(1 for board in suffix_boards if board_utilization(board) < sparse_u)
            prefix_cavity = self._prefix_cavity_area(boards, start_board) / max(EPS, board_area)
            proposals.append(
                RegionProposal(
                    start_idx=int(start_idx),
                    span=len(seq_t) - int(start_idx),
                    score=(float(integer_gap), float(sparse_count), float(prefix_cavity), float(start_idx)),
                    bucket=self._region_bucket(start_idx),
                    source=source,
                )
            )

        proposals.sort(key=lambda item: (item.score, item.start_idx), reverse=True)
        unique: Dict[int, RegionProposal] = {}
        for proposal in proposals:
            prev = unique.get(int(proposal.start_idx))
            if prev is None or proposal.score > prev.score:
                unique[int(proposal.start_idx)] = proposal
        ordered = sorted(unique.values(), key=lambda item: (item.score, item.start_idx), reverse=True)
        return ordered[: self.region_proposal_topk]

    def _region_priority(self, snapshot: LayoutSnapshot) -> Tuple[float, ...]:
        boards = snapshot.boards.materialize()
        sparse_u = float(getattr(self.cfg, "SPILL_SPARSE_UTIL", 0.55))
        tail = boards[-min(2, len(boards)) :] if boards else []
        tail_util_sum = sum(board_utilization(board) for board in tail)
        sparse_count = sum(1 for board in boards if board_utilization(board) < sparse_u)
        spill_area = self._spill_suffix_used_area(snapshot)
        return (
            -float(len(boards)),
            float(tail_util_sum),
            -float(sparse_count),
            -float(spill_area),
            float(_avg_u_gamma(snapshot)),
        )

    def _region_accepts(
        self,
        best_priority: Tuple[float, ...],
        best_score: Tuple[float, ...],
        cand_priority: Tuple[float, ...],
        cand_score: Tuple[float, ...],
    ) -> bool:
        return (cand_priority, cand_score) > (best_priority, best_score)

    def _region_state_key(self, state: RegionRebuildState) -> Tuple:
        return (state.placed_early, state.remaining_region_items, state.parked_items)

    def _prune_region_states(self, states: Sequence[RegionRebuildState]) -> List[RegionRebuildState]:
        ranked: Dict[Tuple, RegionRebuildState] = {}
        for state in states:
            key = self._region_state_key(state)
            prev = ranked.get(key)
            if prev is None or (state.priority, self._terminal_score(state.completion_snapshot), -state.depth) > (
                prev.priority,
                self._terminal_score(prev.completion_snapshot),
                -prev.depth,
            ):
                ranked[key] = state
        ordered = sorted(
            ranked.values(),
            key=lambda item: (item.priority, self._terminal_score(item.completion_snapshot), -item.depth),
            reverse=True,
        )
        return ordered[: self.region_rebuild_beam_width]

    def _region_completion(
        self,
        prefix_sequence: Sequence[int],
        prefix_snapshots: Sequence[LayoutSnapshot],
        placed_early: Sequence[int],
        remaining_region_items: Sequence[int],
        parked_items: Sequence[int],
        current_snapshot: LayoutSnapshot,
    ) -> Tuple[Tuple[int, ...], LayoutSnapshot, List[LayoutSnapshot]]:
        suffix = tuple(int(pid) for pid in remaining_region_items) + tuple(int(pid) for pid in parked_items)
        final_snapshot, suffix_snapshots = self.decoder.replay_order(suffix, prefix_snapshot=current_snapshot)
        full_sequence = tuple(int(pid) for pid in prefix_sequence) + tuple(int(pid) for pid in placed_early) + suffix
        full_snapshots = list(prefix_snapshots)
        full_snapshots.extend(suffix_snapshots[1:])
        return full_sequence, final_snapshot, full_snapshots

    def _region_critical_candidates(self, snapshot: LayoutSnapshot, remaining_region_items: Sequence[int]) -> List[int]:
        if not remaining_region_items:
            return []
        boards = snapshot.boards.materialize()
        front_rects = [fr for board in boards for fr in board.free_rects]
        scan_limit = min(len(remaining_region_items), max(self.region_rebuild_actions_per_state * 2, 16))
        scored: List[Tuple[float, float, float, int]] = []
        for seq_pos, pid in enumerate(remaining_region_items[:scan_limit]):
            part = self.parts_by_id[int(pid)]
            compat_best = 0.0
            for fr in front_rects:
                compat_best = max(compat_best, _compat(part, fr, self.allow_rot))
            scored.append((compat_best, float(self.static_action_score[int(pid)]), -float(seq_pos), int(pid)))
        scored.sort(reverse=True)
        return [pid for _, _, _, pid in scored[: min(4, len(scored))]]

    def _region_blocker_candidates(
        self,
        snapshot: LayoutSnapshot,
        remaining_region_items: Sequence[int],
        critical_ids: Sequence[int],
    ) -> List[int]:
        if not remaining_region_items:
            return []
        boards = snapshot.boards.materialize()
        front_rects = [fr for board in boards for fr in board.free_rects]
        scan_limit = min(len(remaining_region_items), max(self.repair_blocker_window, self.region_rebuild_actions_per_state * 2))
        critical_set = {int(pid) for pid in critical_ids}
        scored: List[Tuple[float, int]] = []
        for seq_pos, pid in enumerate(remaining_region_items[:scan_limit]):
            if int(pid) in critical_set:
                continue
            scored.append((self._estimate_blocker_score(int(pid), front_rects, seq_pos), int(pid)))
        scored.sort(reverse=True)
        return [pid for _, pid in scored[: min(3, len(scored))]]

    def _expand_region_actions(
        self,
        state: RegionRebuildState,
    ) -> List[Tuple[str, int, Optional[int]]]:
        remaining = tuple(int(pid) for pid in state.remaining_region_items)
        if not remaining:
            return []
        critical_ids = self._region_critical_candidates(state.snapshot, remaining)
        blocker_ids = self._region_blocker_candidates(state.snapshot, remaining, critical_ids)
        pos_map = {int(pid): idx for idx, pid in enumerate(remaining)}
        scored: List[Tuple[Tuple[float, ...], str, int, Optional[int]]] = []
        seen = set()

        for pid in critical_ids:
            pos = pos_map.get(int(pid), len(remaining))
            key = ("place", int(pid), None)
            if key in seen:
                continue
            seen.add(key)
            scored.append(((3.0, float(self.static_action_score[int(pid)]), -float(pos)), "place", int(pid), None))

        for pid in blocker_ids:
            pos = pos_map.get(int(pid), len(remaining))
            key = ("park", int(pid), None)
            if key in seen:
                continue
            seen.add(key)
            scored.append(((2.0, float(self.static_action_score[int(pid)]), -float(pos)), "park", int(pid), None))

        for pid in critical_ids[:2]:
            pid_pos = pos_map.get(int(pid))
            if pid_pos is None:
                continue
            for blocker in blocker_ids[:2]:
                blocker_pos = pos_map.get(int(blocker))
                if blocker_pos is None or blocker_pos >= pid_pos:
                    continue
                key = ("pair", int(pid), int(blocker))
                if key in seen:
                    continue
                seen.add(key)
                scored.append(
                    (
                        (
                            1.0,
                            float(self.static_action_score[int(pid)]),
                            float(self.static_action_score[int(blocker)]),
                            -float(pid_pos - blocker_pos),
                        ),
                        "pair",
                        int(pid),
                        int(blocker),
                    )
                )

        scored.sort(reverse=True)
        return [(kind, pid, blocker) for _, kind, pid, blocker in scored[: self.region_rebuild_actions_per_state]]

    def _replay_from_prefix(self, full_sequence: Sequence[int], start_idx: int, snapshots: Sequence[LayoutSnapshot]) -> Tuple[LayoutSnapshot, List[LayoutSnapshot]]:
        prefix_snapshot = snapshots[start_idx]
        suffix = full_sequence[start_idx:]
        final_snapshot, suffix_snapshots = self.decoder.replay_order(suffix, prefix_snapshot=prefix_snapshot)
        out_snapshots = list(snapshots[: start_idx + 1])
        out_snapshots.extend(suffix_snapshots[1:])
        return final_snapshot, out_snapshots

    def _find_spill_boundary(self, snapshot: LayoutSnapshot) -> Optional[int]:
        boards = snapshot.boards.materialize()
        if len(boards) < 2:
            return None
        utils = [board_utilization(board) for board in boards]
        best_idx = None
        best_drop = 0.0
        for i in range(len(utils) - 1):
            drop = utils[i] - utils[i + 1]
            if drop > best_drop:
                best_drop = drop
                best_idx = i + 1
        if best_idx is not None and best_drop >= float(getattr(self.cfg, "SPILL_DROP_THRESHOLD", 0.12)):
            return best_idx
        sparse_u = float(getattr(self.cfg, "SPILL_SPARSE_UTIL", 0.55))
        for i, u in enumerate(utils):
            if u < sparse_u:
                return i
        return None

    def _critical_items(self, snapshot: LayoutSnapshot) -> List[int]:
        boards = snapshot.boards.materialize()
        spill = self._find_spill_boundary(snapshot)
        if spill is None or spill >= len(boards):
            return []
        front_rects = [fr for board in boards[:spill] for fr in board.free_rects]
        if not front_rects:
            return []
        scored: List[Tuple[float, float, float, int]] = []
        for board in boards[spill:]:
            util = board_utilization(board)
            for pp in board.placed:
                part = self.parts_by_id[int(pp.uid)]
                fill_potential = 0.0
                for fr in front_rects:
                    fill_potential = max(fill_potential, _compat(part, fr, bool(getattr(self.cfg, "ALLOW_ROT", True))))
                area = float(part.w0) * float(part.h0)
                scored.append((fill_potential, area, 1.0 - util, int(pp.uid)))
        scored.sort(reverse=True)
        topm = int(getattr(self.cfg, "LOCAL_CRITICAL_TOPM", 8))
        return [uid for _, _, _, uid in scored[:topm]]

    def _position_ladder(self, start: int, delta: int, max_positions: int) -> List[int]:
        out: List[int] = []
        pos = max(0, int(start))
        for _ in range(max(1, int(max_positions))):
            out.append(pos)
            if pos <= 0:
                break
            pos = max(0, pos - max(1, int(delta)))
        return out

    def _pull_forward_candidates(self, sequence: Sequence[int], part_id: int, idx: int, pivot: int) -> List[Tuple[int, ...]]:
        del part_id
        max_positions = max(4, int(getattr(self.cfg, "LOCAL_INSERT_MAX_POS", 8)))
        positions = {
            max(0, pivot),
            max(0, pivot - int(getattr(self.cfg, "LOCAL_PULL_DELTA_1", 3))),
            max(0, idx - int(getattr(self.cfg, "LOCAL_PULL_DELTA_2", 5))),
            max(0, idx - int(getattr(self.cfg, "LOCAL_PULL_DELTA_3", 10))),
            0,
        }
        for pos in self._position_ladder(pivot, int(getattr(self.cfg, "LOCAL_CLEAR_INSERT_DELTA", 4)), max_positions):
            positions.add(pos)
        out: List[Tuple[int, ...]] = []
        for pos in sorted(positions):
            if pos >= idx:
                continue
            seq = list(sequence)
            part = seq.pop(idx)
            seq.insert(pos, part)
            out.append(tuple(seq))
        return out

    def _eject_candidates(self, sequence: Sequence[int], part_id: int, idx: int) -> List[Tuple[int, ...]]:
        del part_id
        out: List[Tuple[int, ...]] = []
        window = int(getattr(self.cfg, "LOCAL_EJECT_WINDOW", 18))
        lo = max(0, idx - window)
        blockers = []
        for j in range(lo, idx):
            pid = int(sequence[j])
            part = self.parts_by_id[pid]
            area = float(part.w0) * float(part.h0)
            aspect = max(float(part.w0), float(part.h0)) / max(EPS, min(float(part.w0), float(part.h0)))
            blockers.append((area * (1.0 + 0.1 * aspect), j))
        blockers.sort(reverse=True)
        for _, j in blockers[: max(1, int(getattr(self.cfg, "LOCAL_BLOCKER_TOPK", 3)))]:
            seq = list(sequence)
            blocker = seq.pop(j)
            ins = min(len(seq), idx)
            seq.insert(ins, blocker)
            out.append(tuple(seq))
        return out

    def _clear_sparse_board_candidates(self, sequence: Sequence[int], snapshot: LayoutSnapshot) -> List[Tuple[int, ...]]:
        boards = snapshot.boards.materialize()
        if len(boards) < 2:
            return []

        spill = self._find_spill_boundary(snapshot)
        sparse_u = float(getattr(self.cfg, "SPILL_SPARSE_UTIL", 0.55))
        target_idx: Optional[int] = None
        if spill is not None and spill < len(boards):
            target_idx = spill
        elif board_utilization(boards[-1]) < sparse_u:
            target_idx = len(boards) - 1
        elif len(boards[-1].placed) <= max(1, int(getattr(self.cfg, "LOCAL_CLEAR_SPARSE_MAX", 6))):
            target_idx = len(boards) - 1
        if target_idx is None or target_idx <= 0:
            return []

        target_ids = [int(pp.uid) for pp in boards[target_idx].placed]
        if not target_ids:
            return []
        idx_map = {int(pid): i for i, pid in enumerate(sequence)}
        front_rects = [fr for board in boards[:target_idx] for fr in board.free_rects]
        scored: List[Tuple[float, float, float, int]] = []
        for pid in target_ids:
            if pid not in idx_map:
                continue
            part = self.parts_by_id[int(pid)]
            fill_potential = 0.0
            for fr in front_rects:
                fill_potential = max(fill_potential, _compat(part, fr, self.allow_rot))
            area = float(part.w0) * float(part.h0)
            scored.append((fill_potential, area, -float(idx_map[pid]), int(pid)))
        if not scored:
            return []

        scored.sort(reverse=True)
        max_move = max(1, int(getattr(self.cfg, "LOCAL_CLEAR_SPARSE_MAX", 6)))
        ordered_ids = [pid for _, _, _, pid in scored[:max_move]]
        pivot = min(idx_map[pid] for pid in target_ids if pid in idx_map)
        insert_delta = int(getattr(self.cfg, "LOCAL_CLEAR_INSERT_DELTA", 4))
        max_positions = max(4, int(getattr(self.cfg, "LOCAL_INSERT_MAX_POS", 8)))
        positions = set(self._position_ladder(pivot, insert_delta, max_positions))
        positions.add(0)
        positions.add(max(0, pivot - insert_delta))
        positions.add(max(0, pivot // 2))
        counts = sorted(
            {
                1,
                max(1, min(len(ordered_ids), (len(ordered_ids) + 1) // 2)),
                min(len(ordered_ids), max_move),
            },
            reverse=True,
        )

        base_sequence = tuple(int(x) for x in sequence)
        seen = set()
        out: List[Tuple[int, ...]] = []
        for count in counts:
            moving = ordered_ids[:count]
            moving_set = set(int(pid) for pid in moving)
            remainder = [int(pid) for pid in sequence if int(pid) not in moving_set]
            for pos in sorted(positions):
                cand = list(remainder)
                ins = min(max(0, pos), len(cand))
                for offset, pid in enumerate(moving):
                    cand.insert(ins + offset, int(pid))
                cand_t = tuple(cand)
                if cand_t == base_sequence or cand_t in seen:
                    continue
                seen.add(cand_t)
                out.append(cand_t)
        return out

    def _legacy_tail_search(self, base_sequence: Tuple[int, ...], base_snapshot: LayoutSnapshot) -> Tuple[Tuple[int, ...], LayoutSnapshot]:
        if not bool(getattr(self.cfg, "LOCAL_SEARCH_ENABLE", True)):
            return base_sequence, base_snapshot

        best_sequence = tuple(int(x) for x in base_sequence)
        best_snapshot, snapshots = self.decoder.replay_order(best_sequence)
        best_score = self._terminal_score(best_snapshot)
        critical = self._critical_items(best_snapshot)
        if not critical:
            return best_sequence, best_snapshot

        failures: Dict[int, int] = {}
        tabu: set[int] = set()
        modes = ("clear", "pull", "eject")
        mode_idx = 0
        mode = modes[mode_idx]
        stalled = 0
        max_iters = int(getattr(self.cfg, "LOCAL_MAX_ITERS", 32))
        fail_limit = int(getattr(self.cfg, "LOCAL_FAIL_LIMIT", 8))
        freeze_after = int(getattr(self.cfg, "TABU_FREEZE_AFTER", 3))

        for _ in range(max_iters):
            critical = [pid for pid in self._critical_items(best_snapshot) if pid not in tabu]
            if not critical:
                break
            idx_map = {int(pid): i for i, pid in enumerate(best_sequence)}
            spill = self._find_spill_boundary(best_snapshot)
            if spill is None:
                break
            boards = best_snapshot.boards.materialize()
            sparse_ids = {int(pp.uid) for board in boards[spill:] for pp in board.placed}
            sparse_positions = [idx_map[pid] for pid in sparse_ids if pid in idx_map]
            pivot = min(sparse_positions) if sparse_positions else 0

            failure_key: Optional[int] = None
            if mode == "clear":
                candidates = self._clear_sparse_board_candidates(best_sequence, best_snapshot)
            else:
                target = critical[0]
                idx = idx_map.get(target)
                if idx is None:
                    tabu.add(target)
                    continue
                failure_key = int(target)
                if mode == "pull":
                    candidates = self._pull_forward_candidates(best_sequence, target, idx, pivot)
                else:
                    candidates = self._eject_candidates(best_sequence, target, idx)

            if not candidates:
                stalled += 1
                if failure_key is not None:
                    failures[failure_key] = failures.get(failure_key, 0) + 1
                    if failures[failure_key] >= freeze_after:
                        tabu.add(failure_key)
                if stalled >= fail_limit:
                    mode_idx = (mode_idx + 1) % len(modes)
                    mode = modes[mode_idx]
                    stalled = 0
                continue

            improved = False
            for cand_seq in candidates:
                first_diff = next((i for i, (a, b) in enumerate(zip(best_sequence, cand_seq)) if a != b), len(cand_seq))
                cand_snapshot, cand_snapshots = self._replay_from_prefix(cand_seq, first_diff, snapshots)
                cand_score = self._terminal_score(cand_snapshot)
                if cand_score > best_score:
                    best_sequence = tuple(int(x) for x in cand_seq)
                    best_snapshot = cand_snapshot
                    snapshots = cand_snapshots
                    best_score = cand_score
                    if failure_key is not None:
                        failures.pop(failure_key, None)
                    stalled = 0
                    improved = True
                    break

            if improved:
                continue

            stalled += 1
            if failure_key is not None:
                failures[failure_key] = failures.get(failure_key, 0) + 1
                if failures[failure_key] >= freeze_after:
                    tabu.add(failure_key)
            if stalled >= fail_limit:
                mode_idx = (mode_idx + 1) % len(modes)
                mode = modes[mode_idx]
                stalled = 0

        return best_sequence, best_snapshot

    def _destroy_repair_candidates(self, sequence: Sequence[int], snapshot: LayoutSnapshot) -> List[Tuple[int, ...]]:
        if not self.lns_enable:
            return []
        seq_t = tuple(int(pid) for pid in sequence)
        if len(seq_t) < 4:
            return []
        boards = snapshot.boards.materialize()
        spill = self._find_spill_boundary(snapshot)
        board_starts = self._board_first_positions(snapshot, seq_t)
        tail_start = board_starts[int(spill)] if spill is not None and spill < len(board_starts) else max(0, len(seq_t) - 8)
        suffix = list(seq_t[tail_start:])
        destroy_n = min(len(suffix) - 1, max(2, int(math.ceil(len(seq_t) * self.lns_destroy_frac))))
        if destroy_n <= 0:
            return []

        critical = [int(pid) for pid in self._critical_items(snapshot) if int(pid) in suffix]
        ordered_ids: List[int] = []
        for pid in critical + suffix:
            if pid not in ordered_ids:
                ordered_ids.append(pid)
        moving = ordered_ids[:destroy_n]
        if not moving:
            return []

        moving_set = set(int(pid) for pid in moving)
        remainder = [int(pid) for pid in seq_t if int(pid) not in moving_set]
        pivot = max(0, tail_start - max(1, destroy_n // 2))
        positions = sorted(set(self._insert_positions(len(seq_t) - 1, pivot)) | {0, pivot, max(0, pivot - 1)})
        permutations = [
            tuple(sorted(moving, key=lambda pid: self.static_action_score[int(pid)], reverse=True)),
            tuple(
                sorted(
                    moving,
                    key=lambda pid: (
                        float(self.parts_by_id[int(pid)].w0) * float(self.parts_by_id[int(pid)].h0),
                        self.static_action_score[int(pid)],
                    ),
                    reverse=True,
                )
            ),
            tuple(self.rng.sample(moving, k=len(moving))),
        ]

        out: List[Tuple[int, ...]] = []
        seen = {seq_t}
        for ordered in permutations:
            for pos in positions[: max(4, self.rand_topk)]:
                cand = list(remainder)
                ins = min(max(0, int(pos)), len(cand))
                for offset, pid in enumerate(ordered):
                    cand.insert(ins + offset, int(pid))
                cand_t = tuple(cand)
                if cand_t in seen:
                    continue
                seen.add(cand_t)
                out.append(cand_t)
        return out

    def _sequence_sa_lns_candidates(self, sequence: Sequence[int], snapshot: LayoutSnapshot) -> List[Tuple[int, ...]]:
        seq_t = tuple(int(pid) for pid in sequence)
        idx_map = {int(pid): i for i, pid in enumerate(seq_t)}
        boards = snapshot.boards.materialize()
        spill = self._find_spill_boundary(snapshot)
        board_starts = self._board_first_positions(snapshot, seq_t) if boards else []
        pivot = board_starts[int(spill)] if spill is not None and spill < len(board_starts) else max(0, len(seq_t) // 2)
        critical = [int(pid) for pid in self._critical_items(snapshot) if int(pid) in idx_map]
        candidates: List[Tuple[int, ...]] = []
        seen = {seq_t}

        def add_many(items: Sequence[Tuple[int, ...]], limit: int) -> None:
            for cand in items[: max(0, int(limit))]:
                cand_t = tuple(int(pid) for pid in cand)
                if cand_t in seen or not self._is_complete_sequence(cand_t):
                    continue
                seen.add(cand_t)
                candidates.append(cand_t)

        add_many(self._clear_sparse_board_candidates(seq_t, snapshot), 4)
        for pid in critical[:3]:
            idx = idx_map.get(int(pid))
            if idx is None:
                continue
            add_many(self._pull_forward_candidates(seq_t, int(pid), idx, pivot), 3)
            add_many(self._eject_candidates(seq_t, int(pid), idx), 2)
        add_many(self._destroy_repair_candidates(seq_t, snapshot), 4)

        if not candidates:
            for seed in self.constructive_warmstarts[:2]:
                if seed.sequence != seq_t:
                    candidates.append(seed.sequence)
        return candidates

    def _sequence_sa_lns(self, base_sequence: Tuple[int, ...], base_snapshot: LayoutSnapshot) -> Tuple[Tuple[int, ...], LayoutSnapshot]:
        if self.sa_iters <= 0:
            return base_sequence, base_snapshot

        search_start = time.perf_counter()
        time_budget_s = 2.0
        if math.isfinite(self.solve_deadline):
            scaled = self.solver_time_limit_s * self.sa_budget_frac if self.solver_time_limit_s > 0 else 2.0
            time_budget_s = max(0.25, min(self.sa_time_cap_s, scaled))
        deadline = min(self.solve_deadline, search_start + time_budget_s)

        current_sequence = tuple(int(x) for x in base_sequence)
        current_snapshot, current_snapshots = self.decoder.replay_order(current_sequence)
        current_score = self._terminal_score(current_snapshot)
        current_scalar = self._reward_scalar(current_snapshot, 0)
        best_sequence = current_sequence
        best_snapshot = current_snapshot
        best_score = current_score
        temperature = float(self.sa_t0)

        for _ in range(self.sa_iters):
            if time.perf_counter() >= deadline:
                break
            candidates = self._sequence_sa_lns_candidates(current_sequence, current_snapshot)
            if not candidates:
                break
            cand_sequence = tuple(self.rng.choice(candidates))
            self.meta["sa_lns_attempts"] += 1.0
            first_diff = self._sequence_first_diff(current_sequence, cand_sequence)
            cand_snapshot, cand_snapshots = self._replay_from_prefix(cand_sequence, first_diff, current_snapshots)
            cand_score = self._terminal_score(cand_snapshot)
            cand_scalar = self._reward_scalar(cand_snapshot, 0)

            accept = cand_score > current_score
            if not accept:
                delta = float(cand_scalar) - float(current_scalar)
                if delta >= 0.0:
                    accept = True
                else:
                    accept = math.exp(delta / max(1e-9, temperature)) > self.rng.random()
            if accept:
                current_sequence = cand_sequence
                current_snapshot = cand_snapshot
                current_snapshots = cand_snapshots
                current_score = cand_score
                current_scalar = cand_scalar
                self.meta["sa_lns_accepts"] += 1.0

            if cand_score > best_score:
                best_sequence = cand_sequence
                best_snapshot = cand_snapshot
                best_score = cand_score
                self.meta["sa_lns_improvements"] += 1.0
                self._update_global_best(cand_sequence, cand_snapshot)

            temperature *= self.sa_alpha

        self.meta["sa_lns_time_s"] += float(time.perf_counter() - search_start)
        return best_sequence, best_snapshot

    def _expand_repair_actions(
        self,
        state: RepairBeamState,
        tail_ids: Sequence[int],
        blocker_ids: Sequence[int],
        allow_blockers: bool,
    ) -> List[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]:
        idx_map = {int(pid): i for i, pid in enumerate(state.sequence)}
        focus_tail = [int(pid) for pid in state.remaining_tail if int(pid) in idx_map]
        if not focus_tail:
            focus_tail = [int(pid) for pid in tail_ids if int(pid) in idx_map]
        pivot = self._focus_pivot(state.sequence, tail_ids)
        tail_positions = [idx_map[pid] for pid in tail_ids if pid in idx_map]
        tail_hi = max(tail_positions) if tail_positions else len(state.sequence) - 1
        scored: List[Tuple[Tuple[float, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = []
        seen = set()

        if state.depth == 0:
            for cand_seq in self._clear_sparse_board_candidates(state.sequence, state.snapshot)[:3]:
                if cand_seq in seen:
                    continue
                seen.add(cand_seq)
                scored.append(
                    (
                        (10.0, 0.0, 0.0),
                        tuple(int(x) for x in cand_seq),
                        tuple(int(pid) for pid in state.remaining_tail),
                        tuple(int(pid) for pid in state.ejected_items),
                    )
                )

        for pid in focus_tail[:2]:
            idx = idx_map[int(pid)]
            part_score = float(self.static_action_score[int(pid)])
            rem_tail = tuple(int(x) for x in state.remaining_tail if int(x) != int(pid))
            for pos in self._insert_positions(idx, pivot):
                cand_seq = self._move_part(state.sequence, idx, pos)
                if cand_seq in seen:
                    continue
                seen.add(cand_seq)
                scored.append(((8.0, part_score, -float(idx - pos)), cand_seq, rem_tail, tuple(state.ejected_items)))

        if allow_blockers:
            available_blockers = [int(pid) for pid in blocker_ids if int(pid) in idx_map]
            for blocker in available_blockers[:2]:
                bidx = idx_map[int(blocker)]
                if bidx < tail_hi:
                    positions = sorted({min(len(state.sequence) - 1, tail_hi), len(state.sequence) - 1}, reverse=True)
                    for pos in positions:
                        if pos <= bidx:
                            continue
                        cand_seq = self._move_part(state.sequence, bidx, pos)
                        if cand_seq in seen:
                            continue
                        seen.add(cand_seq)
                        ejected = tuple(sorted(set(int(x) for x in state.ejected_items + (int(blocker),))))
                        scored.append(
                            (
                                (7.0, float(self.static_action_score[int(blocker)]), -float(pos - bidx)),
                                cand_seq,
                                tuple(state.remaining_tail),
                                ejected,
                            )
                        )
                for pid in focus_tail[:2]:
                    tidx = idx_map.get(int(pid))
                    if tidx is None or bidx >= tidx:
                        continue
                    cand_seq = self._swap_parts(state.sequence, bidx, tidx)
                    if cand_seq in seen:
                        continue
                    seen.add(cand_seq)
                    rem_tail = tuple(int(x) for x in state.remaining_tail if int(x) != int(pid))
                    ejected = tuple(sorted(set(int(x) for x in state.ejected_items + (int(blocker),))))
                    scored.append(
                        (
                            (6.0, float(self.static_action_score[int(pid)]), float(self.static_action_score[int(blocker)])),
                            cand_seq,
                            rem_tail,
                            ejected,
                        )
                    )

        scored.sort(reverse=True)
        return [(seq, rem, ejected) for _, seq, rem, ejected in scored[: self.repair_actions_per_state]]

    def _constructive_completion(
        self,
        prefix_sequence: Sequence[int],
        prefix_snapshots: Sequence[LayoutSnapshot],
        placed_early: Sequence[int],
        remaining_tail: Sequence[int],
        base_suffix: Sequence[int],
        ejected_blockers: Sequence[int],
        current_snapshot: LayoutSnapshot,
    ) -> Tuple[Tuple[int, ...], LayoutSnapshot, List[LayoutSnapshot]]:
        suffix = tuple(int(pid) for pid in remaining_tail) + tuple(
            int(pid) for pid in base_suffix if int(pid) not in set(int(x) for x in ejected_blockers)
        ) + tuple(int(pid) for pid in ejected_blockers)
        final_snapshot, suffix_snapshots = self.decoder.replay_order(suffix, prefix_snapshot=current_snapshot)
        full_sequence = tuple(int(pid) for pid in prefix_sequence) + tuple(int(pid) for pid in placed_early) + suffix
        full_snapshots = list(prefix_snapshots)
        full_snapshots.extend(suffix_snapshots[1:])
        return full_sequence, final_snapshot, full_snapshots

    def _constructive_tail_repair_pass(
        self,
        base_sequence: Tuple[int, ...],
        base_snapshot: LayoutSnapshot,
        base_snapshots: List[LayoutSnapshot],
        tail_boards: int,
        allow_blockers: bool,
    ) -> Tuple[Tuple[int, ...], LayoutSnapshot, List[LayoutSnapshot], int, bool]:
        tail_ids = self._tail_part_ids(base_snapshot, base_sequence, tail_boards)
        if not tail_ids:
            return base_sequence, base_snapshot, base_snapshots, 0, False
        blocker_ids = self._blocker_ids(base_snapshot, base_sequence, tail_ids, tail_boards) if allow_blockers else []
        idx_map = {int(pid): i for i, pid in enumerate(base_sequence)}
        focus_positions = [idx_map[pid] for pid in tail_ids if pid in idx_map]
        for pid in blocker_ids:
            if pid in idx_map:
                focus_positions.append(idx_map[pid])
        if not focus_positions:
            return base_sequence, base_snapshot, base_snapshots, 0, False

        prefix_idx = min(focus_positions)
        prefix_sequence = tuple(int(pid) for pid in base_sequence[:prefix_idx])
        prefix_snapshot = base_snapshots[prefix_idx]
        prefix_head_snapshots = list(base_snapshots[: prefix_idx + 1])
        base_suffix = tuple(int(pid) for pid in base_sequence[prefix_idx:] if int(pid) not in set(int(x) for x in tail_ids))
        base_board_count = len(base_snapshot.boards)
        deadline = time.perf_counter() + self.repair_pass_time_slice_s
        nodes = 0
        improved = False
        best_sequence = tuple(int(x) for x in base_sequence)
        best_snapshot = base_snapshot
        best_snapshots = list(base_snapshots)
        best_score = self._terminal_score(best_snapshot)

        initial_sequence, initial_completion_snapshot, initial_completion_snapshots = self._constructive_completion(
            prefix_sequence,
            prefix_head_snapshots,
            tuple(),
            tuple(int(pid) for pid in tail_ids),
            base_suffix,
            tuple(),
            prefix_snapshot,
        )
        beam: List[ConstructiveRepairState] = [
            ConstructiveRepairState(
                placed_early=tuple(),
                remaining_tail=tuple(int(pid) for pid in tail_ids),
                ejected_blockers=tuple(),
                snapshot=prefix_snapshot,
                prefix_snapshots=[prefix_snapshot],
                completion_sequence=initial_sequence,
                completion_snapshot=initial_completion_snapshot,
                completion_snapshots=initial_completion_snapshots,
                priority=self._repair_priority(initial_completion_snapshot, base_board_count),
                depth=0,
            )
        ]

        while beam and nodes < self.repair_node_limit and time.perf_counter() < deadline:
            next_states: List[ConstructiveRepairState] = []
            for state in beam:
                tail_choices = list(state.remaining_tail[: min(3, len(state.remaining_tail))])
                blocker_choices = []
                if allow_blockers:
                    blocker_choices = [pid for pid in blocker_ids if pid not in state.ejected_blockers][:2]

                for pid in tail_choices:
                    if nodes >= self.repair_node_limit or time.perf_counter() >= deadline:
                        break
                    try:
                        decoded = self.decoder.decode(state.snapshot, int(pid))
                    except Exception:
                        continue
                    placed_early = tuple(int(x) for x in state.placed_early + (int(pid),))
                    remaining_tail = tuple(int(x) for x in state.remaining_tail if int(x) != int(pid))
                    prefix_snapshots = list(state.prefix_snapshots)
                    prefix_snapshots.append(decoded.snapshot)
                    completion_sequence, completion_snapshot, completion_snapshots = self._constructive_completion(
                        prefix_sequence,
                        prefix_head_snapshots + prefix_snapshots[1:],
                        placed_early,
                        remaining_tail,
                        base_suffix,
                        state.ejected_blockers,
                        decoded.snapshot,
                    )
                    nodes += 1
                    cand_score = self._terminal_score(completion_snapshot)
                    if self._repair_accepts(best_score, cand_score):
                        best_sequence = completion_sequence
                        best_snapshot = completion_snapshot
                        best_snapshots = completion_snapshots
                        best_score = cand_score
                        improved = True
                        self.meta["repair_improvements"] += 1.0
                        cleared = max(0, base_board_count - len(completion_snapshot.boards))
                        self.meta["tail_boards_cleared"] = max(self.meta["tail_boards_cleared"], float(cleared))
                        if len(completion_snapshot.boards) < base_board_count:
                            return best_sequence, best_snapshot, best_snapshots, nodes, True
                    next_states.append(
                        ConstructiveRepairState(
                            placed_early=placed_early,
                            remaining_tail=remaining_tail,
                            ejected_blockers=tuple(state.ejected_blockers),
                            snapshot=decoded.snapshot,
                            prefix_snapshots=prefix_snapshots,
                            completion_sequence=completion_sequence,
                            completion_snapshot=completion_snapshot,
                            completion_snapshots=completion_snapshots,
                            priority=self._repair_priority(completion_snapshot, base_board_count),
                            depth=state.depth + 1,
                        )
                    )

                for blocker in blocker_choices:
                    if nodes >= self.repair_node_limit or time.perf_counter() >= deadline:
                        break
                    ejected = tuple(sorted(set(int(x) for x in state.ejected_blockers + (int(blocker),))))
                    completion_sequence, completion_snapshot, completion_snapshots = self._constructive_completion(
                        prefix_sequence,
                        prefix_head_snapshots + list(state.prefix_snapshots[1:]),
                        state.placed_early,
                        state.remaining_tail,
                        base_suffix,
                        ejected,
                        state.snapshot,
                    )
                    nodes += 1
                    cand_score = self._terminal_score(completion_snapshot)
                    if self._repair_accepts(best_score, cand_score):
                        best_sequence = completion_sequence
                        best_snapshot = completion_snapshot
                        best_snapshots = completion_snapshots
                        best_score = cand_score
                        improved = True
                        self.meta["repair_improvements"] += 1.0
                        cleared = max(0, base_board_count - len(completion_snapshot.boards))
                        self.meta["tail_boards_cleared"] = max(self.meta["tail_boards_cleared"], float(cleared))
                        if len(completion_snapshot.boards) < base_board_count:
                            return best_sequence, best_snapshot, best_snapshots, nodes, True
                    next_states.append(
                        ConstructiveRepairState(
                            placed_early=tuple(state.placed_early),
                            remaining_tail=tuple(state.remaining_tail),
                            ejected_blockers=ejected,
                            snapshot=state.snapshot,
                            prefix_snapshots=list(state.prefix_snapshots),
                            completion_sequence=completion_sequence,
                            completion_snapshot=completion_snapshot,
                            completion_snapshots=completion_snapshots,
                            priority=self._repair_priority(completion_snapshot, base_board_count),
                            depth=state.depth + 1,
                        )
                    )

            beam = self._prune_constructive_states(next_states)

        return best_sequence, best_snapshot, best_snapshots, nodes, improved

    def _repair_pass(
        self,
        base_sequence: Tuple[int, ...],
        base_snapshot: LayoutSnapshot,
        base_snapshots: List[LayoutSnapshot],
        tail_boards: int,
        allow_blockers: bool,
    ) -> Tuple[Tuple[int, ...], LayoutSnapshot, List[LayoutSnapshot], int, bool]:
        seq_best, snap_best, snaps_best, nodes_constructive, improved_constructive = self._constructive_tail_repair_pass(
            base_sequence,
            base_snapshot,
            base_snapshots,
            tail_boards,
            allow_blockers,
        )
        if len(snap_best.boards) < len(base_snapshot.boards):
            return seq_best, snap_best, snaps_best, nodes_constructive, True

        base_sequence = seq_best
        base_snapshot = snap_best
        base_snapshots = snaps_best
        tail_ids = self._tail_part_ids(base_snapshot, base_sequence, tail_boards)
        if not tail_ids:
            return base_sequence, base_snapshot, base_snapshots, nodes_constructive, improved_constructive
        blocker_ids = self._blocker_ids(base_snapshot, base_sequence, tail_ids, tail_boards) if allow_blockers else []
        base_board_count = len(base_snapshot.boards)
        deadline = time.perf_counter() + self.repair_pass_time_slice_s
        best_sequence = tuple(int(x) for x in base_sequence)
        best_snapshot = base_snapshot
        best_snapshots = list(base_snapshots)
        best_score = self._terminal_score(best_snapshot)
        improved = bool(improved_constructive)
        nodes = int(nodes_constructive)

        beam: List[RepairBeamState] = [
            RepairBeamState(
                sequence=best_sequence,
                snapshot=best_snapshot,
                snapshots=list(base_snapshots),
                remaining_tail=tuple(int(pid) for pid in tail_ids),
                ejected_items=tuple(),
                depth=0,
                priority=self._repair_priority(best_snapshot, base_board_count),
            )
        ]

        while beam and nodes < self.repair_node_limit and time.perf_counter() < deadline:
            next_states: List[RepairBeamState] = []
            for state in beam:
                expansions = self._expand_repair_actions(state, tail_ids, blocker_ids, allow_blockers)
                for cand_seq, rem_tail, ejected_items in expansions:
                    if nodes >= self.repair_node_limit or time.perf_counter() >= deadline:
                        break
                    first_diff = self._sequence_first_diff(state.sequence, cand_seq)
                    if first_diff >= len(cand_seq):
                        continue
                    try:
                        cand_snapshot, cand_snapshots = self._replay_from_prefix(cand_seq, first_diff, state.snapshots)
                    except Exception:
                        continue
                    nodes += 1
                    cand_score = self._terminal_score(cand_snapshot)
                    cand_priority = self._repair_priority(cand_snapshot, base_board_count)
                    if self._repair_accepts(best_score, cand_score):
                        best_sequence = tuple(int(x) for x in cand_seq)
                        best_snapshot = cand_snapshot
                        best_snapshots = cand_snapshots
                        best_score = cand_score
                        improved = True
                        self.meta["repair_improvements"] += 1.0
                        cleared = max(0, base_board_count - len(cand_snapshot.boards))
                        self.meta["tail_boards_cleared"] = max(self.meta["tail_boards_cleared"], float(cleared))
                        if len(cand_snapshot.boards) < base_board_count:
                            return best_sequence, best_snapshot, best_snapshots, nodes, True
                    next_states.append(
                        RepairBeamState(
                            sequence=tuple(int(x) for x in cand_seq),
                            snapshot=cand_snapshot,
                            snapshots=cand_snapshots,
                            remaining_tail=tuple(int(x) for x in rem_tail),
                            ejected_items=tuple(int(x) for x in ejected_items),
                            depth=state.depth + 1,
                            priority=cand_priority,
                        )
                    )
            beam = self._prune_repair_states(next_states)
        return best_sequence, best_snapshot, best_snapshots, nodes, improved

    def _region_rebuild_pass(
        self,
        base_sequence: Tuple[int, ...],
        base_snapshot: LayoutSnapshot,
        base_snapshots: List[LayoutSnapshot],
        proposal: RegionProposal,
    ) -> Tuple[Tuple[int, ...], LayoutSnapshot, List[LayoutSnapshot], int, bool]:
        start_idx = min(len(base_sequence) - 1, max(0, int(proposal.start_idx)))
        region_items = tuple(int(pid) for pid in base_sequence[start_idx:])
        if not region_items:
            return base_sequence, base_snapshot, base_snapshots, 0, False

        prefix_sequence = tuple(int(pid) for pid in base_sequence[:start_idx])
        prefix_snapshot = base_snapshots[start_idx]
        prefix_head_snapshots = list(base_snapshots[: start_idx + 1])
        deadline = time.perf_counter() + self.region_rebuild_time_slice_s
        nodes = 0

        initial_sequence, initial_completion_snapshot, initial_completion_snapshots = self._region_completion(
            prefix_sequence,
            prefix_head_snapshots,
            tuple(),
            region_items,
            tuple(),
            prefix_snapshot,
        )
        best_sequence = initial_sequence
        best_snapshot = initial_completion_snapshot
        best_snapshots = initial_completion_snapshots
        best_priority = self._region_priority(best_snapshot)
        best_score = self._terminal_score(best_snapshot)
        improved = False

        beam: List[RegionRebuildState] = [
            RegionRebuildState(
                placed_early=tuple(),
                remaining_region_items=region_items,
                parked_items=tuple(),
                snapshot=prefix_snapshot,
                prefix_snapshots=[prefix_snapshot],
                completion_sequence=initial_sequence,
                completion_snapshot=initial_completion_snapshot,
                completion_snapshots=initial_completion_snapshots,
                priority=best_priority,
                depth=0,
            )
        ]

        while beam and nodes < self.region_rebuild_node_limit and time.perf_counter() < deadline:
            next_states: List[RegionRebuildState] = []
            for state in beam:
                expansions = self._expand_region_actions(state)
                for kind, pid, blocker in expansions:
                    if nodes >= self.region_rebuild_node_limit or time.perf_counter() >= deadline:
                        break
                    new_placed_early = tuple(state.placed_early)
                    new_remaining = tuple(state.remaining_region_items)
                    new_parked = tuple(state.parked_items)
                    new_snapshot = state.snapshot
                    new_prefix_snapshots = list(state.prefix_snapshots)

                    if kind == "place":
                        try:
                            decoded = self.decoder.decode(state.snapshot, int(pid))
                        except Exception:
                            continue
                        new_snapshot = decoded.snapshot
                        new_prefix_snapshots.append(decoded.snapshot)
                        new_placed_early = tuple(state.placed_early + (int(pid),))
                        new_remaining = tuple(int(x) for x in state.remaining_region_items if int(x) != int(pid))
                    elif kind == "park":
                        new_remaining = tuple(int(x) for x in state.remaining_region_items if int(x) != int(pid))
                        new_parked = tuple(state.parked_items + (int(pid),))
                    elif kind == "pair" and blocker is not None:
                        try:
                            decoded = self.decoder.decode(state.snapshot, int(pid))
                        except Exception:
                            continue
                        new_snapshot = decoded.snapshot
                        new_prefix_snapshots.append(decoded.snapshot)
                        new_placed_early = tuple(state.placed_early + (int(pid),))
                        new_remaining = tuple(
                            int(x)
                            for x in state.remaining_region_items
                            if int(x) != int(pid) and int(x) != int(blocker)
                        )
                        new_parked = tuple(state.parked_items + (int(blocker),))
                    else:
                        continue

                    completion_sequence, completion_snapshot, completion_snapshots = self._region_completion(
                        prefix_sequence,
                        prefix_head_snapshots + new_prefix_snapshots[1:],
                        new_placed_early,
                        new_remaining,
                        new_parked,
                        new_snapshot,
                    )
                    nodes += 1
                    cand_priority = self._region_priority(completion_snapshot)
                    cand_score = self._terminal_score(completion_snapshot)
                    if self._region_accepts(best_priority, best_score, cand_priority, cand_score):
                        best_sequence = completion_sequence
                        best_snapshot = completion_snapshot
                        best_snapshots = completion_snapshots
                        best_priority = cand_priority
                        best_score = cand_score
                        improved = True
                        self.meta["repair_improvements"] += 1.0
                        self.meta["region_rebuild_improvements"] += 1.0
                        if len(completion_snapshot.boards) < len(base_snapshot.boards):
                            return best_sequence, best_snapshot, best_snapshots, nodes, True
                    next_states.append(
                        RegionRebuildState(
                            placed_early=new_placed_early,
                            remaining_region_items=new_remaining,
                            parked_items=new_parked,
                            snapshot=new_snapshot,
                            prefix_snapshots=new_prefix_snapshots,
                            completion_sequence=completion_sequence,
                            completion_snapshot=completion_snapshot,
                            completion_snapshots=completion_snapshots,
                            priority=cand_priority,
                            depth=state.depth + 1,
                        )
                    )
            beam = self._prune_region_states(next_states)

        return best_sequence, best_snapshot, best_snapshots, nodes, improved

    def _elite_candidates_for_region_repair(
        self,
        fallback_sequence: Tuple[int, ...],
        fallback_snapshot: LayoutSnapshot,
    ) -> List[Tuple[EliteCandidate, List[RegionProposal]]]:
        ordered = self._elite_candidates_for_repair(fallback_sequence, fallback_snapshot)
        enriched: List[Tuple[EliteCandidate, List[RegionProposal]]] = []
        for candidate in ordered:
            proposals = self._region_proposals(candidate.snapshot, candidate.sequence)
            if proposals:
                enriched.append((candidate, proposals))
        if not enriched:
            return []

        enriched.sort(key=lambda item: (item[0].score, item[1][0].score, len(item[0].sequence)), reverse=True)
        selected: List[Tuple[EliteCandidate, List[RegionProposal]]] = []
        seen_buckets = set()
        deferred: List[Tuple[EliteCandidate, List[RegionProposal]]] = []
        for candidate, proposals in enriched:
            bucket = int(proposals[0].bucket)
            if bucket in seen_buckets:
                deferred.append((candidate, proposals))
                continue
            seen_buckets.add(bucket)
            selected.append((candidate, proposals))
            if len(selected) >= self.region_rebuild_topk:
                break
        if len(selected) < self.region_rebuild_topk:
            for candidate, proposals in deferred:
                selected.append((candidate, proposals))
                if len(selected) >= self.region_rebuild_topk:
                    break
        return selected[: self.region_rebuild_topk]

    def _region_guided_repair(
        self,
        fallback_sequence: Tuple[int, ...],
        fallback_snapshot: LayoutSnapshot,
    ) -> Tuple[Tuple[int, ...], LayoutSnapshot]:
        candidates = self._elite_candidates_for_region_repair(fallback_sequence, fallback_snapshot)
        if not candidates:
            return fallback_sequence, fallback_snapshot

        repair_start = time.perf_counter()
        best_sequence = tuple(int(x) for x in fallback_sequence)
        best_snapshot = fallback_snapshot
        best_priority = self._region_priority(best_snapshot)
        best_score = self._terminal_score(best_snapshot)
        best_rank = 0
        best_start_idx = -1
        best_span = 0
        self.meta["region_proposals_total"] += float(sum(len(proposals) for _, proposals in candidates))

        for rank, (candidate, proposals) in enumerate(candidates, start=1):
            try:
                work_snapshot, work_snapshots = self.decoder.replay_order(candidate.sequence)
            except Exception:
                continue
            work_sequence = tuple(int(x) for x in candidate.sequence)
            work_priority = self._region_priority(work_snapshot)
            work_score = self._terminal_score(work_snapshot)
            for proposal in proposals:
                self.meta["repair_attempts"] += 1.0
                self.meta["region_rebuild_attempts"] += 1.0
                seq_cur, snap_cur, snaps_cur, nodes, _ = self._region_rebuild_pass(
                    work_sequence,
                    work_snapshot,
                    work_snapshots,
                    proposal,
                )
                normalized = self._normalize_sequence_candidate(seq_cur)
                if normalized is None:
                    continue
                seq_cur, snap_cur, snaps_cur = normalized
                self.meta["repair_nodes_expanded"] += float(nodes)
                self.meta["region_nodes_expanded"] += float(nodes)
                cand_priority = self._region_priority(snap_cur)
                cand_score = self._terminal_score(snap_cur)
                if self._region_accepts(work_priority, work_score, cand_priority, cand_score):
                    work_sequence = seq_cur
                    work_snapshot = snap_cur
                    work_snapshots = snaps_cur
                    work_priority = cand_priority
                    work_score = cand_score
                    if self._region_accepts(best_priority, best_score, cand_priority, cand_score):
                        best_sequence = seq_cur
                        best_snapshot = snap_cur
                        best_priority = cand_priority
                        best_score = cand_score
                        best_rank = rank
                        best_start_idx = int(proposal.start_idx)
                        best_span = int(proposal.span)
                self._update_global_best(seq_cur, snap_cur)

        self.meta["repair_time_s"] += float(time.perf_counter() - repair_start)
        self.meta["region_time_s"] += float(time.perf_counter() - repair_start)
        if best_rank > 0:
            self.meta["repair_best_from_elite_rank"] = float(best_rank)
            self.meta["region_best_start_idx"] = float(best_start_idx)
            self.meta["region_best_span"] = float(best_span)
        return best_sequence, best_snapshot

    def _pattern_master_status_name(self, status: Optional[int]) -> str:
        model = _load_cp_model()
        if model is None:
            return "ORTOOLS_MISSING"
        names = {
            model.OPTIMAL: "OPTIMAL",
            model.FEASIBLE: "FEASIBLE",
            model.INFEASIBLE: "INFEASIBLE",
            model.MODEL_INVALID: "MODEL_INVALID",
            model.UNKNOWN: "UNKNOWN",
        }
        return names.get(status, f"STATUS_{status}")

    def _repair_window_score(
        self,
        boards: Sequence[Board],
        affected_boards: Sequence[int],
        *,
        start_idx: int,
    ) -> Tuple[float, ...]:
        if not boards or not affected_boards:
            return (0.0, 0.0, 0.0, 0.0, float(start_idx))
        board_area = max(EPS, float(boards[0].W) * float(boards[0].H))
        sparse_u = float(getattr(self.cfg, "SPILL_SPARSE_UTIL", 0.55))
        ordered_boards = sorted({int(idx) for idx in affected_boards})
        used_equiv = sum(board_used_area(boards[idx]) for idx in ordered_boards) / board_area
        reclaim = max(0.0, float(len(ordered_boards)) - math.ceil(max(used_equiv - EPS, 0.0)))
        nearest_integer = max(1.0, float(round(used_equiv)))
        integer_gap = -abs(used_equiv - nearest_integer)
        sparse_count = sum(1 for idx in ordered_boards if board_utilization(boards[idx]) < sparse_u)
        prefix_cavity = self._prefix_cavity_area(boards, min(ordered_boards)) / board_area
        return (float(reclaim), float(integer_gap), -float(sparse_count), float(prefix_cavity), float(start_idx))

    def _window_family(self, window: RepairWindow) -> str:
        source = str(window.source)
        if source.startswith("tail-strike:"):
            return "tail-strike"
        if source.startswith("evict:"):
            return "evict"
        if source.startswith("spill"):
            return "spill"
        if source.startswith("region:"):
            return "region"
        if source.startswith("last"):
            return "last"
        return "generic"

    def _post_repair_state_profile(self, snapshot: LayoutSnapshot) -> Dict[str, float]:
        boards = snapshot.boards.materialize()
        if not boards:
            return {"tail_pressure": 0.0, "sparse_pressure": 0.0, "cavity_pressure": 0.0, "long_pressure": 0.0}
        board_area = max(EPS, float(boards[0].W) * float(boards[0].H))
        tail_u = float(board_utilization(boards[-1]))
        sparse_u = float(getattr(self.cfg, "SPILL_SPARSE_UTIL", 0.55))
        sparse_ratio = float(sum(1 for board in boards if board_utilization(board) < sparse_u)) / max(1.0, float(len(boards)))
        cavity_ratio = self._prefix_cavity_area(boards, max(0, len(boards) - 1)) / board_area
        tail_parts = [self.parts_by_id[int(pp.uid)] for board in boards[-min(2, len(boards)) :] for pp in board.placed if int(pp.uid) in self.parts_by_id]
        if tail_parts:
            aspect_scores = [
                max(float(part.w0), float(part.h0)) / max(EPS, min(float(part.w0), float(part.h0)))
                for part in tail_parts
            ]
            long_pressure = max(0.0, (sum(aspect_scores) / float(len(aspect_scores)) - 2.0) / 4.0)
        else:
            long_pressure = 0.0
        return {
            "tail_pressure": max(0.0, 0.65 - tail_u),
            "sparse_pressure": max(0.0, sparse_ratio),
            "cavity_pressure": max(0.0, cavity_ratio),
            "long_pressure": max(0.0, long_pressure),
        }

    def _window_priority_key(self, snapshot: LayoutSnapshot, window: RepairWindow) -> Tuple[float, ...]:
        base = tuple(float(x) for x in window.score)
        if not self.post_repair_alns_enable:
            return base
        family = self._window_family(window)
        profile = self._post_repair_state_profile(snapshot)
        learned = float(self.post_repair_operator_scores.get(family, 1.0))
        family_bias = 0.0
        if family == "tail-strike":
            family_bias = 2.0 * profile["tail_pressure"] + 0.8 * profile["sparse_pressure"]
        elif family == "evict":
            family_bias = 1.5 * profile["tail_pressure"] + 0.8 * profile["cavity_pressure"]
        elif family == "spill":
            family_bias = 1.2 * profile["sparse_pressure"] + 0.7 * profile["cavity_pressure"]
        elif family == "region":
            family_bias = 0.8 * profile["long_pressure"] + 0.5 * profile["cavity_pressure"]
        elif family == "last":
            family_bias = 1.0 * profile["tail_pressure"] + 0.4 * profile["sparse_pressure"]
        boosted = list(base)
        if boosted:
            boosted[0] += 0.25 * float(learned - 1.0) + 0.35 * float(family_bias)
        return tuple(boosted)

    def _update_post_repair_operator_score(
        self,
        window: RepairWindow,
        before_snapshot: LayoutSnapshot,
        after_snapshot: LayoutSnapshot,
        *,
        attempt_time_s: float,
    ) -> None:
        if not self.post_repair_alns_enable:
            return
        family = self._window_family(window)
        before_boards = len(before_snapshot.boards)
        after_boards = len(after_snapshot.boards)
        reward = 6.0 * float(max(0, before_boards - after_boards))
        reward += 50.0 * max(0.0, _u_global(after_snapshot) - _u_global(before_snapshot))
        reward += 8.0 * max(0.0, _u_last(after_snapshot) - _u_last(before_snapshot))
        reward -= 0.05 * max(0.0, float(attempt_time_s))
        target = max(0.25, 1.0 + reward)
        old = float(self.post_repair_operator_scores.get(family, 1.0))
        alpha = float(self.post_repair_alns_alpha)
        self.post_repair_operator_scores[family] = max(0.25, min(8.0, (1.0 - alpha) * old + alpha * target))
        self.meta["post_repair_alns_updates"] += 1.0

    def _window_board_ids(self, snapshot: LayoutSnapshot, window: RepairWindow) -> List[int]:
        movable = {int(pid) for pid in window.movable_part_ids}
        boards = snapshot.boards.materialize()
        return [
            int(bidx)
            for bidx, board in enumerate(boards)
            if any(int(pp.uid) in movable for pp in board.placed)
        ]

    def _expand_window_to_width(
        self,
        snapshot: LayoutSnapshot,
        sequence: Sequence[int],
        window: RepairWindow,
        target_width: int,
    ) -> Optional[RepairWindow]:
        boards = snapshot.boards.materialize()
        seq_t = tuple(int(pid) for pid in sequence)
        if not boards or not seq_t:
            return None
        board_ids = self._window_board_ids(snapshot, window)
        if not board_ids:
            return None
        target_width = min(len(boards), max(int(target_width), len(board_ids)))
        anchor_last = max(board_ids)
        start_board = max(0, int(anchor_last) - int(target_width) + 1)
        expanded_board_ids = list(range(start_board, anchor_last + 1))
        if len(expanded_board_ids) <= len(board_ids):
            return None
        idx_map = {int(pid): idx for idx, pid in enumerate(seq_t)}
        movable_set: Set[int] = set()
        for bidx in expanded_board_ids:
            movable_set.update(int(pp.uid) for pp in boards[int(bidx)].placed)
        if not movable_set:
            return None
        start_idx = min(idx_map[int(pid)] for pid in movable_set if int(pid) in idx_map)
        score = self._repair_window_score(boards, expanded_board_ids, start_idx=start_idx)
        return self._window_from_movable_set(
            boards,
            seq_t,
            idx_map,
            movable_set,
            blocker_ids=window.blocker_part_ids,
            score=score,
            source=f"{window.source}|expand:{len(expanded_board_ids)}",
        )

    def _shrink_window_to_width(
        self,
        snapshot: LayoutSnapshot,
        sequence: Sequence[int],
        window: RepairWindow,
        target_width: int,
    ) -> Optional[RepairWindow]:
        boards = snapshot.boards.materialize()
        seq_t = tuple(int(pid) for pid in sequence)
        if not boards or not seq_t:
            return None
        board_ids = sorted(set(self._window_board_ids(snapshot, window)))
        if not board_ids:
            return None
        target_width = min(len(board_ids), max(2, int(target_width)))
        if len(board_ids) <= target_width:
            return window
        clipped_board_ids = board_ids[-target_width:]
        idx_map = {int(pid): idx for idx, pid in enumerate(seq_t)}
        movable_set: Set[int] = set()
        for bidx in clipped_board_ids:
            movable_set.update(int(pp.uid) for pp in boards[int(bidx)].placed)
        if not movable_set:
            return None
        start_idx = min(idx_map[int(pid)] for pid in movable_set if int(pid) in idx_map)
        score = self._repair_window_score(boards, clipped_board_ids, start_idx=start_idx)
        return self._window_from_movable_set(
            boards,
            seq_t,
            idx_map,
            movable_set,
            blocker_ids=window.blocker_part_ids,
            score=score,
            source=f"{window.source}|seedw:{len(clipped_board_ids)}",
        )

    def _normalize_repair_window(
        self,
        snapshot: LayoutSnapshot,
        sequence: Sequence[int],
        window: RepairWindow,
    ) -> RepairWindow:
        if not self.post_repair_dynamic_expand_enable:
            return window
        target_width = max(2, int(self.post_repair_initial_window_width))
        if window.base_board_count <= target_width:
            return window
        normalized = self._shrink_window_to_width(snapshot, sequence, window, target_width)
        if normalized is None:
            return window
        if normalized.base_board_count < window.base_board_count:
            self.meta["post_repair_window_normalizations"] += 1.0
        return normalized

    def _window_expansion_chain(
        self,
        snapshot: LayoutSnapshot,
        sequence: Sequence[int],
        window: RepairWindow,
    ) -> List[RepairWindow]:
        seed = self._normalize_repair_window(snapshot, sequence, window)
        variants = [seed]
        if not self.post_repair_dynamic_expand_enable:
            return variants
        seen = {tuple(int(pid) for pid in seed.movable_part_ids)}
        for target_width in self.post_repair_dynamic_window_widths:
            expanded = self._expand_window_to_width(snapshot, sequence, seed, int(target_width))
            if expanded is None:
                continue
            key = tuple(int(pid) for pid in expanded.movable_part_ids)
            if key in seen:
                continue
            seen.add(key)
            variants.append(expanded)
        return variants

    def _window_from_movable_set(
        self,
        boards: Sequence[Board],
        sequence: Sequence[int],
        idx_map: Dict[int, int],
        movable_set: Set[int],
        *,
        blocker_ids: Sequence[int],
        score: Tuple[float, ...],
        source: str,
    ) -> Optional[RepairWindow]:
        if not movable_set:
            return None
        seq_t = tuple(int(pid) for pid in sequence)
        movable_ids = tuple(int(pid) for pid in seq_t if int(pid) in movable_set)
        fixed_prefix = tuple(int(pid) for pid in seq_t if int(pid) not in movable_set)
        if not movable_ids or len(fixed_prefix) + len(movable_ids) != len(seq_t):
            return None
        affected_boards = [
            bidx for bidx, board in enumerate(boards) if any(int(pp.uid) in movable_set for pp in board.placed)
        ]
        if not affected_boards:
            return None
        start_idx = min(idx_map[int(pid)] for pid in movable_ids if int(pid) in idx_map)
        return RepairWindow(
            start_idx=int(start_idx),
            start_board=int(min(affected_boards)),
            span=len(movable_ids),
            movable_part_ids=movable_ids,
            blocker_part_ids=tuple(int(pid) for pid in blocker_ids if int(pid) in movable_set),
            fixed_prefix_sequence=fixed_prefix,
            base_board_count=int(max(1, len(sorted(set(affected_boards))))),
            score=tuple(float(x) for x in score),
            source=str(source),
        )

    def _eviction_windows(
        self,
        snapshot: LayoutSnapshot,
        sequence: Sequence[int],
        idx_map: Dict[int, int],
    ) -> List[RepairWindow]:
        if not self.post_repair_evict_enable:
            return []
        boards = snapshot.boards.materialize()
        seq_t = tuple(int(pid) for pid in sequence)
        if len(boards) < 4 or not seq_t:
            return []

        board_area = max(EPS, float(boards[0].W) * float(boards[0].H))
        tail_candidates = list(range(max(0, len(boards) - 2), len(boards)))
        if not tail_candidates:
            return []
        tail_board = min(tail_candidates, key=lambda idx: (board_utilization(boards[idx]), -idx))

        host_scored: List[Tuple[float, float, float, int]] = []
        for bidx, board in enumerate(boards[: max(0, len(boards) - 1)]):
            if bidx == tail_board:
                continue
            util = board_utilization(board)
            waste = max(0.0, board_area - board_used_area(board))
            cavity = sum(float(fr.w) * float(fr.h) for fr in board.free_rects)
            host_scored.append((waste / board_area, cavity / board_area, -util, int(bidx)))
        host_scored.sort(reverse=True)
        host_candidates = [bidx for _, _, _, bidx in host_scored[: self.post_repair_evict_candidates]]
        if len(host_candidates) < 2:
            return []

        windows: List[RepairWindow] = []
        max_hosts = min(self.post_repair_evict_host_boards, len(host_candidates))
        for host_count in range(2, max_hosts + 1):
            for combo in combinations(host_candidates, host_count):
                board_ids = sorted(set((int(tail_board),) + tuple(int(bidx) for bidx in combo)))
                movable_set: Set[int] = set()
                for bidx in board_ids:
                    movable_set.update(int(pp.uid) for pp in boards[bidx].placed)
                if not movable_set:
                    continue
                start_idx = min(idx_map[int(pid)] for pid in movable_set if int(pid) in idx_map)
                score = self._repair_window_score(boards, board_ids, start_idx=start_idx)
                window = self._window_from_movable_set(
                    boards,
                    seq_t,
                    idx_map,
                    movable_set,
                    blocker_ids=tuple(),
                    score=score,
                    source="evict:" + ",".join(str(bidx + 1) for bidx in board_ids),
                )
                if window is not None:
                    windows.append(window)
        windows.sort(key=lambda item: (item.score, item.start_idx), reverse=True)
        return windows[: self.post_repair_evict_topk]

    def _tail_strike_windows(
        self,
        snapshot: LayoutSnapshot,
        sequence: Sequence[int],
        idx_map: Dict[int, int],
    ) -> List[RepairWindow]:
        if not self.post_repair_tail_strike_enable:
            return []
        boards = snapshot.boards.materialize()
        seq_t = tuple(int(pid) for pid in sequence)
        if len(boards) < 2 or not seq_t:
            return []

        last_board = int(len(boards) - 1)
        last_util = board_utilization(boards[last_board])
        windows: List[RepairWindow] = []
        seen = set()
        for width in self.post_repair_tail_strike_widths:
            width = min(len(boards), max(2, int(width)))
            board_ids = tuple(range(max(0, len(boards) - width), len(boards)))
            if board_ids in seen:
                continue
            seen.add(board_ids)
            movable_set: Set[int] = set()
            for bidx in board_ids:
                movable_set.update(int(pp.uid) for pp in boards[int(bidx)].placed)
            if not movable_set:
                continue
            start_idx = min(idx_map[int(pid)] for pid in movable_set if int(pid) in idx_map)
            score = list(self._repair_window_score(boards, board_ids, start_idx=start_idx))
            score[0] += max(0.0, 0.60 - float(last_util))
            window = self._window_from_movable_set(
                boards,
                seq_t,
                idx_map,
                movable_set,
                blocker_ids=tuple(),
                score=tuple(score),
                source="tail-strike:" + ",".join(str(int(bidx) + 1) for bidx in board_ids),
            )
            if window is not None:
                windows.append(window)
        windows.sort(key=lambda item: (item.score, item.start_idx), reverse=True)
        self.meta["post_repair_tail_strike_windows"] = float(len(windows))
        return windows[: self.post_repair_tail_strike_topk]

    def _repair_windows(
        self,
        snapshot: LayoutSnapshot,
        sequence: Sequence[int],
    ) -> List[RepairWindow]:
        boards = snapshot.boards.materialize()
        seq_t = tuple(int(pid) for pid in sequence)
        if len(boards) < 2 or len(seq_t) < 2:
            return []

        idx_map = {int(pid): idx for idx, pid in enumerate(seq_t)}
        board_positions = self._board_first_positions(snapshot, seq_t)
        if not board_positions:
            return []

        candidate_starts: Dict[int, str] = {}

        def add_start(start_idx: int, source: str) -> None:
            if not seq_t:
                return
            start_idx = min(len(seq_t) - 1, max(0, int(start_idx)))
            candidate_starts.setdefault(start_idx, source)

        proposals = self._region_proposals(snapshot, seq_t)
        for proposal in proposals:
            add_start(int(proposal.start_idx), f"region:{proposal.source}")

        if len(boards) >= 2:
            add_start(min(board_positions[-2:]), "last2")
        if len(boards) >= 3:
            add_start(min(board_positions[-3:]), "last3")

        spill = self._find_spill_boundary(snapshot)
        if spill is not None and spill < len(board_positions):
            spill_start = int(board_positions[int(spill)])
            add_start(spill_start, "spill")
            for backoff in (16, 32):
                add_start(max(0, spill_start - int(backoff)), f"spill-b{int(backoff)}")

        seed_starts = list(candidate_starts.items())
        for start_idx, source in seed_starts:
            for backoff in sorted({16, 32, *self.region_start_backoffs}):
                if backoff <= 0:
                    continue
                add_start(max(0, int(start_idx) - int(backoff)), f"{source}-b{int(backoff)}")

        windows: Dict[Tuple[int, ...], RepairWindow] = {}

        def add_window(window: Optional[RepairWindow]) -> None:
            if window is None:
                return
            key = tuple(int(pid) for pid in window.movable_part_ids)
            prev = windows.get(key)
            if prev is None or window.score > prev.score:
                windows[key] = window

        for start_idx, source in candidate_starts.items():
            region_ids = tuple(int(pid) for pid in seq_t[start_idx:])
            region_set = set(region_ids)
            if not region_set:
                continue
            affected_boards = [
                bidx for bidx, board in enumerate(boards) if any(int(pp.uid) in region_set for pp in board.placed)
            ]
            if not affected_boards:
                continue
            start_board = min(affected_boards)
            blocker_start = max(0, start_board - self.post_repair_blocker_boards)
            front_rects = [fr for board in boards[:start_board] for fr in board.free_rects]
            blocker_scored: List[Tuple[float, float, int]] = []
            for board in boards[blocker_start:start_board]:
                for pp in board.placed:
                    pid = int(pp.uid)
                    pos = idx_map.get(pid)
                    if pos is None or pos >= start_idx:
                        continue
                    blocker_scored.append((self._estimate_blocker_score(pid, front_rects, pos), -float(pos), pid))
            blocker_scored.sort(reverse=True)
            blockers: List[int] = []
            seen_blockers = set()
            for _, _, pid in blocker_scored:
                if pid in seen_blockers:
                    continue
                seen_blockers.add(pid)
                blockers.append(int(pid))
                if len(blockers) >= self.post_repair_blocker_topk:
                    break

            movable_set = set(region_set)
            movable_set.update(int(pid) for pid in blockers)
            score = self._repair_window_score(boards, affected_boards, start_idx=int(start_idx))
            add_window(
                self._window_from_movable_set(
                    boards,
                    seq_t,
                    idx_map,
                    movable_set,
                    blocker_ids=tuple(int(pid) for pid in blockers),
                    score=score,
                    source=source,
                )
            )

        for window in self._eviction_windows(snapshot, seq_t, idx_map):
            add_window(window)
        for window in self._tail_strike_windows(snapshot, seq_t, idx_map):
            add_window(window)

        ordered = sorted(
            (self._normalize_repair_window(snapshot, seq_t, item) for item in windows.values()),
            key=lambda item: (self._window_priority_key(snapshot, item), item.start_idx),
            reverse=True,
        )
        diversified: List[RepairWindow] = []
        seen_families: Set[str] = set()
        for item in ordered:
            family = self._window_family(item)
            if family in seen_families:
                continue
            diversified.append(item)
            seen_families.add(family)
            if len(diversified) >= self.post_repair_windows_topk:
                break
        if len(diversified) < self.post_repair_windows_topk:
            seen_keys = {tuple(int(pid) for pid in item.movable_part_ids) for item in diversified}
            for item in ordered:
                key = tuple(int(pid) for pid in item.movable_part_ids)
                if key in seen_keys:
                    continue
                diversified.append(item)
                seen_keys.add(key)
                if len(diversified) >= self.post_repair_windows_topk:
                    break
        self.meta["post_repair_windows_total"] = float(len(ordered))
        return diversified[: self.post_repair_windows_topk]

    def _pattern_fill_order(
        self,
        movable_part_ids: Sequence[int],
        prefix_snapshot: LayoutSnapshot,
    ) -> Tuple[int, ...]:
        front_rects = [fr for board in prefix_snapshot.boards.materialize() for fr in board.free_rects]
        scored: List[Tuple[float, float, float, int]] = []
        for seq_pos, pid in enumerate(movable_part_ids):
            part = self.parts_by_id[int(pid)]
            compat_best = 0.0
            for fr in front_rects:
                compat_best = max(compat_best, _compat(part, fr, self.allow_rot))
            scored.append((compat_best, float(self.static_action_score[int(pid)]), -float(seq_pos), int(pid)))
        scored.sort(reverse=True)
        return tuple(pid for _, _, _, pid in scored)

    def _pattern_candidate_orders(
        self,
        window: RepairWindow,
        prefix_snapshot: LayoutSnapshot,
    ) -> List[Tuple[str, Tuple[int, ...]]]:
        movable = tuple(int(pid) for pid in window.movable_part_ids)
        if not movable:
            return []

        blocker_set = {int(pid) for pid in window.blocker_part_ids}
        fill_order = self._pattern_fill_order(movable, prefix_snapshot)
        critical = list(fill_order[: min(4, len(fill_order))])
        seen = set()
        orders: List[Tuple[str, Tuple[int, ...]]] = []

        def add(name: str, seq: Sequence[int]) -> None:
            seq_t = tuple(int(pid) for pid in seq)
            if len(seq_t) != len(movable):
                return
            if tuple(sorted(seq_t)) != tuple(sorted(movable)):
                return
            if seq_t in seen:
                return
            seen.add(seq_t)
            orders.append((name, seq_t))

        add("original", movable)
        add("reverse", tuple(reversed(movable)))
        add(
            "area",
            tuple(
                sorted(
                    movable,
                    key=lambda pid: (
                        float(self.parts_by_id[int(pid)].w0) * float(self.parts_by_id[int(pid)].h0),
                        self.static_action_score[int(pid)],
                        -int(pid),
                    ),
                    reverse=True,
                )
            ),
        )
        add(
            "longside",
            tuple(
                sorted(
                    movable,
                    key=lambda pid: (
                        max(float(self.parts_by_id[int(pid)].w0), float(self.parts_by_id[int(pid)].h0)),
                        float(self.parts_by_id[int(pid)].w0) * float(self.parts_by_id[int(pid)].h0),
                        -int(pid),
                    ),
                    reverse=True,
                )
            ),
        )
        add("fill", fill_order)
        add("blocker_park", tuple(pid for pid in movable if pid not in blocker_set) + tuple(pid for pid in movable if pid in blocker_set))
        add("blocker_first", tuple(pid for pid in movable if pid in blocker_set) + tuple(pid for pid in movable if pid not in blocker_set))
        if critical:
            critical_set = set(critical)
            add("critical_first", tuple(pid for pid in movable if pid in critical_set) + tuple(pid for pid in movable if pid not in critical_set))
            add("critical_last", tuple(pid for pid in movable if pid not in critical_set) + tuple(pid for pid in movable if pid in critical_set))
        for crit in critical[:2]:
            for blocker in window.blocker_part_ids[:2]:
                blocker = int(blocker)
                if blocker == int(crit) or blocker not in movable:
                    continue
                seq = list(movable)
                blocker_idx = seq.index(blocker)
                seq.pop(blocker_idx)
                crit_idx = seq.index(int(crit))
                seq.insert(crit_idx + 1, blocker)
                add(f"pair:{int(crit)}:{blocker}", seq)
        for name, seq in self._pattern_random_orders(
            movable,
            prefix_snapshot,
            blocker_set=blocker_set,
            critical_ids=set(critical),
        ):
            add(name, seq)
        return orders

    def _pattern_random_orders(
        self,
        movable_part_ids: Sequence[int],
        prefix_snapshot: LayoutSnapshot,
        *,
        blocker_set: Set[int],
        critical_ids: Set[int],
    ) -> List[Tuple[str, Tuple[int, ...]]]:
        if self.post_repair_random_orders <= 0 or len(movable_part_ids) <= 1:
            return []

        movable = tuple(int(pid) for pid in movable_part_ids)
        pos_map = {int(pid): idx for idx, pid in enumerate(movable)}
        large_ids = {
            int(pid)
            for pid in sorted(
                movable,
                key=lambda pid: (
                    float(self.parts_by_id[int(pid)].w0) * float(self.parts_by_id[int(pid)].h0),
                    self.static_action_score[int(pid)],
                    -int(pid),
                ),
                reverse=True,
            )[: min(8, len(movable))]
        }
        fill_order = self._pattern_fill_order(movable, prefix_snapshot)
        fill_rank = {int(pid): len(fill_order) - idx for idx, pid in enumerate(fill_order)}
        base_scores: Dict[int, float] = {}
        for pid in movable:
            base_scores[int(pid)] = (
                2.5 * float(fill_rank.get(int(pid), 0))
                + 3.0 * float(self.static_action_score[int(pid)])
                + 1.0 * float(int(pid) in critical_ids)
                + 0.75 * float(int(pid) in large_ids)
                - 0.25 * float(int(pid) in blocker_set)
                - 0.005 * float(pos_map[int(pid)])
            )

        orders: List[Tuple[str, Tuple[int, ...]]] = []
        seen = set()
        topk = min(len(movable), max(2, self.post_repair_random_topk))
        for ridx in range(self.post_repair_random_orders):
            remaining = list(movable)
            out: List[int] = []
            while remaining:
                scored: List[Tuple[float, float, int]] = []
                for pid in remaining:
                    jitter = 0.85 + 0.30 * self.rng.random()
                    scored.append((float(base_scores[int(pid)]) * jitter, float(self.static_action_score[int(pid)]), int(pid)))
                scored.sort(reverse=True)
                pool = [int(pid) for _, _, pid in scored[:topk]]
                weights = [max(1e-6, float(base_scores[int(pid)])) for pid in pool]
                picked = int(self.rng.choices(pool, weights=weights, k=1)[0])
                out.append(picked)
                remaining.remove(picked)
            order = tuple(int(pid) for pid in out)
            if order in seen:
                continue
            seen.add(order)
            orders.append((f"rand:{ridx}", order))
        return orders

    def _keep_pattern_candidate(
        self,
        pattern: PatternCandidate,
        *,
        focus_ids: Set[int],
        large_ids: Set[int],
    ) -> bool:
        part_ids = {int(pid) for pid in pattern.part_ids}
        if not part_ids:
            return False
        if len(part_ids) <= 1:
            return True
        if pattern.util >= self.post_repair_pattern_strong_util:
            return True
        if pattern.util < self.post_repair_pattern_min_util:
            return False
        return bool(part_ids.intersection(focus_ids)) or bool(part_ids.intersection(large_ids))

    def _build_single_board_pattern(
        self,
        order: Sequence[int],
        *,
        place_mode: str,
        source: str,
    ) -> Tuple[Optional[PatternCandidate], Tuple[int, ...]]:
        board = make_empty_board(
            1,
            self.cfg.BOARD_W,
            self.cfg.BOARD_H,
            trim=float(getattr(self.cfg, "TRIM", 0.0)),
            safe_gap=float(getattr(self.cfg, "SAFE_GAP", 0.0)),
            touch_tol=float(getattr(self.cfg, "TOUCH_TOL", 1e-6)),
            place_mode=place_mode,
        )
        placed: List[int] = []
        remaining: List[int] = []
        for pid in order:
            part = self.parts_by_id[int(pid)]
            bp = best_local_placement(board, part, self.allow_rot, place_mode=place_mode)
            if bp is None:
                remaining.append(int(pid))
                continue
            board = apply_local_blueprint(board, part, bp)
            placed.append(int(pid))
        if not placed:
            return None, tuple(int(pid) for pid in remaining)
        used_area = board_used_area(board)
        pattern = PatternCandidate(
            part_ids=tuple(sorted(int(pid) for pid in placed)),
            sequence=tuple(int(pid) for pid in placed),
            used_area=float(used_area),
            util=float(board_utilization(board)),
            waste_area=max(0.0, float(board.W) * float(board.H) - float(used_area)),
            place_mode=str(place_mode),
            source=source,
        )
        return pattern, tuple(int(pid) for pid in remaining)

    def _existing_board_patterns(
        self,
        snapshot: LayoutSnapshot,
        window: RepairWindow,
    ) -> List[PatternCandidate]:
        movable_set = {int(pid) for pid in window.movable_part_ids}
        if not movable_set:
            return []
        patterns: List[PatternCandidate] = []
        modes = [self.decoder.place_mode]
        if self.post_repair_use_bssf:
            modes.append("maxrects_bssf")
        for board in snapshot.boards.materialize():
            seq = tuple(int(pid) for pid in window.movable_part_ids if any(int(pp.uid) == int(pid) for pp in board.placed))
            if not seq:
                continue
            for place_mode in modes:
                pattern, _ = self._build_single_board_pattern(
                    seq,
                    place_mode=place_mode,
                    source=f"existing:{board.bid}:{place_mode}",
                )
                if pattern is not None:
                    patterns.append(pattern)
        return patterns

    def _pattern_budget_for_window(self, window: RepairWindow) -> Tuple[int, int]:
        per_mode = int(self.post_repair_patterns_per_mode)
        cap = int(self.post_repair_pattern_cap)
        if str(window.source).startswith("tail-strike:"):
            mult = max(1, int(self.post_repair_tail_strike_pattern_mult))
            per_mode = max(per_mode, int(per_mode * mult))
            cap = max(cap, int(cap * mult))
        return per_mode, cap

    def _generate_pattern_candidates(
        self,
        snapshot: LayoutSnapshot,
        window: RepairWindow,
        prefix_snapshot: LayoutSnapshot,
    ) -> Tuple[int, List[PatternCandidate]]:
        per_mode_budget, pattern_cap = self._pattern_budget_for_window(window)
        orders = self._pattern_candidate_orders(window, prefix_snapshot)
        raw_patterns: List[PatternCandidate] = []
        raw_patterns.extend(self._existing_board_patterns(snapshot, window))
        fill_order = self._pattern_fill_order(window.movable_part_ids, prefix_snapshot)
        focus_ids = {int(pid) for pid in fill_order[: min(8, len(fill_order))]}
        focus_ids.update(int(pid) for pid in window.blocker_part_ids[: self.post_repair_blocker_topk])
        large_ids = {
            int(pid)
            for pid in sorted(
                window.movable_part_ids,
                key=lambda pid: (
                    float(self.parts_by_id[int(pid)].w0) * float(self.parts_by_id[int(pid)].h0),
                    self.static_action_score[int(pid)],
                    -int(pid),
                ),
                reverse=True,
            )[: min(8, len(window.movable_part_ids))]
        }
        place_modes = [self.decoder.place_mode]
        if self.post_repair_use_bssf:
            place_modes.append("maxrects_bssf")

        for place_mode in place_modes:
            built = 0
            for order_name, order in orders:
                if built >= per_mode_budget:
                    break
                remaining = tuple(int(pid) for pid in order)
                while remaining and built < per_mode_budget:
                    pattern, next_remaining = self._build_single_board_pattern(
                        remaining,
                        place_mode=place_mode,
                        source=f"{order_name}:{place_mode}",
                    )
                    if pattern is None:
                        break
                    if self._keep_pattern_candidate(pattern, focus_ids=focus_ids, large_ids=large_ids):
                        raw_patterns.append(pattern)
                    built += 1
                    if len(next_remaining) >= len(remaining):
                        break
                    remaining = next_remaining

        for pid in window.movable_part_ids:
            for place_mode in place_modes[:1]:
                pattern, _ = self._build_single_board_pattern(
                    (int(pid),),
                    place_mode=place_mode,
                    source=f"singleton:{place_mode}",
                )
                if pattern is not None:
                    raw_patterns.append(pattern)

        ranked: Dict[Tuple[int, ...], PatternCandidate] = {}
        for pattern in raw_patterns:
            prev = ranked.get(pattern.part_ids)
            if prev is None or (
                pattern.util,
                pattern.used_area,
                -len(pattern.sequence),
                pattern.source,
            ) > (
                prev.util,
                prev.used_area,
                -len(prev.sequence),
                prev.source,
            ):
                ranked[pattern.part_ids] = pattern

        ordered = sorted(
            ranked.values(),
            key=lambda item: (len(item.part_ids), item.util, item.used_area, -item.waste_area),
            reverse=True,
        )
        return len(raw_patterns), ordered[: pattern_cap]

    def _pattern_quality_coeff(self, pattern: PatternCandidate) -> int:
        sparse_u = float(getattr(self.cfg, "SPILL_SPARSE_UTIL", 0.55))
        quality = (pattern.util * pattern.util) + 0.01 * float(len(pattern.part_ids))
        if pattern.util < sparse_u:
            quality -= 0.2
        return int(round(1_000_000.0 * quality))

    def _solve_pattern_master_greedy(
        self,
        window: RepairWindow,
        patterns: Sequence[PatternCandidate],
    ) -> Tuple[List[PatternCandidate], str]:
        uncovered = {int(pid) for pid in window.movable_part_ids}
        if not uncovered:
            return [], "EMPTY"
        selected: List[PatternCandidate] = []
        remaining = list(patterns)
        while uncovered:
            best_idx = None
            best_key = None
            for idx, pattern in enumerate(remaining):
                part_ids = tuple(int(pid) for pid in pattern.part_ids)
                if not part_ids or any(int(pid) not in uncovered for pid in part_ids):
                    continue
                cover = part_ids
                key = (
                    len(cover),
                    self._pattern_quality_coeff(pattern),
                    float(pattern.used_area),
                    -float(pattern.waste_area),
                    -len(pattern.sequence),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = idx
            if best_idx is None:
                return [], "drop:ORTOOLS_MISSING|min:UNCOVERED"
            best = remaining.pop(best_idx)
            selected.append(best)
            uncovered.difference_update(int(pid) for pid in best.part_ids)
        return selected, "drop:ORTOOLS_MISSING|min:GREEDY|qual:GREEDY"

    def _solve_pattern_master_branch_and_bound(
        self,
        window: RepairWindow,
        patterns: Sequence[PatternCandidate],
        *,
        time_limit_s: float,
    ) -> Tuple[List[PatternCandidate], str]:
        movable = tuple(int(pid) for pid in window.movable_part_ids)
        if not movable:
            return [], "EMPTY"

        greedy_selected, greedy_status = self._solve_pattern_master_greedy(window, patterns)
        if not greedy_selected:
            return [], greedy_status

        pid_bit = {int(pid): 1 << idx for idx, pid in enumerate(movable)}
        pattern_data: List[Tuple[PatternCandidate, int, int, int]] = []
        index_by_part_ids: Dict[Tuple[int, ...], int] = {}
        cover_map: Dict[int, List[int]] = {int(pid): [] for pid in movable}

        for pattern in patterns:
            mask = 0
            valid = True
            for pid in pattern.part_ids:
                bit = pid_bit.get(int(pid))
                if bit is None:
                    valid = False
                    break
                mask |= int(bit)
            if not valid or mask == 0:
                continue
            idx = len(pattern_data)
            coeff = self._pattern_quality_coeff(pattern)
            pattern_data.append((pattern, int(mask), int(coeff), _bit_count(mask)))
            index_by_part_ids[tuple(int(pid) for pid in pattern.part_ids)] = idx
            for pid in pattern.part_ids:
                cover_map[int(pid)].append(idx)

        if any(not cover_map[int(pid)] for pid in movable):
            return [], "drop:ORTOOLS_MISSING|min:UNCOVERED"

        greedy_indices = [
            index_by_part_ids[tuple(int(pid) for pid in pattern.part_ids)]
            for pattern in greedy_selected
            if tuple(int(pid) for pid in pattern.part_ids) in index_by_part_ids
        ]
        best_indices = list(greedy_indices)
        best_count = len(best_indices)
        best_quality = sum(pattern_data[idx][2] for idx in best_indices)
        all_mask = (1 << len(movable)) - 1
        max_cover = max(size for _, _, _, size in pattern_data)
        deadline = time.perf_counter() + max(0.2, min(float(time_limit_s), 6.0))
        timed_out = False
        selected_stack: List[int] = []

        def dfs(uncovered_mask: int, quality_sum: int) -> None:
            nonlocal best_indices, best_count, best_quality, timed_out
            if time.perf_counter() >= deadline:
                timed_out = True
                return
            if uncovered_mask == 0:
                used_count = len(selected_stack)
                if used_count < best_count or (used_count == best_count and quality_sum > best_quality):
                    best_indices = list(selected_stack)
                    best_count = used_count
                    best_quality = quality_sum
                return

            remaining_parts = _bit_count(uncovered_mask)
            lower_bound = int(math.ceil(remaining_parts / max(1, max_cover)))
            if len(selected_stack) + lower_bound > best_count:
                return

            choice_options: Optional[List[int]] = None
            bits = int(uncovered_mask)
            while bits:
                lsb = bits & -bits
                pid = int(movable[lsb.bit_length() - 1])
                options = [
                    idx
                    for idx in cover_map[int(pid)]
                    if int(pattern_data[idx][1]) & ~int(uncovered_mask) == 0
                ]
                if not options:
                    return
                if choice_options is None or len(options) < len(choice_options):
                    choice_options = options
                    if len(choice_options) == 1:
                        break
                bits ^= lsb

            if not choice_options:
                return

            choice_options.sort(
                key=lambda idx: (
                    int(pattern_data[idx][3]),
                    int(pattern_data[idx][2]),
                    float(pattern_data[idx][0].used_area),
                    -float(pattern_data[idx][0].waste_area),
                ),
                reverse=True,
            )

            for idx in choice_options:
                _, mask, coeff, _ = pattern_data[idx]
                selected_stack.append(int(idx))
                dfs(int(uncovered_mask) & ~int(mask), quality_sum + int(coeff))
                selected_stack.pop()
                if timed_out:
                    return

        dfs(all_mask, 0)
        selected = [pattern_data[idx][0] for idx in best_indices]
        status = "drop:ORTOOLS_MISSING|min:DFS_BB|qual:DFS_BB"
        if timed_out:
            status += "_TIME"
        return selected, status

    def _solve_pattern_master(
        self,
        window: RepairWindow,
        patterns: Sequence[PatternCandidate],
        *,
        time_limit_s: float,
        hint_part_sets: Optional[Sequence[Tuple[int, ...]]] = None,
    ) -> Tuple[List[PatternCandidate], str]:
        model = _load_cp_model()
        if model is None:
            self.meta["post_repair_backend"] = "dfs_bb"
            return self._solve_pattern_master_branch_and_bound(window, patterns, time_limit_s=time_limit_s)
        self.meta["post_repair_backend"] = "cp_sat"
        self.meta["post_repair_cp_sat_used"] += 1.0
        movable = tuple(int(pid) for pid in window.movable_part_ids)
        if not movable:
            return [], "EMPTY"
        cover_map = {
            int(pid): [idx for idx, pattern in enumerate(patterns) if int(pid) in pattern.part_ids]
            for pid in movable
        }
        if any(not indices for indices in cover_map.values()):
            return [], "UNCOVERED"
        hint_indices = self._pattern_hint_indices(patterns, movable, hint_part_sets or ())
        hint_index_set = {int(idx) for idx in hint_indices}

        def build_model() -> Tuple[Any, List[Any]]:
            cp = model.CpModel()
            xs = [cp.NewBoolVar(f"x_{idx}") for idx in range(len(patterns))]
            for pid in movable:
                cp.Add(sum(xs[idx] for idx in cover_map[int(pid)]) == 1)
            if hint_index_set:
                for idx, var in enumerate(xs):
                    cp.AddHint(var, 1 if idx in hint_index_set else 0)
            return cp, xs

        target_boards = max(1, int(window.base_board_count) - int(self.post_repair_target_drop))
        time_limit_s = max(0.2, float(time_limit_s))
        statuses: List[str] = [f"hint:{len(hint_indices)}"] if hint_indices else []
        hint_selected = [patterns[idx] for idx in hint_indices]

        if target_boards < int(window.base_board_count):
            model_drop, xs_drop = build_model()
            model_drop.Add(sum(xs_drop) <= int(target_boards))
            coeffs = [self._pattern_quality_coeff(pattern) for pattern in patterns]
            model_drop.Maximize(sum(coeff * var for coeff, var in zip(coeffs, xs_drop)))
            solver_drop = model.CpSolver()
            solver_drop.parameters.num_search_workers = min(self.post_repair_cp_num_workers, max(1, os.cpu_count() or 1))
            solver_drop.parameters.max_time_in_seconds = max(0.2, min(time_limit_s * 0.35, 30.0))
            status_drop = solver_drop.Solve(model_drop)
            statuses.append(f"drop:{self._pattern_master_status_name(status_drop)}")
            if status_drop in (model.OPTIMAL, model.FEASIBLE):
                selected = [patterns[idx] for idx, var in enumerate(xs_drop) if solver_drop.Value(var) > 0]
                return selected, "|".join(statuses)

        model_min, xs_min = build_model()
        model_min.Minimize(sum(xs_min))
        solver_min = model.CpSolver()
        solver_min.parameters.num_search_workers = min(self.post_repair_cp_num_workers, max(1, os.cpu_count() or 1))
        solver_min.parameters.max_time_in_seconds = max(0.2, min(time_limit_s * 0.35, 60.0))
        status_min = solver_min.Solve(model_min)
        statuses.append(f"min:{self._pattern_master_status_name(status_min)}")
        if status_min not in (model.OPTIMAL, model.FEASIBLE):
            if hint_selected:
                statuses.append("fallback:HINT")
                return hint_selected, "|".join(statuses)
            return [], "|".join(statuses)

        best_board_count = int(round(solver_min.ObjectiveValue()))
        model_q, xs_q = build_model()
        model_q.Add(sum(xs_q) == int(best_board_count))
        coeffs = [self._pattern_quality_coeff(pattern) for pattern in patterns]
        model_q.Maximize(sum(coeff * var for coeff, var in zip(coeffs, xs_q)))
        solver_q = model.CpSolver()
        solver_q.parameters.num_search_workers = min(self.post_repair_cp_num_workers, max(1, os.cpu_count() or 1))
        solver_q.parameters.max_time_in_seconds = max(0.2, min(time_limit_s * 0.65, 120.0))
        status_q = solver_q.Solve(model_q)
        statuses.append(f"qual:{self._pattern_master_status_name(status_q)}")
        if status_q not in (model.OPTIMAL, model.FEASIBLE):
            selected = [patterns[idx] for idx, var in enumerate(xs_min) if solver_min.Value(var) > 0]
            return selected, "|".join(statuses)

        selected = [patterns[idx] for idx, var in enumerate(xs_q) if solver_q.Value(var) > 0]
        return selected, "|".join(statuses)

    def _sequence_from_selected_patterns(
        self,
        fixed_prefix_sequence: Sequence[int],
        selected_patterns: Sequence[PatternCandidate],
        original_sequence: Sequence[int],
    ) -> Optional[Tuple[Tuple[int, ...], LayoutSnapshot, List[LayoutSnapshot]]]:
        if not selected_patterns:
            return None
        pos = {int(pid): idx for idx, pid in enumerate(original_sequence)}
        orderings = [
            sorted(selected_patterns, key=lambda item: min(pos[int(pid)] for pid in item.sequence)),
            sorted(selected_patterns, key=lambda item: (-len(item.part_ids), -item.used_area, min(pos[int(pid)] for pid in item.sequence))),
            sorted(selected_patterns, key=lambda item: (-item.util, -item.used_area, min(pos[int(pid)] for pid in item.sequence))),
        ]
        best: Optional[Tuple[Tuple[int, ...], LayoutSnapshot, List[LayoutSnapshot]]] = None
        best_score: Optional[Tuple[float, ...]] = None
        seen_sequences = set()
        prefix = tuple(int(pid) for pid in fixed_prefix_sequence)
        for ordering in orderings:
            candidate_sequence = prefix + tuple(int(pid) for pattern in ordering for pid in pattern.sequence)
            if candidate_sequence in seen_sequences:
                continue
            seen_sequences.add(candidate_sequence)
            normalized = self._normalize_sequence_candidate(candidate_sequence)
            if normalized is None:
                continue
            seq_cur, snap_cur, snaps_cur = normalized
            score = self._terminal_score(snap_cur)
            if best is None or score > best_score:
                best = (seq_cur, snap_cur, snaps_cur)
                best_score = score
        return best

    def _global_pattern_window(
        self,
        base_sequence: Sequence[int],
        base_snapshot: LayoutSnapshot,
    ) -> RepairWindow:
        return RepairWindow(
            start_idx=0,
            start_board=0,
            span=len(base_sequence),
            movable_part_ids=tuple(int(pid) for pid in base_sequence),
            blocker_part_ids=tuple(),
            fixed_prefix_sequence=tuple(),
            base_board_count=max(1, len(base_snapshot.boards)),
            score=(0.0, 0.0, 0.0, 0.0),
            source="global_harvest",
        )

    def _keep_global_pattern_candidate(self, pattern: PatternCandidate) -> bool:
        if not pattern.part_ids:
            return False
        if len(pattern.part_ids) <= 1:
            return True
        if pattern.util >= self.global_pattern_strong_util:
            return True
        return pattern.util >= self.global_pattern_min_util

    def _tail_focus_part_ids(self, snapshot: LayoutSnapshot, *, tail_boards: int) -> Set[int]:
        boards = snapshot.boards.materialize()
        if not boards:
            return set()
        width = min(len(boards), max(1, int(tail_boards)))
        focus: Set[int] = set()
        for board in boards[-width:]:
            focus.update(int(pp.uid) for pp in board.placed)
        return focus

    def _snapshot_hint_part_sets(
        self,
        snapshot: LayoutSnapshot,
        movable_part_ids: Sequence[int],
    ) -> List[Tuple[int, ...]]:
        movable_set = {int(pid) for pid in movable_part_ids}
        if not movable_set:
            return []
        out: List[Tuple[int, ...]] = []
        for board in snapshot.boards.materialize():
            part_ids = tuple(sorted(int(pp.uid) for pp in board.placed if int(pp.uid) in movable_set))
            if part_ids:
                out.append(part_ids)
        return out

    def _pattern_hint_indices(
        self,
        patterns: Sequence[PatternCandidate],
        movable_part_ids: Sequence[int],
        hint_part_sets: Sequence[Tuple[int, ...]],
    ) -> List[int]:
        if not patterns or not hint_part_sets:
            return []
        index_by_part_ids = {
            tuple(int(pid) for pid in pattern.part_ids): idx for idx, pattern in enumerate(patterns)
        }
        selected: List[int] = []
        seen = set()
        covered: Set[int] = set()
        for part_set in hint_part_sets:
            key = tuple(int(pid) for pid in part_set)
            idx = index_by_part_ids.get(key)
            if idx is not None:
                if idx not in seen:
                    selected.append(int(idx))
                    seen.add(int(idx))
                covered.update(int(pid) for pid in key)
                continue
            for pid in key:
                singleton_idx = index_by_part_ids.get((int(pid),))
                if singleton_idx is None or singleton_idx in seen:
                    continue
                selected.append(int(singleton_idx))
                seen.add(int(singleton_idx))
                covered.add(int(pid))
        movable_set = {int(pid) for pid in movable_part_ids}
        if covered != movable_set:
            return []
        return selected

    def _global_pattern_memory_candidates(
        self,
        *,
        focus_part_ids: Optional[Set[int]] = None,
    ) -> List[PatternCandidate]:
        if not self.global_pattern_memory_enable or not self.global_pattern_memory:
            return []
        ordered = sorted(
            (
                item
                for item in self.global_pattern_memory.values()
                if item.util >= self.global_pattern_memory_master_min_util
                and (
                    not focus_part_ids
                    or any(int(pid) in focus_part_ids for pid in item.part_ids)
                )
            ),
            key=lambda item: (len(item.part_ids), item.util, item.used_area, -len(item.sequence), item.source),
            reverse=True,
        )
        result = list(ordered[: self.global_pattern_memory_topk])
        self.meta["global_pattern_memory_relevant"] = float(len(result))
        return result

    def _ensure_global_pattern_cover(
        self,
        selected: Sequence[PatternCandidate],
        base_sequence: Sequence[int],
    ) -> List[PatternCandidate]:
        out = list(selected)
        existing = {tuple(int(pid) for pid in pattern.part_ids) for pattern in out}
        for pid in base_sequence:
            pid_i = int(pid)
            key = (pid_i,)
            if key in existing:
                continue
            pattern, _ = self._build_single_board_pattern(
                (pid_i,),
                place_mode=self.decoder.place_mode,
                source="global:cover-singleton",
            )
            if pattern is None:
                continue
            out.append(pattern)
            existing.add(key)
        return out

    def _harvest_patterns_from_layout(self, layout: HarvestedLayoutCandidate) -> List[PatternCandidate]:
        boards = layout.snapshot.boards.materialize()
        if not boards:
            return []
        seq_t = tuple(int(pid) for pid in layout.sequence)
        modes = [self.decoder.place_mode]
        if self.global_pattern_use_bssf and self.decoder.place_mode != "maxrects_bssf":
            modes.append("maxrects_bssf")

        patterns: List[PatternCandidate] = []
        for board in boards:
            board_set = {int(pp.uid) for pp in board.placed}
            if not board_set:
                continue
            original = tuple(pid for pid in seq_t if pid in board_set)
            if not original:
                continue
            orders = [("original", original)]
            area_order = tuple(
                sorted(
                    original,
                    key=lambda pid: (
                        float(self.parts_by_id[int(pid)].w0) * float(self.parts_by_id[int(pid)].h0),
                        self.static_action_score[int(pid)],
                        -int(pid),
                    ),
                    reverse=True,
                )
            )
            if area_order != original:
                orders.append(("area", area_order))
            for place_mode in modes:
                for order_name, order in orders:
                    pattern, _ = self._build_single_board_pattern(
                        order,
                        place_mode=place_mode,
                        source=f"harvest:{layout.source}:{board.bid}:{order_name}:{place_mode}",
                    )
                    if pattern is None or not self._keep_global_pattern_candidate(pattern):
                        continue
                    patterns.append(pattern)
        return patterns

    def _global_pattern_candidates(
        self,
        base_sequence: Sequence[int],
        base_snapshot: LayoutSnapshot,
    ) -> Tuple[int, List[PatternCandidate]]:
        self._record_layout_candidate(base_sequence, base_snapshot, source="global_base")
        layouts = sorted(
            self.harvested_layouts,
            key=lambda item: (item.score, -len(item.board_signature), item.source),
            reverse=True,
        )[: self.global_pattern_layout_topk]

        raw_patterns: List[PatternCandidate] = []
        for layout in layouts:
            raw_patterns.extend(self._harvest_patterns_from_layout(layout))
        memory_patterns: List[PatternCandidate] = []
        max_focus_width = max(1, int(self.global_pattern_memory_focus_tail_boards))
        for width in range(1, max_focus_width + 1):
            focus_part_ids = self._tail_focus_part_ids(base_snapshot, tail_boards=width)
            memory_patterns = self._global_pattern_memory_candidates(focus_part_ids=focus_part_ids)
            if memory_patterns or width >= max_focus_width:
                break
        raw_patterns.extend(memory_patterns)

        for pid in base_sequence:
            pattern, _ = self._build_single_board_pattern(
                (int(pid),),
                place_mode=self.decoder.place_mode,
                source="global:singleton",
            )
            if pattern is not None:
                raw_patterns.append(pattern)

        ranked: Dict[Tuple[int, ...], PatternCandidate] = {}
        for pattern in raw_patterns:
            prev = ranked.get(pattern.part_ids)
            if prev is None or (
                pattern.util,
                pattern.used_area,
                -len(pattern.sequence),
                pattern.source,
            ) > (
                prev.util,
                prev.used_area,
                -len(prev.sequence),
                prev.source,
            ):
                ranked[pattern.part_ids] = pattern

        ordered = sorted(
            ranked.values(),
            key=lambda item: (len(item.part_ids), item.util, item.used_area, -len(item.sequence), item.source),
            reverse=True,
        )
        selected = list(ordered[: self.global_pattern_cap])
        selected_keys = {tuple(int(pid) for pid in pattern.part_ids) for pattern in selected}
        memory_injected = 0
        memory_reserve = min(len(memory_patterns), max(32, self.global_pattern_cap // 8), 512)
        for pattern in memory_patterns:
            key = tuple(int(pid) for pid in pattern.part_ids)
            if key in selected_keys:
                continue
            selected.append(pattern)
            selected_keys.add(key)
            memory_injected += 1
            if memory_injected >= memory_reserve:
                break
        self.meta["global_pattern_memory_injected"] = float(memory_injected)
        selected = self._ensure_global_pattern_cover(selected, base_sequence)
        memory_keys = {tuple(int(pid) for pid in pattern.part_ids) for pattern in memory_patterns}
        self.meta["global_pattern_memory_selected"] = float(
            sum(1 for pattern in selected if tuple(int(pid) for pid in pattern.part_ids) in memory_keys)
        )
        return len(raw_patterns), selected

    def _global_pattern_master_repair(
        self,
        base_sequence: Tuple[int, ...],
        base_snapshot: LayoutSnapshot,
    ) -> Tuple[Tuple[int, ...], LayoutSnapshot]:
        if not self.global_pattern_master_enable:
            return base_sequence, base_snapshot

        if self.global_pattern_include_baseline:
            normalized = self._normalize_sequence_candidate(self._area_aspect_sequence())
            if normalized is not None:
                seq_t, snap_t, _ = normalized
                self._record_layout_candidate(seq_t, snap_t, source="baseline")

        self.meta["global_pattern_attempts"] += 1.0
        generated_count, patterns = self._global_pattern_candidates(base_sequence, base_snapshot)
        self.meta["global_pattern_patterns_generated"] += float(generated_count)
        self.meta["global_pattern_patterns_kept"] += float(len(patterns))
        if not patterns:
            return base_sequence, base_snapshot

        time_limit_s = self.global_pattern_time_limit_s
        if math.isfinite(self.solve_deadline):
            time_limit_s = min(time_limit_s, max(0.2, self.solve_deadline - time.perf_counter()))
        if time_limit_s <= 0.0:
            return base_sequence, base_snapshot

        window = self._global_pattern_window(base_sequence, base_snapshot)
        hint_part_sets = self._snapshot_hint_part_sets(base_snapshot, window.movable_part_ids)
        self.meta["global_pattern_hint_patterns"] = float(len(hint_part_sets))
        try:
            selected_patterns, status = self._solve_pattern_master(
                window,
                patterns,
                time_limit_s=time_limit_s,
                hint_part_sets=hint_part_sets,
            )
        except Exception:
            return base_sequence, base_snapshot

        self.meta["global_pattern_master_status"] = status
        self.meta["global_pattern_backend"] = str(self.meta.get("post_repair_backend", ""))
        if not selected_patterns:
            return base_sequence, base_snapshot

        candidate = self._sequence_from_selected_patterns(tuple(), selected_patterns, base_sequence)
        if candidate is None:
            return base_sequence, base_snapshot

        seq_cur, snap_cur, _ = candidate
        if self._terminal_score(snap_cur) > self._terminal_score(base_snapshot):
            boards_saved = max(0, len(base_snapshot.boards) - len(snap_cur.boards))
            self.meta["global_pattern_improvements"] += 1.0
            self.meta["global_pattern_boards_saved"] = max(
                float(self.meta.get("global_pattern_boards_saved", 0.0)),
                float(boards_saved),
            )
            self._update_global_best(seq_cur, snap_cur)
            return seq_cur, snap_cur
        return base_sequence, base_snapshot

    def _pattern_master_post_repair(
        self,
        base_sequence: Tuple[int, ...],
        base_snapshot: LayoutSnapshot,
    ) -> Tuple[Tuple[int, ...], LayoutSnapshot]:
        if not self.post_repair_enable:
            return base_sequence, base_snapshot

        repair_start = time.perf_counter()
        best_sequence = tuple(int(x) for x in base_sequence)
        best_snapshot = base_snapshot
        best_score = self._terminal_score(best_snapshot)
        deadline = repair_start + self.post_repair_time_limit_s
        windows = self._repair_windows(best_snapshot, best_sequence)
        if not windows:
            return best_sequence, best_snapshot

        for window in windows:
            variants = self._window_expansion_chain(best_snapshot, best_sequence, window)
            for expand_idx, window_cur in enumerate(variants):
                if time.perf_counter() >= deadline:
                    break
                if expand_idx > 0:
                    self.meta["post_repair_dynamic_expansions"] += 1.0
                before_snapshot = best_snapshot
                self.meta["repair_attempts"] += 1.0
                self.meta["post_repair_attempts"] += 1.0
                attempt_start = time.perf_counter()
                try:
                    prefix_snapshot, _ = self.decoder.replay_order(window_cur.fixed_prefix_sequence)
                except Exception:
                    self._update_post_repair_operator_score(
                        window_cur,
                        before_snapshot,
                        before_snapshot,
                        attempt_time_s=time.perf_counter() - attempt_start,
                    )
                    continue
                generated_count, patterns = self._generate_pattern_candidates(best_snapshot, window_cur, prefix_snapshot)
                self.meta["post_repair_patterns_generated"] += float(generated_count)
                self.meta["post_repair_patterns_kept"] += float(len(patterns))
                if not patterns:
                    self._update_post_repair_operator_score(
                        window_cur,
                        before_snapshot,
                        before_snapshot,
                        attempt_time_s=time.perf_counter() - attempt_start,
                    )
                    continue

                time_left = max(0.2, deadline - time.perf_counter())
                solve_start = time.perf_counter()
                hint_part_sets = self._snapshot_hint_part_sets(before_snapshot, window_cur.movable_part_ids)
                self.meta["post_repair_hint_patterns"] = float(len(hint_part_sets))
                try:
                    selected_patterns, status = self._solve_pattern_master(
                        window_cur,
                        patterns,
                        time_limit_s=time_left,
                        hint_part_sets=hint_part_sets,
                    )
                except Exception:
                    self._update_post_repair_operator_score(
                        window_cur,
                        before_snapshot,
                        before_snapshot,
                        attempt_time_s=time.perf_counter() - attempt_start,
                    )
                    continue

                solve_time_s = time.perf_counter() - solve_start
                self.meta["post_repair_master_status"] = status
                candidate = self._sequence_from_selected_patterns(window_cur.fixed_prefix_sequence, selected_patterns, best_sequence)
                improved = False
                if candidate is not None:
                    seq_cur, snap_cur, _ = candidate
                    cand_score = self._terminal_score(snap_cur)
                    if cand_score > best_score:
                        boards_saved = max(0, len(best_snapshot.boards) - len(snap_cur.boards))
                        best_sequence = seq_cur
                        best_snapshot = snap_cur
                        best_score = cand_score
                        self.meta["repair_improvements"] += 1.0
                        self.meta["post_repair_improvements"] += 1.0
                        self.meta["post_repair_best_window_start"] = float(window_cur.start_idx)
                        self.meta["post_repair_boards_saved"] = max(
                            float(self.meta.get("post_repair_boards_saved", 0.0)),
                            float(boards_saved),
                        )
                        if boards_saved > 0:
                            self.meta["tail_boards_cleared"] += float(boards_saved)
                        self._update_global_best(seq_cur, snap_cur)
                        improved = True
                attempt_time_s = time.perf_counter() - attempt_start
                self._update_post_repair_operator_score(
                    window_cur,
                    before_snapshot,
                    best_snapshot if improved else before_snapshot,
                    attempt_time_s=attempt_time_s,
                )
                if improved:
                    break
                if expand_idx + 1 >= len(variants):
                    break
                if solve_time_s > self.post_repair_dynamic_fast_solve_s:
                    break
                if time.perf_counter() >= deadline:
                    break
            if time.perf_counter() >= deadline:
                break

        self.meta["repair_time_s"] += float(time.perf_counter() - repair_start)
        self.meta["post_repair_time_s"] += float(time.perf_counter() - repair_start)
        return best_sequence, best_snapshot

    def _elite_candidates_for_repair(
        self,
        fallback_sequence: Tuple[int, ...],
        fallback_snapshot: LayoutSnapshot,
    ) -> List[EliteCandidate]:
        self._record_elite_candidate(fallback_sequence, fallback_snapshot)
        ordered = list(self.elite_archive)
        if self.global_best_sequence is not None and self.global_best_snapshot is not None:
            self._record_elite_candidate(self.global_best_sequence, self.global_best_snapshot)
            ordered = list(self.elite_archive)
        ordered.sort(key=lambda item: (item.score, len(item.sequence)), reverse=True)
        return ordered[: self.repair_elite_topk]

    def _focused_tail_collapse(
        self,
        base_sequence: Tuple[int, ...],
        base_snapshot: LayoutSnapshot,
    ) -> Tuple[Tuple[int, ...], LayoutSnapshot]:
        repair_start = time.perf_counter()
        try:
            work_snapshot, work_snapshots = self.decoder.replay_order(base_sequence)
        except Exception:
            return base_sequence, base_snapshot

        work_sequence = tuple(int(x) for x in base_sequence)
        work_priority = self._region_priority(work_snapshot)
        work_score = self._terminal_score(work_snapshot)
        improved_any = False

        for pass_idx in range(1, self.repair_tail_boards + 1):
            allow_blockers = pass_idx >= min(3, self.repair_tail_boards)
            self.meta["repair_attempts"] += 1.0
            self.meta["tail_collapse_attempts"] += 1.0
            seq_cur, snap_cur, snaps_cur, nodes, improved = self._repair_pass(
                work_sequence,
                work_snapshot,
                work_snapshots,
                tail_boards=pass_idx,
                allow_blockers=allow_blockers,
            )
            normalized = self._normalize_sequence_candidate(seq_cur)
            if normalized is None:
                continue
            seq_cur, snap_cur, snaps_cur = normalized
            self.meta["repair_nodes_expanded"] += float(nodes)
            cand_priority = self._region_priority(snap_cur)
            cand_score = self._terminal_score(snap_cur)
            if self._region_accepts(work_priority, work_score, cand_priority, cand_score):
                work_sequence = seq_cur
                work_snapshot = snap_cur
                work_snapshots = snaps_cur
                work_priority = cand_priority
                work_score = cand_score
                improved_any = improved_any or improved
                if improved:
                    self.meta["tail_collapse_improvements"] += 1.0
            self._update_global_best(seq_cur, snap_cur)

        self.meta["repair_time_s"] += float(time.perf_counter() - repair_start)
        return work_sequence, work_snapshot

    def _board_closing_repair(
        self,
        fallback_sequence: Tuple[int, ...],
        fallback_snapshot: LayoutSnapshot,
    ) -> Tuple[Tuple[int, ...], LayoutSnapshot]:
        return self._focused_tail_collapse(fallback_sequence, fallback_snapshot)

    def _memetic_tail_search(self, base_sequence: Tuple[int, ...], base_snapshot: LayoutSnapshot) -> Tuple[Tuple[int, ...], LayoutSnapshot]:
        best_sequence = tuple(int(x) for x in base_sequence)
        best_snapshot = base_snapshot
        best_score = self._terminal_score(best_snapshot)

        region_sequence, region_snapshot = self._region_guided_repair(best_sequence, best_snapshot)
        if self._terminal_score(region_snapshot) > best_score:
            best_sequence = region_sequence
            best_snapshot = region_snapshot
            best_score = self._terminal_score(region_snapshot)

        post_sequence, post_snapshot = self._pattern_master_post_repair(best_sequence, best_snapshot)
        if self._terminal_score(post_snapshot) > best_score:
            best_sequence = post_sequence
            best_snapshot = post_snapshot
            best_score = self._terminal_score(post_snapshot)

        collapsed_sequence, collapsed_snapshot = self._focused_tail_collapse(best_sequence, best_snapshot)
        if self._terminal_score(collapsed_snapshot) > best_score:
            best_sequence = collapsed_sequence
            best_snapshot = collapsed_snapshot
            best_score = self._terminal_score(collapsed_snapshot)

        polished_sequence, polished_snapshot = self._legacy_tail_search(collapsed_sequence, collapsed_snapshot)
        if self._terminal_score(polished_snapshot) > best_score:
            best_sequence = polished_sequence
            best_snapshot = polished_snapshot

        return best_sequence, best_snapshot

    def solve(self) -> SearchResult:
        self._prime_constructive_warmstarts()
        root = SearchNode(snapshot=_empty_snapshot(), remaining_mask=self.full_remaining_mask, prefix_sequence=tuple())
        self._warm_start(root)
        committed_snapshot = root.snapshot
        overall_deadline = time.perf_counter() + self.solver_time_limit_s if self.solver_time_limit_s > 0.0 else math.inf
        reserve_s = self._post_mcts_reserve_s()
        search_deadline = overall_deadline
        if math.isfinite(overall_deadline) and reserve_s > 0.0:
            search_deadline = max(time.perf_counter() + 0.25, overall_deadline - reserve_s)
        self.solve_deadline = overall_deadline
        self.meta["post_mcts_reserve_s"] = float(reserve_s)

        while root.remaining_mask:
            if time.perf_counter() >= search_deadline:
                self.meta["solver_timeout_hit"] = 1.0
                break
            self.meta["commit_rounds"] += 1.0
            round_floor = max(self.sims_per_step, self.commit_min_root_visits)
            round_sims = 0
            confirm_passes = 0
            chain: List[Tuple[int, SearchNode]] = []
            last_chain_signature: Tuple[int, ...] = tuple()
            round_cap = max(round_floor, self._round_sim_budget())

            while time.perf_counter() < search_deadline:
                sims_now = max(1, self.sims_per_step)
                self._run_simulations(root, sims_now)
                round_sims += sims_now
                if round_sims < round_floor:
                    continue
                candidate_chain = self._commit_chain(root)
                if candidate_chain and len(candidate_chain) >= max(1, self.commit_min_steps):
                    chain_signature = self._chain_signature(candidate_chain)
                    if chain_signature == last_chain_signature:
                        confirm_passes += 1
                    else:
                        confirm_passes = 1
                        last_chain_signature = chain_signature
                    chain = candidate_chain
                    if confirm_passes >= self.commit_confirm_passes:
                        break
                else:
                    confirm_passes = 0
                    last_chain_signature = tuple()
                if self.commit_force_enable and round_sims >= round_cap:
                    chain = self._commit_chain(root, force=True)
                    if chain:
                        self.meta["commit_force_rounds"] += 1.0
                        break

            if not chain:
                if time.perf_counter() >= search_deadline:
                    self.meta["solver_timeout_hit"] = 1.0
                    if self.commit_timeout_force:
                        chain = self._commit_chain(root, force=True)
                        if chain:
                            self.meta["commit_force_rounds"] += 1.0
            if not chain:
                break
            if len(chain) > 1:
                self.meta["commit_combo_rounds"] += 1.0
            for action, child in chain:
                committed_snapshot = child.snapshot
                root = self._detach_root(child)
                self._warm_start(root)
                self.meta["commit_steps"] += 1.0
                if not root.remaining_mask:
                    break

        committed_sequence_t = tuple(int(pid) for pid in root.prefix_sequence)
        if self._is_complete_sequence(committed_sequence_t):
            committed_score = self._terminal_score(committed_snapshot)
            self._update_global_best(committed_sequence_t, committed_snapshot)
            base_sequence = committed_sequence_t
            base_snapshot = committed_snapshot
        else:
            completed_snapshot, completion_suffix = self._rollout(committed_snapshot, root.remaining_mask)
            base_sequence = committed_sequence_t + tuple(int(x) for x in completion_suffix)
            base_snapshot = completed_snapshot
            committed_score = self._terminal_score(base_snapshot)
            self._update_global_best(base_sequence, base_snapshot)
        if self.global_best_sequence is not None and self.global_best_snapshot is not None:
            if self.global_best_score is not None and self.global_best_score > committed_score:
                base_sequence = self.global_best_sequence
                base_snapshot = self.global_best_snapshot

        annealed_sequence, annealed_snapshot = self._sequence_sa_lns(base_sequence, base_snapshot)
        polished_sequence, polished_snapshot = self._memetic_tail_search(annealed_sequence, annealed_snapshot)
        self._update_global_best(polished_sequence, polished_snapshot)
        harvested_sequence, harvested_snapshot = self._global_pattern_master_repair(polished_sequence, polished_snapshot)
        self._update_global_best(harvested_sequence, harvested_snapshot)

        final_sequence = harvested_sequence
        result_source = "post_pipeline"
        if self.global_best_sequence is not None and self.global_best_snapshot is not None:
            if self.global_best_score is not None and self.global_best_score > self._terminal_score(harvested_snapshot):
                final_sequence = self.global_best_sequence
                result_source = "global_best"
        if tuple(final_sequence) == self._area_aspect_sequence():
            result_source = "baseline_floor"
        final_snapshot, _ = OmnipotentDecoder(self.parts_by_id, self.cfg).replay_order(final_sequence)
        self.meta["result_source"] = result_source
        self.meta["elite_archive_size"] = float(len(self.elite_archive))
        self.meta["rollout_cache_entries"] = float(len(self.rollout_cache))
        self.meta["decode_cache_hits"] = float(self.decoder.decode_hits)
        self.meta["decode_cache_misses"] = float(self.decoder.decode_misses)
        self.meta["decode_cache_entries"] = float(len(self.decoder.decode_cache))
        self.meta["micro_cache_hits"] = float(self.decoder.micro_hits)
        self.meta["micro_cache_misses"] = float(self.decoder.micro_misses)
        self.meta["micro_cache_entries"] = float(len(self.decoder.micro_cache))
        self.meta["macro_cache_entries"] = float(len(self.macro_cache))
        self.meta["coarse_fit_cache_entries"] = float(len(self.coarse_fit_cache))

        return SearchResult(
            boards=list(final_snapshot.boards),
            sequence=final_sequence,
            score_global=self._terminal_score(final_snapshot),
            scalar_reward=self._reward_scalar(final_snapshot, 0),
            meta=dict(self.meta),
        )


def solve_with_receding_horizon_mcts(parts: Sequence, cfg, seed: int) -> SearchResult:
    return RecedingHorizonMCTS(parts, cfg, seed).solve()
