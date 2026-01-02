# src/inrp/packer.py
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import random
import math
import importlib.util

# --- Numba JIT 导入与回退机制 ---
if importlib.util.find_spec("numba") is not None:
    from numba import jit
    HAS_NUMBA = True
else:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
# --------------------------------------

EPS = 1e-9

# --- 静态 JIT 函数 (极速几何核心) ---
@jit(nopython=True)
def _jit_overlap(ax, ay, aw, ah, bx, by, bw, bh, eps):
    # strict overlap
    return (ax < bx + bw - eps and ax + aw > bx + eps and
            ay < by + bh - eps and ay + ah > by + eps)

@jit(nopython=True)
def _jit_inside(rx, ry, rw, rh, W, H, eps):
    return (rx >= -eps and ry >= -eps and
            rx + rw <= W + eps and ry + rh <= H + eps)
# ------------------------------------------

@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float

@dataclass
class PlacedPart:
    uid: int
    pid_raw: str
    rect: Rect              # inflated rect
    w0: float               # original width
    h0: float               # original height
    rotated: bool = False

@dataclass
class Board:
    bid: int
    W: float
    H: float
    placed: List[PlacedPart] = field(default_factory=list)

# --- 使用 JIT 核心的包装函数 ---
def _overlap(a: Rect, b: Rect) -> bool:
    return _jit_overlap(a.x, a.y, a.w, a.h, b.x, b.y, b.w, b.h, EPS)

def _inside_board(r: Rect, W: float, H: float) -> bool:
    return _jit_inside(r.x, r.y, r.w, r.h, W, H, EPS)

def _can_place(board: Board, r: Rect) -> bool:
    # 快速解包，减少对象访问开销
    rx, ry, rw, rh = r.x, r.y, r.w, r.h

    if not _jit_inside(rx, ry, rw, rh, board.W, board.H, EPS):
        return False

    for pp in board.placed:
        pr = pp.rect
        # 调用 JIT 函数
        if _jit_overlap(rx, ry, rw, rh, pr.x, pr.y, pr.w, pr.h, EPS):
            return False
    return True
# --------------------------------------

def _candidate_points(board: Board) -> List[Tuple[float, float]]:
    pts = {(0.0, 0.0)}
    for pp in board.placed:
        x, y, w, h = pp.rect.x, pp.rect.y, pp.rect.w, pp.rect.h
        pts.add((x + w, y))
        pts.add((x, y + h))
    return sorted(pts, key=lambda t: (t[1], t[0]))

def _touch_shared_len(a: Rect, b: Rect) -> float:
    shared = 0.0
    if abs((a.x + a.w) - b.x) <= 1e-6 or abs((b.x + b.w) - a.x) <= 1e-6:
        y0 = max(a.y, b.y)
        y1 = min(a.y + a.h, b.y + b.h)
        if y1 - y0 > 1e-6:
            shared += (y1 - y0)
    if abs((a.y + a.h) - b.y) <= 1e-6 or abs((b.y + b.h) - a.y) <= 1e-6:
        x0 = max(a.x, b.x)
        x1 = min(a.x + a.w, b.x + b.w)
        if x1 - x0 > 1e-6:
            shared += (x1 - x0)
    return shared

def _border_contact(r: Rect, W: float, H: float) -> float:
    c = 0.0
    if abs(r.x - 0.0) <= 1e-6: c += r.h
    if abs((r.x + r.w) - W) <= 1e-6: c += r.h
    if abs(r.y - 0.0) <= 1e-6: c += r.w
    if abs((r.y + r.h) - H) <= 1e-6: c += r.w
    return c

def _place_one(board: Board, uid: int, pid_raw: str,
               w: float, h: float, w0: float, h0: float,
               allow_rot: bool,
               place_mode: str = "blf",
               rng: Optional[random.Random] = None,
               rand_topk: int = 3) -> bool:
    pts = _candidate_points(board)
    dims = [(w, h, False)]
    if allow_rot and abs(w - h) > EPS:
        dims.append((h, w, True))

    if place_mode == "blf":
        for (x, y) in pts:
            for (cw, ch, rot) in dims:
                r = Rect(x, y, cw, ch)
                if _can_place(board, r):
                    board.placed.append(PlacedPart(uid=uid, pid_raw=pid_raw, rect=r,
                                                   w0=w0, h0=h0, rotated=rot))
                    return True
        return False

    if place_mode == "blf_rand":
        rng = rng or random.Random(0)
        topk = max(1, int(rand_topk))
        dims2 = dims[:]
        rng.shuffle(dims2)
        i = 0
        while i < len(pts):
            y_level = pts[i][1]
            cand = []
            while i < len(pts) and abs(pts[i][1] - y_level) <= 1e-12:
                x, y = pts[i]
                for (cw, ch, rot) in dims2:
                    r = Rect(x, y, cw, ch)
                    if _can_place(board, r):
                        cand.append((x, r, rot))
                i += 1
            if cand:
                cand.sort(key=lambda t: t[0])
                pick_pool = cand[:min(topk, len(cand))]
                x_pick, r_best, rot_best = rng.choice(pick_pool)
                board.placed.append(PlacedPart(uid=uid, pid_raw=pid_raw, rect=r_best,
                                               w0=w0, h0=h0, rotated=rot_best))
                return True
        return False

    if place_mode == "blf_ccf":
        i = 0
        while i < len(pts):
            y_level = pts[i][1]
            cand = []
            while i < len(pts) and abs(pts[i][1] - y_level) <= 1e-12:
                x, y = pts[i]
                for (cw, ch, rot) in dims:
                    r = Rect(x, y, cw, ch)
                    if _can_place(board, r):
                        shared = 0.0
                        for pp in board.placed:
                            shared += _touch_shared_len(r, pp.rect)
                        border = _border_contact(r, board.W, board.H)
                        key = (-shared, -border, x)
                        cand.append((key, r, rot))
                i += 1
            if cand:
                cand.sort(key=lambda t: t[0])
                _, r_best, rot_best = cand[0]
                board.placed.append(PlacedPart(uid=uid, pid_raw=pid_raw, rect=r_best,
                                               w0=w0, h0=h0, rotated=rot_best))
                return True
        return False

    raise ValueError(f"unknown place_mode: {place_mode}")

def pack_one_board(parts: List, W: float, H: float, allow_rot: bool, bid: int,
                   order: str = "size", place_mode: str = "blf",
                   rng: Optional[random.Random] = None,
                   rand_topk: int = 3) -> Tuple[Board, List]:
    board = Board(bid=bid, W=W, H=H)
    for p in parts:
        ok = (p.w <= W + EPS and p.h <= H + EPS) or (allow_rot and p.h <= W + EPS and p.w <= H + EPS)
        if not ok:
            raise RuntimeError(f"Part too big: uid={p.uid}")

    if order == "size":
        parts_sorted = sorted(parts, key=lambda p: (max(p.w, p.h), p.w * p.h), reverse=True)
    elif order == "area":
        parts_sorted = sorted(parts, key=lambda p: p.w * p.h, reverse=True)
    elif order == "none":
        parts_sorted = parts[:]
    else:
        parts_sorted = parts[:]

    remaining = []
    for p in parts_sorted:
        ok = _place_one(board, p.uid, p.pid_raw, p.w, p.h, p.w0, p.h0,
                        allow_rot, place_mode=place_mode, rng=rng, rand_topk=rand_topk)
        if not ok:
            remaining.append(p)
    return board, remaining

def pack_multi_board(parts: List, W: float, H: float, allow_rot: bool,
                     max_boards: int = 10_000, order: str = "size", place_mode: str = "blf",
                     rng: Optional[random.Random] = None,
                     rand_topk: int = 3) -> List[Board]:
    left = parts[:]
    boards: List[Board] = []
    bid = 1
    while left:
        if bid > max_boards:
            raise RuntimeError(f"Max boards limit reached")
        board, left2 = pack_one_board(
            left, W, H, allow_rot, bid=bid,
            order=order, place_mode=place_mode,
            rng=rng, rand_topk=rand_topk
        )
        if len(board.placed) == 0:
            raise RuntimeError(f"Cannot place part: uid={left[0].uid}")
        boards.append(board)
        left = left2
        bid += 1
    return boards

def pack_baselineA(parts: List, W: float, H: float, allow_rot: bool) -> List[Board]:
    return pack_multi_board(parts, W, H, allow_rot, order="size", place_mode="blf")

def _avg_util(boards: List[Board]) -> float:
    if not boards: return 0.0
    s = 0.0
    for b in boards:
        used = sum(pp.rect.w * pp.rect.h for pp in b.placed)
        s += used / (b.W * b.H)
    return s / len(boards)

def pack_baselineB(parts: List, W: float, H: float, allow_rot: bool,
                   restarts: int = 50, seed: int = 0,
                   rand_place: bool = True,
                   rand_topk: int = 3) -> List[Board]:
    rng = random.Random(seed)
    base = parts[:]

    def key_size(p): return (max(p.w, p.h), p.w * p.h)
    def key_area(p): return (p.w * p.h, max(p.w, p.h))
    def key_maxside(p): return (max(p.w, p.h), p.w * p.h)
    def key_minside(p): return (min(p.w, p.h), p.w * p.h)
    def key_ratio(p): return (max(p.w, p.h) / max(EPS, min(p.w, p.h)), p.w * p.h)

    recipes: List[Callable[[List], List]] = []
    recipes.append(lambda arr: sorted(arr, key=key_size, reverse=True))
    recipes.append(lambda arr: sorted(arr, key=key_area, reverse=True))
    recipes.append(lambda arr: sorted(arr, key=key_maxside, reverse=True))
    recipes.append(lambda arr: sorted(arr, key=key_ratio, reverse=True))
    recipes.append(lambda arr: sorted(arr, key=key_minside, reverse=True))
    def _shuffle(arr):
        arr2 = arr[:]
        rng.shuffle(arr2)
        return arr2
    recipes.append(_shuffle)
    def _noisy_size(arr):
        return sorted(arr, key=lambda p: (key_size(p), rng.random()), reverse=True)
    recipes.append(_noisy_size)

    place_mode = "blf_rand" if rand_place else "blf"
    arr0 = recipes[0](base)
    boards0 = pack_multi_board(arr0, W, H, allow_rot, order="none", place_mode=place_mode, rng=rng, rand_topk=rand_topk)
    best: Optional[List[Board]] = boards0
    best_key = (len(boards0), -_avg_util(boards0))

    R = max(1, int(restarts))
    for t in range(1, R):
        rec = recipes[t % len(recipes)]
        arr = rec(base)
        boards = pack_multi_board(arr, W, H, allow_rot, order="none", place_mode=place_mode, rng=rng, rand_topk=rand_topk)
        key = (len(boards), -_avg_util(boards))
        if key < best_key:
            best = boards
            best_key = key
    return best

# --- 新增：Stage 1 SA 序列优化器 ---
def pack_sequence_sa(
        parts: List,
        W: float,
        H: float,
        allow_rot: bool,
        seed: int,
        iters: int = 2000,
        T0: float = 0.02,
        alpha: float = 0.995,
) -> List[Board]:
    """
    Simulated Annealing to optimize the input sequence of parts.
    Objective: Minimize N_boards, then Maximize Utilization.
    """
    rng = random.Random(seed)

    # 1. Initial Solution (Sort by Area Descending - strong baseline)
    current_seq = sorted(parts, key=lambda p: p.w * p.h, reverse=True)

    # Eval: deterministic BLF is fast and stable for outer loop
    current_boards = pack_multi_board(current_seq, W, H, allow_rot, order="none", place_mode="blf")
    current_u = _avg_util(current_boards)
    current_n = len(current_boards)

    best_boards = current_boards
    best_key = (current_n, -current_u) # Minimize this tuple

    T = T0
    n_parts = len(parts)

    for _ in range(iters):
        # 2. Neighbor: Swap two random parts
        if n_parts < 2:
            break
        idx1, idx2 = rng.sample(range(n_parts), 2)
        new_seq = current_seq[:]
        new_seq[idx1], new_seq[idx2] = new_seq[idx2], new_seq[idx1]

        # 3. Eval
        try:
            new_boards = pack_multi_board(new_seq, W, H, allow_rot, order="none", place_mode="blf")
        except RuntimeError:
            continue # Skip infeasible permutations if any

        new_u = _avg_util(new_boards)
        new_n = len(new_boards)

        # 4. Acceptance Criterion
        delta_n = new_n - current_n
        delta_u = new_u - current_u # Higher is better

        accept = False
        if delta_n < 0:
            # Improvement in board count (primary objective)
            accept = True
        elif delta_n > 0:
            # Degradation in board count - usually reject, unless T is very high
            # For strict nesting, we usually reject increasing N_boards.
            accept = False
        else:
            # Same board count, look at utilization
            if delta_u > 0:
                accept = True
            else:
                # Metropolis: exp( (new_U - curr_U) / T )
                # delta_u is negative, T is positive
                if rng.random() < math.exp(delta_u / max(1e-9, T)):
                    accept = True

        if accept:
            current_seq = new_seq
            current_boards = new_boards
            current_n = new_n
            current_u = new_u

            # Update Global Best
            if (current_n, -current_u) < best_key:
                best_boards = current_boards
                best_key = (current_n, -current_u)

        T *= alpha

    return best_boards

def _board_score_shared_border_trail(
        board: Board,
        *,
        cut_mode: str = "trail",
        tab_enable: bool = False,
        tab_per_part: int = 0,
        tab_len: float = 0.0,
        tab_corner_clear: float = 0.0,
        line_snap_eps: float = 0.0,
        min_shared_len: float = 0.0,
        nd_coord: int = 6,
) -> Tuple[float, float, int]:
    from .routing import build_segments_from_board, estimate_cut_and_strokes
    segs, L_shared, _n_tabs = build_segments_from_board(
        board,
        share_mode="union",
        tab_enable=tab_enable,
        tab_per_part=tab_per_part,
        tab_len=tab_len,
        tab_corner_clear=tab_corner_clear,
        line_snap_eps=line_snap_eps,
        min_shared_len=min_shared_len,
        nd=nd_coord,
    )
    _L_cut_base, _n_comp, n_stroke, _reps, _comps = estimate_cut_and_strokes(segs, cut_mode=cut_mode)
    border = sum(_border_contact(pp.rect, board.W, board.H) for pp in board.placed)
    return float(L_shared), float(border), int(n_stroke)

def _repack_fixed_parts_ccf(
        parts_on_board: List,
        W: float,
        H: float,
        allow_rot: bool,
        bid: int,
        restarts: int,
        seed: int,
        *,
        border_w: float = 0.05,
        trail_penalty: float = 0.02,
        cut_mode: str = "trail",
        tab_enable: bool = False,
        tab_per_part: int = 0,
        tab_len: float = 0.0,
        tab_corner_clear: float = 0.0,
        line_snap_eps: float = 0.0,
        min_shared_len: float = 0.0,
        nd_coord: int = 6,
        method: str = "restarts",
        sa_iters: int = 600,
        sa_T0: float = 5000.0,
        sa_alpha: float = 0.995,
        sa_max_starts: int = 5,
) -> Optional[Board]:
    rng = random.Random(seed)
    base = sorted(parts_on_board, key=lambda p: (max(p.w, p.h), p.w * p.h), reverse=True)
    best_board: Optional[Board] = None
    best_key = None

    def _eval_order(arr) -> Tuple[Optional[Board], Optional[Tuple[float, float, float, int]]]:
        b, remaining = pack_one_board(arr, W, H, allow_rot, bid=bid, order="none", place_mode="blf_ccf")
        if remaining: return None, None
        shared, border, n_stroke = _board_score_shared_border_trail(
            b,
            cut_mode=cut_mode,
            tab_enable=tab_enable,
            tab_per_part=tab_per_part,
            tab_len=tab_len,
            tab_corner_clear=tab_corner_clear,
            line_snap_eps=line_snap_eps,
            min_shared_len=min_shared_len,
            nd_coord=nd_coord,
        )
        score = float(shared) + float(border_w) * float(border) - float(trail_penalty) * float(n_stroke)
        key = (-score, -shared, -border, n_stroke)
        return b, (score, shared, border, n_stroke, key)

    def _update_best(b: Board, info):
        nonlocal best_board, best_key
        _score, _shared, _border, _n_stroke, key = info
        if best_board is None or key < best_key:
            best_board, best_key = b, key

    method = (method or "restarts").lower().strip()
    starts = max(1, int(restarts))
    if method == "sa":
        starts = min(starts, max(1, int(sa_max_starts)))

    for t in range(starts):
        cur_order = base[:]
        if t > 0: rng.shuffle(cur_order)
        b0, info0 = _eval_order(cur_order)
        if b0 is None: continue
        _update_best(b0, info0)

        if method != "sa" or sa_iters <= 0 or len(cur_order) < 2:
            continue

        cur_board = b0
        cur_info = info0
        cur_score = float(cur_info[0])
        T = float(sa_T0)
        n = len(cur_order)
        iters = int(sa_iters)

        for _ in range(iters):
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j: continue
            cand_order = cur_order[:]
            cand_order[i], cand_order[j] = cand_order[j], cand_order[i]

            b1, info1 = _eval_order(cand_order)
            if b1 is None:
                T *= float(sa_alpha)
                continue

            cand_score = float(info1[0])
            delta = cand_score - cur_score

            if delta >= 0.0:
                accept = True
            else:
                accept = (rng.random() < math.exp(delta / max(1e-9, T)))

            if accept:
                cur_order = cand_order
                cur_board = b1
                cur_info = info1
                cur_score = cand_score
                _update_best(cur_board, cur_info)
            T *= float(sa_alpha)

    return best_board

def pack_proposed_shared(
        parts: List,
        W: float,
        H: float,
        allow_rot: bool,
        restarts: int = 10,
        seed: int = 0,
        inner_restarts: int = 30,
        *,
        border_w: float = 0.05,
        trail_penalty: float = 0.02,
        cut_mode: str = "trail",
        tab_enable: bool = False,
        tab_per_part: int = 0,
        tab_len: float = 0.0,
        tab_corner_clear: float = 0.0,
        line_snap_eps: float = 0.0,
        min_shared_len: float = 0.0,
        nd_coord: int = 6,
        stage2_method: str = "restarts",
        stage2_sa_iters: int = 600,
        stage2_sa_T0: float = 5000.0,
        stage2_sa_alpha: float = 0.995,
        stage2_sa_max_starts: int = 5,
        stage1_method: str = "sa",  # 新增：默认为 "sa" 启用方案 1
        stage1_sa_iters: int = 2000, # 新增：Stage 1 SA 迭代次数
) -> List[Board]:
    """
    Proposed(shared) Route-2 Two-stage (Paper-friendly):
    Stage1: Sequence optimization (SA) or BaselineB to get minimal N_board.
    Stage2: Fixed-membership internal repacking (SA) for shared edges.
    """
    # Stage 1: Sequence Optimization
    if stage1_method == "sa":
        boards_stage1 = pack_sequence_sa(parts, W, H, allow_rot, seed=seed, iters=stage1_sa_iters)
    else:
        boards_stage1 = pack_baselineB(parts, W, H, allow_rot, restarts=restarts, seed=seed)

    # Stage 2: Shared-Edge Repacking (fixed membership)
    uid2part = {p.uid: p for p in parts}
    improved: List[Board] = []

    for b in boards_stage1:
        subset = [uid2part[pp.uid] for pp in b.placed]
        b2 = _repack_fixed_parts_ccf(
            subset, W, H, allow_rot,
            bid=b.bid,
            restarts=inner_restarts,
            seed=seed * 100000 + b.bid,
            border_w=border_w,
            trail_penalty=trail_penalty,
            cut_mode=cut_mode,
            tab_enable=tab_enable,
            tab_per_part=tab_per_part,
            tab_len=tab_len,
            tab_corner_clear=tab_corner_clear,
            line_snap_eps=line_snap_eps,
            min_shared_len=min_shared_len,
            nd_coord=nd_coord,
            method=stage2_method,
            sa_iters=stage2_sa_iters,
            sa_T0=stage2_sa_T0,
            sa_alpha=stage2_sa_alpha,
            sa_max_starts=stage2_sa_max_starts,
        )
        improved.append(b2 if b2 is not None else b)

    return improved