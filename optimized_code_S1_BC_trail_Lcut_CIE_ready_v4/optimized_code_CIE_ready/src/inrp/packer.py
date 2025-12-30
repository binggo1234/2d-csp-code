# packer.py
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random

EPS = 1e-9


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
    rect: Rect              # inflated rect (already includes trim/gap/tool)
    w0: float               # original width  (no inflation)
    h0: float               # original height (no inflation)
    rotated: bool = False


@dataclass
class Board:
    bid: int
    W: float
    H: float
    placed: List[PlacedPart] = field(default_factory=list)


def _overlap(a: Rect, b: Rect) -> bool:
    # strict overlap (touching edge is allowed)
    return (a.x < b.x + b.w - EPS and a.x + a.w > b.x + EPS and
            a.y < b.y + b.h - EPS and a.y + a.h > b.y + EPS)


def _inside_board(r: Rect, W: float, H: float) -> bool:
    return (r.x >= -EPS and r.y >= -EPS and
            r.x + r.w <= W + EPS and r.y + r.h <= H + EPS)


def _can_place(board: Board, r: Rect) -> bool:
    if not _inside_board(r, board.W, board.H):
        return False
    for pp in board.placed:
        if _overlap(r, pp.rect):
            return False
    return True


def _candidate_points(board: Board) -> List[Tuple[float, float]]:
    """
    Classic BLF candidates: (0,0) + each placed rect's (x+w,y) and (x,y+h)
    Sorted by (y, x) => Bottom-Left
    """
    pts = {(0.0, 0.0)}
    for pp in board.placed:
        x, y, w, h = pp.rect.x, pp.rect.y, pp.rect.w, pp.rect.h
        pts.add((x + w, y))
        pts.add((x, y + h))
    return sorted(pts, key=lambda t: (t[1], t[0]))


def _touch_shared_len(a: Rect, b: Rect) -> float:
    """
    Shared edge contact length between 2 rects if they touch on side.
    (Used for "CCF-like" preference inside one board)
    """
    shared = 0.0

    # vertical touch: a.right == b.left OR b.right == a.left
    if abs((a.x + a.w) - b.x) <= 1e-6 or abs((b.x + b.w) - a.x) <= 1e-6:
        y0 = max(a.y, b.y)
        y1 = min(a.y + a.h, b.y + b.h)
        if y1 - y0 > 1e-6:
            shared += (y1 - y0)

    # horizontal touch: a.top == b.bottom OR b.top == a.bottom
    if abs((a.y + a.h) - b.y) <= 1e-6 or abs((b.y + b.h) - a.y) <= 1e-6:
        x0 = max(a.x, b.x)
        x1 = min(a.x + a.w, b.x + b.w)
        if x1 - x0 > 1e-6:
            shared += (x1 - x0)

    return shared


def _border_contact(r: Rect, W: float, H: float) -> float:
    """How much rect touches board border."""
    c = 0.0
    if abs(r.x - 0.0) <= 1e-6:
        c += r.h
    if abs((r.x + r.w) - W) <= 1e-6:
        c += r.h
    if abs(r.y - 0.0) <= 1e-6:
        c += r.w
    if abs((r.y + r.h) - H) <= 1e-6:
        c += r.w
    return c


def _place_one(board: Board, uid: int, pid_raw: str,
               w: float, h: float, w0: float, h0: float,
               allow_rot: bool,
               place_mode: str = "blf",
               rng: Optional[random.Random] = None,
               rand_topk: int = 3) -> bool:
    """Place one part on current board.

    w/h are inflated sizes.

    place_mode:
      - "blf"      : first feasible at candidates (deterministic)
      - "blf_rand" : within the lowest y-level that has feasible placements,
                     pick randomly among the smallest-x top-K candidates
                     (diversifies BaselineB solutions)
      - "blf_ccf"  : within the lowest y-level, choose best by (shared, border, x)

    rng/rand_topk are only used by "blf_rand".
    """
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
        # shuffle rotation preference (mild diversification)
        dims2 = dims[:]
        rng.shuffle(dims2)

        i = 0
        while i < len(pts):
            y_level = pts[i][1]
            cand = []  # (x, r, rot)
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
        # Only search candidates at the lowest y-level that has feasible placements,
        # then choose by: max shared, max border, min x
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
    """Pack as many parts as possible into one board.

    parts elements must have:
      uid, pid_raw, w, h, w0, h0

    Returns (board, remaining_parts)

    Notes:
      - rng/rand_topk only affect place_mode="blf_rand".
    """
    board = Board(bid=bid, W=W, H=H)

    # feasibility filter
    for p in parts:
        ok = (p.w <= W + EPS and p.h <= H + EPS) or (allow_rot and p.h <= W + EPS and p.w <= H + EPS)
        if not ok:
            raise RuntimeError(f"存在无法放入板材的零件：uid={p.uid} raw={p.pid_raw} "
                               f"inflate={p.w}x{p.h} board={W}x{H}")

    # ordering
    if order == "size":
        parts_sorted = sorted(parts, key=lambda p: (max(p.w, p.h), p.w * p.h), reverse=True)
    elif order == "area":
        parts_sorted = sorted(parts, key=lambda p: p.w * p.h, reverse=True)
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
    """Multi-board packing."""
    left = parts[:]
    boards: List[Board] = []
    bid = 1

    while left:
        if bid > max_boards:
            raise RuntimeError(f"超过最大板数限制 max_boards={max_boards}，剩余零件={len(left)}")

        board, left2 = pack_one_board(
            left, W, H, allow_rot, bid=bid,
            order=order, place_mode=place_mode,
            rng=rng, rand_topk=rand_topk
        )

        if len(board.placed) == 0:
            bad = left[0]
            raise RuntimeError(f"无法放置任何零件：uid={bad.uid} raw={bad.pid_raw}")

        boards.append(board)
        left = left2
        bid += 1

    return boards


# ---------- Baselines / Proposed ----------

def pack_baselineA(parts: List, W: float, H: float, allow_rot: bool) -> List[Board]:
    """
    BaselineA: deterministic greedy BLF + size ordering.
    """
    return pack_multi_board(parts, W, H, allow_rot, order="size", place_mode="blf")


def _avg_util(boards: List[Board]) -> float:
    if not boards:
        return 0.0
    s = 0.0
    for b in boards:
        used = sum(pp.rect.w * pp.rect.h for pp in b.placed)  # inflated area
        s += used / (b.W * b.H)
    return s / len(boards)


def pack_baselineB(parts: List, W: float, H: float, allow_rot: bool,
                   restarts: int = 50, seed: int = 0,
                   rand_place: bool = True,
                   rand_topk: int = 3) -> List[Board]:
    """BaselineB: multi-start search for minimal N_board (and then higher utilization).

    Why this exists:
      - BaselineA is deterministic; some instances make BaselineB collapse into the same solution.
      - Here we diversify BOTH the input ordering and BLF tie-breaking (optional),
        so BaselineB becomes a meaningful "min-board" baseline for the paper.

    Selection key: (min N_board, max avgU).

    Params:
      restarts: number of starts (>=1). Includes a deterministic start.
      rand_place: if True, use place_mode="blf_rand" with a random tie-break among top-K candidates.
      rand_topk: K for the above.
    """
    rng = random.Random(seed)

    # base list (keep object identity)
    base = parts[:]

    def key_size(p):
        return (max(p.w, p.h), p.w * p.h)

    def key_area(p):
        return (p.w * p.h, max(p.w, p.h))

    def key_maxside(p):
        return (max(p.w, p.h), p.w * p.h)

    def key_minside(p):
        return (min(p.w, p.h), p.w * p.h)

    def key_ratio(p):
        r = max(p.w, p.h) / max(EPS, min(p.w, p.h))
        return (r, p.w * p.h)

    # ordering recipes (deterministic + noisy/tie-broken)
    recipes: List[Callable[[List], List]] = []

    # 0) deterministic size (same as BaselineA ordering)
    recipes.append(lambda arr: sorted(arr, key=key_size, reverse=True))

    # 1) deterministic area
    recipes.append(lambda arr: sorted(arr, key=key_area, reverse=True))

    # 2) max-side then area
    recipes.append(lambda arr: sorted(arr, key=key_maxside, reverse=True))

    # 3) aspect-ratio then area
    recipes.append(lambda arr: sorted(arr, key=key_ratio, reverse=True))

    # 4) min-side then area
    recipes.append(lambda arr: sorted(arr, key=key_minside, reverse=True))

    # 5) pure random shuffle
    def _shuffle(arr):
        arr2 = arr[:]
        rng.shuffle(arr2)
        return arr2
    recipes.append(_shuffle)

    # 6) noisy size key (break ties + add mild diversification)
    def _noisy_size(arr):
        return sorted(arr, key=lambda p: (key_size(p), rng.random()), reverse=True)
    recipes.append(_noisy_size)

    place_mode = "blf_rand" if rand_place else "blf"

    # candidate 0 (deterministic): avoid being worse than BaselineA
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

def _board_score_shared_border_trail(
    board: Board,
    *,
    cut_mode: str = "trail",
    tab_enable: bool = False,
    tab_per_part: int = 0,
    tab_len: float = 0.0,
    tab_corner_clear: float = 0.0,
    # shared-edge detection robustness
    line_snap_eps: float = 0.0,
    min_shared_len: float = 0.0,
    nd_coord: int = 6,
) -> Tuple[float, float, int]:
    """Score a board for Stage2 re-pack.

    Returns:
      L_shared (union mode), border contact length, n_stroke (trail count).

    Notes
    - We compute L_shared using the same "union" semantics as metrics/routing,
      so the Stage2 objective matches what we finally report.
    - n_stroke is derived from the cut-segment graph (after union + optional tabs).
    """
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
    # Stage2 scoring
    border_w: float = 0.05,
    trail_penalty: float = 0.02,
    cut_mode: str = "trail",
    # tabs (optional)
    tab_enable: bool = False,
    tab_per_part: int = 0,
    tab_len: float = 0.0,
    tab_corner_clear: float = 0.0,
    # shared-edge detection robustness
    line_snap_eps: float = 0.0,
    min_shared_len: float = 0.0,
    nd_coord: int = 6,
) -> Optional[Board]:
    """
    Re-pack a fixed set of parts (do NOT change board membership),
    aiming to increase shared edges. Must place all parts, otherwise discard.
    """
    rng = random.Random(seed)

    base = sorted(parts_on_board, key=lambda p: (max(p.w, p.h), p.w * p.h), reverse=True)

    best_board: Optional[Board] = None
    best_key = None

    for t in range(max(1, restarts)):
        arr = base[:]
        if t > 0:
            rng.shuffle(arr)

        b, remaining = pack_one_board(arr, W, H, allow_rot, bid=bid, order="none", place_mode="blf_ccf")
        if remaining:
            continue

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

        # Scalar score = shared + w_border*border - w_trail*n_trail
        # (shared stays dominant; the penalty mainly breaks over-shared but
        #  highly fragmented layouts that would cause many lifts later.)
        score = float(shared) + float(border_w) * float(border) - float(trail_penalty) * float(n_stroke)

        # Tie-breakers: prefer larger shared, then border, then fewer strokes.
        key = (-score, -shared, -border, n_stroke)
        if best_board is None or key < best_key:
            best_board, best_key = b, key

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
    # Stage2 scoring
    border_w: float = 0.05,
    trail_penalty: float = 0.02,
    cut_mode: str = "trail",
    # tabs (optional)
    tab_enable: bool = False,
    tab_per_part: int = 0,
    tab_len: float = 0.0,
    tab_corner_clear: float = 0.0,
    # shared-edge detection robustness
    line_snap_eps: float = 0.0,
    min_shared_len: float = 0.0,
    nd_coord: int = 6,
) -> List[Board]:
    """
    ✅ Proposed(shared) Route-2 Two-stage (paper-friendly):

    Stage1: baselineB packing to get minimal N_board solution.
    Stage2: for EACH board, keep its part set fixed, re-pack internally using BLF+CCF
            to increase shared edges, without changing N_board.
    """
    # Stage 1
    boards_stage1 = pack_baselineB(parts, W, H, allow_rot, restarts=restarts, seed=seed)

    # Stage 2 (fixed-membership repack)
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
        )

        improved.append(b2 if b2 is not None else b)

    return improved
