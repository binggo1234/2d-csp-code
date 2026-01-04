# src/inrp/routing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Union
from functools import lru_cache
import math
import heapq
import random

Point = Tuple[float, float]


# =========================
# Data structures
# =========================

@dataclass(frozen=True)
class Segment:
    a: Point
    b: Point
    shared: bool

    def length(self) -> float:
        return math.hypot(self.a[0] - self.b[0], self.a[1] - self.b[1])


# =========================
# Geometry utils
# =========================

def _round_pt(p: Point, nd: int = 6) -> Point:
    return (round(p[0], nd), round(p[1], nd))


def _snap_coord(v: float, pool: List[float], eps: float) -> float:
    for u in pool:
        if abs(v - u) <= eps:
            return u
    pool.append(v)
    return v


def _coord_tol_default(nd: int = 6) -> float:
    # small but not too small; intended to "snap" nearly-equal coords after float ops
    return 1e-4


def _sweep_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float, int]]:
    """
    Given intervals [(s,e),...], return disjoint pieces with coverage count.
    Output: [(y0,y1,cnt), ...] where cnt>0.
    """
    events: Dict[float, int] = {}
    for s, e in intervals:
        if e <= s:
            continue
        events[s] = events.get(s, 0) + 1
        events[e] = events.get(e, 0) - 1
    ys = sorted(events.keys())
    out: List[Tuple[float, float, int]] = []
    cnt = 0
    for i in range(len(ys) - 1):
        y = ys[i]
        cnt += events[y]
        y2 = ys[i + 1]
        if cnt > 0 and y2 > y:
            out.append((y, y2, cnt))
    return out


# =========================
# Tabs (bridges)
# =========================

def _merge_intervals_1d(ints: List[Tuple[float, float]], eps: float = 1e-9) -> List[Tuple[float, float]]:
    if not ints:
        return []
    ints2 = sorted(
        ((min(a, b), max(a, b)) for a, b in ints if max(a, b) - min(a, b) > eps),
        key=lambda t: t[0],
    )
    if not ints2:
        return []
    out = [ints2[0]]
    for s, e in ints2[1:]:
        ps, pe = out[-1]
        if s <= pe + eps:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def _apply_tabs_to_segments(
        segs: List[Segment],
        tabs: Dict[Tuple[str, float], List[Tuple[float, float]]],
        nd: int = 6,
        eps: float = 1e-9,
) -> List[Segment]:
    if not segs or not tabs:
        return segs
    out: List[Segment] = []
    for s in segs:
        ax, ay = s.a
        bx, by = s.b
        if abs(ax - bx) <= 1e-9:  # vertical
            x = round(ax, nd)
            y0, y1 = (ay, by) if ay <= by else (by, ay)
            cuts = tabs.get(("v", x), [])
            cuts = _merge_intervals_1d(
                [(max(y0, a), min(y1, b)) for a, b in cuts if min(y1, b) - max(y0, a) > eps]
            )
            if not cuts:
                out.append(s)
                continue
            cur = y0
            for a, b in cuts:
                if a > cur + eps:
                    out.append(Segment(_round_pt((x, cur), nd), _round_pt((x, a), nd), s.shared))
                cur = max(cur, b)
            if y1 > cur + eps:
                out.append(Segment(_round_pt((x, cur), nd), _round_pt((x, y1), nd), s.shared))
        else:  # horizontal
            y = round(ay, nd)
            x0, x1 = (ax, bx) if ax <= bx else (bx, ax)
            cuts = tabs.get(("h", y), [])
            cuts = _merge_intervals_1d(
                [(max(x0, a), min(x1, b)) for a, b in cuts if min(x1, b) - max(x0, a) > eps]
            )
            if not cuts:
                out.append(s)
                continue
            cur = x0
            for a, b in cuts:
                if a > cur + eps:
                    out.append(Segment(_round_pt((cur, y), nd), _round_pt((a, y), nd), s.shared))
                cur = max(cur, b)
            if x1 > cur + eps:
                out.append(Segment(_round_pt((cur, y), nd), _round_pt((x1, y), nd), s.shared))
    return [ss for ss in out if ss.length() > 1e-6]


def _gen_tabs_for_board(
        board,
        tab_per_part: int,
        tab_len: float,
        corner_clear: float,
        nd: int = 6,
) -> Tuple[Dict[Tuple[str, float], List[Tuple[float, float]]], int]:
    if tab_per_part <= 0 or tab_len <= 0:
        return {}, 0
    tabs: Dict[Tuple[str, float], List[Tuple[float, float]]] = {}
    seen = set()

    def _add_tab(orient: str, line: float, s: float, e: float) -> None:
        if e <= s:
            return
        line_r = round(line, nd)
        c = round((s + e) * 0.5, nd)
        key = (orient, line_r, c)
        if key in seen:
            return
        seen.add(key)
        tabs.setdefault((orient, line_r), []).append((s, e))

    for pp in board.placed:
        r = pp.rect
        x0, x1 = r.x, r.x + r.w
        y0, y1 = r.y, r.y + r.h
        edges = [
            ("h", y0, x0, x1, r.w),
            ("h", y1, x0, x1, r.w),
            ("v", x0, y0, y1, r.h),
            ("v", x1, y0, y1, r.h),
        ]
        edges_sorted = sorted(edges, key=lambda t: t[4], reverse=True)
        alloc = [0, 0, 0, 0]
        for k in range(tab_per_part):
            alloc[k % 4] += 1
        for idx_edge, (orient, line, s0, s1, _) in enumerate(edges_sorted):
            m = alloc[idx_edge]
            if m <= 0:
                continue
            lo = s0 + corner_clear
            hi = s1 - corner_clear
            if hi - lo <= tab_len + 1e-9:
                continue
            for j in range(1, m + 1):
                c = lo + (hi - lo) * (j / (m + 1))
                a = max(lo, c - tab_len * 0.5)
                b = min(hi, c + tab_len * 0.5)
                if b - a > 1e-6:
                    _add_tab(orient, line, a, b)
    return tabs, len(seen)


# =========================
# Segment building
# =========================

def build_segments_from_board(
        board,
        share_mode: str,
        tab_enable: bool = False,
        tab_per_part: int = 0,
        tab_len: float = 0.0,
        tab_corner_clear: float = 0.0,
        *,
        line_snap_eps: float = 0.0,
        min_shared_len: float = 0.0,
        nd: int = 6,
) -> Tuple[List[Segment], float, int]:
    segs: List[Segment] = []
    L_shared = 0.0
    if line_snap_eps <= 0.0:
        line_snap_eps = _coord_tol_default(nd)
    if min_shared_len < 0.0:
        min_shared_len = 0.0

    if share_mode == "none":
        for pp in board.placed:
            r = pp.rect
            pts = [(r.x, r.y), (r.x + r.w, r.y), (r.x + r.w, r.y + r.h), (r.x, r.y + r.h)]
            for i in range(4):
                segs.append(Segment(_round_pt(pts[i], nd), _round_pt(pts[(i + 1) % 4], nd), False))
        if tab_enable:
            tabs, n_tabs = _gen_tabs_for_board(board, tab_per_part, tab_len, tab_corner_clear)
            segs = _apply_tabs_to_segments(segs, tabs, nd=nd)
        return segs, 0.0, 0 if not tab_enable else n_tabs

    if share_mode != "union":
        raise ValueError(f"unknown share_mode: {share_mode}")

    vertical: Dict[float, List[Tuple[float, float]]] = {}
    horizontal: Dict[float, List[Tuple[float, float]]] = {}
    x_pool: List[float] = []
    y_pool: List[float] = []

    for pp in board.placed:
        r = pp.rect
        x0 = _snap_coord(float(r.x), x_pool, line_snap_eps)
        x1 = _snap_coord(float(r.x + r.w), x_pool, line_snap_eps)
        y0 = _snap_coord(float(r.y), y_pool, line_snap_eps)
        y1 = _snap_coord(float(r.y + r.h), y_pool, line_snap_eps)
        vertical.setdefault(x0, []).append((y0, y1))
        vertical.setdefault(x1, []).append((y0, y1))
        horizontal.setdefault(y0, []).append((x0, x1))
        horizontal.setdefault(y1, []).append((x0, x1))

    for x, ints in vertical.items():
        pieces = _sweep_intervals(ints)
        for s0, e0, cnt in pieces:
            L = e0 - s0
            if L <= 1e-12:
                continue
            shared = (cnt >= 2) and (L >= float(min_shared_len))
            segs.append(Segment(_round_pt((x, s0), nd), _round_pt((x, e0), nd), shared))
            if shared:
                L_shared += L

    for y, ints in horizontal.items():
        pieces = _sweep_intervals(ints)
        for s0, e0, cnt in pieces:
            L = e0 - s0
            if L <= 1e-12:
                continue
            shared = (cnt >= 2) and (L >= float(min_shared_len))
            segs.append(Segment(_round_pt((s0, y), nd), _round_pt((e0, y), nd), shared))
            if shared:
                L_shared += L

    if tab_enable:
        tabs, n_tabs = _gen_tabs_for_board(board, tab_per_part, tab_len, tab_corner_clear)
        segs = _apply_tabs_to_segments(segs, tabs, nd=nd)
    else:
        n_tabs = 0
    return segs, L_shared, n_tabs


# =========================
# Graph
# =========================

def build_graph(segs: List[Segment]) -> Tuple[Dict[Point, List[Tuple[Point, float]]], float]:
    g: Dict[Point, List[Tuple[Point, float]]] = {}
    total = 0.0
    for s in segs:
        a, b = s.a, s.b
        w = s.length()
        total += w
        g.setdefault(a, []).append((b, w))
        g.setdefault(b, []).append((a, w))
    return g, total


def connected_components(g: Dict[Point, List[Tuple[Point, float]]]) -> List[Set[Point]]:
    seen: Set[Point] = set()
    comps: List[Set[Point]] = []
    for v in g.keys():
        if v in seen:
            continue
        stack = [v]
        seen.add(v)
        comp = {v}
        while stack:
            x = stack.pop()
            for y, _ in g.get(x, []):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
                    comp.add(y)
        comps.append(comp)
    return comps


def _dijkstra(g: Dict[Point, List[Tuple[Point, float]]], src: Point) -> Dict[Point, float]:
    dist: Dict[Point, float] = {src: 0.0}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist.get(u, None):
            continue
        for v, w in g.get(u, []):
            ndv = d + w
            if ndv < dist.get(v, 1e100):
                dist[v] = ndv
                heapq.heappush(pq, (ndv, v))
    return dist


# =========================
# CPP extra length (matching cost)
# =========================

def _mwpm_pairs_exact(
        odds: List[Point],
        dist_map: Dict[Point, Dict[Point, float]],
) -> Tuple[List[Tuple[Point, Point]], float]:
    """
    Exact MWPM (DP) returning explicit pairs and cost.
    Complexity O(k^2 2^k). Use for k<=20.
    """
    k = len(odds)
    if k <= 1 or k % 2 == 1:
        return [], 0.0

    D = [[0.0] * k for _ in range(k)]
    for i in range(k):
        di = dist_map.get(odds[i], {})
        for j in range(k):
            if i == j:
                continue
            D[i][j] = float(di.get(odds[j], 1e100))

    FULL = (1 << k) - 1

    @lru_cache(maxsize=None)
    def dp(mask: int) -> Tuple[float, Tuple[Tuple[int, int], ...]]:
        if mask == 0:
            return 0.0, ()
        lsb = mask & -mask
        i = lsb.bit_length() - 1
        mask_wo_i = mask ^ lsb

        best_cost = 1e100
        best_pairs: Tuple[Tuple[int, int], ...] = ()
        m = mask_wo_i
        while m:
            lsbj = m & -m
            j = lsbj.bit_length() - 1
            rest_cost, rest_pairs = dp(mask_wo_i ^ lsbj)
            cand = D[i][j] + rest_cost
            if cand < best_cost:
                best_cost = cand
                best_pairs = ((i, j),) + rest_pairs
            m ^= lsbj
        return best_cost, best_pairs

    cost, pairs_idx = dp(FULL)
    pairs = [(odds[i], odds[j]) for (i, j) in pairs_idx]
    return pairs, float(cost)


def _mwpm_pairs_greedy(
        odds: List[Point],
        dist_map: Dict[Point, Dict[Point, float]],
) -> Tuple[List[Tuple[Point, Point]], float]:
    remain = set(odds)
    pairs: List[Tuple[Point, Point]] = []
    cost = 0.0
    while len(remain) >= 2:
        u = next(iter(remain))
        remain.remove(u)
        best_v = None
        best_d = 1e100
        du = dist_map.get(u, {})
        for v in remain:
            d = float(du.get(v, 1e100))
            if d < best_d:
                best_d, best_v = d, v
        if best_v is None:
            break
        remain.remove(best_v)
        pairs.append((u, best_v))
        cost += best_d
    return pairs, float(cost)


def _cpp_pairs_and_cost(
        g: Dict[Point, List[Tuple[Point, float]]],
        comp: Set[Point],
) -> Tuple[List[Tuple[Point, Point]], float, Dict[Point, Dict[Point, float]]]:
    deg = {u: len(g.get(u, [])) for u in comp}
    odds = [u for u, d in deg.items() if d % 2 == 1]
    if len(odds) <= 1:
        return [], 0.0, {}

    dist_map = {u: _dijkstra(g, u) for u in odds}
    if len(odds) <= 20:
        pairs, cost = _mwpm_pairs_exact(odds, dist_map)
    else:
        pairs, cost = _mwpm_pairs_greedy(odds, dist_map)
    return pairs, float(cost), dist_map


def approx_cpp_extra(g: Dict[Point, List[Tuple[Point, float]]], comp: Set[Point]) -> float:
    # kept for backward compatibility; now uses the same pairing logic we use for trail construction
    _, cost, _ = _cpp_pairs_and_cost(g, comp)
    return float(cost)


def estimate_cut_lengths(segs: List[Segment], cut_mode: str = "cpp") -> Tuple[float, int, List[Set[Point]]]:
    g, base = build_graph(segs)
    if not g:
        return 0.0, 0, []
    comps = connected_components(g)
    n_comp = len(comps)
    if cut_mode == "trail":
        return base, n_comp, comps
    elif cut_mode == "cpp":
        extra = sum(approx_cpp_extra(g, c) for c in comps)
        return base + extra, n_comp, comps
    else:
        raise ValueError(f"unknown cut_mode: {cut_mode}")


# =========================
# TSP (legacy point-based)
# =========================

def _euclid(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nn_tour(points: List[Point], start: Point = (0.0, 0.0)) -> Tuple[List[int], float]:
    if not points:
        return [], 0.0
    left = set(range(len(points)))
    cur = start
    order: List[int] = []
    L = 0.0
    while left:
        best_i, best_d = None, 1e100
        for i in left:
            d = _euclid(cur, points[i])
            if d < best_d:
                best_d, best_i = d, i
        left.remove(best_i)
        order.append(best_i)
        L += best_d
        cur = points[best_i]
    L += _euclid(cur, start)
    return order, float(L)


def two_opt(order: List[int], points: List[Point], start: Point = (0.0, 0.0), iters: int = 200) -> Tuple[List[int], float]:
    """
    FIXED: for small n, compute length of the GIVEN order (not a re-run nn_tour on subset).
    """
    def tour_len(ord_idx: List[int]) -> float:
        if not ord_idx:
            return 0.0
        L = _euclid(start, points[ord_idx[0]])
        for k in range(len(ord_idx) - 1):
            L += _euclid(points[ord_idx[k]], points[ord_idx[k + 1]])
        L += _euclid(points[ord_idx[-1]], start)
        return float(L)

    if len(order) < 4:
        return order[:], tour_len(order)

    best = order[:]
    bestL = tour_len(best)
    n = len(best)
    for _ in range(iters):
        improved = False
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                cand = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                Lc = tour_len(cand)
                if Lc + 1e-9 < bestL:
                    best, bestL = cand, Lc
                    improved = True
        if not improved:
            break
    return best, float(bestL)


def simulated_annealing_tour(
        points: List[Point],
        start: Point = (0.0, 0.0),
        iters: int = 3000,
        T0: float = 100.0,
        alpha: float = 0.99,
) -> Tuple[List[int], float]:
    # simplified (kept): nn + 2opt
    order, L = nn_tour(points, start)
    order, L = two_opt(order, points, start, iters=iters // 10)
    return order, float(L)


def air_length_by_strokes(stroke_reps: List[Point], start: Point = (0.0, 0.0)) -> float:
    _, L = simulated_annealing_tour(stroke_reps, start=start)
    return float(L)


# =========================
# Endpoint-aware air moves (paper-grade alignment)
# =========================

def _trail_endpoints(trail: List[Point]) -> Tuple[Point, Point]:
    p0 = (float(trail[0][0]), float(trail[0][1]))
    p1 = (float(trail[-1][0]), float(trail[-1][1]))
    return p0, p1


def _trail_start_end(p0: Point, p1: Point, dir_flag: int) -> Tuple[Point, Point]:
    # dir_flag: 0 => p0->p1, 1 => p1->p0
    return (p0, p1) if int(dir_flag) == 0 else (p1, p0)


def endpoint_nn_order(
        trails: List[List[Point]],
        start: Point = (0.0, 0.0),
        return_to_start: bool = False,
) -> Tuple[List[int], List[int], float]:
    """
    Greedy NN on strokes with endpoint choice.
    Returns (order, dirs, L_air).
    dirs[i]=0 means enter at trail[0], dirs[i]=1 means enter at trail[-1] (reverse).
    """
    n = len(trails)
    if n == 0:
        return [], [], 0.0

    left = set(range(n))
    cur = (float(start[0]), float(start[1]))
    order: List[int] = []
    dirs: List[int] = []
    L = 0.0

    while left:
        best_i = None
        best_dir = 0
        best_d = 1e100

        for i in left:
            p0, p1 = _trail_endpoints(trails[i])
            d0 = _euclid(cur, p0)
            d1 = _euclid(cur, p1)
            if d0 <= d1:
                d, dir_i = d0, 0
            else:
                d, dir_i = d1, 1

            if d < best_d:
                best_d, best_i, best_dir = d, i, dir_i

        left.remove(best_i)
        order.append(best_i)
        dirs.append(best_dir)
        L += best_d

        p0, p1 = _trail_endpoints(trails[best_i])
        _, ept = _trail_start_end(p0, p1, best_dir)
        cur = ept

    if return_to_start and order:
        L += _euclid(cur, (float(start[0]), float(start[1])))

    return order, dirs, float(L)


def endpoint_dp_orient_for_order(
        order: List[int],
        trails: List[List[Point]],
        start: Point = (0.0, 0.0),
        return_to_start: bool = False,
) -> Tuple[List[int], float]:
    """
    For a fixed order, choose best directions via DP (2 states per stroke).
    Returns (dirs_aligned_with_order, L_air).
    """
    n = len(order)
    if n == 0:
        return [], 0.0

    start_pt = (float(start[0]), float(start[1]))

    P0: List[Point] = []
    P1: List[Point] = []
    for idx in order:
        p0, p1 = _trail_endpoints(trails[idx])
        P0.append(p0)
        P1.append(p1)

    INF = 1e100
    dp0 = [_euclid(start_pt, P0[0]), _euclid(start_pt, P1[0])]
    prev_choice = [[-1, -1] for _ in range(n)]

    for i in range(1, n):
        new_dp = [INF, INF]
        for s in (0, 1):
            cur_start = P0[i] if s == 0 else P1[i]
            best_val = INF
            best_prev = -1
            for ps in (0, 1):
                prev_end = P1[i - 1] if ps == 0 else P0[i - 1]
                cand = dp0[ps] + _euclid(prev_end, cur_start)
                if cand < best_val:
                    best_val, best_prev = cand, ps
            new_dp[s] = best_val
            prev_choice[i][s] = best_prev
        dp0 = new_dp

    end_cost = dp0[:]
    if return_to_start:
        end_cost[0] += _euclid(P1[-1], start_pt)
        end_cost[1] += _euclid(P0[-1], start_pt)

    last_state = 0 if end_cost[0] <= end_cost[1] else 1
    bestL = float(end_cost[last_state])

    dirs = [0] * n
    dirs[-1] = last_state
    for i in range(n - 1, 0, -1):
        dirs[i - 1] = prev_choice[i][dirs[i]]

    return dirs, bestL


def endpoint_two_opt(
        order: List[int],
        trails: List[List[Point]],
        start: Point = (0.0, 0.0),
        iters: int = 120,
        return_to_start: bool = False,
) -> Tuple[List[int], List[int], float]:
    """
    Random-sampled 2-opt over order; each candidate evaluated with DP-optimal directions.
    Returns (best_order, best_dirs_aligned, best_L_air).
    """
    if len(order) < 3:
        dirs, L = endpoint_dp_orient_for_order(order, trails, start=start, return_to_start=return_to_start)
        return order[:], dirs, float(L)

    best_order = order[:]
    best_dirs, bestL = endpoint_dp_orient_for_order(best_order, trails, start=start, return_to_start=return_to_start)

    n = len(best_order)
    rng = random.Random(0)

    for _ in range(iters):
        i = rng.randint(0, n - 3)
        k = rng.randint(i + 1, n - 2)
        cand = best_order[:i] + best_order[i:k + 1][::-1] + best_order[k + 1:]

        cand_dirs, candL = endpoint_dp_orient_for_order(cand, trails, start=start, return_to_start=return_to_start)
        if candL + 1e-9 < bestL:
            best_order, best_dirs, bestL = cand, cand_dirs, candL

    return best_order, best_dirs, float(bestL)


def air_moves_from_order_and_dirs(
        order: List[int],
        dirs: List[int],
        trails: List[List[Point]],
        start: Point = (0.0, 0.0),
        return_to_start: bool = False,
) -> Tuple[List[Tuple[Point, Point]], float]:
    """
    Build actual air segments consistent with (order, dirs). Returns (air_segments, L_air).
    """
    if not order:
        return [], 0.0

    cur = (float(start[0]), float(start[1]))
    air: List[Tuple[Point, Point]] = []
    L = 0.0

    for idx, dflag in zip(order, dirs):
        p0, p1 = _trail_endpoints(trails[idx])
        spt, ept = _trail_start_end(p0, p1, dflag)
        L += _euclid(cur, spt)
        air.append((cur, spt))
        cur = ept

    if return_to_start:
        st = (float(start[0]), float(start[1]))
        L += _euclid(cur, st)
        air.append((cur, st))

    return air, float(L)


# =========================
# Stroke decomposition (legacy: local follow)
# =========================

def build_graph_with_edges(segs: List[Segment]) -> Dict[Point, List[Tuple[Point, float, int]]]:
    adj: Dict[Point, List[Tuple[Point, float, int]]] = {}
    for eid, s in enumerate(segs):
        w = s.length()
        adj.setdefault(s.a, []).append((s.b, w, eid))
        adj.setdefault(s.b, []).append((s.a, w, eid))
    return adj


def stroke_representatives(
        adj: Dict[Point, List[Tuple[Point, float, int]]],
        comps: List[Set[Point]],
        return_trails: bool = False,
) -> Union[Tuple[List[Point], int], Tuple[List[Point], List[List[Point]], int]]:
    """
    Legacy heuristic decomposition:
    - follow paths through degree-2 chains, break at deg!=2
    NOTE: kept for compatibility; paper-grade uses Eulerization below.
    """
    used_edges: Set[int] = set()
    reps: List[Point] = []
    trails: List[List[Point]] = []

    def follow(u, v, eid0):
        trail = [u, v]
        prev_eid = eid0
        cur = v
        while True:
            if len(adj.get(cur, [])) != 2:
                break
            nxt = None
            for w, _, eid in adj.get(cur, []):
                if eid != prev_eid and eid not in used_edges:
                    nxt = (w, eid)
                    break
            if not nxt:
                break
            w, eid = nxt
            used_edges.add(eid)
            trail.append(w)
            prev_eid = eid
            cur = w
            if cur == u:
                break
        return trail

    for comp in comps:
        if not comp:
            continue
        deg = {u: len(adj.get(u, [])) for u in comp}
        for u in comp:
            if deg.get(u, 0) == 2:
                continue
            for v, _, eid in adj.get(u, []):
                if eid in used_edges:
                    continue
                used_edges.add(eid)
                trail = follow(u, v, eid)
                reps.append(trail[0])
                trails.append(trail)

        for u in comp:
            for v, _, eid in adj.get(u, []):
                if eid in used_edges:
                    continue
                used_edges.add(eid)
                trail = follow(u, v, eid)
                reps.append(trail[0])
                trails.append(trail)

    if return_trails:
        return reps, trails, len(reps)
    else:
        return reps, len(reps)


# =========================
# Paper-grade stroke decomposition (Eulerization)
# =========================

def _dijkstra_prev(
        g: Dict[Point, List[Tuple[Point, float]]],
        src: Point,
) -> Tuple[Dict[Point, float], Dict[Point, Point]]:
    """
    Dijkstra returning (dist, prev) for path reconstruction.
    prev[v] = u indicates u->v is on shortest path tree.
    """
    dist: Dict[Point, float] = {src: 0.0}
    prev: Dict[Point, Point] = {}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist.get(u, None):
            continue
        for v, w in g.get(u, []):
            ndv = d + w
            if ndv < dist.get(v, 1e100):
                dist[v] = ndv
                prev[v] = u
                heapq.heappush(pq, (ndv, v))
    return dist, prev


def _reconstruct_path(prev: Dict[Point, Point], src: Point, dst: Point) -> List[Point]:
    if src == dst:
        return [src]
    if dst not in prev:
        return []
    cur = dst
    path = [cur]
    while cur != src:
        cur = prev.get(cur)
        if cur is None:
            return []
        path.append(cur)
    path.reverse()
    return path


def _hierholzer_undirected(
        adj: Dict[Point, List[Tuple[Point, int]]],
        n_edges: int,
        start: Point,
) -> List[int]:
    """
    Hierholzer for undirected multigraph.
    adj is mutated (popping used edges).
    Returns traversal as a list of edge instance IDs.
    """
    used = [False] * n_edges
    stack = [start]
    edge_stack: List[int] = []
    tour_edges: List[int] = []

    while stack:
        v = stack[-1]
        # skip already-used edges at the end
        while adj.get(v) and used[adj[v][-1][1]]:
            adj[v].pop()

        if not adj.get(v):
            stack.pop()
            if edge_stack:
                tour_edges.append(edge_stack.pop())
            continue

        w, eid = adj[v].pop()
        if used[eid]:
            continue
        used[eid] = True
        stack.append(w)
        edge_stack.append(eid)

    tour_edges.reverse()
    return tour_edges


def _split_euler_edges_into_trails(
        tour_edges: List[int],
        inst_u: List[Point],
        inst_v: List[Point],
        inst_virtual: List[bool],
) -> List[List[Point]]:
    """
    Convert Euler edge sequence into trails.
    - virtual edges act as pen-up separators (only used in trail-mode).
    """
    trails: List[List[Point]] = []
    cur: List[Point] = []

    def push(p: Point):
        nonlocal cur
        if not cur:
            cur = [p]
        else:
            if cur[-1] != p:
                cur.append(p)

    for eid in tour_edges:
        a = inst_u[eid]
        b = inst_v[eid]
        if inst_virtual[eid]:
            if len(cur) >= 2:
                trails.append(cur)
            cur = []
            continue

        if not cur:
            push(a)
            push(b)
        else:
            if cur[-1] == a:
                push(b)
            elif cur[-1] == b:
                push(a)
            else:
                if len(cur) >= 2:
                    trails.append(cur)
                cur = [a, b]

    if len(cur) >= 2:
        trails.append(cur)
    return trails


def decompose_strokes_euler(
        segs: List[Segment],
        *,
        cut_mode: str = "trail",  # "trail" or "cpp"
) -> Tuple[List[List[Point]], List[Point], int, List[Set[Point]]]:
    """
    Paper-grade decomposition:
    - trail: add virtual matching edges between odd nodes to Eulerize, then split on virtual edges.
    - cpp  : duplicate real edges along shortest paths of matched odd pairs (Eulerization by edge duplication).
            Output will be one (or few) long trails; and L_cut is consistent with duplicated edges traversal.

    Returns: (stroke_trails, reps, n_strokes, comps)
    """
    g, _ = build_graph(segs)
    if not g:
        return [], [], 0, []

    comps = connected_components(g)
    all_trails: List[List[Point]] = []

    for comp in comps:
        if not comp:
            continue

        # base real edge instances for this component
        inst_u: List[Point] = []
        inst_v: List[Point] = []
        inst_virtual: List[bool] = []

        for s in segs:
            a, b = s.a, s.b
            if a in comp and b in comp:
                inst_u.append(a)
                inst_v.append(b)
                inst_virtual.append(False)

        # odds and matching
        deg = {u: len(g.get(u, [])) for u in comp}
        odds = [u for u, d in deg.items() if d % 2 == 1]

        pairs: List[Tuple[Point, Point]] = []
        if len(odds) >= 2:
            # For consistency with estimate_cut_lengths: use same pairing rule (exact<=20 else greedy)
            dist_map = {u: _dijkstra(g, u) for u in odds}
            if len(odds) <= 20:
                pairs, _ = _mwpm_pairs_exact(odds, dist_map)
            else:
                pairs, _ = _mwpm_pairs_greedy(odds, dist_map)

            if cut_mode == "trail":
                # add virtual separators directly between matched odds (not counted in L_cut)
                for (u, v) in pairs:
                    inst_u.append(u)
                    inst_v.append(v)
                    inst_virtual.append(True)
            elif cut_mode == "cpp":
                # duplicate real edges along shortest paths between matched odds (counted in L_cut)
                prev_map: Dict[Point, Dict[Point, Point]] = {}
                for u in odds:
                    _, prev_u = _dijkstra_prev(g, u)
                    prev_map[u] = prev_u
                for (u, v) in pairs:
                    path = _reconstruct_path(prev_map.get(u, {}), u, v)
                    if len(path) < 2:
                        continue
                    for i in range(len(path) - 1):
                        a = path[i]
                        b = path[i + 1]
                        inst_u.append(a)
                        inst_v.append(b)
                        inst_virtual.append(False)
            else:
                raise ValueError(f"unknown cut_mode: {cut_mode}")

        # build adjacency for Hierholzer
        adj: Dict[Point, List[Tuple[Point, int]]] = {}
        for eid, (a, b) in enumerate(zip(inst_u, inst_v)):
            adj.setdefault(a, []).append((b, eid))
            adj.setdefault(b, []).append((a, eid))

        # deterministic start
        start = min(comp, key=lambda p: (p[0], p[1]))
        tour_edges = _hierholzer_undirected(adj, len(inst_u), start=start)

        trails = _split_euler_edges_into_trails(tour_edges, inst_u, inst_v, inst_virtual)
        all_trails.extend(trails)

    reps = [t[0] for t in all_trails if t]
    return all_trails, reps, len(reps), comps


# =========================
# Cut + strokes (public API)
# =========================

def estimate_cut_and_strokes(segs: List[Segment], cut_mode: str = "trail", return_trails: bool = False) -> Tuple:
    """
    Returns:
      if return_trails:
        (L_cut, n_comp, n_strokes, stroke_reps, stroke_trails, comps)
      else:
        (L_cut, n_comp, n_strokes, stroke_reps, comps)

    Paper alignment:
      - when return_trails=True: use Eulerization trails (decompose_strokes_euler),
        so the returned trails can be used to compute/plot air moves consistently.
      - L_cut is consistent with cut_mode:
          * trail: base length (each segment once)
          * cpp  : base + matching cost (and trails include duplicated edges)
    """
    g, base = build_graph(segs)
    if not g:
        if return_trails:
            return 0.0, 0, 0, [], [], []
        else:
            return 0.0, 0, 0, [], []

    comps = connected_components(g)
    if cut_mode == "trail":
        L_cut = float(base)
    elif cut_mode == "cpp":
        L_cut = float(base + sum(approx_cpp_extra(g, c) for c in comps))
    else:
        raise ValueError(f"unknown cut_mode: {cut_mode}")

    if return_trails:
        stroke_trails, reps, n_strokes, comps2 = decompose_strokes_euler(segs, cut_mode=cut_mode)
        # comps2 should match comps in practice; return comps2 for consistency with decomposition
        return L_cut, len(comps2), n_strokes, reps, stroke_trails, comps2

    # legacy fast path (kept): representatives only
    adj_e = build_graph_with_edges(segs)
    reps, n_strokes = stroke_representatives(adj_e, comps, return_trails=False)
    return L_cut, len(comps), n_strokes, reps, comps


# =========================
# Validation helpers
# =========================

def trails_lower_bound_for_component(g, comp):
    if not comp:
        return 0, 0
    odd = sum(1 for u in comp if len(g.get(u, [])) % 2 == 1)
    min_trails = max(1, odd // 2)
    return odd, min_trails


def trails_lower_bound_from_segments(segs):
    g, _ = build_graph(segs)
    if not g:
        return 0, 0, []
    comps = connected_components(g)
    per = []
    odd_total = 0
    min_total = 0
    for i, comp in enumerate(comps):
        odd, mn = trails_lower_bound_for_component(g, comp)
        odd_total += odd
        min_total += mn
        per.append({"comp_id": i, "n_nodes": len(comp), "odd": odd, "min_trails": mn})
    return odd_total, min_total, per
