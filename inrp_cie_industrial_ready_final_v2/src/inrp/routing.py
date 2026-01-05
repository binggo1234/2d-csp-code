# src/inrp/routing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from functools import lru_cache
import math
import heapq
import random

Point = Tuple[float, float]


@dataclass(frozen=True)
class Segment:
    a: Point
    b: Point
    shared: bool  # internal shared edge?

    def length(self) -> float:
        return math.hypot(self.a[0] - self.b[0], self.a[1] - self.b[1])


def _round_pt(p: Point, nd: int = 6) -> Point:
    return (round(p[0], nd), round(p[1], nd))

def _snap_coord(v: float, pool: List[float], eps: float) -> float:
    """Snap coordinate to an existing value within eps (deterministic: first match)."""
    for u in pool:
        if abs(v - u) <= eps:
            return u
    pool.append(v)
    return v


def _coord_tol_default(nd: int = 6) -> float:
    # Conservative default tolerance relative to rounding digits (mm units).
    return 1e-4


def _sweep_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float, int]]:
    """Split intervals into (s,e,count) pieces by multiplicity sweep."""
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

# ---------------- Tab (bridges) ----------------


def _merge_intervals_1d(ints: List[Tuple[float, float]], eps: float = 1e-9) -> List[Tuple[float, float]]:
    if not ints:
        return []
    ints2 = sorted(((min(a, b), max(a, b)) for a, b in ints if max(a, b) - min(a, b) > eps), key=lambda t: t[0])
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
    """Remove (not-cut) tab intervals from segments by splitting them."""
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
            cuts = _merge_intervals_1d([(max(y0, a), min(y1, b)) for a, b in cuts if min(y1, b) - max(y0, a) > eps])
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
            cuts = _merge_intervals_1d([(max(x0, a), min(x1, b)) for a, b in cuts if min(x1, b) - max(x0, a) > eps])
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

    # filter very tiny
    out2 = [ss for ss in out if ss.length() > 1e-6]
    return out2


def _gen_tabs_for_board(
        board,
        tab_per_part: int,
        tab_len: float,
        corner_clear: float,
        nd: int = 6,
) -> Tuple[Dict[Tuple[str, float], List[Tuple[float, float]]], int]:
    """Generate tabs on each part perimeter (axis-aligned rectangles)."""
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
        w = x1 - x0

        edges = [
            ("h", y0, x0, x1, w),  # bottom
            ("h", y1, x0, x1, w),  # top
            ("v", x0, y0, y1, r.h),  # left
            ("v", x1, y0, y1, r.h),  # right
        ]
        # prefer longer edges
        edges_sorted = sorted(edges, key=lambda t: t[4], reverse=True)

        alloc = [0, 0, 0, 0]
        for k in range(tab_per_part):
            alloc[k % 4] += 1

        for idx_edge, (orient, line, s0, s1, L) in enumerate(edges_sorted):
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

# ---------------- Segment building (none / union) ----------------


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
    """Build cut segments from a board."""
    segs: List[Segment] = []
    L_shared = 0.0

    if line_snap_eps <= 0.0:
        line_snap_eps = _coord_tol_default(nd)
    if min_shared_len < 0.0:
        min_shared_len = 0.0

    if share_mode == "none":
        for pp in board.placed:
            r = pp.rect
            p1 = (r.x, r.y)
            p2 = (r.x + r.w, r.y)
            p3 = (r.x + r.w, r.y + r.h)
            p4 = (r.x, r.y + r.h)
            segs.extend([
                Segment(_round_pt(p1, nd), _round_pt(p2, nd), False),
                Segment(_round_pt(p2, nd), _round_pt(p3, nd), False),
                Segment(_round_pt(p3, nd), _round_pt(p4, nd), False),
                Segment(_round_pt(p4, nd), _round_pt(p1, nd), False),
            ])

        n_tabs = 0
        if tab_enable:
            tabs, n_tabs = _gen_tabs_for_board(board, tab_per_part, tab_len, tab_corner_clear)
            segs = _apply_tabs_to_segments(segs, tabs, nd=nd)
        return segs, 0.0, n_tabs

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

    # vertical segments
    for x, ints in vertical.items():
        pieces = _sweep_intervals(ints)
        for s0, e0, cnt in pieces:
            L = e0 - s0
            if L <= 1e-12:
                continue
            shared = (cnt >= 2) and (L >= float(min_shared_len))
            a = _round_pt((x, s0), nd)
            b = _round_pt((x, e0), nd)
            segs.append(Segment(a, b, shared))
            if shared:
                L_shared += L

    # horizontal segments
    for y, ints in horizontal.items():
        pieces = _sweep_intervals(ints)
        for s0, e0, cnt in pieces:
            L = e0 - s0
            if L <= 1e-12:
                continue
            shared = (cnt >= 2) and (L >= float(min_shared_len))
            a = _round_pt((s0, y), nd)
            b = _round_pt((e0, y), nd)
            segs.append(Segment(a, b, shared))
            if shared:
                L_shared += L

    n_tabs = 0
    if tab_enable:
        tabs, n_tabs = _gen_tabs_for_board(board, tab_per_part, tab_len, tab_corner_clear)
        segs = _apply_tabs_to_segments(segs, tabs, nd=nd)

    return segs, L_shared, n_tabs


# ---------------- Graph / components / cut length ----------------


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
            nd = d + w
            if nd < dist.get(v, 1e100):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def _mwpm_extra_exact(odds: List[Point], dist_map: Dict[Point, Dict[Point, float]]) -> float:
    """Exact Minimum-Weight Perfect Matching (MWPM) on odd vertices.
    Complexity: O(k^2 * 2^k). Recommended for k<=20.
    """
    k = len(odds)
    if k <= 1 or k % 2 == 1:
        return 0.0

    D = [[0.0] * k for _ in range(k)]
    for i in range(k):
        di = dist_map.get(odds[i], {})
        for j in range(k):
            if i == j: continue
            D[i][j] = float(di.get(odds[j], 1e100))

    FULL = (1 << k) - 1

    @lru_cache(maxsize=None)
    def dp(mask: int) -> float:
        if mask == 0:
            return 0.0
        lsb = mask & -mask
        i = lsb.bit_length() - 1
        mask_wo_i = mask ^ lsb
        best = 1e100
        m = mask_wo_i
        while m:
            lsbj = m & -m
            j = lsbj.bit_length() - 1
            cand = D[i][j] + dp(mask_wo_i ^ lsbj)
            if cand < best:
                best = cand
            m ^= lsbj
        return best

    return float(dp(FULL))


def approx_cpp_extra(g: Dict[Point, List[Tuple[Point, float]]], comp: Set[Point]) -> float:
    """Extra length for undirected Chinese Postman (CPP)."""
    deg = {u: len(g.get(u, [])) for u in comp}
    odds = [u for u, d in deg.items() if d % 2 == 1]
    if len(odds) <= 1:
        return 0.0

    dist_map: Dict[Point, Dict[Point, float]] = {}
    for u in odds:
        dist_map[u] = _dijkstra(g, u)

    k = len(odds)
    if k <= 20:
        return _mwpm_extra_exact(odds, dist_map)

    # Fallback: greedy
    remain = set(odds)
    extra = 0.0
    while len(remain) >= 2:
        u = next(iter(remain))
        remain.remove(u)
        best_v, best_d = None, 1e100
        for v in remain:
            d = dist_map[u].get(v, 1e100)
            if d < best_d:
                best_d, best_v = d, v
        if best_v is None: break
        remain.remove(best_v)
        extra += best_d
    return extra


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


# ---------------- Air move: TSP Optimization ----------------


def _euclid(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tsp_tour_length(points: List[Point], order: List[int], start: Point = (0.0, 0.0)) -> float:
    """Compute tour length for start -> points[order...] -> start."""
    if not order:
        return 0.0
    L = _euclid(start, points[order[0]])
    for i in range(len(order) - 1):
        L += _euclid(points[order[i]], points[order[i + 1]])
    L += _euclid(points[order[-1]], start)
    return L


def nn_tour(points: List[Point], start: Point = (0.0, 0.0)) -> Tuple[List[int], float]:
    """Nearest-neighbor TSP tour."""
    if not points:
        return [], 0.0
    left = set(range(len(points)))
    cur = start
    order: List[int] = []
    L = 0.0
    while left:
        best_i = None
        best_d = 1e100
        for i in left:
            d = _euclid(cur, points[i])
            if d < best_d:
                best_d = d
                best_i = i
        left.remove(best_i)
        order.append(best_i)
        L += best_d
        cur = points[best_i]
    L += _euclid(cur, start)
    return order, L


def two_opt(
        points: List[Point],
        order: List[int],
        start: Point = (0.0, 0.0),
        max_iter: int = 2000
) -> Tuple[List[int], float]:
    """Classic 2-opt local improvement for an *open* tour used in our tool-path ordering.

    Notes
    -----
    - We model the air-move ordering as: start -> p[order[0]] -> ... -> p[order[-1]] -> start.
    - This keeps behaviour consistent with :func:`nn_tour` and :func:`sa_tsp_opt`.

    Returns
    -------
    (order2, L): improved order and its tour length.
    """
    if len(order) <= 3:
        return order, tsp_tour_length(points, order, start=start)

    best = order[:]
    best_L = tsp_tour_length(points, best, start=start)

    n = len(best)
    it = 0
    improved = True
    while improved and it < max_iter:
        improved = False
        it += 1
        # 2-opt: reverse a segment (i, j)
        for i in range(n - 2):
            for j in range(i + 2, n):
                if j - i == 1:
                    continue
                cand = best[:]
                cand[i:j] = reversed(cand[i:j])
                L = tsp_tour_length(points, cand, start=start)
                if L + 1e-9 < best_L:
                    best, best_L = cand, L
                    improved = True
        # loop until no improving move
    return best, best_L


def simulated_annealing_tour(
        points: List[Point],
        start: Point = (0.0, 0.0),
        iters: int = 5000,
        T0: float = 100.0,
        alpha: float = 0.99
) -> Tuple[List[int], float]:
    """
    Optimize TSP using Simulated Annealing.
    Starting from NN tour, tries to improve by 2-opt swaps.
    """
    n = len(points)
    if n <= 3:
        return nn_tour(points, start)

    # 1. Init with NN
    current_order, _ = nn_tour(points, start)

    def calc_len(ord_idx):
        if not ord_idx: return 0.0
        L = _euclid(start, points[ord_idx[0]])
        for k in range(len(ord_idx)-1):
            L += _euclid(points[ord_idx[k]], points[ord_idx[k+1]])
        L += _euclid(points[ord_idx[-1]], start)
        return L

    current_len = calc_len(current_order)
    best_order = current_order[:]
    best_len = current_len

    T = T0
    rng = random.Random(0) # deterministic SA

    for _ in range(iters):
        # 2. Swap (2-opt move)
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)

        # Reverse segment [i, j]
        new_order = current_order[:i] + current_order[i:j+1][::-1] + current_order[j+1:]
        new_len = calc_len(new_order)

        delta = new_len - current_len

        if delta < 0 or (rng.random() < math.exp(-delta / max(1e-9, T))):
            current_order = new_order
            current_len = new_len
            if current_len < best_len:
                best_len = current_len
                best_order = current_order[:]

        T *= alpha

    return best_order, best_len


def air_length_by_components(comps: List[Set[Point]], start: Point = (0.0, 0.0)) -> float:
    """
    Compute Air Move Length using a Greedy Endpoint Strategy.
    Simulates a realistic toolpath where the cutter flies to the nearest
    entry point of any unvisited component.
    """
    if not comps:
        return 0.0

    unvisited = [list(c) for c in comps]
    current_pos = start
    total_air = 0.0

    while unvisited:
        best_comp_idx = -1
        best_node_idx = -1
        min_dist = 1e100

        # Global Nearest Neighbor search for next entry point
        # (This effectively solves the "Continuous Cutting Problem" greedily)
        for i, nodes in enumerate(unvisited):
            for j, p in enumerate(nodes):
                d = _euclid(current_pos, p)
                if d < min_dist:
                    min_dist = d
                    best_comp_idx = i
                    best_node_idx = j

        total_air += min_dist

        nodes = unvisited[best_comp_idx]
        current_pos = nodes[best_node_idx] # Move to entry
        unvisited.pop(best_comp_idx)

    return total_air


# ---------------- Stroke decomposition ----------------


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
) -> Tuple:
    """Decompose segment graph into *strokes* (continuous cuts without lifting).

    Parameters
    ----------
    adj:
        Edge-adjacency returned by :func:`build_graph_with_edges`.
    comps:
        Connected components from :func:`connected_components`.
    return_trails:
        If True, also return the polyline point list for each stroke.

    Returns
    -------
    reps, n_strokes [, trails]
        reps: representative point per stroke (used for routing).
        n_strokes: number of strokes.
        trails: list of polyline points for each stroke (optional).
    """
    used_edges: Set[int] = set()
    reps: List[Point] = []
    trails: List[List[Point]] = []

    def _pick_rep(trail: List[Point]) -> Point:
        return trail[0] if trail else (0.0, 0.0)

    for comp in comps:
        if not comp:
            continue
        deg = {u: len(adj.get(u, [])) for u in comp}

        def follow(u, v, eid0):
            # Build a maximal trail by walking through degree-2 nodes.
            trail = [u, v]
            prev_eid = eid0
            cur = v
            while True:
                if cur not in comp or deg.get(cur, 0) != 2:
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

        # 1) Prefer starting from odd/junction nodes (deg != 2)
        for u in comp:
            if deg.get(u, 0) == 2:
                continue
            for v, _, eid in adj.get(u, []):
                if eid in used_edges:
                    continue
                used_edges.add(eid)
                tr = follow(u, v, eid)
                reps.append(_pick_rep(tr))
                if return_trails:
                    trails.append(tr)

        # 2) Cleanup remaining cycles (deg == 2)
        for u in comp:
            for v, _, eid in adj.get(u, []):
                if eid in used_edges:
                    continue
                used_edges.add(eid)
                tr = follow(u, v, eid)
                reps.append(_pick_rep(tr))
                if return_trails:
                    trails.append(tr)

    if return_trails:
        return reps, len(reps), trails
    return reps, len(reps)


def air_length_by_strokes(stroke_reps: List[Point], start: Point = (0.0, 0.0)) -> float:
    """
    Optimize air moves between strokes using Simulated Annealing TSP.
    This provides a much tighter L_air bound than greedy/2-opt.
    """
    if not stroke_reps:
        return 0.0

    # UPGRADE: Use Simulated Annealing instead of simple NN/2-opt
    # This usually finds a better sequence for the "Drill/Lift" points.
    _, L = simulated_annealing_tour(stroke_reps, start=start, iters=3000)
    return L


def estimate_cut_and_strokes(
        segs: List[Segment],
        cut_mode: str = "trail",
        *,
        return_trails: bool = False,
) -> Tuple:
    """Estimate cut length and extract strokes.

    Notes
    -----
    This function is intentionally lightweight: it does **not** solve the full
    routing problem. It only provides (1) cutting length estimate under
    different cut models and (2) a set of continuous *strokes* that can be used
    as routing nodes.

    When ``return_trails=True``, this also returns the polyline point list per
    stroke (for visualization).
    """
    g, base = build_graph(segs)
    if not g:
        if return_trails:
            return 0.0, 0, 0, [], [], []
        return 0.0, 0, 0, [], []
    comps = connected_components(g)

    if cut_mode == "trail":
        L_cut = base
    elif cut_mode == "cpp":
        L_cut = base + sum(approx_cpp_extra(g, c) for c in comps)
    else:
        raise ValueError(f"unknown cut_mode: {cut_mode}")

    adj_e = build_graph_with_edges(segs)
    if return_trails:
        reps, n_strokes, trails = stroke_representatives(adj_e, comps, return_trails=True)
        return L_cut, len(comps), n_strokes, reps, comps, trails
    reps, n_strokes = stroke_representatives(adj_e, comps, return_trails=False)
    return L_cut, len(comps), n_strokes, reps, comps


# ---------------- Validation helpers ----------------

def trails_lower_bound_for_component(g, comp):
    if not comp: return 0, 0
    odd = sum(1 for u in comp if len(g.get(u, [])) % 2 == 1)
    min_trails = max(1, odd // 2)
    return odd, min_trails

def trails_lower_bound_from_segments(segs):
    g, _ = build_graph(segs)
    if not g: return 0, 0, []
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
