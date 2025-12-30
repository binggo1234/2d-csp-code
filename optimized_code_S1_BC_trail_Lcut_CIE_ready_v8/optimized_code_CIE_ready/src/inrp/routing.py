# routing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from functools import lru_cache
import math
import heapq

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
    # For nd=6, round-off is 1e-6; we use a slightly larger tolerance by default.
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
    """Generate tabs on each part perimeter (axis-aligned rectangles).

    Return:
      tabs: dict key=("v"/"h", line_coord_rounded) -> list of intervals on the varying axis
      n_tabs: number of unique tab intervals (after de-dup)

    Notes:
      - Tabs on shared edges are allowed but de-duplicated globally.
      - We avoid corners by `corner_clear`.
      - A tab is represented as an interval of length `tab_len` centered at a chosen position.
    """
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
        h = y1 - y0

        # Decide how many tabs on each edge (round-robin on longer edges)
        edges = [
            ("h", y0, x0, x1, w),  # bottom
            ("h", y1, x0, x1, w),  # top
            ("v", x0, y0, y1, h),  # left
            ("v", x1, y0, y1, h),  # right
        ]
        # prefer longer edges
        edges_sorted = sorted(edges, key=lambda t: t[4], reverse=True)

        # allocate counts
        alloc = [0, 0, 0, 0]
        for k in range(tab_per_part):
            alloc[k % 4] += 1
        # map alloc to sorted edges
        for idx_edge, (orient, line, s0, s1, L) in enumerate(edges_sorted):
            m = alloc[idx_edge]
            if m <= 0:
                continue
            lo = s0 + corner_clear
            hi = s1 - corner_clear
            if hi - lo <= tab_len + 1e-9:
                continue
            # equally spaced centers
            for j in range(1, m + 1):
                c = lo + (hi - lo) * (j / (m + 1))
                a = c - tab_len * 0.5
                b = c + tab_len * 0.5
                a = max(a, lo)
                b = min(b, hi)
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
    """Build cut segments from a board.

    share_mode:
      - "none": no common-cut union (each part contributes its own 4 edges)
      - "union": multiplicity sweep (overlapped collinear edges merged; count>=2 marked as shared)

    Robustness knobs (paper-friendly):
      - line_snap_eps: treat coordinates within this tolerance as the same cutting line.
      - min_shared_len: ignore extremely short overlaps that are likely numerical artifacts.

    Tabs:
      - if tab_enable, remove tab intervals from segments by splitting.

    Return: (segments_after_tabs, L_shared_before_tabs, n_tabs)
    """
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

    # union: collect all edges into line buckets then sweep by multiplicity.
    # We snap line coordinates within `line_snap_eps` to avoid missing shared edges due to float noise.
    vertical: Dict[float, List[Tuple[float, float]]] = {}
    horizontal: Dict[float, List[Tuple[float, float]]] = {}
    x_pool: List[float] = []
    y_pool: List[float] = []

    for pp in board.placed:
        r = pp.rect
        x0, x1 = r.x, r.x + r.w
        y0, y1 = r.y, r.y + r.h

        x0s = _snap_coord(float(x0), x_pool, line_snap_eps)
        x1s = _snap_coord(float(x1), x_pool, line_snap_eps)
        y0s = _snap_coord(float(y0), y_pool, line_snap_eps)
        y1s = _snap_coord(float(y1), y_pool, line_snap_eps)

        vertical.setdefault(x0s, []).append((y0s, y1s))
        vertical.setdefault(x1s, []).append((y0s, y1s))
        horizontal.setdefault(y0s, []).append((x0s, x1s))
        horizontal.setdefault(y1s, []).append((x0s, x1s))

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

    For undirected Chinese Postman Problem (CPP), we need to pair all odd-degree
    vertices by minimum total shortest-path distance.

    Here we compute MWPM exactly with a bitmask dynamic program, which is
    practical because the number of odd vertices per connected component is
    usually small in panel-cutting graphs.

    Complexity: O(k^2 * 2^k) where k=len(odds). We recommend k<=20.
    """
    k = len(odds)
    if k <= 1:
        return 0.0
    if k % 2 == 1:
        # Should not happen for undirected graphs, but keep safe.
        return 0.0

    # Distance matrix between odd vertices (shortest-path distances on g)
    D = [[0.0] * k for _ in range(k)]
    for i in range(k):
        di = dist_map.get(odds[i], {})
        for j in range(k):
            if i == j:
                continue
            D[i][j] = float(di.get(odds[j], 1e100))

    FULL = (1 << k) - 1

    @lru_cache(maxsize=None)
    def dp(mask: int) -> float:
        if mask == 0:
            return 0.0

        # pick the first set bit i
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
    """Extra length for undirected Chinese Postman (CPP) on one connected component.

    Standard CPP requires adding duplicate edges so that all vertices have even degree.
    This reduces to pairing odd-degree vertices by *Minimum-Weight Perfect Matching* (MWPM)
    on the metric closure (shortest-path distances) of odd vertices.

    Implementation details:
    - Compute all-pairs shortest paths from each odd vertex via Dijkstra.
    - Solve MWPM exactly by bitmask DP when the number of odd vertices is small (k<=20).
    - Fall back to greedy matching if k is large (rare in our panel-cutting graphs).
    """
    deg = {u: len(g.get(u, [])) for u in comp}
    odds = [u for u, d in deg.items() if d % 2 == 1]
    if len(odds) <= 1:
        return 0.0

    dist_map: Dict[Point, Dict[Point, float]] = {}
    for u in odds:
        dist_map[u] = _dijkstra(g, u)

    # Exact MWPM for small odd sets (paper-grade CPP).
    k = len(odds)
    if k <= 20:
        return _mwpm_extra_exact(odds, dist_map)

    # Fallback: greedy matching (kept as a safeguard).
    remain = set(odds)
    extra = 0.0
    while len(remain) >= 2:
        u = next(iter(remain))
        remain.remove(u)
        best_v = None
        best_d = 1e100
        for v in remain:
            d = dist_map[u].get(v, 1e100)
            if d < best_d:
                best_d = d
                best_v = v
        if best_v is None or best_d >= 1e90:
            break
        remain.remove(best_v)
        extra += best_d
    return extra


def estimate_cut_lengths(segs: List[Segment], cut_mode: str = "cpp") -> Tuple[float, int, List[Set[Point]]]:
    """Return (L_cut_est, n_components, components_nodes)."""
    g, base = build_graph(segs)
    if not g:
        return 0.0, 0, []
    comps = connected_components(g)
    n_comp = len(comps)

    if cut_mode == "trail":
        return base, n_comp, comps

    if cut_mode != "cpp":
        raise ValueError(f"unknown cut_mode: {cut_mode}")

    extra = 0.0
    for comp in comps:
        extra += approx_cpp_extra(g, comp)
    return base + extra, n_comp, comps


# ---------------- Air move: component-level NN + 2-opt ----------------


def _euclid(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def component_representatives(comps: List[Set[Point]]) -> List[Point]:
    """Pick a representative point for each component.

    Strategy: centroid of vertices -> nearest vertex (stable, geometry-respecting).
    """
    reps: List[Point] = []
    for comp in comps:
        pts = list(comp)
        if not pts:
            continue
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        c = (cx, cy)
        best = min(pts, key=lambda p: _euclid(p, c))
        reps.append(best)
    return reps


def nn_tour(points: List[Point], start: Point = (0.0, 0.0)) -> Tuple[List[int], float]:
    """Nearest-neighbor TSP tour visiting all points and returning to start."""
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


def two_opt(order: List[int], points: List[Point], start: Point = (0.0, 0.0), iters: int = 200) -> Tuple[List[int], float]:
    """Simple 2-opt improvement on a cyclic tour (includes return to origin)."""
    if len(order) < 4:
        _, L = nn_tour([points[i] for i in order], start=start)
        return order, L

    def tour_len(ord_idx: List[int]) -> float:
        if not ord_idx:
            return 0.0
        L = _euclid(start, points[ord_idx[0]])
        for a, b in zip(ord_idx, ord_idx[1:]):
            L += _euclid(points[a], points[b])
        L += _euclid(points[ord_idx[-1]], start)
        return L

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
    return best, bestL


def air_length_by_components(comps: List[Set[Point]], start: Point = (0.0, 0.0)) -> float:
    """Air-move length estimated by a component-level TSP (NN + 2-opt)."""
    reps = component_representatives(comps)
    if not reps:
        return 0.0
    order, _ = nn_tour(reps, start=start)
    _, L2 = two_opt(order, reps, start=start, iters=200)
    return L2


# ---------------- Stroke decomposition (manufacturing-friendly lifts) ----------------


def build_graph_with_edges(segs: List[Segment]) -> Dict[Point, List[Tuple[Point, float, int]]]:
    """Adjacency list with edge ids for stroke decomposition."""
    adj: Dict[Point, List[Tuple[Point, float, int]]] = {}
    for eid, s in enumerate(segs):
        a, b = s.a, s.b
        w = s.length()
        adj.setdefault(a, []).append((b, w, eid))
        adj.setdefault(b, []).append((a, w, eid))
    return adj


def stroke_representatives(
    adj: Dict[Point, List[Tuple[Point, float, int]]],
    comps: List[Set[Point]],
) -> Tuple[List[Point], int]:
    """Decompose each connected component into edge-disjoint trails (strokes).

    Stroke rule (paper-friendly, closer to CNC practice):
      - Break at nodes with degree != 2 (branching / endpoints / junctions).
      - Each resulting trail is a "stroke" that can be cut without lifting.
      - Pure cycles (all degree==2) count as one stroke.

    Return:
      reps: representative points (one per stroke, used as entry points for TSP)
      n_strokes: number of strokes
    """
    used_edges: Set[int] = set()
    reps: List[Point] = []

    def _pick_rep(trail: List[Point]) -> Point:
        if not trail:
            return (0.0, 0.0)
        if len(trail) == 1:
            return trail[0]
        # centroid of vertices -> pick closer endpoint (stable)
        cx = sum(p[0] for p in trail) / len(trail)
        cy = sum(p[1] for p in trail) / len(trail)
        c = (cx, cy)
        a = trail[0]
        b = trail[-1]
        return a if _euclid(a, c) <= _euclid(b, c) else b

    for comp in comps:
        if not comp:
            continue
        deg = {u: len(adj.get(u, [])) for u in comp}

        def follow(u: Point, v: Point, eid0: int) -> List[Point]:
            """Follow a trail starting with edge (u-v), stopping when reaching a break node or dead end."""
            trail = [u, v]
            prev_eid = eid0
            cur = v
            while True:
                if cur not in comp:
                    break
                if deg.get(cur, 0) != 2:
                    break
                nxt = None
                for w, _, eid in adj.get(cur, []):
                    if eid == prev_eid or eid in used_edges:
                        continue
                    nxt = (w, eid)
                    break
                if nxt is None:
                    break
                w, eid = nxt
                used_edges.add(eid)
                trail.append(w)
                prev_eid = eid
                cur = w
                # if closed cycle returns to start, stop
                if cur == u:
                    break
            return trail

        # 1) trails starting from break nodes
        for u in comp:
            if deg.get(u, 0) == 2:
                continue
            for v, _, eid in adj.get(u, []):
                if eid in used_edges:
                    continue
                used_edges.add(eid)
                trail = follow(u, v, eid)
                reps.append(_pick_rep(trail))

        # 2) leftover cycles (all deg==2 regions)
        for u in comp:
            for v, _, eid in adj.get(u, []):
                if eid in used_edges:
                    continue
                used_edges.add(eid)
                trail = follow(u, v, eid)
                reps.append(_pick_rep(trail))

    return reps, len(reps)


def air_length_by_strokes(stroke_reps: List[Point], start: Point = (0.0, 0.0)) -> float:
    """Air-move length estimated by a stroke-level TSP (NN + 2-opt)."""
    if not stroke_reps:
        return 0.0
    order, _ = nn_tour(stroke_reps, start=start)
    _, L2 = two_opt(order, stroke_reps, start=start, iters=200)
    return L2


def estimate_cut_and_strokes(
    segs: List[Segment],
    cut_mode: str = "trail",
) -> Tuple[float, int, int, List[Point], List[Set[Point]]]:
    """Convenience wrapper for metrics:

    Returns:
      L_cut, n_comp, n_strokes, stroke_reps, comps

    Notes:
      - L_cut uses `cut_mode` (trail or cpp).
      - Strokes are computed on the underlying segment graph (after tabs and union).
    """
    # graph for components (weights-only) is already handled by estimate_cut_lengths
    g, base = build_graph(segs)
    if not g:
        return 0.0, 0, 0, [], []
    comps = connected_components(g)

    if cut_mode == "trail":
        L_cut = base
    elif cut_mode == "cpp":
        L_cut = base + sum(approx_cpp_extra(g, comp) for comp in comps)
    else:
        raise ValueError(f"unknown cut_mode: {cut_mode}")

    adj_e = build_graph_with_edges(segs)
    reps, n_strokes = stroke_representatives(adj_e, comps)
    return L_cut, len(comps), n_strokes, reps, comps


# ---------------- Validation helpers (paper-friendly) ----------------

def trails_lower_bound_for_component(g, comp):
    """Lower bound on number of edge-disjoint trails to cover all edges in an undirected component.

    For a connected undirected graph, the minimum number of trails that cover all edges
    equals max(1, #odd/2). This is a classic result.

    Returns: (odd_count, min_trails)
    """
    if not comp:
        return 0, 0
    odd = 0
    for u in comp:
        deg = len(g.get(u, []))
        if deg % 2 == 1:
            odd += 1
    min_trails = max(1, odd // 2)  # odd is always even in an undirected graph
    return odd, min_trails


def trails_lower_bound_from_segments(segs):
    """Compute odd-degree statistics and lower-bound trails from segments.

    Returns:
      odd_total: sum of odd-degree nodes over all components
      min_trails_total: sum of max(1, odd/2) over all components
      per_comp: list of dicts
    """
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
