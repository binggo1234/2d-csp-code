from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Set, Optional, Iterable
import math
import random
import heapq

Point = Tuple[float, float]
MIN_EDGE_LEN = 0.05  # mm; drop degenerate micro-edges to keep topology stable


# ---------------- geometry helpers ----------------

def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _round_pt(p: Point, nd: int = 6) -> Point:
    return (round(p[0], nd), round(p[1], nd))


def _unit(vx: float, vy: float, eps: float = 1e-12) -> Tuple[float, float]:
    n = math.hypot(vx, vy)
    if n <= eps:
        return (0.0, 0.0)
    return (vx / n, vy / n)


def _dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _is_collinear(p1: Point, p2: Point, p3: Point, eps: float = 1e-6) -> bool:
    ax, ay = p2[0] - p1[0], p2[1] - p1[1]
    bx, by = p3[0] - p2[0], p3[1] - p2[1]
    la = math.hypot(ax, ay)
    lb = math.hypot(bx, by)
    if la <= eps or lb <= eps:
        return True
    cross = ax * by - ay * bx
    return abs(cross) <= eps * la * lb


def _angle_score(prev: Point, cur: Point, nxt: Point) -> float:
    ux, uy = _unit(cur[0] - prev[0], cur[1] - prev[1])
    vx, vy = _unit(nxt[0] - cur[0], nxt[1] - cur[1])
    return _dot((ux, uy), (vx, vy))


# ---------------- core data structures ----------------

@dataclass(frozen=True)
class Segment:
    a: Point
    b: Point
    shared: bool

    def length(self) -> float:
        return _dist(self.a, self.b)


@dataclass
class Edge:
    eid: int
    u: Point
    v: Point
    geom: List[Point]
    shared: bool

    def length(self) -> float:
        L = 0.0
        for i in range(len(self.geom) - 1):
            L += _dist(self.geom[i], self.geom[i + 1])
        return L


class Graph:
    """
    Undirected multigraph with deterministic adjacency ordering.
    """

    def __init__(self) -> None:
        self.edges: Dict[int, Edge] = {}
        self.adj: Dict[Point, List[int]] = {}

    def add_edge(self, u: Point, v: Point, geom: List[Point], shared: bool) -> int:
        eid = len(self.edges)
        e = Edge(eid=eid, u=u, v=v, geom=geom, shared=shared)
        self.edges[eid] = e
        self.adj.setdefault(u, []).append(eid)
        self.adj.setdefault(v, []).append(eid)
        return eid

    def other(self, eid: int, node: Point) -> Point:
        e = self.edges[eid]
        return e.v if node == e.u else e.u

    def degree(self, node: Point) -> int:
        return len(self.adj.get(node, []))

    def nodes(self) -> List[Point]:
        return list(self.adj.keys())

    def sort_adjacency(self) -> None:
        for u, eids in self.adj.items():
            def key(eid: int):
                e = self.edges[eid]
                v = e.v if u == e.u else e.u
                ang = math.atan2(v[1] - u[1], v[0] - u[0])
                return (ang, e.length(), v[0], v[1], eid)
            eids.sort(key=key)


# ---------------- build graph from segments ----------------

def build_graph_from_segments(segs: List[Segment], *, nd: int = 6) -> Graph:
    """
    Build raw graph where each segment is an edge (polyline with two points).
    """
    def _coerce(seg) -> Segment:
        if isinstance(seg, Segment):
            return seg
        # Allow tuples/lists: (a,b,shared) or (a,b)
        if isinstance(seg, (tuple, list)):
            if len(seg) == 3:
                a, b, sh = seg
                return Segment(a, b, bool(sh))
            if len(seg) == 2:
                a, b = seg
                return Segment(a, b, False)
        raise TypeError(f"Unsupported segment type: {type(seg)}")

    g = Graph()
    for s_raw in segs:
        s = _coerce(s_raw)
        u = _round_pt(s.a, nd)
        v = _round_pt(s.b, nd)
        if _dist(u, v) <= MIN_EDGE_LEN:
            continue
        g.add_edge(u, v, [u, v], shared=s.shared)
    g.sort_adjacency()
    return g


# ---------------- graph simplification ----------------

def simplify_graph_collinear(g: Graph, *, col_eps: float = 1e-6) -> Graph:
    """
    Contract nodes with degree=2 and collinear neighbors into longer edges.
    """
    keep: Set[Point] = set()
    for u in g.nodes():
        d = g.degree(u)
        if d != 2:
            keep.add(u)
            continue
        e1, e2 = g.adj[u][0], g.adj[u][1]
        v1 = g.other(e1, u)
        v2 = g.other(e2, u)
        if not _is_collinear(v1, u, v2, eps=col_eps):
            keep.add(u)

    used_e: Set[int] = set()
    ng = Graph()

    for u in g.nodes():
        if u not in keep:
            continue
        for eid in g.adj.get(u, []):
            if eid in used_e:
                continue

            chain_geom: List[Point] = [u]
            shared_any = g.edges[eid].shared
            used_e.add(eid)
            cur = g.other(eid, u)
            prev = u

            egeom = g.edges[eid].geom
            if egeom[0] == u and egeom[-1] == cur:
                chain_geom.extend(egeom[1:])
            elif egeom[0] == cur and egeom[-1] == u:
                chain_geom.extend(list(reversed(egeom))[1:])
            else:
                chain_geom.append(cur)

            while cur not in keep:
                eids = g.adj.get(cur, [])
                if len(eids) != 2:
                    break
                cand = eids[0] if g.other(eids[0], cur) != prev else eids[1]
                if cand in used_e:
                    break

                used_e.add(cand)
                shared_any = shared_any or g.edges[cand].shared
                nxt = g.other(cand, cur)

                egeom2 = g.edges[cand].geom
                if egeom2[0] == cur and egeom2[-1] == nxt:
                    chain_geom.extend(egeom2[1:])
                elif egeom2[0] == nxt and egeom2[-1] == cur:
                    chain_geom.extend(list(reversed(egeom2))[1:])
                else:
                    chain_geom.append(nxt)

                prev, cur = cur, nxt

            v = cur
            if v != u and _dist(u, v) > 1e-9:
                ng.add_edge(u, v, chain_geom, shared=shared_any)

    ng.sort_adjacency()
    return ng


# ---------------- trail decomposition ----------------

def _is_true_corner(node: Point, g: Graph, eps: float = 1e-6) -> bool:
    if g.degree(node) != 2:
        return False
    e1, e2 = g.adj[node][0], g.adj[node][1]
    v1 = g.other(e1, node)
    v2 = g.other(e2, node)
    return not _is_collinear(v1, node, v2, eps=eps)


def _start_priority(node: Point, g: Graph) -> int:
    """
    CNC 起刀偏好：
    1) 端点 (deg=1)
    2) 真角点 (deg=2 且非共线)
    3) 其它（含 T 结点/odd 节点）
    """
    d = g.degree(node)
    if d == 1:
        return 1000
    if _is_true_corner(node, g):
        return 900
    if d % 2 == 1:
        return 200
    return 100


def decompose_to_trails(
    g: Graph,
    *,
    straight_cos: float = 0.995,
    nd: int = 6,
    random_tie: float = 0.0,
    rng: Optional[random.Random] = None,
) -> List[List[Point]]:
    if rng is None:
        rng = random.Random(0)

    used: Set[int] = set()
    trails: List[List[Point]] = []

    # components over nodes
    seen: Set[Point] = set()
    comps: List[Set[Point]] = []
    for s in g.nodes():
        if s in seen:
            continue
        st = [s]
        seen.add(s)
        comp = {s}
        while st:
            u = st.pop()
            for eid in g.adj.get(u, []):
                v = g.other(eid, u)
                if v not in seen:
                    seen.add(v)
                    st.append(v)
                    comp.add(v)
        comps.append(comp)

    for comp in comps:
        nodes_sorted = sorted(list(comp), key=lambda p: (-_start_priority(p, g), p[0], p[1]))

        while True:
            start_u = None
            start_eid = None
            for u in nodes_sorted:
                for eid in g.adj.get(u, []):
                    if eid not in used:
                        start_u = u
                        start_eid = eid
                        break
                if start_u is not None:
                    break
            if start_u is None:
                break

            trail_pts: List[Point] = [start_u]
            used.add(start_eid)
            cur = g.other(start_eid, start_u)
            prev = start_u

            e = g.edges[start_eid]
            geom = e.geom
            if geom[0] == start_u and geom[-1] == cur:
                trail_pts.extend(geom[1:])
            else:
                trail_pts.extend(list(reversed(geom))[1:])

            prev_dir = _unit(cur[0] - prev[0], cur[1] - prev[1])

            for _ in range(200000):
                cand_eids = [eid for eid in g.adj.get(cur, []) if eid not in used]
                if not cand_eids:
                    break

                scored: List[Tuple[float, float, int]] = []
                for eid in cand_eids:
                    nxt = g.other(eid, cur)
                    dvec = _unit(nxt[0] - cur[0], nxt[1] - cur[1])
                    dp = _dot(prev_dir, dvec)
                    scored.append((dp, g.edges[eid].length(), eid))

                scored.sort(key=lambda t: (-t[0], -t[1], t[2]))

                # 强制直行：若存在足够直的候选，仅在这些里选
                straight = [t for t in scored if t[0] >= straight_cos]
                if straight:
                    best_eid = straight[0][2]
                else:
                    if random_tie > 0.0 and len(scored) >= 2:
                        top = scored[0][0]
                        k = 1
                        for i in range(1, len(scored)):
                            if abs(scored[i][0] - top) <= 1e-6:
                                k += 1
                            else:
                                break
                        if k > 1 and rng.random() < random_tie:
                            _, _, best_eid = rng.choice(scored[:k])
                        else:
                            best_eid = scored[0][2]
                    else:
                        best_eid = scored[0][2]

                used.add(best_eid)
                nxt = g.other(best_eid, cur)

                e2 = g.edges[best_eid]
                geom2 = e2.geom
                if geom2[0] == cur and geom2[-1] == nxt:
                    trail_pts.extend(geom2[1:])
                else:
                    trail_pts.extend(list(reversed(geom2))[1:])

                prev, cur = cur, nxt
                prev_dir = _unit(cur[0] - prev[0], cur[1] - prev[1])

            trails.append([_round_pt(p, nd) for p in trail_pts])

    return trails


# ---------------- global ordering + flip DP ----------------

def _trail_endpoints(tr: List[Point]) -> Tuple[Point, Point]:
    return tr[0], tr[-1]


def nn_order_by_endpoints(trails: List[List[Point]], origin: Point = (0.0, 0.0)) -> List[int]:
    n = len(trails)
    left = set(range(n))
    cur = origin
    order: List[int] = []
    while left:
        best_i = None
        best_d = 1e100
        for i in left:
            a, b = _trail_endpoints(trails[i])
            d = min(_dist(cur, a), _dist(cur, b))
            if d < best_d:
                best_d = d
                best_i = i
        left.remove(best_i)
        order.append(best_i)
        a, b = _trail_endpoints(trails[best_i])
        cur = a if _dist(cur, a) <= _dist(cur, b) else b
    return order


def two_opt_order(order: List[int], trails: List[List[Point]], origin: Point = (0.0, 0.0), iters: int = 200) -> List[int]:
    if len(order) < 4:
        return order[:]

    mids: Dict[int, Point] = {}
    for i in order:
        a, b = _trail_endpoints(trails[i])
        mids[i] = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

    def tour_len(ord_idx: List[int]) -> float:
        if not ord_idx:
            return 0.0
        L = _dist(origin, mids[ord_idx[0]])
        for k in range(len(ord_idx) - 1):
            L += _dist(mids[ord_idx[k]], mids[ord_idx[k + 1]])
        return L + _dist(mids[ord_idx[-1]], origin)

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
    return best


def optimize_flips_dp(order: List[int], trails: List[List[Point]], origin: Point = (0.0, 0.0)) -> Tuple[List[bool], float]:
    n = len(order)
    if n == 0:
        return [], 0.0

    ends = []
    for idx in order:
        a, b = _trail_endpoints(trails[idx])
        ends.append((a, b))

    INF = 1e100
    dp0 = [INF] * n
    dp1 = [INF] * n
    prev0 = [-1] * n
    prev1 = [-1] * n

    a0, b0 = ends[0]
    dp0[0] = _dist(origin, a0)
    dp1[0] = _dist(origin, b0)

    for i in range(1, n):
        ai, bi = ends[i]
        a_prev, b_prev = ends[i - 1]

        cand00 = dp0[i - 1] + _dist(b_prev, ai)
        cand10 = dp1[i - 1] + _dist(a_prev, ai)
        if cand00 <= cand10:
            dp0[i] = cand00
            prev0[i] = 0
        else:
            dp0[i] = cand10
            prev0[i] = 1

        cand01 = dp0[i - 1] + _dist(b_prev, bi)
        cand11 = dp1[i - 1] + _dist(a_prev, bi)
        if cand01 <= cand11:
            dp1[i] = cand01
            prev1[i] = 0
        else:
            dp1[i] = cand11
            prev1[i] = 1

    end_state = 0 if dp0[-1] <= dp1[-1] else 1
    L_air = dp0[-1] if end_state == 0 else dp1[-1]

    flips = [False] * n
    s = end_state
    for i in reversed(range(n)):
        flips[i] = (s == 1)
        if i == 0:
            break
        s = prev0[i] if s == 0 else prev1[i]
    return flips, L_air


def apply_order_and_flips(order: List[int], flips: List[bool], trails: List[List[Point]]) -> List[List[Point]]:
    out: List[List[Point]] = []
    for k, idx in enumerate(order):
        tr = trails[idx]
        out.append(list(reversed(tr)) if flips[k] else tr)
    return out


# ---------------- high-level toolpath ----------------

def compute_toolpath_trails(
    segs: List[Segment],
    *,
    origin: Point = (0.0, 0.0),
    nd: int = 6,
    col_eps: float = 1e-6,
    n_restarts: int = 1,
    random_tie: float = 0.0,
    seed: int = 0,
) -> Tuple[List[List[Point]], Dict[str, float]]:
    best = None
    best_metrics = None
    best_oriented: List[List[Point]] = []

    base_g = build_graph_from_segments(segs, nd=nd)
    simp_g = simplify_graph_collinear(base_g, col_eps=col_eps)

    for r in range(max(1, n_restarts)):
        rng = random.Random(seed + r)
        trails = decompose_to_trails(simp_g, nd=nd, random_tie=random_tie, rng=rng)

        order = nn_order_by_endpoints(trails, origin=origin)
        order = two_opt_order(order, trails, origin=origin, iters=120)
        flips, L_air = optimize_flips_dp(order, trails, origin=origin)
        oriented = apply_order_and_flips(order, flips, trails)

        N_pierce = float(len(oriented))
        L_cut = 0.0
        for tr in oriented:
            for i in range(len(tr) - 1):
                L_cut += _dist(tr[i], tr[i + 1])

        metrics = {
            "L_cut": float(L_cut),
            "L_air": float(L_air),
            "N_pierce": float(N_pierce),
            "N_trails": float(len(oriented)),
        }

        score = (metrics["L_air"], metrics["N_pierce"], metrics["L_cut"])
        if best is None or score < best:
            best = score
            best_metrics = metrics
            best_oriented = oriented

    return best_oriented, best_metrics if best_metrics is not None else {"L_cut": 0.0, "L_air": 0.0, "N_pierce": 0.0, "N_trails": 0.0}


def diagnose_pierce_points(oriented_trails: List[List[Point]], g: Graph) -> Dict[str, int]:
    """
    Optional diagnostic helper: count pierce points by node degree.
    Returns a dict with counts for endpoint/deg2/junction/total.
    """
    pierce = [tr[0] for tr in oriented_trails if tr]
    out = {"pierce": len(pierce), "endpoint": 0, "deg2": 0, "junction": 0}
    for p in pierce:
        d = g.degree(p)
        if d == 1:
            out["endpoint"] += 1
        elif d == 2:
            out["deg2"] += 1
        elif d >= 3:
            out["junction"] += 1
    return out


# ---------------- segment generation (from board) ----------------

def _coord_tol_default(nd: int = 6) -> float:
    return 1e-4


def _snap_coord(v: float, pool: List[float], eps: float) -> float:
    for u in pool:
        if abs(v - u) <= eps:
            return u
    pool.append(v)
    return v


def _sweep_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float, int]]:
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


def _apply_tabs_to_segments(segs: List[Segment], tabs: Dict[Tuple[str, float], List[Tuple[float, float]]], nd: int = 6, eps: float = 1e-9) -> List[Segment]:
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
    return [ss for ss in out if _dist(ss.a, ss.b) > 1e-6]


def _gen_tabs_for_board(board, tab_per_part: int, tab_len: float, corner_clear: float, nd: int = 6) -> Tuple[Dict[Tuple[str, float], List[Tuple[float, float]]], int]:
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
        edges = [("h", y0, x0, x1, r.w), ("h", y1, x0, x1, r.w), ("v", x0, y0, y1, r.h), ("v", x1, y0, y1, r.h)]
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
            tabs, n_tabs = _gen_tabs_for_board(board, tab_per_part, tab_len, tab_corner_clear, nd=nd)
            segs = _apply_tabs_to_segments(segs, tabs, nd=nd)
        else:
            n_tabs = 0
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
        tabs, n_tabs = _gen_tabs_for_board(board, tab_per_part, tab_len, tab_corner_clear, nd=nd)
        segs = _apply_tabs_to_segments(segs, tabs, nd=nd)
    else:
        n_tabs = 0
    return segs, L_shared, n_tabs


# ---------------- basic graph helpers for metrics ----------------

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


# ---------------- classic CPP helpers (for compatibility) ----------------

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
    k = len(odds)
    if k <= 1 or k % 2 == 1:
        return 0.0
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
    deg = {u: len(g.get(u, [])) for u in comp}
    odds = [u for u, d in deg.items() if d % 2 == 1]
    if len(odds) <= 1:
        return 0.0
    dist_map = {u: _dijkstra(g, u) for u in odds}
    k = len(odds)
    if k <= 20:
        return _mwpm_extra_exact(odds, dist_map)
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
        if best_v is None:
            break
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


# ---------------- public APIs for strokes ----------------

def _is_point(obj) -> bool:
    try:
        return isinstance(obj, (tuple, list)) and len(obj) == 2 and isinstance(obj[0], (int, float)) and isinstance(obj[1], (int, float))
    except Exception:
        return False


def air_length_by_strokes(items, start: Point = (0.0, 0.0)) -> float:
    """
    Compatibility helper:
    - If given a list of Segments (or segment-like tuples), compute toolpath trails and return L_air.
    - If given a list of points (stroke representatives), do a simple NN+2opt tour to estimate air.
    """
    lst = list(items) if items is not None else []
    if not lst:
        return 0.0

    # Case 1: list of points
    if all(_is_point(x) for x in lst):
        order, L = nn_tour(lst, start=start)
        order, L = two_opt(order, lst, start=start, iters=100)
        return float(L)

    # Case 2: assume segments
    try:
        trails, metrics = compute_toolpath_trails(lst, origin=start)
        return metrics.get("L_air", 0.0)
    except Exception:
        # fallback to 0 on malformed inputs
        return 0.0


def estimate_cut_and_strokes(
    segs: List[Segment],
    cut_mode: str = "trail",
    return_trails: bool = False,
    *,
    origin: Point = (0.0, 0.0),
    nd: int = 6,
) -> Tuple:
    g, base = build_graph(segs)
    if not g:
        if return_trails:
            return 0.0, 0, 0, [], [], []
        else:
            return 0.0, 0, 0, [], []

    comps = connected_components(g)
    if cut_mode == "trail":
        L_cut_base = base
    elif cut_mode == "cpp":
        L_cut_base = base + sum(approx_cpp_extra(g, c) for c in comps)
    else:
        raise ValueError(f"unknown cut_mode: {cut_mode}")

    oriented_trails, metrics = compute_toolpath_trails(segs, origin=origin, nd=nd)
    L_cut = metrics.get("L_cut", L_cut_base)
    n_strokes = len(oriented_trails)
    stroke_reps = [tr[0] for tr in oriented_trails]

    if return_trails:
        return L_cut, len(comps), n_strokes, stroke_reps, oriented_trails, comps
    else:
        return L_cut, len(comps), n_strokes, stroke_reps, comps


# ---------------- legacy helpers (NN + 2-opt on points) ----------------

def _euclid(a: Point, b: Point) -> float:
    return _dist(a, b)


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
    return order, L


def two_opt(order: List[int], points: List[Point], start: Point = (0.0, 0.0), iters: int = 200) -> Tuple[List[int], float]:
    if len(order) < 4:
        _, L = nn_tour([points[i] for i in order], start=start)
        return order, L

    def tour_len(ord_idx):
        if not ord_idx:
            return 0.0
        L = _euclid(start, points[ord_idx[0]])
        for k in range(len(ord_idx) - 1):
            L += _euclid(points[ord_idx[k]], points[ord_idx[k + 1]])
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


# ---------------- validation helpers ----------------

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
