from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Tuple, Set, Optional, Iterable
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


def _trail_bbox_area(tr: List[Point]) -> float:
    if not tr:
        return 0.0
    xs = [p[0] for p in tr]
    ys = [p[1] for p in tr]
    return max(0.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))


def _trail_bbox_xmin(tr: List[Point]) -> float:
    if not tr:
        return 0.0
    return min(p[0] for p in tr)


def _trail_bbox_ymin(tr: List[Point]) -> float:
    if not tr:
        return 0.0
    return min(p[1] for p in tr)


def _trail_length(tr: List[Point]) -> float:
    L = 0.0
    for i in range(len(tr) - 1):
        L += _dist(tr[i], tr[i + 1])
    return L


def _trail_signed_area(tr: List[Point]) -> float:
    if len(tr) < 3:
        return 0.0
    pts = tr if tr[0] == tr[-1] else tr + [tr[0]]
    s = 0.0
    for i in range(len(pts) - 1):
        s += pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1]
    return 0.5 * s


def _choose_edge_start_origin(
    trails: List[List[Point]],
    fallback: Point = (0.0, 0.0),
    board_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Point:
    if not trails:
        return fallback
    if board_bounds is None:
        xs = [p[0] for tr in trails for p in tr]
        ys = [p[1] for tr in trails for p in tr]
        if not xs or not ys:
            return fallback
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
    else:
        xmin, xmax, ymin, ymax = board_bounds

    corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    best = None
    for tr in trails:
        for p in (tr[0], tr[-1]):
            d_edge = min(abs(p[0] - xmin), abs(p[0] - xmax), abs(p[1] - ymin), abs(p[1] - ymax))
            d_corner = min(_dist(p, c) for c in corners)
            key = (d_edge, d_corner, p[1], p[0])
            if best is None or key < best[0]:
                best = (key, p)
    return best[1] if best is not None else fallback


def _priority_nn_order(
    trails: List[List[Point]],
    origin: Point = (0.0, 0.0),
    route_priority: str = "none",
    entry_penalty_fn: Optional[Callable[[Point], float]] = None,
) -> List[int]:
    n = len(trails)
    left = set(range(n))
    cur = origin
    order: List[int] = []
    mode = (route_priority or "none").lower().strip()
    while left:
        best_i = None
        best_key = None
        for i in left:
            a, b = _trail_endpoints(trails[i])
            pa = float(entry_penalty_fn(a)) if entry_penalty_fn is not None else 0.0
            pb = float(entry_penalty_fn(b)) if entry_penalty_fn is not None else 0.0
            d = min(_dist(cur, a) + pa, _dist(cur, b) + pb)
            if mode == "small_first":
                key = (_trail_bbox_area(trails[i]), _trail_length(trails[i]), d, i)
            else:
                key = (d, i)
            if best_key is None or key < best_key:
                best_key = key
                best_i = i
        left.remove(best_i)
        order.append(best_i)
        a, b = _trail_endpoints(trails[best_i])
        cur = a if _dist(cur, a) <= _dist(cur, b) else b
    return order


def _best_orient_from_current(cur: Point, tr: List[Point]) -> Tuple[bool, float, Point]:
    """
    Choose trail direction (forward or reversed) by shortest current->trail entry move.
    Returns:
      flip     : whether to reverse trail
      d_air    : entry air distance
      end_point: resulting trail end point after applying chosen direction
    """
    a, b = _trail_endpoints(tr)
    d_to_a = _dist(cur, a)
    d_to_b = _dist(cur, b)
    if d_to_a <= d_to_b:
        return False, d_to_a, b
    return True, d_to_b, a


def _order_and_orient_regional_nn(
    trails: List[List[Point]],
    *,
    origin: Point = (0.0, 0.0),
    feed_air: float = 30000.0,
    t_lift: float = 0.8,
    window_size: int = 10,
    backtrack: int = 2,
    entry_penalty_fn: Optional[Callable[[Point], float]] = None,
) -> Tuple[List[List[Point]], float, List[int], List[bool]]:
    """
    Regional ordering for CNC-friendly routing:
      1) coarse left->right by trail bbox.x_min
      2) local-window nearest-next using time cost:
         cost = air_distance / V_air + t_lift
      3) each candidate trail can be entered from either direction
         (flip selected by current-point shortest entry distance)
    """
    n = len(trails)
    if n == 0:
        return [], 0.0, [], []

    sorted_idx = sorted(
        range(n),
        key=lambda i: (_trail_bbox_xmin(trails[i]), _trail_bbox_ymin(trails[i]), i),
    )
    pos_of = {idx: pos for pos, idx in enumerate(sorted_idx)}
    visited = [False] * n

    oriented: List[List[Point]] = []
    order: List[int] = []
    flips: List[bool] = []
    cur = origin
    L_air = 0.0
    ptr = 0
    win = max(1, int(window_size))
    back = max(0, int(backtrack))
    v_air_eff = _air_speed_mm_per_sec(feed_air)
    lift_eff = max(float(t_lift), 0.0)

    while len(order) < n:
        while ptr < n and visited[sorted_idx[ptr]]:
            ptr += 1

        lo = max(0, ptr - back)
        hi = min(n, ptr + win)
        cand = [sorted_idx[p] for p in range(lo, hi) if not visited[sorted_idx[p]]]
        if not cand:
            cand = [idx for idx in sorted_idx if not visited[idx]]
            if not cand:
                break

        best = None
        for idx in cand:
            tr = trails[idx]
            if not tr:
                continue
            a, b = _trail_endpoints(tr)
            pa = float(entry_penalty_fn(a)) if entry_penalty_fn is not None else 0.0
            pb = float(entry_penalty_fn(b)) if entry_penalty_fn is not None else 0.0
            d_to_a = _dist(cur, a)
            d_to_b = _dist(cur, b)
            if d_to_a + pa <= d_to_b + pb:
                flip, d_air, end_pt, entry_pen = False, d_to_a, b, pa
            else:
                flip, d_air, end_pt, entry_pen = True, d_to_b, a, pb
            move_t = ((d_air + entry_pen) / v_air_eff) + lift_eff
            # Keep the coarse x-order but optimize within local window.
            region_delta = abs(pos_of[idx] - ptr)
            key = (move_t, d_air + entry_pen, region_delta, _trail_bbox_xmin(tr), idx)
            if best is None or key < best[0]:
                best = (key, idx, flip, d_air, end_pt)

        if best is None:
            break

        _, idx, flip, d_air, end_pt = best
        visited[idx] = True
        order.append(idx)
        flips.append(bool(flip))
        L_air += float(d_air)
        tr = trails[idx]
        oriented.append(list(reversed(tr)) if flip else tr)
        cur = end_pt

    return oriented, float(L_air), order, flips


def _force_ccw_closed_trails(trails: List[List[Point]]) -> List[List[Point]]:
    out: List[List[Point]] = []
    for tr in trails:
        if tr and tr[0] == tr[-1] and _trail_signed_area(tr) < 0.0:
            out.append(list(reversed(tr)))
        else:
            out.append(tr)
    return out


def _snap_and_merge_axis_segments(
    segs: List[Segment],
    *,
    snap_eps: float = 1e-4,
    nd: int = 6,
) -> List[Segment]:
    """
    Endpoint snapping + axis-collinear interval merge.
    Applied for shared-edge union outputs to reduce tiny fragmentation that
    inflates stroke count.
    """
    if not segs:
        return []

    x_pool: List[float] = []
    y_pool: List[float] = []
    vertical: Dict[Tuple[float, bool], List[Tuple[float, float]]] = {}
    horizontal: Dict[Tuple[float, bool], List[Tuple[float, float]]] = {}
    misc: List[Segment] = []

    for s in segs:
        ax0, ay0 = s.a
        bx0, by0 = s.b
        ax = _snap_coord(float(ax0), x_pool, snap_eps)
        ay = _snap_coord(float(ay0), y_pool, snap_eps)
        bx = _snap_coord(float(bx0), x_pool, snap_eps)
        by = _snap_coord(float(by0), y_pool, snap_eps)

        if _dist((ax, ay), (bx, by)) <= max(1e-9, snap_eps * 0.1):
            continue

        if abs(ax - bx) <= snap_eps:
            x = round((ax + bx) * 0.5, nd)
            y0, y1 = (ay, by) if ay <= by else (by, ay)
            if y1 - y0 > 1e-9:
                vertical.setdefault((x, bool(s.shared)), []).append((y0, y1))
        elif abs(ay - by) <= snap_eps:
            y = round((ay + by) * 0.5, nd)
            x0, x1 = (ax, bx) if ax <= bx else (bx, ax)
            if x1 - x0 > 1e-9:
                horizontal.setdefault((y, bool(s.shared)), []).append((x0, x1))
        else:
            misc.append(Segment(_round_pt((ax, ay), nd), _round_pt((bx, by), nd), bool(s.shared)))

    out: List[Segment] = []
    for (x, sh), ints in vertical.items():
        for y0, y1 in _merge_intervals_1d(ints, eps=snap_eps):
            out.append(Segment(_round_pt((x, y0), nd), _round_pt((x, y1), nd), sh))
    for (y, sh), ints in horizontal.items():
        for x0, x1 in _merge_intervals_1d(ints, eps=snap_eps):
            out.append(Segment(_round_pt((x0, y), nd), _round_pt((x1, y), nd), sh))

    out.extend(misc)
    return [s for s in out if _dist(s.a, s.b) > 1e-9]


def _split_axis_segments_at_junctions(
    segs: List[Segment],
    *,
    snap_eps: float = 1e-4,
    nd: int = 6,
) -> List[Segment]:
    """
    Junction merge for axis-aligned segments:
    if an endpoint of a vertical segment lies on a horizontal segment interior
    (or vice versa), split the host segment at that endpoint so both strokes
    are topologically connected in the graph.
    """
    if not segs:
        return []

    x_pool: List[float] = []
    y_pool: List[float] = []
    vertical: List[Tuple[float, float, float, bool]] = []
    horizontal: List[Tuple[float, float, float, bool]] = []
    misc: List[Segment] = []

    eps_keep = max(1e-9, snap_eps * 0.1)
    for s in segs:
        ax0, ay0 = s.a
        bx0, by0 = s.b
        ax = _snap_coord(float(ax0), x_pool, snap_eps)
        ay = _snap_coord(float(ay0), y_pool, snap_eps)
        bx = _snap_coord(float(bx0), x_pool, snap_eps)
        by = _snap_coord(float(by0), y_pool, snap_eps)

        if _dist((ax, ay), (bx, by)) <= eps_keep:
            continue

        if abs(ax - bx) <= snap_eps:
            x = round((ax + bx) * 0.5, nd)
            y0, y1 = (ay, by) if ay <= by else (by, ay)
            if y1 - y0 > eps_keep:
                vertical.append((x, y0, y1, bool(s.shared)))
        elif abs(ay - by) <= snap_eps:
            y = round((ay + by) * 0.5, nd)
            x0, x1 = (ax, bx) if ax <= bx else (bx, ax)
            if x1 - x0 > eps_keep:
                horizontal.append((y, x0, x1, bool(s.shared)))
        else:
            misc.append(Segment(_round_pt((ax, ay), nd), _round_pt((bx, by), nd), bool(s.shared)))

    v_end_by_y: Dict[float, List[float]] = {}
    h_end_by_x: Dict[float, List[float]] = {}
    for x, y0, y1, _ in vertical:
        y0k = round(y0, nd)
        y1k = round(y1, nd)
        v_end_by_y.setdefault(y0k, []).append(x)
        v_end_by_y.setdefault(y1k, []).append(x)
    for y, x0, x1, _ in horizontal:
        x0k = round(x0, nd)
        x1k = round(x1, nd)
        h_end_by_x.setdefault(x0k, []).append(y)
        h_end_by_x.setdefault(x1k, []).append(y)

    out: List[Segment] = []
    for y, x0, x1, sh in horizontal:
        yk = round(y, nd)
        cuts = [x0, x1]
        for x in v_end_by_y.get(yk, []):
            if (x0 + snap_eps) < x < (x1 - snap_eps):
                cuts.append(x)
        cuts2 = sorted(set(round(v, nd) for v in cuts))
        for i in range(len(cuts2) - 1):
            a = cuts2[i]
            b = cuts2[i + 1]
            if b - a > eps_keep:
                out.append(Segment(_round_pt((a, yk), nd), _round_pt((b, yk), nd), sh))

    for x, y0, y1, sh in vertical:
        xk = round(x, nd)
        cuts = [y0, y1]
        for y in h_end_by_x.get(xk, []):
            if (y0 + snap_eps) < y < (y1 - snap_eps):
                cuts.append(y)
        cuts2 = sorted(set(round(v, nd) for v in cuts))
        for i in range(len(cuts2) - 1):
            a = cuts2[i]
            b = cuts2[i + 1]
            if b - a > eps_keep:
                out.append(Segment(_round_pt((xk, a), nd), _round_pt((xk, b), nd), sh))

    out.extend(misc)

    uniq: Dict[Tuple[Point, Point], bool] = {}
    for s in out:
        if _dist(s.a, s.b) <= eps_keep:
            continue
        a, b = s.a, s.b
        key = (a, b) if a <= b else (b, a)
        uniq[key] = bool(uniq.get(key, False) or s.shared)

    out2: List[Segment] = []
    for (a, b), sh in sorted(
        uniq.items(),
        key=lambda kv: (kv[0][0][0], kv[0][0][1], kv[0][1][0], kv[0][1][1], int(kv[1])),
    ):
        out2.append(Segment(a, b, sh))
    return out2


def _bridge_shared_axis_gaps(
    segs: List[Segment],
    *,
    bridge_eps: float = 0.0,
    snap_eps: float = 1e-4,
    nd: int = 6,
) -> List[Segment]:
    """
    Merge short collinear gaps on shared seams.
    This is mainly for union+tabs output where both neighboring pieces are
    shared and a short tab gap causes excessive stroke fragmentation.
    """
    if not segs or bridge_eps <= 0.0:
        return list(segs)

    x_pool: List[float] = []
    y_pool: List[float] = []
    vertical: Dict[float, List[Tuple[float, float, bool]]] = {}
    horizontal: Dict[float, List[Tuple[float, float, bool]]] = {}
    misc: List[Segment] = []

    eps_keep = max(1e-9, snap_eps * 0.1)
    for s in segs:
        ax0, ay0 = s.a
        bx0, by0 = s.b
        ax = _snap_coord(float(ax0), x_pool, snap_eps)
        ay = _snap_coord(float(ay0), y_pool, snap_eps)
        bx = _snap_coord(float(bx0), x_pool, snap_eps)
        by = _snap_coord(float(by0), y_pool, snap_eps)

        if _dist((ax, ay), (bx, by)) <= eps_keep:
            continue

        if abs(ax - bx) <= snap_eps:
            x = round((ax + bx) * 0.5, nd)
            y0, y1 = (ay, by) if ay <= by else (by, ay)
            if y1 - y0 > eps_keep:
                vertical.setdefault(x, []).append((y0, y1, bool(s.shared)))
        elif abs(ay - by) <= snap_eps:
            y = round((ay + by) * 0.5, nd)
            x0, x1 = (ax, bx) if ax <= bx else (bx, ax)
            if x1 - x0 > eps_keep:
                horizontal.setdefault(y, []).append((x0, x1, bool(s.shared)))
        else:
            misc.append(Segment(_round_pt((ax, ay), nd), _round_pt((bx, by), nd), bool(s.shared)))

    out: List[Segment] = []

    def _emit_axis(intervals: List[Tuple[float, float, bool]], fixed: float, is_vertical: bool) -> None:
        if not intervals:
            return
        ints = sorted(intervals, key=lambda t: (t[0], t[1], int(t[2])))
        cs, ce, csh = ints[0]
        for s0, s1, sh in ints[1:]:
            # Do not let shared color "bleed" into adjacent non-shared pieces.
            # Direct merge is allowed only when the shared flag is consistent.
            if s0 <= ce + snap_eps and sh == csh:
                ce = max(ce, s1)
                continue
            # Mixed-flag overlap should be rare (mostly numeric jitter).
            # Clip it to the current end to keep a clean boundary.
            if sh != csh and s0 < ce:
                s0 = ce
                if s1 - s0 <= eps_keep:
                    continue
            gap = s0 - ce
            if gap <= bridge_eps and csh and sh:
                # Bridge only when both adjacent pieces are shared.
                ce = max(ce, s1)
                csh = True
                continue
            if ce - cs > eps_keep:
                if is_vertical:
                    out.append(Segment(_round_pt((fixed, cs), nd), _round_pt((fixed, ce), nd), csh))
                else:
                    out.append(Segment(_round_pt((cs, fixed), nd), _round_pt((ce, fixed), nd), csh))
            cs, ce, csh = s0, s1, sh

        if ce - cs > eps_keep:
            if is_vertical:
                out.append(Segment(_round_pt((fixed, cs), nd), _round_pt((fixed, ce), nd), csh))
            else:
                out.append(Segment(_round_pt((cs, fixed), nd), _round_pt((ce, fixed), nd), csh))

    for x, ints in vertical.items():
        _emit_axis(ints, x, True)
    for y, ints in horizontal.items():
        _emit_axis(ints, y, False)

    out.extend(misc)
    return [s for s in out if _dist(s.a, s.b) > eps_keep]


def _postprocess_union_segments(
    segs: List[Segment],
    *,
    snap_eps: float = 1e-4,
    shared_gap_bridge_eps: float = 0.0,
    nd: int = 6,
) -> List[Segment]:
    """
    Union-output cleanup pipeline:
      1) endpoint snapping
      2) collinear merge
      3) shared seam gap-bridge
      4) junction merge (split at T-junction endpoints)
    """
    if not segs:
        return []
    eps_eff = max(1e-9, float(snap_eps))
    s1 = _snap_and_merge_axis_segments(segs, snap_eps=eps_eff, nd=nd)
    s2 = _bridge_shared_axis_gaps(
        s1,
        bridge_eps=max(0.0, float(shared_gap_bridge_eps)),
        snap_eps=eps_eff,
        nd=nd,
    )
    return _split_axis_segments_at_junctions(s2, snap_eps=eps_eff, nd=nd)


def _air_speed_mm_per_sec(feed_air: float) -> float:
    # feed_air is configured in mm/min; convert to mm/s for time cost.
    va_min = float(feed_air)
    if va_min <= 1e-12:
        return 1.0
    return va_min / 60.0


def _time_obj_from_air(L_air: float, n_lift: int, feed_air: float, t_lift: float) -> float:
    v_air = _air_speed_mm_per_sec(feed_air)
    return (float(L_air) / v_air) + float(n_lift) * max(float(t_lift), 0.0)


def _two_opt_order_time(
    order: List[int],
    trails: List[List[Point]],
    *,
    origin: Point = (0.0, 0.0),
    feed_air: float = 30000.0,
    t_lift: float = 0.8,
    iters: int = 120,
    entry_penalty_fn: Optional[Callable[[Point], float]] = None,
) -> List[int]:
    if len(order) < 4:
        return order[:]

    best = order[:]
    _flips0, L_air0 = optimize_flips_dp(best, trails, origin=origin, entry_penalty_fn=entry_penalty_fn)
    best_t = _time_obj_from_air(L_air0, len(best), feed_air, t_lift)
    n = len(best)

    for _ in range(max(1, int(iters))):
        improved = False
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                cand = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                _flips_c, L_air_c = optimize_flips_dp(cand, trails, origin=origin, entry_penalty_fn=entry_penalty_fn)
                t_c = _time_obj_from_air(L_air_c, len(cand), feed_air, t_lift)
                if t_c + 1e-12 < best_t:
                    best = cand
                    best_t = t_c
                    improved = True
        if not improved:
            break
    return best


def _compress_adjacent_trails(oriented: List[List[Point]], *, eps: float = 1e-6) -> List[List[Point]]:
    if not oriented:
        return []
    out: List[List[Point]] = [oriented[0][:]]
    for tr in oriented[1:]:
        if not tr:
            continue
        last_end = out[-1][-1]
        if _dist(last_end, tr[0]) <= eps:
            out[-1].extend(tr[1:])
            continue
        if _dist(last_end, tr[-1]) <= eps:
            tr2 = list(reversed(tr))
            out[-1].extend(tr2[1:])
            continue
        out.append(tr[:])
    return out


def _merge_collinear_edges(tr: List[Point], *, eps: float = 1e-9, col_eps: float = 1e-6) -> List[Point]:
    """
    Merge consecutive collinear edges inside one trail polyline.
    This reduces tiny vertex fragmentation (stop/go jitter points) without
    changing the geometric path.
    """
    if len(tr) <= 2:
        return tr[:]

    out: List[Point] = [tr[0]]
    for p in tr[1:]:
        if _dist(out[-1], p) <= eps:
            continue
        out.append(p)
        while len(out) >= 3 and _is_collinear(out[-3], out[-2], out[-1], eps=col_eps):
            out.pop(-2)
    return out


def _air_len_by_oriented(oriented: List[List[Point]], *, origin: Point = (0.0, 0.0)) -> float:
    cur = origin
    L = 0.0
    for tr in oriented:
        if not tr:
            continue
        L += _dist(cur, tr[0])
        cur = tr[-1]
    return float(L)


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


def optimize_flips_dp(
    order: List[int],
    trails: List[List[Point]],
    origin: Point = (0.0, 0.0),
    entry_penalty_fn: Optional[Callable[[Point], float]] = None,
) -> Tuple[List[bool], float]:
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
    p0 = float(entry_penalty_fn(a0)) if entry_penalty_fn is not None else 0.0
    p1 = float(entry_penalty_fn(b0)) if entry_penalty_fn is not None else 0.0
    dp0[0] = _dist(origin, a0) + p0
    dp1[0] = _dist(origin, b0) + p1

    for i in range(1, n):
        ai, bi = ends[i]
        a_prev, b_prev = ends[i - 1]
        pai = float(entry_penalty_fn(ai)) if entry_penalty_fn is not None else 0.0
        pbi = float(entry_penalty_fn(bi)) if entry_penalty_fn is not None else 0.0

        cand00 = dp0[i - 1] + _dist(b_prev, ai) + pai
        cand10 = dp1[i - 1] + _dist(a_prev, ai) + pai
        if cand00 <= cand10:
            dp0[i] = cand00
            prev0[i] = 0
        else:
            dp0[i] = cand10
            prev0[i] = 1

        cand01 = dp0[i - 1] + _dist(b_prev, bi) + pbi
        cand11 = dp1[i - 1] + _dist(a_prev, bi) + pbi
        if cand01 <= cand11:
            dp1[i] = cand01
            prev1[i] = 0
        else:
            dp1[i] = cand11
            prev1[i] = 1

    end_state = 0 if dp0[-1] <= dp1[-1] else 1

    flips = [False] * n
    s = end_state
    for i in reversed(range(n)):
        flips[i] = (s == 1)
        if i == 0:
            break
        s = prev0[i] if s == 0 else prev1[i]

    # Return physical air length (without entry penalties).
    cur = origin
    L_air = 0.0
    for i in range(n):
        a, b = ends[i]
        if flips[i]:
            start, end = b, a
        else:
            start, end = a, b
        L_air += _dist(cur, start)
        cur = end
    return flips, float(L_air)


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
    route_start_policy: str = "none",
    route_priority: str = "none",
    route_ccw: bool = False,
    board_bounds: Optional[Tuple[float, float, float, float]] = None,
    feed_air: float = 30000.0,
    t_lift: float = 0.8,
    route_local_window: int = 10,
    route_local_backtrack: int = 2,
    # --- CNC process-rule layer (optional) ---
    part_boxes: Optional[List[Dict[str, float]]] = None,
    route_hierarchical: bool = False,
    route_large_frac: float = 0.20,
    route_small_first_area_mm2: float = 0.0,
    route_entry_junction_penalty_mm: float = 0.0,
    ramp_enable: bool = False,
    ramp_len: float = 0.0,
    anti_shift_enable: bool = False,
    anti_shift_area_m2: float = 0.05,
    anti_shift_ar: float = 5.0,
    two_pass_enable: bool = False,
) -> Tuple[List[List[Point]], Dict[str, float]]:
    best = None
    best_metrics = None
    best_oriented: List[List[Point]] = []

    segs2 = list(segs)
    if any(bool(getattr(s, "shared", False)) for s in segs2):
        segs2 = _postprocess_union_segments(
            segs2,
            snap_eps=max(1e-9, _coord_tol_default(nd)),
            nd=nd,
        )

    base_g = build_graph_from_segments(segs2, nd=nd)
    simp_g = simplify_graph_collinear(base_g, col_eps=col_eps)

    entry_penalty_mm = max(0.0, float(route_entry_junction_penalty_mm))

    def _entry_penalty(p: Point) -> float:
        if entry_penalty_mm <= 0.0:
            return 0.0
        d = simp_g.degree(p)
        if d <= 1:
            return 0.0
        if d == 2 and _is_true_corner(p, simp_g):
            return 0.0
        if d >= 3:
            return entry_penalty_mm
        return 0.25 * entry_penalty_mm

    for r in range(max(1, n_restarts)):
        rng = random.Random(seed + r)
        trails = decompose_to_trails(simp_g, nd=nd, random_tie=random_tie, rng=rng)

        policy = (route_start_policy or "none").lower().strip()
        if policy == "near_board_edge":
            origin_eff = _choose_edge_start_origin(trails, fallback=origin, board_bounds=board_bounds)
        else:
            origin_eff = origin

        prio = (route_priority or "none").lower().strip()

        def _order_one_group(_trails: List[List[Point]], _origin: Point) -> Tuple[List[List[Point]], float]:
            """Order + orient a group of trails with existing heuristics."""
            if not _trails:
                return [], 0.0
            if prio == "regional_nn":
                oriented_g, L_air_g, order_g, flips_g = _order_and_orient_regional_nn(
                    _trails,
                    origin=_origin,
                    feed_air=feed_air,
                    t_lift=t_lift,
                    window_size=route_local_window,
                    backtrack=route_local_backtrack,
                    entry_penalty_fn=_entry_penalty,
                )
                if len(order_g) >= 4:
                    order2_g = _two_opt_order_time(
                        order_g,
                        _trails,
                        origin=_origin,
                        feed_air=feed_air,
                        t_lift=t_lift,
                        iters=80,
                        entry_penalty_fn=_entry_penalty,
                    )
                    flips2_g, L_air2_g = optimize_flips_dp(
                        order2_g,
                        _trails,
                        origin=_origin,
                        entry_penalty_fn=_entry_penalty,
                    )
                    oriented2_g = apply_order_and_flips(order2_g, flips2_g, _trails)
                    if _time_obj_from_air(L_air2_g, len(order2_g), feed_air, t_lift) < _time_obj_from_air(L_air_g, len(order_g), feed_air, t_lift):
                        oriented_g, L_air_g = oriented2_g, L_air2_g
                return oriented_g, float(L_air_g)
            # default path
            order_g = _priority_nn_order(
                _trails,
                origin=_origin,
                route_priority=route_priority,
                entry_penalty_fn=_entry_penalty,
            )
            if prio == "none":
                order_g = two_opt_order(order_g, _trails, origin=_origin, iters=120)
            if len(order_g) >= 4:
                order_g = _two_opt_order_time(
                    order_g,
                    _trails,
                    origin=_origin,
                    feed_air=feed_air,
                    t_lift=t_lift,
                    iters=80,
                    entry_penalty_fn=_entry_penalty,
                )
            flips_g, L_air_g = optimize_flips_dp(
                order_g,
                _trails,
                origin=_origin,
                entry_penalty_fn=_entry_penalty,
            )
            oriented_g = apply_order_and_flips(order_g, flips_g, _trails)
            return oriented_g, float(L_air_g)

        oriented: List[List[Point]] = []
        L_air = 0.0

        # --- Process-rule layer: hierarchical ordering + anti-shifting two-pass ---
        if route_hierarchical and part_boxes:
            # Assign each trail to a part (by midpoint containment / nearest bbox), then build groups.
            def _trail_mid(tr: List[Point]) -> Point:
                if not tr:
                    return (0.0, 0.0)
                # midpoint by arclength would be better; use simple vertex mid for robustness
                return tr[len(tr) // 2]

            def _bbox_dist2(p: Point, bb: Dict[str, float]) -> float:
                x0, x1, y0, y1 = bb["xmin"], bb["xmax"], bb["ymin"], bb["ymax"]
                dx = 0.0
                if p[0] < x0:
                    dx = x0 - p[0]
                elif p[0] > x1:
                    dx = p[0] - x1
                dy = 0.0
                if p[1] < y0:
                    dy = y0 - p[1]
                elif p[1] > y1:
                    dy = p[1] - y1
                return dx * dx + dy * dy

            def _assign_part(tr: List[Point]) -> Optional[Dict[str, float]]:
                p = _trail_mid(tr)
                # 1) strict containment with eps
                eps = max(1e-9, _coord_tol_default(nd))
                for bb in part_boxes:
                    if (bb["xmin"] - eps) <= p[0] <= (bb["xmax"] + eps) and (bb["ymin"] - eps) <= p[1] <= (bb["ymax"] + eps):
                        return bb
                # 2) nearest bbox (for shared edges / boundary trails)
                best_bb, best_d2 = None, 1e100
                for bb in part_boxes:
                    d2 = _bbox_dist2(p, bb)
                    if d2 < best_d2:
                        best_d2, best_bb = d2, bb
                return best_bb

            # Split by part area (small-first, large-last)
            areas = sorted([float(bb.get("area_mm2", 0.0)) for bb in part_boxes if float(bb.get("area_mm2", 0.0)) > 0.0])
            area_cut = areas[int((1.0 - max(0.0, min(1.0, route_large_frac))) * (len(areas) - 1))] if areas else 0.0

            critical_small_group: List[List[Point]] = []
            small_group: List[List[Point]] = []
            large_group: List[List[Point]] = []
            finish_group: List[List[Point]] = []
            small_abs_th = max(0.0, float(route_small_first_area_mm2))

            for tr in trails:
                bb = _assign_part(tr)
                a_mm2 = float(bb.get("area_mm2", 0.0)) if bb else 0.0
                ar = float(bb.get("ar", 1.0)) if bb else 1.0
                # anti-shifting trigger (small + slender)
                a_m2 = a_mm2 / 1e6
                need_two_pass = bool(two_pass_enable) and bool(anti_shift_enable) and (a_m2 < float(anti_shift_area_m2)) and (ar >= float(anti_shift_ar))
                # Level 2/3 split
                if small_abs_th > 0.0 and a_mm2 > 0.0 and a_mm2 <= small_abs_th:
                    critical_small_group.append(tr)
                elif a_mm2 >= area_cut:
                    large_group.append(tr)
                else:
                    small_group.append(tr)
                if need_two_pass:
                    # finish pass duplicated and scheduled at the end
                    finish_group.append(tr)

            # Level 1 (inner holes/slots): not available in rectangle-only geometry, keep placeholder.
            inner_group: List[List[Point]] = []

            groups = [inner_group, critical_small_group, small_group, large_group, finish_group]
            cur_origin = origin_eff
            for g_trails in groups:
                oriented_g, L_air_g = _order_one_group(g_trails, cur_origin)
                oriented.extend(oriented_g)
                if oriented_g:
                    cur_origin = oriented_g[-1][-1]
                L_air += L_air_g
        else:
            oriented, L_air = _order_one_group(trails, origin_eff)

        if route_ccw:
            oriented = _force_ccw_closed_trails(oriented)
        oriented = [_merge_collinear_edges(tr, eps=max(1e-9, _coord_tol_default(nd)), col_eps=col_eps) for tr in oriented if tr]

        # Ramping / lead-in (simplified): add a short segment at the start of each trail.
        if ramp_enable and ramp_len > 0:
            oriented2: List[List[Point]] = []
            for tr in oriented:
                if len(tr) >= 2:
                    p0, p1 = tr[0], tr[1]
                    ux, uy = _unit(p1[0] - p0[0], p1[1] - p0[1])
                    pr = (p0[0] + float(ramp_len) * ux, p0[1] + float(ramp_len) * uy)
                    oriented2.append([pr, p0] + tr[1:])
                else:
                    oriented2.append(tr)
            oriented = oriented2

        # Optional consolidation: if two consecutive trails are already connected
        # at endpoints after snapping, merge them into one continuous stroke.
        oriented = _compress_adjacent_trails(oriented, eps=max(1e-9, _coord_tol_default(nd)))
        if not ramp_enable:
            oriented = [_merge_collinear_edges(tr, eps=max(1e-9, _coord_tol_default(nd)), col_eps=col_eps) for tr in oriented if tr]
        L_air = _air_len_by_oriented(oriented, origin=origin_eff)

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
            "N_seg_in": float(len(segs)),
            "N_seg_eff": float(len(segs2)),
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


def _sweep_intervals_with_meta(intervals: List[Tuple[float, float, dict]]) -> List[Tuple[float, float, int, List[dict]]]:
    events: Dict[float, List[Tuple[int, int]]] = {}
    meta: Dict[int, dict] = {}
    iid = 0
    for s, e, payload in intervals:
        if e <= s:
            continue
        meta[iid] = payload
        events.setdefault(s, []).append((1, iid))
        events.setdefault(e, []).append((-1, iid))
        iid += 1

    xs = sorted(events.keys())
    out: List[Tuple[float, float, int, List[dict]]] = []
    active: Set[int] = set()
    for i in range(len(xs) - 1):
        x = xs[i]
        for typ, idx in events.get(x, []):
            if typ > 0:
                active.add(idx)
            else:
                active.discard(idx)
        x2 = xs[i + 1]
        if x2 > x and active:
            metas = [meta[idx] for idx in active]
            out.append((x, x2, len(active), metas))
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


def _limit_shared_continuous_cut(
    segs: List[Segment],
    *,
    max_run_len: float = 0.0,
    hold_bridge_len: float = 0.0,
    nd: int = 6,
    eps: float = 1e-9,
) -> List[Segment]:
    """
    Split over-long shared segments into shorter cuts separated by hold-bridges.
    This reduces one-shot long shared cuts that may quickly release vacuum.
    """
    max_run = max(0.0, float(max_run_len))
    hold_len = max(0.0, float(hold_bridge_len))
    if not segs or max_run <= eps or hold_len <= eps:
        return list(segs)

    out: List[Segment] = []
    for s in segs:
        if not bool(s.shared):
            out.append(s)
            continue

        a = s.a
        b = s.b
        L = _dist(a, b)
        if L <= max_run + eps:
            out.append(s)
            continue

        ux = (b[0] - a[0]) / max(L, eps)
        uy = (b[1] - a[1]) / max(L, eps)
        cur = 0.0
        while cur < L - eps:
            cut_len = min(max_run, L - cur)
            if cut_len > eps:
                p0 = (a[0] + ux * cur, a[1] + uy * cur)
                p1 = (a[0] + ux * (cur + cut_len), a[1] + uy * (cur + cut_len))
                if _dist(p0, p1) > eps:
                    out.append(Segment(_round_pt(p0, nd), _round_pt(p1, nd), True))
            cur += cut_len
            if cur >= L - eps:
                break
            # Keep a small uncut bridge before continuing.
            cur += min(hold_len, L - cur)

    return [ss for ss in out if _dist(ss.a, ss.b) > eps]


def _gen_tabs_for_board(
    board,
    tab_per_part: int,
    tab_len: float,
    corner_clear: float,
    tab_skip_trim_edge: bool = False,
    tab_adaptive: bool = False,
    tab_slender_ratio: float = 6.0,
    tab_slender_extra: int = 0,
    tab_small_area_extra: int = 0,
    nd: int = 6,
) -> Tuple[Dict[Tuple[str, float], List[Tuple[float, float]]], int]:
    if tab_len <= 0:
        return {}, 0
    tabs: Dict[Tuple[str, float], List[Tuple[float, float]]] = {}
    seen = set()
    board_area = max(float(getattr(board, "W", 0.0)) * float(getattr(board, "H", 0.0)), 1.0)

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
        tabs_for_part = max(0, int(tab_per_part))
        if tab_adaptive:
            w0 = max(float(getattr(pp, "w0", r.w)), 1e-9)
            h0 = max(float(getattr(pp, "h0", r.h)), 1e-9)
            slender = max(w0, h0) / max(1e-9, min(w0, h0))
            if slender >= float(tab_slender_ratio):
                tabs_for_part += max(0, int(tab_slender_extra))
            # Small parts are easy to drift; add a lightweight extra tab.
            if (w0 * h0) <= 0.02 * board_area:
                tabs_for_part += max(0, int(tab_small_area_extra))
            # For robust large non-slender parts, one fewer tab usually keeps
            # stability while reducing trail fragmentation.
            if slender < float(tab_slender_ratio) and (w0 * h0) >= 0.08 * board_area:
                tabs_for_part = max(2, tabs_for_part - 1)
        if tabs_for_part <= 0:
            continue

        x0, x1 = r.x, r.x + r.w
        y0, y1 = r.y, r.y + r.h
        edges = [("h", y0, x0, x1, r.w), ("h", y1, x0, x1, r.w), ("v", x0, y0, y1, r.h), ("v", x1, y0, y1, r.h)]
        if tab_skip_trim_edge:
            t = float(getattr(board, "trim", 0.0) or 0.0)
            W = float(getattr(board, "W", 0.0) or 0.0)
            H = float(getattr(board, "H", 0.0) or 0.0)
            edge_eps = max(1e-6, _coord_tol_default(nd))
            x_left = t
            x_right = W - t
            y_bot = t
            y_top = H - t
            edges = [
                e for e in edges
                if not (
                    (e[0] == "h" and (abs(float(e[1]) - y_bot) <= edge_eps or abs(float(e[1]) - y_top) <= edge_eps))
                    or (e[0] == "v" and (abs(float(e[1]) - x_left) <= edge_eps or abs(float(e[1]) - x_right) <= edge_eps))
                )
            ]
        edges_sorted = sorted(edges, key=lambda t: t[4], reverse=True)
        alloc = [0, 0, 0, 0]
        for k in range(tabs_for_part):
            alloc[k % 4] += 1
        for idx_edge, (orient, line, s0, s1, L) in enumerate(edges_sorted):
            m = alloc[idx_edge]
            if m <= 0:
                continue
            # Adaptive mode keeps extra tabs away from corners on slender strips.
            corner_clear_eff = corner_clear
            if tab_adaptive and tabs_for_part > tab_per_part:
                corner_clear_eff = max(corner_clear_eff, 0.1 * min(r.w, r.h))
            lo = s0 + corner_clear_eff
            hi = s1 - corner_clear_eff
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
    tab_skip_trim_edge: bool = False,
    *,
    line_snap_eps: float = 0.0,
    min_shared_len: float = 0.0,
    shared_enable_edgeband: bool = False,
    shared_min_len_edgeband: float = 0.0,
    shared_max_continuous_cut: float = 0.0,
    shared_hold_bridge_len: float = 0.0,
    tab_adaptive: bool = False,
    tab_slender_ratio: float = 6.0,
    tab_slender_extra: int = 0,
    tab_small_area_extra: int = 0,
    nd: int = 6,
) -> Tuple[List[Segment], float, int, Dict[str, float]]:
    segs: List[Segment] = []
    L_shared = 0.0
    info: Dict[str, float] = {
        "L_shared": 0.0,
        "L_shared_edgeband": 0.0,
        "L_shared_candidate": 0.0,
        "L_shared_candidate_edgeband": 0.0,
        "N_shared": 0.0,
        "N_shared_edgeband": 0.0,
        "N_shared_candidate": 0.0,
        "N_shared_candidate_edgeband": 0.0,
    }
    if line_snap_eps <= 0.0:
        line_snap_eps = _coord_tol_default(nd)
    if min_shared_len < 0.0:
        min_shared_len = 0.0
    if shared_min_len_edgeband < 0.0:
        shared_min_len_edgeband = 0.0

    if share_mode == "none":
        for pp in board.placed:
            r = pp.rect
            pts = [(r.x, r.y), (r.x + r.w, r.y), (r.x + r.w, r.y + r.h), (r.x, r.y + r.h)]
            for i in range(4):
                segs.append(Segment(_round_pt(pts[i], nd), _round_pt(pts[(i + 1) % 4], nd), False))
        if tab_enable:
            tabs, n_tabs = _gen_tabs_for_board(
                board,
                tab_per_part,
                tab_len,
                tab_corner_clear,
                tab_skip_trim_edge=tab_skip_trim_edge,
                tab_adaptive=tab_adaptive,
                tab_slender_ratio=tab_slender_ratio,
                tab_slender_extra=tab_slender_extra,
                tab_small_area_extra=tab_small_area_extra,
                nd=nd,
            )
            segs = _apply_tabs_to_segments(segs, tabs, nd=nd)
        else:
            n_tabs = 0
        return segs, 0.0, n_tabs, info

    if share_mode != "union":
        raise ValueError(f"unknown share_mode: {share_mode}")

    vertical: Dict[float, List[Tuple[float, float, dict]]] = {}
    horizontal: Dict[float, List[Tuple[float, float, dict]]] = {}
    x_pool: List[float] = []
    y_pool: List[float] = []

    for pp in board.placed:
        r = pp.rect
        x0 = _snap_coord(float(r.x), x_pool, line_snap_eps)
        x1 = _snap_coord(float(r.x + r.w), x_pool, line_snap_eps)
        y0 = _snap_coord(float(r.y), y_pool, line_snap_eps)
        y1 = _snap_coord(float(r.y + r.h), y_pool, line_snap_eps)

        vertical.setdefault(x0, []).append(
            (y0, y1, {"part_uid": getattr(pp, "uid", None), "side": "L", "is_edgeband": bool(getattr(pp, "eb_L", 0))})
        )
        vertical.setdefault(x1, []).append(
            (y0, y1, {"part_uid": getattr(pp, "uid", None), "side": "R", "is_edgeband": bool(getattr(pp, "eb_R", 0))})
        )
        horizontal.setdefault(y0, []).append(
            (x0, x1, {"part_uid": getattr(pp, "uid", None), "side": "B", "is_edgeband": bool(getattr(pp, "eb_B", 0))})
        )
        horizontal.setdefault(y1, []).append(
            (x0, x1, {"part_uid": getattr(pp, "uid", None), "side": "T", "is_edgeband": bool(getattr(pp, "eb_T", 0))})
        )

    for x, ints in vertical.items():
        pieces = _sweep_intervals_with_meta(ints)
        for s0, e0, cnt, metas in pieces:
            L = e0 - s0
            if L <= 1e-12:
                continue
            part_ids = {m.get("part_uid", None) for m in metas if m.get("part_uid", None) is not None}
            has_edgeband = any(bool(m.get("is_edgeband", False)) for m in metas)
            shared_candidate = (len(part_ids) >= 2) and (L >= float(min_shared_len))
            if shared_candidate:
                info["L_shared_candidate"] += L
                info["N_shared_candidate"] += 1.0
                if has_edgeband:
                    info["L_shared_candidate_edgeband"] += L
                    info["N_shared_candidate_edgeband"] += 1.0

            shared = False
            if shared_candidate:
                if has_edgeband:
                    shared = bool(shared_enable_edgeband) and (L >= float(shared_min_len_edgeband))
                else:
                    shared = True

            segs.append(Segment(_round_pt((x, s0), nd), _round_pt((x, e0), nd), shared))
            if shared:
                L_shared += L
                info["N_shared"] += 1.0
                if has_edgeband:
                    info["L_shared_edgeband"] += L
                    info["N_shared_edgeband"] += 1.0

    for y, ints in horizontal.items():
        pieces = _sweep_intervals_with_meta(ints)
        for s0, e0, cnt, metas in pieces:
            L = e0 - s0
            if L <= 1e-12:
                continue
            part_ids = {m.get("part_uid", None) for m in metas if m.get("part_uid", None) is not None}
            has_edgeband = any(bool(m.get("is_edgeband", False)) for m in metas)
            shared_candidate = (len(part_ids) >= 2) and (L >= float(min_shared_len))
            if shared_candidate:
                info["L_shared_candidate"] += L
                info["N_shared_candidate"] += 1.0
                if has_edgeband:
                    info["L_shared_candidate_edgeband"] += L
                    info["N_shared_candidate_edgeband"] += 1.0

            shared = False
            if shared_candidate:
                if has_edgeband:
                    shared = bool(shared_enable_edgeband) and (L >= float(shared_min_len_edgeband))
                else:
                    shared = True

            segs.append(Segment(_round_pt((s0, y), nd), _round_pt((e0, y), nd), shared))
            if shared:
                L_shared += L
                info["N_shared"] += 1.0
                if has_edgeband:
                    info["L_shared_edgeband"] += L
                    info["N_shared_edgeband"] += 1.0

    info["L_shared"] = float(L_shared)
    if tab_enable:
        tabs, n_tabs = _gen_tabs_for_board(
            board,
            tab_per_part,
            tab_len,
            tab_corner_clear,
            tab_skip_trim_edge=tab_skip_trim_edge,
            tab_adaptive=tab_adaptive,
            tab_slender_ratio=tab_slender_ratio,
            tab_slender_extra=tab_slender_extra,
            tab_small_area_extra=tab_small_area_extra,
            nd=nd,
        )
        segs = _apply_tabs_to_segments(segs, tabs, nd=nd)
    else:
        n_tabs = 0
    segs = _postprocess_union_segments(
        segs,
        snap_eps=max(line_snap_eps, _coord_tol_default(nd)),
        shared_gap_bridge_eps=(float(tab_len) + float(line_snap_eps)) if (tab_enable and float(tab_len) > 0.0) else 0.0,
        nd=nd,
    )
    segs = _limit_shared_continuous_cut(
        segs,
        max_run_len=max(0.0, float(shared_max_continuous_cut)),
        hold_bridge_len=max(0.0, float(shared_hold_bridge_len)),
        nd=nd,
    )
    if float(shared_max_continuous_cut) > 0.0 and float(shared_hold_bridge_len) > 0.0:
        segs = _postprocess_union_segments(
            segs,
            snap_eps=max(line_snap_eps, _coord_tol_default(nd)),
            shared_gap_bridge_eps=0.0,
            nd=nd,
        )
    return segs, L_shared, n_tabs, info


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
    route_start_policy: str = "none",
    route_priority: str = "none",
    route_ccw: bool = False,
    # process-rule layer
    part_boxes: Optional[List[Dict[str, float]]] = None,
    route_hierarchical: bool = False,
    route_large_frac: float = 0.20,
    route_small_first_area_mm2: float = 0.0,
    route_entry_junction_penalty_mm: float = 0.0,
    ramp_enable: bool = False,
    ramp_len: float = 0.0,
    anti_shift_enable: bool = False,
    anti_shift_area_m2: float = 0.05,
    anti_shift_ar: float = 5.0,
    two_pass_enable: bool = False,
    board_bounds: Optional[Tuple[float, float, float, float]] = None,
    feed_air: float = 30000.0,
    t_lift: float = 0.8,
    route_local_window: int = 10,
    route_local_backtrack: int = 2,
    return_route_metrics: bool = False,
) -> Tuple:
    g, base = build_graph(segs)
    if not g:
        if return_trails:
            if return_route_metrics:
                return 0.0, 0, 0, [], [], [], {"L_cut": 0.0, "L_air": 0.0, "N_pierce": 0.0, "N_trails": 0.0}
            return 0.0, 0, 0, [], [], []
        else:
            if return_route_metrics:
                return 0.0, 0, 0, [], [], {"L_cut": 0.0, "L_air": 0.0, "N_pierce": 0.0, "N_trails": 0.0}
            return 0.0, 0, 0, [], []

    comps = connected_components(g)
    if cut_mode == "trail":
        L_cut_base = base
    elif cut_mode == "cpp":
        L_cut_base = base + sum(approx_cpp_extra(g, c) for c in comps)
    else:
        raise ValueError(f"unknown cut_mode: {cut_mode}")

    oriented_trails, metrics = compute_toolpath_trails(
        segs,
        origin=origin,
        nd=nd,
        route_start_policy=route_start_policy,
        route_priority=route_priority,
        route_ccw=route_ccw,
        part_boxes=part_boxes,
        route_hierarchical=route_hierarchical,
        route_large_frac=route_large_frac,
        route_small_first_area_mm2=route_small_first_area_mm2,
        route_entry_junction_penalty_mm=route_entry_junction_penalty_mm,
        ramp_enable=ramp_enable,
        ramp_len=ramp_len,
        anti_shift_enable=anti_shift_enable,
        anti_shift_area_m2=anti_shift_area_m2,
        anti_shift_ar=anti_shift_ar,
        two_pass_enable=two_pass_enable,
        board_bounds=board_bounds,
        feed_air=feed_air,
        t_lift=t_lift,
        route_local_window=route_local_window,
        route_local_backtrack=route_local_backtrack,
    )
    L_cut = metrics.get("L_cut", L_cut_base)
    n_strokes = len(oriented_trails)
    stroke_reps = [tr[0] for tr in oriented_trails]

    if return_trails:
        if return_route_metrics:
            return L_cut, len(comps), n_strokes, stroke_reps, oriented_trails, comps, metrics
        return L_cut, len(comps), n_strokes, stroke_reps, oriented_trails, comps
    else:
        if return_route_metrics:
            return L_cut, len(comps), n_strokes, stroke_reps, comps, metrics
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
