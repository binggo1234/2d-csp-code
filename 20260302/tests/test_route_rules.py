from __future__ import annotations

from inrp.routing import (
    Segment,
    _limit_shared_continuous_cut,
    build_segments_from_board,
    compute_toolpath_trails,
    optimize_flips_dp,
)
from inrp.packer import Board, PlacedPart, Rect


def _bbox_area(tr):
    xs = [p[0] for p in tr]
    ys = [p[1] for p in tr]
    return max(0.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))


def test_small_first_priority_prefers_small_trail_even_if_farther() -> None:
    # Large square near origin.
    large = [
        Segment((0.0, 0.0), (100.0, 0.0), False),
        Segment((100.0, 0.0), (100.0, 100.0), False),
        Segment((100.0, 100.0), (0.0, 100.0), False),
        Segment((0.0, 100.0), (0.0, 0.0), False),
    ]
    # Small square farther on +x.
    small = [
        Segment((220.0, 0.0), (235.0, 0.0), False),
        Segment((235.0, 0.0), (235.0, 15.0), False),
        Segment((235.0, 15.0), (220.0, 15.0), False),
        Segment((220.0, 15.0), (220.0, 0.0), False),
    ]
    oriented, _ = compute_toolpath_trails(
        large + small,
        origin=(0.0, 0.0),
        route_priority="small_first",
        route_hierarchical=False,
    )

    assert len(oriented) == 2
    assert _bbox_area(oriented[0]) < _bbox_area(oriented[1])


def test_entry_penalty_affects_flip_choice() -> None:
    trails = [[(0.0, 0.0), (10.0, 0.0)]]
    order = [0]

    def penalty(p):
        # Penalize starting at x=0 side.
        return 100.0 if abs(p[0]) < 1e-9 else 0.0

    flips, _ = optimize_flips_dp(order, trails, origin=(5.0, 0.0), entry_penalty_fn=penalty)
    assert flips == [True]


def test_limit_shared_continuous_cut_inserts_hold_bridges() -> None:
    segs = [Segment((0.0, 0.0), (100.0, 0.0), True)]
    out = _limit_shared_continuous_cut(segs, max_run_len=30.0, hold_bridge_len=5.0, nd=6)

    got = sorted((round(s.a[0], 6), round(s.b[0], 6), s.shared) for s in out)
    assert got == [
        (0.0, 30.0, True),
        (35.0, 65.0, True),
        (70.0, 100.0, True),
    ]


def test_tab_skip_trim_edge_keeps_trim_boundary_continuous() -> None:
    b = Board(bid=1, W=100.0, H=60.0, trim=5.0)
    b.placed = [
        PlacedPart(uid=1, pid_raw="p1", rect=Rect(5.0, 5.0, 50.0, 20.0), w0=50.0, h0=20.0),
    ]

    segs_with_tabs, _, _, _ = build_segments_from_board(
        b,
        share_mode="none",
        tab_enable=True,
        tab_per_part=2,
        tab_len=10.0,
        tab_corner_clear=5.0,
        tab_skip_trim_edge=False,
        nd=6,
    )
    segs_skip_trim, _, _, _ = build_segments_from_board(
        b,
        share_mode="none",
        tab_enable=True,
        tab_per_part=2,
        tab_len=10.0,
        tab_corner_clear=5.0,
        tab_skip_trim_edge=True,
        nd=6,
    )

    yb = 5.0
    n_bottom_with_tabs = sum(
        1
        for s in segs_with_tabs
        if abs(s.a[1] - yb) < 1e-9 and abs(s.b[1] - yb) < 1e-9 and abs(s.a[0] - s.b[0]) > 1e-9
    )
    n_bottom_skip_trim = sum(
        1
        for s in segs_skip_trim
        if abs(s.a[1] - yb) < 1e-9 and abs(s.b[1] - yb) < 1e-9 and abs(s.a[0] - s.b[0]) > 1e-9
    )

    assert n_bottom_with_tabs > 1
    assert n_bottom_skip_trim == 1
