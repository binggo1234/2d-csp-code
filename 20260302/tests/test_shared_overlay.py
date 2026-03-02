from __future__ import annotations

from inrp.routing import Segment, _postprocess_union_segments


def _segments_key(segs):
    out = []
    for s in segs:
        a, b = s.a, s.b
        if b < a:
            a, b = b, a
        out.append((a, b, bool(s.shared)))
    return sorted(out)


def test_shared_flag_does_not_bleed_to_adjacent_nonshared() -> None:
    segs = [
        Segment((0.0, 0.0), (5.0, 0.0), True),
        Segment((5.0, 0.0), (10.0, 0.0), False),
    ]
    out = _postprocess_union_segments(segs, snap_eps=1e-6, shared_gap_bridge_eps=0.0, nd=6)

    assert _segments_key(out) == [
        ((0.0, 0.0), (5.0, 0.0), True),
        ((5.0, 0.0), (10.0, 0.0), False),
    ]


def test_shared_gap_bridge_only_when_both_sides_are_shared() -> None:
    segs = [
        Segment((0.0, 0.0), (5.0, 0.0), True),
        Segment((5.2, 0.0), (10.0, 0.0), False),
    ]
    out = _postprocess_union_segments(segs, snap_eps=1e-6, shared_gap_bridge_eps=0.3, nd=6)

    assert _segments_key(out) == [
        ((0.0, 0.0), (5.0, 0.0), True),
        ((5.2, 0.0), (10.0, 0.0), False),
    ]


def test_shared_gap_bridge_keeps_shared_when_both_shared() -> None:
    segs = [
        Segment((0.0, 0.0), (5.0, 0.0), True),
        Segment((5.2, 0.0), (10.0, 0.0), True),
    ]
    out = _postprocess_union_segments(segs, snap_eps=1e-6, shared_gap_bridge_eps=0.3, nd=6)

    assert _segments_key(out) == [
        ((0.0, 0.0), (10.0, 0.0), True),
    ]
