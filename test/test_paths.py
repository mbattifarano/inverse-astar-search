from __future__ import annotations

from inverse_astar_search.types import Paths, Path
from hypothesis import given, infer, assume

from inverse_astar_search.paths import edges, origin, destination, trip, is_valid


@given(path=infer)
def test_edges(path: Path):
    assume(is_valid(path))
    actual = list(edges(path))
    assert len(actual) == len(path) - 1
    for i, (u, v) in enumerate(actual):
        assert path[i] == u
        assert path[i+1] == v


@given(path=infer)
def test_endpoints(path: Path):
    assume(is_valid(path))
    o = origin(path)
    d = destination(path)
    od_pair = trip(path)
    assert od_pair == (o, d)

