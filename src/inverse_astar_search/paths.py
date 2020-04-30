from inverse_astar_search.types import (
    Paths, Path, NodePair, Node, TripIndexedPaths, NodeIndexedPaths
)

from collections import defaultdict
from typing import Iterable, Tuple


def edges(path: Path) -> Iterable[NodePair]:
    if not path:
        return
    nodes = iter(path)
    u = next(nodes)
    for v in nodes:
        yield (u, v)
        u = v

def origin(path: Path) -> Node:
    return path[0]


def destination(path: Path) -> Node:
    return path[-1]


def trip(path: Path) -> NodePair:
    return (
        origin(path),
        destination(path),
    )

def chain(a: Path, b: Path) -> Path:
    assert a[-1] == b[0]
    return a + b[1:]


def destination_indexed_paths(paths: Paths) -> NodeIndexedPaths:
    d = {}
    for path in paths:
        d.setdefault(dest(path), list()).append(path)
    return d


def trip_indexed_paths(paths: Path) -> TripIndexedPaths:
    d = {}
    for path in paths:
        d.setdefault(trip(path), list()).append(path)
    return d


def is_valid(path: Path) -> bool:
    """A path is valid if it has two or more nodes
    """
    return len(path) >= 2
