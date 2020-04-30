import networkx as nx
import numpy as np

from inverse_astar_search.settings import WEIGHT_KEY
from inverse_astar_search.types import Path
from inverse_astar_search.paths import edges


def assign_weights(graph: nx.DiGraph, weights: np.ndarray) -> nx.DiGraph:
    for e, w in zip(graph.edges, weights):
        graph.edges[e][WEIGHT_KEY] = w
    return graph


def path_cost(graph: nx.DiGraph, path: Path) -> float:
    return sum(
        graph.edges[e][WEIGHT_KEY]
        for e in edges(path)
    )

