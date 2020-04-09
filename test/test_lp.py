from inverse_astar_search.optim import linear_program
from inverse_astar_search.settings import WEIGHT_KEY
from inverse_astar_search.graph import assign_weights, path_cost

import networkx as nx
import cvxpy as cp
import numpy as np

import pytest

import hypothesis
from hypothesis import given, assume, settings
from hypothesis.strategies import integers
from hypothesis.extra.numpy import arrays

NUMBER_OF_NODES = 3


weighted_adjacency_matrix = (
        arrays(
            dtype=np.float,
            shape=(NUMBER_OF_NODES, NUMBER_OF_NODES),
            elements=integers(min_value=0, max_value=10))
        .map(lambda A: A * (1 - np.eye(NUMBER_OF_NODES)))  # set the diagonal to zero
        .filter(lambda A: A.sum() > 0)  # at least one edge
        .map(lambda A: A / A.sum())  # normalize the edge weights
)


@given(A=weighted_adjacency_matrix)
@settings(max_examples=5, deadline=None)
def test_linear_program(network_factory, A):
    graph = network_factory(A)
    n_nodes = graph.number_of_nodes()
    hypothesis.note(f"number of nodes = {n_nodes}")
    old_weights = np.array([graph.edges[e][WEIGHT_KEY] for e in graph.edges])
    _paths = nx.shortest_path(graph, weight=WEIGHT_KEY)
    paths = [p
             for _, ps in _paths.items()
             for _, p in ps.items()
             ]
    hypothesis.note(f"paths:\n{paths}")
    result = linear_program(graph, paths)
    result.problem.solve()
    assert result.problem.status == cp.OPTIMAL
    hypothesis.note(f"discovered paths:\n{result.discovered_paths}")
    weights = result.edge_cost.value
    hypothesis.note(f"Did we recover the edge weights? {'yes' if np.allclose(weights, old_weights) else 'no'}; max error = {abs(weights - old_weights).max()}")
    assert weights is not None
    assert len(weights) == graph.number_of_edges()
    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() >= 0.0
    graph = assign_weights(graph, weights)
    new_A = np.zeros_like(A)
    for i, j, data in graph.edges(data=True):
        new_A[i, j] = data[WEIGHT_KEY]
    hypothesis.note(f"Recovered adjacency matrix:")
    hypothesis.note(new_A)
    expected_path_length = nx.shortest_path_length(graph, weight=WEIGHT_KEY)
    n_path_checks = 0
    for s, costs in expected_path_length:
        for t, cost in costs.items():
            n_path_checks += 1
            hypothesis.note(f"{s}->{t} path = {_paths[s][t]}")
            actual_cost = result.min_trip_cost_of((s, t))
            expected_cost = path_cost(graph, _paths[s][t])
            hypothesis.note(f"least cost from {s}->{t} by edge weight: {cost}")
            hypothesis.note(f"least cost from {s}->{t} by lp variable: {actual_cost}")
            hypothesis.note(f"least cost from {s}->{t} by actual shortest path: {expected_cost}")
            assert expected_cost == pytest.approx(cost)
            assert cost == pytest.approx(actual_cost)

