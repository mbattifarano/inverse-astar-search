from inverse_astar_search.optim import linear_program, get_by_node_pair, entropy
from inverse_astar_search.settings import WEIGHT_KEY
from inverse_astar_search.graph import assign_weights, path_cost

from typing import Set, Tuple, List

import networkx as nx
import cvxpy as cp
import numpy as np
import os

import pytest

import hypothesis
from hypothesis import given, assume, settings, example, infer
from hypothesis.strategies import integers, lists, nothing
from hypothesis.extra.numpy import arrays

NUMBER_OF_NODES = 8
CI = os.environ.get('CI', False)  # are we on CI or not


weighted_adjacency_matrix = (
        arrays(
            dtype=np.float,
            shape=integers(min_value=3, max_value=10).map(lambda n: (n, n)),
            elements=integers(min_value=0, max_value=10),
            fill=nothing(),
            unique=False,
        )
        .map(lambda A: A * (1 - np.eye(A.shape[0])))  # set the diagonal to zero
        .filter(lambda A: (A.sum(1) > 0).all())  # at least one out edge from every node
)


@given(pairs=infer)
def test_get_by_node_pair(pairs: Set[Tuple[str, str]]):
    assume(pairs)
    index = {pair: i for i, pair in enumerate(pairs)}
    x = cp.Variable(len(index))
    x.value = np.arange(len(index))
    for pair, i in index.items():
        assert get_by_node_pair(index, x, pair).value == i

def test_entropy():
    n = 5
    values = np.arange(n) + 1
    x = cp.Variable(n)
    x.value = values
    actual = entropy((x[i] for i in range(n))).value
    expected = -(values * np.log(values)).sum()
    assert actual == pytest.approx(expected)


def hash_graph(a: np.ndarray):
    idx = a.nonzero()
    w = a[idx]
    i,j = idx
    return hash(tuple(sorted(zip(i, j, w))))

example_network_3x3 = np.array([
    [0., 3., 1.],
    [1., 0., 1.],
    [1., 1., 0.]
])

example_network_5x5 = np.array([
    [0., 0., 0., 1., 1.],
    [1., 0., 1., 1., 1.],
    [1., 1., 0., 1., 1.],
    [1., 1., 1., 0., 1.],
    [1., 1., 1., 1., 0.]
])

@given(A=weighted_adjacency_matrix)
@example(A=example_network_3x3)
@example(A=example_network_5x5)
@settings(max_examples=10, deadline=None)
def test_linear_program(network_factory, A):
    graph = network_factory(A)
    if A.shape == example_network_3x3.shape and np.allclose(A, example_network_3x3):
        fname = "example_3x3"
    elif A.shape == example_network_5x5.shape and np.allclose(A, example_network_5x5):
        fname = "example_5x5"
    else:
        fname = hash_graph(A)
    nx.write_graphml(graph, f"test/artifacts/{fname}.graphml")
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
    hypothesis.note(f"number of constraints = {len(result.problem.constraints)}")
    result.penalty.value = 0
    result.problem.solve()
    assert result.problem.status == cp.OPTIMAL
    hypothesis.note(f"objective value = {result.problem.value}")
    weights = result.edge_cost.value
    np.save(f"test/artifacts/{fname}.weights.npy", weights)
    hypothesis.note(f"Did we recover the edge weights? {'yes' if np.allclose(weights, old_weights) else 'no'}; max error = {abs(weights - old_weights).max()}")
    assert weights is not None
    assert len(weights) == graph.number_of_edges()
    assert weights.min() >= 0.0
    graph = assign_weights(graph, weights)
    for u in graph.nodes:
        assert sum(graph.edges[u, v][WEIGHT_KEY]
                   for v in graph.successors(u)) == pytest.approx(1.0)
    new_A = np.zeros_like(A)
    for i, j, data in graph.edges(data=True):
        new_A[i, j] = data[WEIGHT_KEY]
    old_A = A / A.sum(1).reshape(-1, 1)
    hypothesis.note(f"Input adjancency matrix (row-normalized):")
    hypothesis.note(old_A)
    hypothesis.note(f"Recovered adjacency matrix:")
    hypothesis.note(new_A)
    errors = [abs(old - new)
              for old, new in zip(old_A.reshape(-1), new_A.reshape(-1))
              if old > 0
              ]
    hypothesis.note(f"Errors: mean={np.mean(errors)}; min={np.min(errors)}, max={np.max(errors)}")
    # Run shortest paths on graph with recovered weights
    expected_path_length = nx.shortest_path_length(graph, weight=WEIGHT_KEY)
    n_path_checks = 0
    # Everything being compared is in [0, 1] so absolute tolerance is sufficient
    approx = lambda expected: pytest.approx(expected, abs=1e-6)
    for s, costs in expected_path_length:
        for t, cost in costs.items():
            n_path_checks += 1
            hypothesis.note(f"{s}->{t} path = {_paths[s][t]}")
            # These two should ALWAYS be the same
            actual_cost = result.min_trip_cost_of((s, t))  # min path cost from solution
            expected_cost = path_cost(graph, _paths[s][t])  # cost of the observed path on the network
            hypothesis.note(f"least cost from {s}->{t} by edge weight: {cost}")
            hypothesis.note(f"least cost from {s}->{t} by lp variable: {actual_cost}")
            hypothesis.note(f"least cost from {s}->{t} by actual shortest path: {expected_cost}")
            assert expected_cost == approx(actual_cost)
            if not CI:
                # we are interested in counter examples locally, but not on CI
                assert cost == expected_cost

