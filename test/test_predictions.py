from hypothesis import given, settings
from hypothesis.strategies import lists, integers, nothing
from hypothesis.extra.numpy import arrays

import numpy as np
import networkx as nx
import cvxpy as cp

from inverse_astar_search.settings import WEIGHT_KEY
from inverse_astar_search.optim import linear_program
from inverse_astar_search.graph import path_cost, assign_weights


def _sublists(l):
    for i in range(len(l)):
        yield l[:i]

def _get_lp_predicted_weights(graph, observations):
    result = linear_program(graph, observations)
    result.problem.solve()
    assert result.problem.status == cp.OPTIMAL
    return result.edge_cost.value

def _get_rlm_predicted_weights(graph, observations):
    pass

def _accuracy(graph, shortest_paths, weights, tol=1e-6):
    g = assign_weights(graph, weights)
    _costs = dict(nx.shortest_path_length(g, weight=WEIGHT_KEY))
    _errors = np.array([
        path_cost(g, p) - _costs[p[0]][p[-1]]
        for p in shortest_paths
    ])
    return (_errors < tol).mean()

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

@given(A=weighted_adjacency_matrix, path_i=lists(integers(min_value=0), min_size=1))
@settings(max_examples=10, deadline=None)
def test_predictions(data_collector, network_factory, A, path_i):
    graph = network_factory(A)
    _paths = nx.shortest_path(graph, weight=WEIGHT_KEY)
    paths = [p
             for _, ps in _paths.items()
             for _, p in ps.items()
             if p
             ]
    n_paths = len(paths)
    assert paths
    for idx in filter(None, _sublists(path_i)):
        path_ids = [i % n_paths for i in idx]
        observations = [tuple(paths[i]) for i in path_ids]
        assert observations
        lp_weights = _get_lp_predicted_weights(graph, observations)
        rlm_weights = _get_rlm_predicted_weights(graph, observations)
        data_collector['accuracy'].append(dict(
            n_nodes=graph.number_of_nodes(),
            n_edges=graph.number_of_edges(),
            n_unique_training=len(set(observations)),
            lp_accuracy=_accuracy(graph, paths, lp_weights),
            rlm_accuracy=np.inf,
        ))

