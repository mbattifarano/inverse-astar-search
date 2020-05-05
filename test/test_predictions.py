from hypothesis import given, settings
from hypothesis.strategies import lists, integers, nothing
from hypothesis.extra.numpy import arrays

import pytest
import time
import numpy as np
import networkx as nx
import cvxpy as cp
import datetime as dt

from inverse_astar_search.settings import WEIGHT_KEY
from inverse_astar_search.optim import linear_program
from inverse_astar_search.graph import path_cost, assign_weights
from inverse_astar_search.pastar import chosen_edges
from inverse_astar_search.logit import create_decision_matrix_from_observed_path, logit_estimation


def _sublists(l):
    for i in range(len(l)):
        yield l[:i]

def hash_graph(a: np.ndarray):
    idx = a.nonzero()
    w = a[idx]
    i,j = idx
    return hash(tuple(sorted(zip(i, j, w))))

def _get_lp_predicted_weights(graph, observations):
    result = linear_program(graph, observations)
    result.problem.solve()
    assert result.problem.status == cp.OPTIMAL
    return result.edge_cost.value

def _utility_to_edge_weight(utility):
    neg_utility = - utility
    offset = -min(0, neg_utility.min())
    return neg_utility + offset

def _get_rlm_predicted_weights(graph, observations):
    observations = dict(enumerate(observations))
    features = graph.graph['features']
    data = graph.graph['data']
    X = {feature: nx.adjacency_matrix(graph, weight=feature)
         for feature in features}
    y = chosen_edges(observations)
    decisions = {
        i: create_decision_matrix_from_observed_path(graph, path)
        for i, path in observations.items()
    }

    thetas = logit_estimation(
        X=X,
        y=y,
        avail=decisions,
        attributes=features,
    )
    theta = np.array([thetas[feature] for feature in features])
    utility = data @ theta
    return _utility_to_edge_weight(utility)

def _accuracy(graph, shortest_paths, weights, tol=1e-6):
    g = assign_weights(graph, weights)
    _costs = dict(nx.shortest_path_length(g, weight=WEIGHT_KEY))
    _errors = np.array([
        path_cost(g, p) - _costs[p[0]][p[-1]]
        for p in shortest_paths
    ])
    return (_errors < tol).mean()

shapes = (
    integers(min_value=2, max_value=4)
    .map(lambda n: n * 3)
    .map(lambda n: (n, n))
)

di_graph = (
    arrays(
        dtype=np.float,
        shape=shapes,
        elements=integers(min_value=0, max_value=1),
        fill=nothing(),
        unique=False,
    )
    .map(lambda A: A * (1 - np.eye(A.shape[0])))  # set the diagonal to zero
    .filter(lambda A: (A.sum(1) > 0).all())  # at least one out edge from every node
    .map(nx.DiGraph)
)


_indices = arrays(  # arrays of 10 to 100 non-negative unique integers
        dtype=np.int,
        shape=20,
        elements=integers(min_value=0, max_value=1e5),
        unique=True,
    )
_noise = integers(min_value=-2, max_value=1).map(lambda n: 10**n)


timestamp = dt.datetime.utcnow().isoformat()


@given(graph=di_graph, path_i=_indices)
@settings(max_examples=50, deadline=None)
def test_predictions(data_collector, graph, path_i):
    t0 = time.time()
    data_list = data_collector[f"accuracy-{timestamp}"]
    experiment = len(data_list)
    print(f"Running experiment {experiment} on a graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    features = list('abcd')
    theta_true = np.ones(len(features))
    X = np.random.random((graph.number_of_edges(), len(features)))
    utility = X @ theta_true
    for noise in [0, 0.1, 1]:
        epsilon = np.random.normal(0, noise, graph.number_of_edges())
        edge_weights = _utility_to_edge_weight(utility)
        assert edge_weights.min() >= 0
        for i, (_, _, edata) in enumerate(graph.edges(data=True)):
            edata[WEIGHT_KEY] = edge_weights[i]
            for j, name in enumerate(features):
                edata[name] = X[i, j]
        graph.graph['features'] = features
        graph.graph['data'] = X

        _paths = nx.shortest_path(graph, weight=WEIGHT_KEY)
        paths = [p
                 for _, ps in _paths.items()
                 for _, p in ps.items()
                 if len(p) >= 2
                 ]
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        n_paths = len(paths)
        assert n_paths > 0
        for sample_id, idx in enumerate(filter(lambda a: len(a) > 0, _sublists(path_i))):
            path_ids = [i % n_paths for i in idx]
            observations = [tuple(paths[i]) for i in path_ids]
            assert observations
            n_unique_observations = len(set(observations))
            estimated_weights = {
                'linear program': _get_lp_predicted_weights(graph, observations),
                'recursive logit': _get_rlm_predicted_weights(graph, observations),
            }
            for model, weights in estimated_weights.items():
                data_collector[f"accuracy-{timestamp}"].append(dict(
                    experiment=experiment, # graph id
                    sample_id=sample_id, #observation id
                    n_nodes=n_nodes,
                    n_edges=n_edges,
                    noise=noise, # noise used
                    n_unique_training=n_unique_observations,
                    model=model,
                    accuracy=_accuracy(graph, paths, weights),
                ))
        print(f"Ran models in {time.time()-t0:0.4f}s")

