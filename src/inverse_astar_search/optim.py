from __future__ import annotations

from inverse_astar_search.types import (
    Path, Paths, TripIndexedPaths, NodePairIndex, NodePair, Node
)
from inverse_astar_search.paths import trip_indexed_paths, edges, trip, chain

import cvxpy as cp
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression

import networkx as nx

from typing import List, Iterable, Tuple, NamedTuple, Callable
from itertools import product, starmap
from toolz import curry, juxt, concat, compose
from toolz.curried import map

starmap = curry(starmap)


def edge_cost(number_of_edges: int) -> cp.Variable:
    return cp.Variable(
        number_of_edges,
        name="edge_cost",
        nonneg=True,
    )


def min_path_cost(number_of_trips: int) -> cp.Variable:
    return cp.Variable(
        number_of_trips,
        name="min_trip_cost",
        nonneg=True
    )

NodePairToVariable = Callable[[NodePair], cp.Variable]

@curry
def get_by_node_pair(index: NodePairIndex, variable: cp.Variable,
                     node_pair: NodePair) -> cp.Varable:
    return variable[index[node_pair]]


@curry
def successors(graph: nx.DiGraph, node: Node) -> Iterable[Node]:
    return graph.successors(node)


@curry
def out_edges(successors: Callable[Node, Iterable[Node]],
              node: Node) -> Iterable[NodePair]:
    return (
        (node, v)
        for v in successors(node)
    )


def entropy(variables: Iterable[cp.Variable]) -> Expression:
    return sum(map(cp.entr, variables))


@curry
def edges_entropy(edge_cost: NodePairToVariable, edges: Iterable[NodePair]) -> cp.Expression:
    return entropy(map(edge_cost, edges))


@curry
def edge_cost_is_normalized(edge_cost: NodePairToVariable,
                            edges: Iterable[NodePair]) -> Constraint:
    return sum(map(edge_cost, edges)) == 1.0


@curry
def path_cost(edge_cost: NodePairToVariable, path: Path) -> Expression:
    return sum(map(edge_cost, edges(path)))


@curry
def path_cost_gap(min_trip_cost: cp.Variable, path_cost: Expression
                  ) -> Expression:
    return path_cost - min_trip_cost


def gap_is_optimal(gap: Expression) -> Constraint:
    return gap == 0.0


def gap_is_suboptimal(gap: Expression) -> Constraint:
    return gap >= 0.0


@curry
def all_paths_via(trip_indexed_paths: TripIndexedPaths,
                  trip: NodePair,
                  node: Node) -> Iterable[Path]:
    """Generate additional paths completing a trip by stitching together known paths"""
    s, t = trip
    return starmap(chain,
                product(trip_indexed_paths.get((s, node), []),  # all paths s->node
                        trip_indexed_paths.get((node, t), [])   # all paths node->t
                )
        )


@curry
def suboptimal_paths(nodes: Iterable[Node],
                     trip_indexed_paths: TripIndexedPaths,
                     trip: NodePair) -> Paths:
    return concat(map(all_paths_via(trip_indexed_paths, trip),
                      nodes)
                  )


def linear_program(graph: nx.DiGraph, paths: Paths) -> Result:
    cost = edge_cost(graph.number_of_edges())

    trips = list(set(map(trip, paths)))
    least_path_cost = min_path_cost(len(trips))
    gamma = cp.Parameter(nonneg=True)

    edge_index = {e: i for i, e in enumerate(graph.edges)}
    trip_index = {t: i for i, t in enumerate(trips)}

    edge_cost_of = get_by_node_pair(edge_index, cost)
    trip_cost_of = get_by_node_pair(trip_index, least_path_cost)

    out_edges_of_node = out_edges(successors(graph))  # Node -> Iterable[NodePair]

    min_trip_cost_from_path = compose(
        trip_cost_of,  # NodePair -> cp.Variable
        trip  # Path -> NodePair
    )  # Path -> cp.Variable

    compute_gaps = compose(
        starmap(path_cost_gap),
        map(juxt(min_trip_cost_from_path,
                 path_cost(edge_cost_of))
            )
    )  # Iterable[Path] -> Iterable[float]

    out_edge_cost_is_normalized = compose(
        edge_cost_is_normalized(edge_cost_of),
        out_edges_of_node,
    )  # Node -> Constraint

    constraints = []

    constraints.extend(map(out_edge_cost_is_normalized, graph.nodes))

    constraints.extend(map(gap_is_optimal, compute_gaps(paths)))

    other_paths = concat(map(suboptimal_paths(graph.nodes, trip_indexed_paths(paths)),
                             trips))
    other_paths = list(other_paths)
    suboptimal_gaps = compute_gaps(other_paths)
    constraints.extend(map(gap_is_suboptimal, suboptimal_gaps))

    out_edge_entropy = compose(
        edges_entropy(edge_cost_of),  # Iterable[NodePair] -> cp.Expression
        out_edges_of_node,  # Node -> Iterable[NodePair]
    )  # Node -> cp.Expression

    total_out_edge_entropy = compose(
        sum,  # Iterable[cp.Expression] -> cp.Expression
        map(out_edge_entropy) # Iterable[Node] -> Iterable[cp.Expression]
    )  # Iterable[Node] -> cp.Expression

    objective = total_out_edge_entropy(graph.nodes) - cp.sum(least_path_cost)

    problem = cp.Problem(
        cp.Maximize(objective),
        constraints
    )

    return Result(
        problem=problem,
        penalty=gamma,
        edge_cost=cost,
        min_trip_cost=least_path_cost,
        edge_index=edge_index,
        trip_index=trip_index,
        discovered_paths=other_paths,
    )


class Result(NamedTuple):
    problem: cp.Problem
    penalty: cp.Parameter
    edge_cost: cp.Variable
    min_trip_cost: cp.Variable
    edge_index: NodePairIndex
    trip_index: NodePairIndex
    discovered_paths: Paths

    def edge_cost_of(self, e: NodePair) -> float:
        return self.edge_cost[self.edge_index[e]].value

    def min_trip_cost_of(self, t: NodePair) -> float:
        return self.min_trip_cost[self.trip_index[t]].value

