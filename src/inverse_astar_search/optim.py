from __future__ import annotations

from inverse_astar_search.types import (
    Path, Paths, TripIndexedPaths, NodePairIndex, NodePair, Node
)
from inverse_astar_search.paths import trip_indexed_paths, edges, trip, chain

import cvxpy as cp
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression

import networkx as nx

from typing import List, Iterable, Tuple, NamedTuple
from itertools import product, starmap
from toolz import curry, juxt, concat, compose
from toolz.curried import map

starmap = curry(starmap)


def edge_cost(graph: nx.DiGraph) -> cp.Variable:
    return cp.Variable(
        graph.number_of_edges(),
        nonneg=True
    )


def min_path_cost(number_of_trips: int) -> cp.Variable:
    return cp.Variable(
        number_of_trips,
        nonneg=True
    )


def edge_cost_is_normalized(edge_cost: cp.Variable) -> Constraint:
    return cp.sum(edge_cost) == 1


@curry
def path_cost(edge_index: NodePairIndex, edge_cost: cp.Variable, path: Path
              ) -> Expression:
    return cp.sum([edge_cost[edge_index[e]]
                   for e in edges(path)])

@curry
def path_cost_gap(min_trip_cost: cp.Variable, path_cost: Expression
                  ) -> Expression:
    return path_cost - min_trip_cost


def gap_is_optimal(gap: Expression) -> Constraint:
    return gap == 0.0


def gap_is_suboptimal(gap: Expression) -> Constraint:
    return gap >= 0.0


@curry
def min_trip_cost(trip_index: NodePairIndex, min_path_cost: cp.Variable,
                  trip: NodePair) -> cp.Variable:
    return min_path_cost[trip_index[trip]]


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
    cost = edge_cost(graph)

    trips = list(set(map(trip, paths)))
    least_path_cost = min_path_cost(len(trips))

    edge_index = {e: i for i, e in enumerate(graph.edges)}
    trip_index = {t: i for i, t in enumerate(trips)}

    min_trip_cost_from_path = compose(
        min_trip_cost(trip_index, least_path_cost),
        trip
    )

    compute_gaps = compose(
        starmap(path_cost_gap),
        map(juxt(min_trip_cost_from_path,
                 path_cost(edge_index, cost))
            )
    )

    constraints = [edge_cost_is_normalized(cost)]
    constraints.extend(map(gap_is_optimal, compute_gaps(paths)))

    other_paths = concat(map(suboptimal_paths(graph.nodes, trip_indexed_paths(paths)),
                             trips))
    other_paths = list(other_paths)
    suboptimal_gaps = list(compute_gaps(other_paths))
    constraints.extend(map(gap_is_suboptimal, suboptimal_gaps))

    objective = cp.sum(suboptimal_gaps) if suboptimal_gaps else -cp.sum(least_path_cost)

    problem = cp.Problem(
        cp.Maximize(objective),
        constraints
    )

    return Result(
        problem=problem,
        edge_cost=cost,
        min_trip_cost=least_path_cost,
        edge_index=edge_index,
        trip_index=trip_index,
        discovered_paths=other_paths,
    )


class Result(NamedTuple):
    problem: cp.Problem
    edge_cost: cp.Variable
    min_trip_cost: cp.Variable
    edge_index: NodePairIndex
    trip_index: NodePairIndex
    discovered_paths: Paths

    def edge_cost_of(self, e: NodePair) -> float:
        return self.edge_cost[self.edge_index[e]].value

    def min_trip_cost_of(self, t: NodePair) -> float:
        return self.min_trip_cost[self.trip_index[t]].value

