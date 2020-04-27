from pytest import fixture

import networkx as nx


@fixture(scope="session")
def network_factory():
    def from_adjacency_matrix(A):
        graph = nx.DiGraph()
        n, m = A.shape
        assert n == m, "Adjacency matrix must be square"
        graph.add_nodes_from(range(n))
        for (i, j) in zip(*A.nonzero()):
            graph.add_edge(i, j, weight=A[i,j])
        return graph
    return from_adjacency_matrix
