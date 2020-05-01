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

@fixture(params=[0, 1])
def network_lecture_AI(request):
    networks = []
    # g1 = nx.graph(directed=True)

    g1 = nx.DiGraph(directed=True)

    g1.add_edge('s', 'a', weight=2)
    g1.add_edge('s', 'b', weight=2)
    g1.add_edge('a', 't', weight=2)
    g1.add_edge('b', 't', weight=3)

    networks.append(g1)

    g2 = nx.DiGraph(directed=True)

    g2.add_edge('s', 'a', weight=1)
    g2.add_edge('s', 'b', weight=1)
    g2.add_edge('a', 'c', weight=1)
    g2.add_edge('b', 'c', weight=2)
    g2.add_edge('c', 't', weight=3)

    networks.append(g2)

    return networks[request.param]

