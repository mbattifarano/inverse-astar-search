import networkx as nx

from inverse_astar_search.pastar import max_admissible_heuristic, heuristic_bounds


def test_network_lecture_AI():
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

    return networks


def test_astar_admissibility():
    # Toy network
    g1 = test_network_lecture_AI()[1]

    # Shortest path
    optimal_path = nx.shortest_path(g1, source='s', target='t')

    # Admissible heuristic
    h_cost_g1 = {'s': 2, 'a': 4, 'b': 1, 'c': 1, 't': 0}

    # max_admissible_heuristic(G = g0, target = 't')

    def astar_heuristic(a, b):
        return h_cost_g1[a]
        # return max_admissible_heuristic(G=g0, target=b)[a]

    astar_optimal_path = nx.astar_path(G=g1, source='s', target='t', heuristic=astar_heuristic)

    assert optimal_path == astar_optimal_path


def test_astar_inadmissibility():
    # Toy network
    g1 = test_network_lecture_AI()[1]

    # Shortest path
    optimal_path = nx.shortest_path(g1, source='s', target='t')

    # Innadmisible heuristic
    h_cost_g1 = {'s': 2, 'a': 6, 'b': 1, 'c': 1, 't': 0}

    def astar_heuristic(a, b):
        return h_cost_g1[a]
        # return max_admissible_heuristic(G=g0, target=b)[a]

    astar_optimal_path = nx.astar_path(G=g1, source='s', target='t', heuristic=astar_heuristic)

    assert optimal_path != astar_optimal_path

def test_max_admissible_heuristic():
    # Toy network
    g1 = test_network_lecture_AI()[1]

    # Shortest path
    optimal_path = nx.shortest_path(g1, source='s', target='t')

    def astar_heuristic(a, b):
        return max_admissible_heuristic(G=g1, target=b)[a]

    astar_optimal_path = nx.astar_path(G=g1, source='s', target='t', heuristic=astar_heuristic)

    assert optimal_path == astar_optimal_path