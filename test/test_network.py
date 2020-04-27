from inverse_astar_search.network import show_multiDiNetwork
from inverse_astar_search.network import random_edge_weights
from inverse_astar_search.network import get_adjacency_toynetwork
from inverse_astar_search.network import create_network

import networkx as nx
import matplotlib.pyplot as plt

def test_show_multiDiNetwork():
    A = get_adjacency_toynetwork(1)
    W = random_edge_weights(A=A, limits=(1, 3))
    show_multiDiNetwork(G0 = create_network(W), show_labels = False)

