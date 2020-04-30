
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import math

def create_network(W):
    '''Create a graph object compatible with network x package

    :arg W: weight or adjacency matrix

    Notes: edges with 0 cost and loops are ignored.

    '''
    graph = nx.DiGraph()
    n, m = W.shape
    assert n == m, "Adjacency matrix must be square"
    graph.add_nodes_from(range(n))
    for (i, j) in zip(*W.nonzero()):
        if i != j:
            graph.add_edge(i, j, weight=W[i, j])


    #Add attribute for the heuristic cost associated to each node
    nx.set_node_attributes(graph, 0,'heuristic')

    return graph

def set_random_nodes_coordinates(G, attribute_label, factor = 1):

    if factor != 1:
        pos = {k:factor*v for k,v in nx.random_layout(G.copy(), dim=2).items()}

    nx.set_node_attributes(G,pos, attribute_label)

    return G

def set_edges_euclidean_distances(G, attribute_label, nodes_coordinate_label = 'pos'):

    pos_nodes = nx.get_node_attributes(G,nodes_coordinate_label)
    len_edges = {}
    for edge in G.edges():
        len_edges[edge] = np.linalg.norm(np.array(pos_nodes[edge[0]]) - np.array(pos_nodes[edge[1]]))

    nx.set_edge_attributes(G,len_edges,attribute_label)

    return G

def random_edge_weights(A, limits, type = int):
    '''
    Assign random integer weights to non-zero cells in adjacency matrix A

    :arg limits: tuple with lower and upper bound for the random integer numbers
    '''

    # for (u, v) in G.edges():
    #     G.edges[u, v]['weight'] = random.randint(0, 10)

    for (i, j) in zip(*A.nonzero()):

        if type is int:
            A[(i,j)] = random.randint(*limits)
        else:
            A[(i, j)] = random.random(*limits)

    return A

def random_parallel_edges(A, limits):

    return random_edge_weights(A, limits, type = int)

def create_MultiDiGraph_network(DG):
    ''' Receive a Digraph and return MultiDiGraph by
    randomly creating additional edges between nodes
    with a existing edge

    '''

    #Get adjacency matrix
    A0 = nx.convert_matrix.to_numpy_matrix(DG)

    # Randomly generate extra edges
    UM = np.random.randint(low = 0, high = 2, size = A0.shape)
    A1 = np.multiply(UM,random_edge_weights(A0, (np.min(A0), np.max(A0))))

    # Generate MultiGraphNetwork
    MG = nx.compose(nx.from_numpy_matrix(A1),nx.DiGraph(nx.from_numpy_matrix(A1)))

    return MG

def show_network(G):
    '''Visualization of network.
    :arg G: graph object

    '''
    fig = plt.subplots()
    # fig.set_tight_layout(False) #Avoid warning using matplot

    pos = nx.get_node_attributes(G, 'pos')

    if len(pos) == 0:
        pos = nx.spring_layout(G)

    nx.draw(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw(G, with_labels=True, arrows=True, connectionstyle='arc3, rad = 0.1')

    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    nx.draw_networkx_labels(G, pos)

    plt.show()

def show_multiDiNetwork(G0, show_labels = False):

    #https://stackoverflow.com/questions/60067022/multidigraph-edges-from-networkx-draw-with-connectionstyle

    def new_add_edge(G, a, b):
        if (a, b) in G.edges:
            max_rad = max(x[2]['rad'] for x in G.edges(data=True) if sorted(x[:2]) == sorted([a,b]))
        else:
            max_rad = 0
        G.add_edge(a, b, rad=max_rad+0.1)

    G = nx.MultiDiGraph()

    edges = list(G0.edges)

    for edge in edges:
        new_add_edge(G, edge[0], edge[1])

    plt.figure(figsize=(10,10))

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)

    labels = nx.get_edge_attributes(G0, 'weight')

    if show_labels == True:
        for label_key in list(labels.keys()):
            labels[label_key] = str(list(label_key))+ ' = ' + str(labels[label_key])

            if labels.get(label_key[1],label_key[0]) is not None:
                labels[label_key] += ' , ' + str((label_key[1],label_key[0])) + ' = ' + str(labels.get(label_key[1],label_key[0]))


    # nx.draw(G, with_labels=True, arrows=True, connectionstyle='arc3, rad = 0.1')

    for edge in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(edge[0],edge[1])], connectionstyle=f'arc3, rad = {edge[2]["rad"]}')

        if show_labels == True:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, label_pos= 0.5)

    plt.show()

def get_adjacency_toynetwork(index):
    ''':argument index: index of the toy network'''

    toynetworks = []
    toynetworks.append(
        np.array([
        [0., 1., 1.],
        [1., 0., 1.],
        [1., 1., 0.]
    ]))

    toynetworks.append(
        np.array([
        [0., 0., 0., 1., 1.],
        [1., 0., 1., 1., 1.],
        [1., 1., 0., 1., 1.],
        [1., 1., 1., 0., 1.],
        [1., 1., 1., 1., 0.]
    ])
    )
    return toynetworks[index]