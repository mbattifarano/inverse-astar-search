# =============================================================================
# Imports
# =============================================================================

# Internal modules
from inverse_astar_search.network import *

from inverse_astar_search.pastar import heuristic_bounds, astar_path_heuristic_nodes\
    , set_heuristic_costs_nodes, set_heuristic_costs_edges, get_edge_attributes_labels,path_generator\
    , chosen_edges, path_length, paths_lengths, neighbors_path

from inverse_astar_search.logit import create_decision_matrix_from_observed_path\
    , logit_estimation, compute_edge_utility, logit_path_predictions, accuracy_pastar_predictions

# External modules
import numpy as np
import networkx as nx

# =============================================================================
# Network Factory
# =============================================================================
def create_network_data(n_nodes, n_sample_paths: int, theta_training: dict, attributes_thresholds: dict = None):

    nodes_G = n_nodes

    # Adjacency matrix
    A = np.random.randint(0, 2, [nodes_G, nodes_G])

    # Create networkX graph
    G = create_network(A)

    # Node attributes

    # - Coordinates
    G = set_random_nodes_coordinates(G, attribute_label = 'pos', factor=1000)
    nx.get_node_attributes(G, name='pos')

    # Edges attributes

    # - Distance [km]
    G = set_edges_euclidean_distances(G, attribute_label = 'distance', nodes_coordinate_label = 'pos')
    # nx.get_edge_attributes(G, name='distance')

    # - Cost ($)
    cost_edges = dict(zip(dict(G.edges).keys(), np.random.randint(0, 20, len(list(G.edges)))))
    nx.set_edge_attributes(G, values=cost_edges, name='cost')

    # - Travel time (mins)
    speed = 20  # km/hr
    travel_time_edges = dict(zip(dict(G.edges).keys(),
                                 np.round(60 / speed * np.array(list(nx.get_edge_attributes(G, 'distance').values())),
                                          1)))
    nx.set_edge_attributes(G, values=travel_time_edges, name='travel_time')

    # Utility at edges
    utility_edges = compute_edge_utility(G, theta=theta_training)
    nx.set_edge_attributes(G, utility_edges, name='utility')

    # Edge weight equals utility
    weight_edges = {key: -val for key, val in nx.get_edge_attributes(G, 'utility').items()}
    nx.set_edge_attributes(G, values=weight_edges, name='weight')

    # Simulate observed paths
    observed_paths = path_generator(G = G, n_paths =n_sample_paths, attribute='utility')


    # Heuristics

    # Aerial (euclidian) distance
    # TODO:

    # - Heuristic cost at edge level: TODO: needs to be implemented properly
    G = heuristic_bounds(G=G, observed_path=observed_paths[0])
    G = set_heuristic_costs_nodes(G)
    G = set_heuristic_costs_edges(G)

    # # Observed path (real observation)
    # print((observed_paths[0], path_length(G, path=observed_paths[0], attribute='distance')))

    # nx.get_node_attributes(G, 'f_bound_neighbor')
    # nx.get_node_attributes(G, 'h_bound_neighbor')
    # nx.get_node_attributes(G, 'h_bound_optimal')

    return G, observed_paths

# Preference parameters
theta_G_training = {'travel_time': -4, 'cost': -2} #Signs are not required to be negative
vot = theta_G_training['travel_time'] / theta_G_training['cost']

G_training, observed_paths = create_network_data(n_nodes = 40, n_sample_paths = 100, theta_training = theta_G_training)

# =============================================================================
# Logit Estimation
# =============================================================================

#Attributes in logit model
get_edge_attributes_labels(G_training) # Candidate attributes
attributes_G_training = ['travel_time','cost','h']

# Matrix with choice sets generated at each node
decision_matrix_paths_G_training = {key:create_decision_matrix_from_observed_path(G = G_training, observed_path = observed_path) for key,observed_path  in observed_paths.items()}

#Dictionary with matrix of attributes between every OD pair
X_G_training = {attribute:nx.adjacency_matrix(G_training, weight = attribute).todense() for attribute in attributes_G_training}

#Chosen edges
y_G_training = chosen_edges(paths = observed_paths)

#Logit estimates

# - With heuristic
theta_logit_heuristic = logit_estimation(X = X_G_training, y = y_G_training, avail = decision_matrix_paths_G_training, attributes = attributes_G_training)

# - With no heuristic (standard logit)
attributes_logit_standard_G =  [i for i in attributes_G_training if i != 'h']
theta_logit_standard = logit_estimation(X = X_G_training, y = y_G_training, avail = decision_matrix_paths_G_training, attributes = attributes_logit_standard_G)

# =============================================================================
# Prediction in training data
# =============================================================================

#Observed paths
# print(*observed_paths.values(), sep='\n')
observed_paths
paths_lengths(G_training, paths = observed_paths, attribute = 'utility')
logit_path_predictions(G = G_training, observed_paths = observed_paths, theta_logit = theta_G_training)

#Path predictions based on logit estimates

# - Considering heuristic cost (not informative so far because they are set based on a single path [0]
prediction_logit_heuristic = logit_path_predictions(G = G_training, observed_paths = observed_paths, theta_logit = theta_logit_heuristic)

# - Ignoring heuristic cost from prediction but in estimation
theta_logit_no_heuristic = {key:val for key,val in theta_logit_heuristic.items() if key != 'h'}
prediction_logit_no_heuristic =  logit_path_predictions(G = G_training, observed_paths = observed_paths, theta_logit = theta_logit_no_heuristic)

# - Exclusion of heuristic cost in estimation and prediction
prediction_logit_standard = logit_path_predictions(G = G_training, observed_paths = observed_paths, theta_logit = theta_logit_standard)

# =============================================================================
# Generalization power (testing data)
# =============================================================================

# Create testing data (using same preference parameters, but different network and sample of paths)
G_testing, observed_paths_testing = create_network_data(n_nodes = 60, n_sample_paths = 200, theta_training = theta_G_training)

# Predictions
prediction_logit_heuristic_testing = logit_path_predictions(G = G_testing, observed_paths = observed_paths_testing, theta_logit = theta_logit_heuristic)
prediction_logit_no_heuristic_testing =  logit_path_predictions(G = G_testing, observed_paths = observed_paths_testing, theta_logit = theta_logit_no_heuristic)
prediction_logit_standard_testing = logit_path_predictions(G = G_testing, observed_paths = observed_paths_testing, theta_logit = theta_logit_standard)

# =============================================================================
# Summary tables
# =============================================================================

# a) Accuracy training sample

# Observed paths (accuracy 100%)
print('Observed paths')
print(accuracy_pastar_predictions(G = G_training, predicted_paths = observed_paths, observed_paths= observed_paths))

#Standard logit model
print('Standard logit model')
print(accuracy_pastar_predictions(G = G_training, predicted_paths = prediction_logit_standard['predicted_path'], observed_paths= observed_paths))

#Logit model and heuristic (pastar)
print('Pastar with heuristic')
print(accuracy_pastar_predictions(G = G_training, predicted_paths = prediction_logit_heuristic['predicted_path'], observed_paths= observed_paths))

#Pastar with no heuristic
print('Pastar with no heuristic')
print(accuracy_pastar_predictions(G = G_training, predicted_paths = prediction_logit_no_heuristic['predicted_path'], observed_paths= observed_paths))

# b) Accuracy testing sample

# Observed paths (accuracy 100%)
print('Observed paths')
print(accuracy_pastar_predictions(G = G_testing, predicted_paths = observed_paths_testing, observed_paths= observed_paths_testing))

#Standard logit model
print('Standard logit model')
print(accuracy_pastar_predictions(G = G_testing, predicted_paths = prediction_logit_standard_testing['predicted_path'], observed_paths= observed_paths_testing))

#Logit model and heuristic (pastar)
print('Pastar with heuristic')
print(accuracy_pastar_predictions(G = G_testing, predicted_paths = prediction_logit_heuristic_testing['predicted_path'], observed_paths= observed_paths_testing))

#Pastar with no heuristic
print('Pastar with no heuristic')
print(accuracy_pastar_predictions(G = G_testing, predicted_paths = prediction_logit_no_heuristic_testing['predicted_path'], observed_paths= observed_paths_testing))


#Table path predictions

# list(prediction_logit_heuristic['predicted_path'].values())
# list(prediction_logit_standard['predicted_path'].values())

# =============================================================================
# Summary plots
# =============================================================================

## Analysis of parameter estimates in logit model

# #Visualize network
# show_multiDiNetwork(G_training)
# show_multiDiNetwork(G_testing)


# =============================================================================
# Single path analysis of heuristic cost
# =============================================================================

#TODO: Generalize heuristic assignment when having multiple paths.

# Pick first random path
print((observed_paths[0], path_length(G_training, path=observed_paths[0], attribute='distance')))

# Path predicted with dijsktra (h = 0)
dijkstra_path_prediction = nx.astar_path(G = G_training
                           ,source = observed_paths[0][0], target = observed_paths[0][-1]
                           ,weight = 'distance')

print((dijkstra_path_prediction,path_length(G_training, dijkstra_path_prediction,attribute = 'distance')))

# Path predicted with A*star and the computed heuristic costs
astar_heuristic_path_prediction = astar_path_heuristic_nodes(G = G_training,heuristic_costs= nx.get_node_attributes(G_training, 'h')
                           ,source = observed_paths[0][0], target = observed_paths[0][-1]
                           ,weight = 'distance')

print((astar_heuristic_path_prediction,path_length(G_training, astar_heuristic_path_prediction, attribute = 'distance')))
