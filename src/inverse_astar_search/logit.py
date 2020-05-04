import numpy as np
import cvxpy as cp
import networkx as nx

from inverse_astar_search.pastar import neighbors_path

def create_decision_matrix_from_observed_path(G, observed_path):
    ''' Receive a dictionary with the connected nodes for each node
        Return a matrix to represent the choice set at each node
    '''

    nNodes = len(np.unique(list(G.nodes)))
    expanded_nodes = observed_path[:-1] #All nodes except target node
    # next_nodes = dict(zip(optimal_path[:-1], optimal_path[1:]))
    connected_nodes = neighbors_path(G = G, path = observed_path)

    avail_matrix = np.zeros([nNodes,nNodes])

    for expanded_node in expanded_nodes:
        avail_matrix[expanded_node, connected_nodes[expanded_node]] = 1
        # nRow += 1

    return avail_matrix
    # return avail_matrix

def get_avail_attribute(avail_matrix, Xk):

    Xk_avail = []

    for i in range(avail_matrix.shape[0]):
        Xk_avail.append(Xk[i,np.where(avail_matrix[i, :] == 1)[0]])

    return Xk_avail

def widetolong(wide_matrix):
    """Wide to long format
    The new matrix has one rows per alternative
    """

    wide_matrix = wide_matrix.astype(int)

    if wide_matrix.ndim == 1:
        wide_matrix = wide_matrix.reshape(1, wide_matrix.shape[0])

    long_matrix = np.repeat(wide_matrix, repeats=np.sum(wide_matrix, axis=1), axis=0)

    return long_matrix

def compute_edge_utility(G, theta: dict):
    attributes = list(theta.keys())

    utility = np.zeros(len(G.edges))

    for attribute in attributes:
        utility += theta[attribute] * np.array(list(nx.get_edge_attributes(G, attribute).values()))

    return dict(zip(G.edges, utility))

def logit_estimation(X,y: dict,avail: dict,attributes):
    '''

    :argument y: chosen edges in path i
    :argument avail: choice scenarios (set) for trip i
    :argument X: network attributes
    :argument attributes: attributes to fit discrete choice model
    '''

    #Estimated parameters to be optimized (learned)
    cp_theta = {i:cp.Variable(1) for i in attributes}

    nodes_decision = {i:[y_j[0] for y_j in y_i] for i,y_i in zip(range(len(y)),y.values())}
    nodes_chosen = {i:[y_j[1] for y_j in y_i] for i,y_i in zip(range(len(y)),y.values())}

    X_avail = {}
    for i, avail_path in avail.items():
        X_avail[i] = {attribute:get_avail_attribute(avail_path, X[attribute]) for attribute in attributes}

    # X_avail = {attribute: get_avail_attribute(avail, X[attribute]) for attribute in attributes}

    # Loglikelihood function
    Z = []
    for i,observed_path in avail.items():
        Z_i = []
        for j,k in zip(nodes_decision[i],nodes_chosen[i]):
            Z_chosen_attr = []
            Z_logsum_attr = []
            for attribute in attributes:
                Z_chosen_attr.append(X[attribute][j,k] * cp_theta[attribute])
                Z_logsum_attr.append(X_avail[i][attribute][j]* cp_theta[attribute])

            Z_i.append(cp.sum(Z_chosen_attr) - cp.log_sum_exp(cp.sum(Z_logsum_attr)))

        Z.append(cp.sum(Z_i))


    # Z = [X['travel_time'][i,j] * cp_theta['travel_time'] + X['cost'][i,j] *  cp_theta['cost'] + X['h'][i,j] * cp_theta['h']
    #       - cp.log_sum_exp(X_avail['travel_time'][i] * cp_theta['travel_time'] + X_avail['cost'][i] *  cp_theta['cost'] + X_avail['h'][i] * cp_theta['h'])
    #      for i,j in zip(nodes_decision,nodes_chosen)
    #      ]  # axis = 1 is for rows

    cp_objective_logit = cp.Maximize(cp.sum(Z))

    cp_problem_logit = cp.Problem(cp_objective_logit, constraints = []) #Excluding extra attributes

    cp_problem_logit.solve()
    assert cp_problem_logit.status == cp.OPTIMAL

    return {key:val.value for key,val in cp_theta.items()}

# def non_negative_costs_edges():

def logit_path_predictions(G, observed_paths: dict, theta_logit: dict):

    G_copy = G.copy()

    weights_logit = {key: -val for key, val in theta_logit.items()}

    #Make sure all utilities are positive so a-star can run properly
    min_u = min(list(compute_edge_utility(G, theta=weights_logit).values()))

    valid_utilities_astar = {i:u+abs(min_u) for i,u in compute_edge_utility(G, theta=weights_logit).items()}
    
    nx.set_edge_attributes(G_copy, values = valid_utilities_astar, name = 'utility_prediction')

    # nx.get_edge_attributes(G_copy, 'utility_prediction')

    predicted_paths = {}

    for key, observed_path in observed_paths.items():
        predicted_paths[key] = nx.astar_path(G=G_copy, source=observed_path[0], target=observed_path[-1], weight='utility_prediction')

    predicted_paths_length = paths_lengths(G_copy,predicted_paths,'utility_prediction') #Utility acts as a proxy of the path length (negative)

    return {'predicted_path': predicted_paths, 'length': predicted_paths_length}

def accuracy_pastar_predictions(G, predicted_paths: dict, observed_paths:dict):

    # paths_lengths(G, paths = predicted_paths, attribute = 'utility')

    edge_acc = {}
    for key,observed_path in observed_paths.items():
        edge_acc[key] = sum(el in observed_path for el in predicted_paths[key])/len(observed_path)

    path_acc = dict(zip(edge_acc.keys(),np.where(np.array(list(edge_acc.values())) < 1, 0, 1)))

    x_correct_edges = np.round(np.mean(np.array(list(edge_acc.values()))),2)
    x_correct_paths = np.round(np.mean(np.array(list(path_acc.values()))),2)

    # utility_diff = abs(sum(paths_lengths(G, paths= predicted_paths, attribute='utility').values())
    #                   -abs(sum(paths_lengths(G, paths= observed_paths, attribute='utility').values()))
    #                   )

    return {'acc_edges': x_correct_edges, 'acc_paths': x_correct_paths}
