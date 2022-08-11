

from scipy import io as sio
from pygsp import graphs
import pickle
import numpy as np
import torch

def graph_setup(unthresholding, weights):
    """Function to finalize the graph setup -- with options to threshold the graph by overriding the graph weights

    Args:
        unthresholding (bool): do you want a graph unthresholded ?
        weights (Float): Weight matrix

    Returns:
        Graph: Returns the Graph, be it un- or thresholded, which the latter is done using the 8Nearest-Neighbour
    """
    coordinates = sio.loadmat(
        '/homes/v20subra/S4B2/GSP/Glasser360_2mm_codebook.mat')['codeBook']

    G = graphs.Graph(weights, gtype = 'HCP subject',
                     lap_type = 'combinatorial', coords = coordinates)
    G.set_coordinates('spring')
    print('{} nodes, {} edges'.format(G.N, G.Ne))

    if unthresholding:
        pickle_file = '/homes/v20subra/S4B2/GSP/MMP_RSFC_brain_graph_fullgraph.pkl'

        with open(pickle_file, 'rb') as f:
            [connectivity] = pickle.load(f)
        np.fill_diagonal(connectivity, 0)

        G = graphs.Graph(connectivity)
        print(G.is_connected())
        print('{} nodes, {} edges'.format(G.N, G.Ne))

    return G


def NNgraph():
    """Nearest Neighbour graph Setup.

    Returns:
        Matrix of floats: A weight matrix for the thresholded graph
    """

    pickle_file = '/homes/v20subra/S4B2/GSP/MMP_RSFC_brain_graph_fullgraph.pkl'

    # with open(pickle_file, 'rb') as f:
    #     [connectivity] = pickle.load(f)
    # np.fill_diagonal(connectivity, 0)
    connectivity = sio.loadmat('/homes/v20subra/S4B2/GSP/SC_avg56.mat')['SC_avg56']
    graph = torch.from_numpy(connectivity)
    knn_graph = torch.zeros(graph.shape)
    for i in range(knn_graph.shape[0]):
        graph[i, i] = 0
        best_k = torch.sort(graph[i, :])[1][-8:]
        knn_graph[i, best_k] = 1
        knn_graph[best_k, i] = 1

    degree = torch.diag(torch.pow(knn_graph.sum(dim = 0), -0.5))

    weight_matrix_after_NN = torch.matmul(
        degree, torch.matmul(knn_graph, degree))
    return weight_matrix_after_NN

def graph_setup_main():
    G = graph_setup(False, NNgraph())
    return G
