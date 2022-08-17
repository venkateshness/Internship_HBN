
#%%
from math import degrees
from cv2 import Laplacian
from scipy import io as sio
from pygsp import graphs
import pickle
import numpy as np
import torch
import networkx as nx
def NNgraph():
    """Nearest Neighbour graph Setup.

    Returns:
        Matrix of floats: A weight matrix for the thresholded graph
    """

    pickle_file = '/homes/v20subra/S4B2/GSP/MMP_RSFC_brain_graph_fullgraph.pkl'

    with open(pickle_file, 'rb') as f:
        [connectivity] = pickle.load(f)
    np.fill_diagonal(connectivity, 0)
    # connectivity = sio.loadmat('/homes/v20subra/S4B2/GSP/SC_avg56.mat')['SC_avg56']
    graph = torch.from_numpy(connectivity)
    knn_graph = torch.zeros(graph.shape)
    for i in range(knn_graph.shape[0]):
        graph[i, i] = 0
        best_k = torch.sort(graph[i, :])[1][-8:]
        knn_graph[i, best_k] = 1
        knn_graph[best_k, i] = 1

    degree = torch.diag(knn_graph.sum(dim = 0))
    adjacency = knn_graph
    laplacian   = degree - adjacency
    values, eigs = torch.linalg.eigh(laplacian)
    return laplacian, adjacency

