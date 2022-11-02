
#%%
from math import degrees
from cv2 import Laplacian
from scipy import io as sio
from pygsp import graphs
import pickle
import numpy as np
import scipy
import torch
import networkx as nx
def NNgraph(graph_type, connectivity_matrix = None):
    """Nearest Neighbour graph Setup.

    Returns:
        Matrix of floats: A weight matrix for the thresholded graph
    """
    if graph_type =='FC':

        pickle_file = '/homes/v20subra/S4B2/GSP/MMP_RSFC_brain_graph_fullgraph.pkl'

        with open(pickle_file, 'rb') as f:
            [connectivity] = pickle.load(f)
        np.fill_diagonal(connectivity, 0)

    elif graph_type == 'SC':
        connectivity = sio.loadmat('/homes/v20subra/S4B2/GSP/SC_avg56.mat')['SC_avg56']
    
    # elif graph == 'kalofiasFC':
    #     connectivity = sio.loadmat('/homes/v20subra/S4B2/gspbox/kalofiasbraingraphs.mat')['FC_smooth_log']
    
    elif graph_type == 'individual':
        connectivity = connectivity_matrix
    # elif graph == 'kalofiasSC':
    #     connectivity = sio.loadmat('/homes/v20subra/S4B2/gspbox/kalofiasbraingraphs.mat')['SC_smooth_log']

    graph = torch.from_numpy(connectivity)
    # knn_graph = torch.zeros(graph.shape)
    
    # for i in range(knn_graph.shape[0]):
    #     graph[i, i] = 0
    #     best_k = torch.sort(graph[i, :])[1][-8:]
    #     knn_graph[i, best_k] = 1 #graph[i, best_k].float()
    #     knn_graph[best_k, i] = 1 #graph[best_k, i].float()

    
    # G = nx.from_numpy_matrix(graph.numpy())
    # laplacian = nx.normalized_laplacian_matrix(G).toarray()
    
    degree = torch.diag(np.power(sum(graph != 0), -0.5))
    laplacian = torch.eye(graph.shape[0]) - torch.matmul(degree, torch.matmul(graph, degree))
    # values, eigs = torch.linalg.eigh(laplacian)
    # print(l - laplacian.numpy())

    return laplacian
#%%
# _, adjacency_FC = NNgraph()
# NNgraph('SC')
# # from scipy import stats
# # from nilearn.plotting import plot_matrix
# # plot_matrix(np.corrcoef(, np.random.rand(25,25)))

# from scipy import stats
# from nilearn.plotting import plot_matrix
# plot_matrix(np.corrcoef( adjacency_FC, adjacency_SC))
# # %%
