
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
def NNgraph(graph):
    """Nearest Neighbour graph Setup.

    Returns:
        Matrix of floats: A weight matrix for the thresholded graph
    """
    if graph =='FC':

        pickle_file = '/homes/v20subra/S4B2/GSP/MMP_RSFC_brain_graph_fullgraph.pkl'

        with open(pickle_file, 'rb') as f:
            [connectivity] = pickle.load(f)
        np.fill_diagonal(connectivity, 0)

    elif graph == 'SC':
        connectivity = sio.loadmat('/homes/v20subra/S4B2/GSP/SC_avg56.mat')['SC_avg56']
    
    # elif graph == 'kalofiasFC':
    #     connectivity = sio.loadmat('/homes/v20subra/S4B2/gspbox/kalofiasbraingraphs.mat')['FC_smooth_log']
    
    # elif graph == 'kalofiasSC':
    #     connectivity = sio.loadmat('/homes/v20subra/S4B2/gspbox/kalofiasbraingraphs.mat')['SC_smooth_log']

    graph = torch.from_numpy(connectivity)
    graph.fill_diagonal_(0)
    knn_graph = torch.zeros(graph.shape)
    
    for i in range(knn_graph.shape[0]):
        graph[i, i] = 0
        best_k = torch.sort(graph[i, :])[1][-8:]
        knn_graph[i, best_k] = 1 #graph[i, best_k].float()
        knn_graph[best_k, i] = 1 #graph[best_k, i].float()


    degree = torch.diag(knn_graph.sum(dim = 0)) #torch.tensor(np.diag(sum(connectivity!=0))) 
    adjacency = knn_graph
    
    laplacian   = degree - adjacency
    values, eigs = torch.linalg.eigh(laplacian)
    return laplacian

#%%
# _, adjacency_FC = NNgraph()

# # from scipy import stats
# # from nilearn.plotting import plot_matrix
# # plot_matrix(np.corrcoef(, np.random.rand(25,25)))

# #%%
# _, adjacency_SC = NNgraph()

# from scipy import stats
# from nilearn.plotting import plot_matrix
# plot_matrix(np.corrcoef( adjacency_FC, adjacency_SC))
# # %%
