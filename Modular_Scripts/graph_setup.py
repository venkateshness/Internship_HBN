#%%
from math import degrees
from cv2 import Laplacian
from scipy import io as sio
import pickle
import numpy as np
import scipy
import networkx as nx


def NNgraph(graph_type, connectivity_matrix=None):
    """Nearest Neighbour graph Setup.

    Returns:
        Matrix of floats: A weight matrix for the thresholded graph
    """
    if graph_type == "FC":

        pickle_file = "/homes/v20subra/S4B2/GSP/MMP_RSFC_brain_graph_fullgraph.pkl"

        with open(pickle_file, "rb") as f:
            [connectivity] = pickle.load(f)
        np.fill_diagonal(connectivity, 0)

    elif graph_type == "SC":
        connectivity = sio.loadmat("/homes/v20subra/S4B2/GSP/SC_avg56.mat")["SC_avg56"]

    # elif graph == 'kalofiasFC':
    #     connectivity = sio.loadmat('/homes/v20subra/S4B2/gspbox/kalofiasbraingraphs.mat')['FC_smooth_log']

    elif graph_type == "individual":
        connectivity = connectivity_matrix
    # elif graph == 'kalofiasSC':
    #     connectivity = sio.loadmat('/homes/v20subra/S4B2/gspbox/kalofiasbraingraphs.mat')['SC_smooth_log']

    graph = connectivity
    # knn_graph = torch.zeros(graph.shape)

    # for i in range(knn_graph.shape[0]):
    #     graph[i, i] = 0
    #     best_k = torch.sort(graph[i, :])[1][-8:]
    #     knn_graph[i, best_k] = 1 #graph[i, best_k].float()
    #     knn_graph[best_k, i] = 1 #graph[best_k, i].float()

    # G = nx.from_numpy_matrix(graph.numpy())
    # laplacian = nx.normalized_laplacian_matrix(G).toarray()

    degree = np.diag(np.power(np.sum(graph, axis=1), -0.5))
    laplacian = np.eye(graph.shape[0]) - np.matmul(degree, np.matmul(graph, degree))
    # values, eigs = torch.linalg.eigh(laplacian)
    # print(l - laplacian.numpy())

    return laplacian


#%%
# _, adjacency_FC = NNgraph()
laplacian =NNgraph('SC')
import scipy.linalg as la

[eigvals, eigevecs] = la.eigh(laplacian)
# # # from scipy import stats
# import matplotlib.pyplot as plt
# plt.figure(figsize=(25,25))
# plt.imshow(adjacency, cmap='gray')

# # plot_matrix(np.corrcoef(, np.random.rand(25,25)))

# from scipy import stats
# from nilearn.plotting import plot_matrix
# plot_matrix(np.corrcoef( adjacency_FC, adjacency_SC))
# # %%

import matplotlib
from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image, plotting
import matplotlib.pyplot as plt


path_Glasser = "/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz"

mnitemp = fetch_icbm152_2009()
mask_mni = image.load_img(mnitemp["mask"])
glasser_atlas = image.load_img(path_Glasser)

signal = np.expand_dims(eigevecs[:,1], axis=0)  # add dimension 1 to signal array

U0_brain = signals_to_img_labels(signal, path_Glasser, mnitemp["mask"])

plotting.plot_img_on_surf(
U0_brain,
colorbar=True, views = ['lateral'], cmap='cold_hot', threshold=0.01)# %%
# plt.savefig('eig_lf.png', transparent=True)

# %%
