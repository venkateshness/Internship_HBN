#%%
from cProfile import label
from turtle import addshape, shape
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster
from sklearn.decomposition import randomized_svd
from torch import rand
sns.set_theme()

import scipy.stats
import importlib
os.chdir('/homes/v20subra/S4B2/')
from Modular_Scripts import graph_setup
importlib.reload(graph_setup)
import numpy as np
from nilearn import datasets, plotting, image, maskers
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
#%%

from bct.utils import BCTParamError, binarize, get_rng
from bct.utils import pick_four_unique_nodes_quickly
def randmio_und_connected(R, itr, seed=None):
    '''
    This function randomizes an undirected network, while preserving the
    degree distribution. The function does not preserve the strength
    distribution in weighted networks. The function also ensures that the
    randomized network maintains connectedness, the ability for every node
    to reach every other node in the network. The input network for this
    function must be connected.
    NOTE the changes to the BCT matlab function of the same name
    made in the Jan 2016 release
    have not been propagated to this function because of substantially
    decreased time efficiency in the implementation. Expect these changes
    to be merged eventually.
    Parameters
    ----------
    W : NxN np.ndarray
        undirected binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.
    seed : hashable, optional
        If None (default), use the np.random's global random state to generate random numbers.
        Otherwise, use a new np.random.RandomState instance seeded with the given value.
    Returns
    -------
    R : NxN np.ndarray
        randomized network
    eff : int
        number of actual rewirings carried out
    '''
    if not np.allclose(R, R.T):
        raise BCTParamError("Input must be undirected")

    # if number_of_components(R) > 1:
    #     raise BCTParamError("Input is not connected")
    rng = get_rng(seed)

    R = R.copy()
    n = len(R)
    i, j = np.where(np.tril(R))
    k = len(i)
    # itr = 2
    # maximum number of rewiring attempts per iteration
    max_attempts = np.round(n * k / (n * (n - 1)))
    # actual number of successful rewirings
    eff = 0

    for it in range(int(itr)):
        att = 0
        while att <= max_attempts:  # while not rewired
            rewire = True
            while True:
                e1 = rng.randint(k)
                e2 = rng.randint(k)
                while e1 == e2:
                    e2 = rng.randint(k)
                a = i[e1]
                b = j[e1]
                c = i[e2]
                d = j[e2]

                if a != c and a != d and b != c and b != d:
                    break  # all 4 vertices must be different

            if rng.random_sample() > .5:

                i.setflags(write=True)
                j.setflags(write=True)
                i[e2] = d
                j[e2] = c  # flip edge c-d with 50% probability
                c = i[e2]
                d = j[e2]  # to explore all potential rewirings

            # rewiring condition
            if not (R[a, d] or R[c, b]):
                # connectedness condition
                if not (R[a, c] or R[b, d]):
                    P = R[(a, d), :].copy()
                    P[0, b] = 0
                    P[1, c] = 0
                    PN = P.copy()
                    PN[:, d] = 1
                    PN[:, a] = 1
                    while True:
                        P[0, :] = np.any(R[P[0, :] != 0, :], axis=0)
                        P[1, :] = np.any(R[P[1, :] != 0, :], axis=0)
                        P *= np.logical_not(PN)
                        if not np.all(np.any(P, axis=1)):
                            rewire = False
                            break
                        elif np.any(P[:, (b, c)]):
                            break
                        PN += P
                # end connectedness testing

                if rewire:
                    R[a, d] = R[a, b]
                    R[a, b] = 0
                    R[d, a] = R[b, a]
                    R[b, a] = 0
                    R[c, b] = R[c, d]
                    R[c, d] = 0
                    R[b, c] = R[d, c]
                    R[d, c] = 0

                    j.setflags(write=True)
                    j[e1] = d
                    j[e2] = b  # reassign edge indices
                    eff += 1
                    break
            att += 1

    return R, eff

#%%

envelope_signal_bandpassed_bc_corrected = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/0_percentile.npz')

envelope_signal_bandpassed_bc_corrected_thresholded = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/98_percentile.npz')

def packaging_bands(signal_array):
    theta = signal_array['theta']
    alpha = signal_array['alpha']
    low_beta = signal_array['low_beta']
    high_beta = signal_array['high_beta']

    dic =  {'theta' : theta, 'alpha' : alpha, 'low_beta' : low_beta, 'high_beta' : high_beta}

    return dic

dic_of_envelope_signals_unthresholded = packaging_bands(envelope_signal_bandpassed_bc_corrected)
dic_of_envelope_signals_thresholded = packaging_bands(envelope_signal_bandpassed_bc_corrected_thresholded)

laplacian,_ = graph_setup.NNgraph()


subjects = 25
number_of_clusters = 5
fs = 125
pre_stim_in_samples = 25
post_stim_in_samples = 63
seconds_per_event = pre_stim_in_samples + post_stim_in_samples
regions = 360
events = np.load('/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events.npz')


video_duration = seconds_per_event

def smoothness_computation(band, laplacian):
    """The main function that does GFT, function-calls the temporal slicing, frequency summing, pre- post- graph-power accumulating 

    Args:
        band (array): Envelope band to use

    Returns:
        dict: Baseline-corrected ERD for all trials 
    """

    
    one = np.array(band).T # dim(one) = entire_video_duration x ROIs x subjects
    two = np.swapaxes(one,axis1=1,axis2=2) # dim (two) = entire_video_duration x subjects x ROIs

    signal = np.expand_dims(two,2) # dim (signal) = entire_video_duration x subjects x 1 x ROIs

    stage1 = np.tensordot(signal,laplacian,axes=(3,0)) # dim (laplacian) = (ROIs x ROIs).... dim (stage1) = same as dim (signal)

    signal_stage2 = np.swapaxes(signal,2,3) # dim(signal_stage2) = (entire_video_duration x subjects x ROIs x 1)

    assert np.shape(signal_stage2) == (video_duration, subjects, regions, 1)


    smoothness_roughness_time_series = np.squeeze( np.matmul(stage1,signal_stage2) ) # dim = entire_video_duration x subjects
    assert np.shape(smoothness_roughness_time_series) == (video_duration, subjects)
    
    return smoothness_roughness_time_series
# %%
smoothness_computed = dict()

for labels, signal in dic_of_envelope_signals_thresholded.items():
    placeholder_gsv = list()
    for event in range(number_of_clusters):
        signal_for_gsv = np.array(signal)[:, event, :, :]

        placeholder_gsv.append( smoothness_computation (  signal_for_gsv, laplacian))

    smoothness_computed[f'{   labels  }'] = placeholder_gsv
# %%
smoothness_computed_bootstapping = defaultdict(dict)

for labels, signal in dic_of_envelope_signals_thresholded.items():

    for i in tqdm.tqdm(range(1000)):
        _, adjacency = graph_setup.NNgraph()
        G = adjacency.numpy()

        randomized_graph, eff = randmio_und_connected(G, itr = 500, seed = i)
        randomized_graph_adjacency = randomized_graph
        
        # print(sum(sum(randomized_graph_adjacency * G)))
        # print('eff',eff)
        laplacian_stats = randomized_graph

        placeholder_gsv = list()

        for event in range(number_of_clusters):
            signal_for_gsv = np.array(signal)[:, event, :, :]
            placeholder_gsv.append( smoothness_computation(signal_for_gsv, laplacian = laplacian_stats) )
        
        assert np.shape(placeholder_gsv) == (number_of_clusters, seconds_per_event, subjects)
        smoothness_computed_bootstapping[f'{labels}'][f'{i}'] = placeholder_gsv
# %%
arr = np.array(list(smoothness_computed_bootstapping['high_beta'].values()))

s_ = np.mean( smoothness_computed['high_beta'][3], axis=1)
s_bs = np.mean( arr[:,3,:,:]  , axis=2).T
plt.plot(s_bs, color='r', label = 'null')
plt.plot(s_)
plt.xlabel('time (samples)')
# %%
np.shape(arr)
# %%
