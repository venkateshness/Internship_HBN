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

os.chdir("/homes/v20subra/S4B2/")

from Modular_Scripts import graph_setup

importlib.reload(graph_setup)
from nilearn import datasets, plotting, image, maskers
import networkx as nx
from collections import defaultdict
from tqdm import tqdm

from bct.utils import BCTParamError, binarize, get_rng
from bct.utils import pick_four_unique_nodes_quickly


def randmio_und_connected(R, itr, seed=None):

    """
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
    """
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

            if rng.random_sample() > 0.5:

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


envelope_signal_bandpassed_bc_corrected = np.load(
    f"/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/0_percentile.npz"
)

envelope_signal_bandpassed_bc_corrected_thresholded = np.load(
    f"/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/98_percentile.npz"
)


def packaging_bands(signal_array):
    theta = signal_array["theta"]
    alpha = signal_array["alpha"]
    low_beta = signal_array["low_beta"]
    high_beta = signal_array["high_beta"]

    dic = {"theta": theta, "alpha": alpha, "low_beta": low_beta, "high_beta": high_beta}

    return dic


dic_of_envelope_signals_unthresholded = packaging_bands(
    envelope_signal_bandpassed_bc_corrected
)
dic_of_envelope_signals_thresholded = packaging_bands(
    envelope_signal_bandpassed_bc_corrected_thresholded
)

laplacian, _ = graph_setup.NNgraph()


subjects = 25
number_of_clusters = 5
fs = 125
pre_stim_in_samples = 25
post_stim_in_samples = 63
seconds_per_event = pre_stim_in_samples + post_stim_in_samples
regions = 360
events = np.load("/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events.npz")


video_duration = seconds_per_event


def smoothness_computation(band, laplacian):
    """The main function that does GFT, function-calls the temporal slicing, frequency summing, pre- post- graph-power accumulating

    Args:
        band (array): Envelope band to use

    Returns:
        dict: Baseline-corrected ERD for all trials
    """
    for_all_subjects = list()

    for subject in range(subjects):
        subject_wise = list()
        for timepoints in range(seconds_per_event):
            sig = band[subject, :, timepoints]

            stage1 = np.matmul(sig.T, laplacian)

            final = np.matmul(stage1, sig)
            subject_wise.append(final)
        for_all_subjects.append(subject_wise)

    data_to_return = np.array(for_all_subjects).T
    assert np.shape(data_to_return) == (video_duration, subjects)
    return data_to_return


smoothness_computed = dict()

for labels, signal in dic_of_envelope_signals_thresholded.items():
    placeholder_gsv = list()
    for event in range(number_of_clusters):
        signal_for_gsv = np.array(signal)[:, event, :, :]

        placeholder_gsv.append(smoothness_computation(signal_for_gsv, laplacian))

    smoothness_computed[f"{   labels  }"] = placeholder_gsv

#%%
smoothness_computed_bootstapping = defaultdict(dict)
from joblib import Parallel, delayed

import multiprocessing

NB_CPU = multiprocessing.cpu_count()

rewiring_proportion_investigation = defaultdict(dict)
n_trials = 1000

_, adjacency = graph_setup.NNgraph()
total_n_edges = sum(sum(adjacency.numpy()))

_10_total_n_edges = total_n_edges / 10
_to_swap_10_n_edges = int(_10_total_n_edges / 4)

# def permute_ROIs(time_series):
#     assert np.shape(time_series) == (subjects, number_of_clusters, regions, seconds_per_event)
#     subject_wise = list()
#     for subject in range(subjects):
#         event_wise = list()
#         for event in range(number_of_clusters):
#             signal = time_series[subject, event]
#             assert np.shape(signal) == (regions, seconds_per_event)

#             perm_idx = np.random.permutation(360)

#             signal_permuted = signal[perm_idx]
#             event_wise.append(signal_permuted)

#         subject_wise.append(event_wise)

#     return subject_wise

# dic_of_envelope_signals_thresholded_permuted = defaultdict(dict)

# for labels, signals in dic_of_envelope_signals_thresholded.items():
#     dic_of_envelope_signals_thresholded_permuted[f'{labels}'] = permute_ROIs(signals)


def surrogate_signal(signal, eig_vector_original, is_sc_ignorant):
    all_subject_reconstructed = list()

    random_signs = np.round(np.random.rand(np.shape(eig_vector_original)[1]))
    random_signs[random_signs == 0] = -1
    random_signs_diag = np.diag(random_signs)

    for subject in range(subjects):
        subject_wise_signal = np.array(signal)[subject]

        if is_sc_ignorant:
            eig_vectors_manip = np.matmul(
                np.fliplr(eig_vector_original), random_signs_diag
            )

        else:
            eig_vectors_manip = np.matmul(eig_vector_original, random_signs_diag)

        spatial = np.array(np.matmul(subject_wise_signal.T, eig_vectors_manip))

        signal_reconstructed = np.matmul(eig_vector_original, spatial.T)
        assert np.shape(signal_reconstructed) == (regions, seconds_per_event)

        all_subject_reconstructed.append(signal_reconstructed)
    return np.array(all_subject_reconstructed)


# 1. Degree re-calculation
# 2. Surrogated Laplacian
# 3. Permute ROIs
# 4. Surrogate signal


def surrogate_laplacian(eig_vectors, eig_vals, is_sc_ignorant):

    random_signs = np.round(np.random.rand(np.shape(eig_vectors)[1]))
    random_signs[random_signs == 0] = -1
    random_signs_diag = np.diag(random_signs)
    eig_vals_diag = np.diag(eig_vals)

    if is_sc_ignorant:
        eig_vectors_manip = np.matmul(np.fliplr(eig_vectors), random_signs_diag)

    else:
        eig_vectors_manip = np.matmul(eig_vectors, random_signs_diag)

    eig_vectors_inv = np.linalg.inv(eig_vectors_manip)
    laplacian_reconstructed = np.matmul(
        np.matmul(eig_vectors_manip, eig_vals_diag), eig_vectors_inv
    )

    return laplacian_reconstructed


rewiring_edges = [_to_swap_10_n_edges, _to_swap_10_n_edges * 2, _to_swap_10_n_edges * 5]

for n_iter in rewiring_edges:
    for labels, signal in dic_of_envelope_signals_thresholded.items():
        if labels == "theta":
            # _, adjacency = graph_setup.NNgraph()
            # G = adjacency.numpy()

            adjacency = networkx.erdos_renyi_graph(360, 0.5)
            G = networkx.to_numpy_array(adjacency)

            def parallelisation(i):

                randomized_graph_adj, eff = randmio_und_connected(G, itr=n_iter, seed=i)
                degree = np.diag(np.sum(randomized_graph_adj, axis=0))

                laplacian_for_stats = degree - randomized_graph_adj
                [eig_vals, eig_vectors] = np.linalg.eigh(laplacian_for_stats)

                placeholder_gsv = list()
                for event in range(number_of_clusters):
                    signal_for_gsv = np.array(signal)[:, event, :, :]
                    # signal_reconstructed = surrogate_signal(signal_for_gsv, eig_vectors, is_sc_ignorant=True)

                    placeholder_gsv.append(
                        smoothness_computation(
                            signal_for_gsv, laplacian=laplacian_for_stats
                        )
                    )

                assert np.shape(placeholder_gsv) == (
                    number_of_clusters,
                    seconds_per_event,
                    subjects,
                )

                return placeholder_gsv

            placeholder_gsv = Parallel(n_jobs=NB_CPU - 1, max_nbytes=None)(
                delayed(parallelisation)(j) for j in tqdm(range(n_trials))
            )
            smoothness_computed_bootstapping[f"{labels}"] = placeholder_gsv

    rewiring_proportion_investigation[f"{n_iter}"] = smoothness_computed_bootstapping[
        "theta"
    ]

# %%
a = 1
b = 3
c = 1

fig = plt.figure(figsize=(15, 5))
n_graph_edges = 3300

rewiring_edges_labels = ["10", "20", "50"]
for subplot_label, subplot_signal in rewiring_proportion_investigation.items():
    plt.subplot(a, b, c)
    plt.style.use("fivethirtyeight")
    assert np.shape(subplot_signal) == (
        n_trials,
        number_of_clusters,
        seconds_per_event,
        subjects,
    )

    subplot_signal_mean_bs = np.mean(np.array(subplot_signal)[:, 0, :, :], axis=2)
    plt.plot(subplot_signal_mean_bs.T, c="r")

    signal = smoothness_computed["theta"][0]
    signal_mean = np.mean(signal, axis=1)
    signal_std = scipy.stats.sem(signal, axis=1)

    upper_bound = signal_mean + signal_std
    lower_bound = signal_mean - signal_std

    plt.plot(signal_mean, c="deepskyblue", label="original")
    plt.fill_between(
        range(seconds_per_event),
        upper_bound,
        lower_bound,
        alpha=0.2,
        color="deepskyblue",
    )

    plt.title(f"Rewiring prop = {rewiring_edges_labels[c-1]}%")
    plt.xticks(np.arange(0, 88, 25), labels=["-200", "0", "200", "400"])
    plt.axvline(pre_stim_in_samples, color="g", linestyle="-.")
    plt.xlabel("time (ms)")
    plt.ylabel("relative variation")
    plt.axvspan(0, pre_stim_in_samples, color="orange", alpha=0.2)
    plt.legend()
    c += 1
fig.suptitle("GSV / graph-ignorant Surrogate Signal. FC graph/ Theta/ event 0")


# %%
# 1. Degree re-calculation
# 2. Surrogated Laplacian
# 3. Permute ROIs
# 4. Surrogate signal

# %%
