#%%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import mne

envelope_signal_bandpassed = np.load(
    "/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/0_percentile.npz",
    mmap_mode="r",
)
alpha = envelope_signal_bandpassed["alpha"]
theta = envelope_signal_bandpassed["theta"]
low_beta = envelope_signal_bandpassed["low_beta"]
high_beta = envelope_signal_bandpassed["high_beta"]

dict_of_unthresholded_signals_for_all_bands = dict()
dict_of_unthresholded_signals_for_all_bands = {
    "theta": theta,
    "alpha": alpha,
    "low_beta": low_beta,
    "high_beta": high_beta,
}

duration = 21250
subjects = 25
regions = 360
event_type = "19_events"
total_events = 19

clusters = np.load(
    f"/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events_{event_type}.npz"
)

fs = 125
pre_stim = 25
post_stim = 63
second_in_sample = pre_stim + post_stim
number_of_clusters = 3


def baseline_correction(signal):
    """Subject-wise signal to apply the baseline correction on. Dim =  regions x seconds_in_sample

    Args:
        signal (array): epoch array per subject

    Returns:
        array: Baseline-corrected signal
    """
    region, _ = np.shape(signal)
    baseline_corr = list()

    for r in range(region):

        signal_1s_window = signal[r]
        signal_baseline = signal_1s_window[:pre_stim]

        mean_signal_baseline = np.mean(signal_baseline)
        baseline_corr.append(
            (signal_1s_window - mean_signal_baseline) / np.mean(mean_signal_baseline)
        )

    return baseline_corr


# def standardisation(signal):
#     """Standardisation of the signal. Dim = subject x regions x time

#     Args:
#         signal (array): epoch array for all subjects

#     Returns:
#         array: standardised signal
#     """
#     first, second, third = np.shape(signal)
#     signal_standardised_first_dim = list()

#     for first_dim in range(first):
#         signal_standardised_second_dim = list()

#         for second_dim in range(second):

#             signal_1s_window = signal[first_dim,second_dim]
#             signal_standardised_second_dim.append((signal_1s_window - np.mean(signal_1s_window))/ np.std(signal_1s_window))

#         signal_standardised_first_dim.append(signal_standardised_second_dim)

#     return signal_standardised_first_dim

dict_of_sliced_bc_averaged = defaultdict(dict)


def slicing_averaging(band):
    """Slicing and Averaging of the cortical signal according to cluster groups. Dim = subjects x regions x time

    Args:
        band (string): frequency bands
    """
    subject_level = list()

    for subject in range(subjects):  # Subject-wise

        cluster_level = list()
        for event_label, event_time in clusters.items():  # Cluster-wise

            signal = dict_of_unthresholded_signals_for_all_bands[f"{band}"][
                subject, :, :
            ]
            event_level = list()

            for event in event_time:  # event-wise
                signal_sliced_non_bc = signal[
                    :, event * fs - pre_stim : event * fs + post_stim
                ]

                signal_sliced_bc = mne.baseline.rescale(
                    signal_sliced_non_bc,
                    times=np.array(list(range(second_in_sample))) / fs,
                    baseline=(None, 0.1),
                    mode="zscore",
                    verbose=False,
                )  # apply baseline correction
                # event_level.append(signal_sliced_bc)
                cluster_level.append(signal_sliced_bc)
                

            # assert np.shape(event_level) == (len(event_time), regions, second_in_sample)
        
        # cluster_level = standardisation(np.array(cluster_level))
        assert np.shape(cluster_level) == (
            total_events,
            regions,
            second_in_sample,
        )
        subject_level.append(cluster_level)

    dict_of_sliced_bc_averaged[f"{band}"] = subject_level


for labels, signal in dict_of_unthresholded_signals_for_all_bands.items():
    slicing_averaging(labels)
    assert np.shape(dict_of_sliced_bc_averaged[f"{labels}"]) == (
        subjects,
        total_events,
        regions,
        second_in_sample,
    )


# Thresholding the sliced signal

dict_of_thresholded_signals_for_all_bands = dict()
percentile = [0]

for perc in percentile:  # each percentile

    for labels, signal in dict_of_sliced_bc_averaged.items():  # band-wise

        all_subject = list()

        for subject in range(subjects):
            per_subject = list()

            for event_group in range(total_events):

                the_array_of_interest_new = np.array(signal)[subject, event_group, :, :]
                percentile_value = np.percentile(
                    the_array_of_interest_new, perc, axis=0
                )

                ready_array = (
                    the_array_of_interest_new
                    * (the_array_of_interest_new > percentile_value)
                ).T
                per_subject.append(ready_array)

            assert np.shape(per_subject) == (
                total_events,
                second_in_sample,
                regions,
            )
            all_subject.append(per_subject)

        all_subject_swapped = np.swapaxes(all_subject, 2, 3)

        assert np.shape(all_subject_swapped) == (
            subjects,
            total_events,
            regions,
            second_in_sample,
        )
        
        dict_of_thresholded_signals_for_all_bands[f"{labels}"] = all_subject_swapped
    print("writing")

    np.savez(
        f"/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/{event_type}/{perc}_percentile_with_zscore_events_wise",
        **dict_of_thresholded_signals_for_all_bands,
    )

# %%
