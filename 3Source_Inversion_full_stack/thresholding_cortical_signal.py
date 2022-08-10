#%%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


envelope_signal_bandpassed = np.load(
    '/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/eloreta_cortical_signal_thresholded/0_percentile.npz', mmap_mode='r')
alpha = envelope_signal_bandpassed['alpha']
theta = envelope_signal_bandpassed['theta']
low_beta = envelope_signal_bandpassed['low_beta']
high_beta = envelope_signal_bandpassed['high_beta']

dict_of_unthresholded_signals_for_all_bands = dict()
dict_of_unthresholded_signals_for_all_bands = {'theta':theta, 'alpha': alpha, 'low_beta':low_beta, 'high_beta':high_beta}

duration = 21250
subjects = 25
regions = 360
# #%%
events = np.load('/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events.npz')
dict_of_unthresholded_signals_for_all_bands_sliced_and_averaged = defaultdict(dict)
fs = 125
pre_stim = 62
post_stim = 63
second_in_sample = pre_stim + post_stim
number_of_clusters = 5


def slicing_averaging(band):
    subject_level = list()

    for subject in range(subjects):
        cluster_level = list()
        for event_label, event_time in events.items():
            signal = dict_of_thresholded_signals_for_all_bands[f'{band}'][subject,    :,  :]
    
            event_level = list()
            for event in event_time:
                event_level.append(signal[:,   event * fs - pre_stim : event * fs + post_stim])
            

            assert np.shape(event_level) == (len(event_time), regions, second_in_sample)
            cluster_level.append(np.mean(event_level, axis=0))
        
        assert np.shape(cluster_level) == (number_of_clusters, regions, second_in_sample)
        subject_level.append(cluster_level)

    dict_of_unthresholded_signals_for_all_bands_sliced_and_averaged[f'{band}'] = subject_level


for labels, signal in  dict_of_unthresholded_signals_for_all_bands.items():
    slicing_averaging(labels)
    assert np.shape(dict_of_unthresholded_signals_for_all_bands_sliced_and_averaged[f'{labels}']) == (subjects, number_of_clusters, regions, second_in_sample)
#%%
dic_of_ERD  = dict()

def erd_setup(band):
    all_subject = list()

    for subject in range(subjects):
        event_level = list()
        for event in range(number_of_clusters):

            signal = np.array(dict_of_unthresholded_signals_for_all_bands_sliced_and_averaged[f'{band}'])[subject, event,:,:]
            mean_signal = np.expand_dims(   np.mean(signal[:, : pre_stim], axis = 1),   axis = 1)
            std_signal = np.expand_dims(   np.std(signal[:, : pre_stim], axis = 1),   axis = 1)


            baseline_corrected = np.subtract(signal,mean_signal) / std_signal
            event_level.append(baseline_corrected)
        all_subject.append(event_level)

    dic_of_ERD[f'{band}'] = all_subject

for labels, signal in dict_of_unthresholded_signals_for_all_bands_sliced_and_averaged.items():
    erd_setup(labels)
    assert np.shape(dic_of_ERD[f'{labels}']) == (subjects, number_of_clusters, regions, second_in_sample)


#%%
dict_of_thresholded_signals_for_all_bands = dict()
percentile = [98, 95, 90, 50, 0]
for perc in percentile:
    for labels, signal in dic_of_ERD.items():
        all_subject = list()
        
        for subject in range(subjects):
            per_subject = list()
            
            for event_group in range(number_of_clusters):
                per_event = list()
                for sample in range(second_in_sample):
                    the_array_of_interest = np.array(signal)[subject, event_group, :, sample]
                    percentile_value = np.percentile (the_array_of_interest,  perc)
                    index_where_its_met = np.where (the_array_of_interest >   percentile_value)[0]
                    array_of_zeros = np.zeros( (regions,) )
                    array_of_zeros[index_where_its_met] = the_array_of_interest[index_where_its_met]
                    assert np.shape(array_of_zeros) == (regions,)

                    per_event.append(array_of_zeros)
            
                assert np.shape(per_event) == (second_in_sample, regions)

                per_subject.append(per_event)
        
            assert np.shape(per_subject) == (number_of_clusters, second_in_sample, regions)
            all_subject.append(per_subject)

        all_subject_swapped = np.swapaxes(all_subject, 2, 3)

        assert np.shape(all_subject_swapped) == (subjects, number_of_clusters, regions, second_in_sample)

        dict_of_thresholded_signals_for_all_bands[f'{labels}'] = all_subject_swapped

    np.savez(f'/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/{perc}_percentile',**dict_of_thresholded_signals_for_all_bands)

#%%




#%%

# import mpl_interactions.ipyplot as iplt
# import plotly.express as px

# # %matplotlib notebook
# video_duration = 21250
# band = 'alpha'
# percentile = 98
# data = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/eloreta_cortical_signal_thresholded/{percentile}_percentile.npz')[band]
# events = np.load('/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events.npz')
# fs = 125
# pre_stim = 62
# post_stim = 63
# events['0']
# def axvspanning():
#     color_groups = ['darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgrey']
#     for event_label, event_time in events.items():
#         for seconds in event_time:
#             fig.add_vrect(x0 = seconds * fs - pre_stim, x1 = seconds * fs + post_stim,  line_width=0, fillcolor = color_groups[int(event_label)], opacity=0.2)

# plt.figure(figsize=(25,5))
# signal_to_plot = np.mean(data[:,np.where(np.array(match)==0)[0],:], axis = 1)
# mean_signal = np.mean(signal_to_plot, axis = 0)
# sem_signal = sem(signal_to_plot, axis=0)
# import plotly.graph_objects as go

# plt.style.use('fivethirtyeight')
# fig = px.line(mean_signal.T)
# lower = mean_signal - sem_signal
# upper = mean_signal + sem_signal
# fig.add_trace(go.Scatter (x = np.array(range(video_duration)), y = upper, marker=dict(color="#444"),name = 'SEM upper bound',
#         line=dict(width=0),
#         mode='lines',
#         fillcolor='rgba(68, 68, 68, 0.3)',
#         showlegend=True))

# fig.add_trace(go.Scatter (x = np.array(range(video_duration)), y = lower, marker=dict(color="#444"), name = 'SEM = subjects',
#         line=dict(width=0),
#         mode='lines',
#         fillcolor='rgba(68, 68, 68, 0.3)',
#         fill='tonexty',
#         showlegend=True))
# fig.update_layout(
#     title=f"cortical signal / Visual ROIs averaged - {band} @ {percentile}",
#     xaxis_title="Samples @ 125/second",
#     yaxis_title="eLORETA value",
#     # legend_title="Legend Title",
#     font=dict(
#         family="Courier New, monospace",
#         size=18,
#         color="RebeccaPurple"
#     )
# )
# axvspanning()
# fig.write_html(f'/homes/v20subra/S4B2/3Source_Inversion_full_stack/Validation_cortical_thresholding/{percentile}_{band}.html')
# # %%
# lower
# # %%
# mean_signal + sem_signal
# # %%

# %%
