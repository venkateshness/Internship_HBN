#%%
from os import stat
from typing_extensions import final
import fs
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu, rankdata, ttest_1samp
from sklearn.covariance import empirical_covariance
from tqdm import tqdm
from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image, plotting
import mne
from scipy.signal import butter, lfilter
import scipy
from tqdm import tqdm
#%%
empirical_activity = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_activation_sliced_for_all_subjects_all_bands.npz')['theta']
bced_activity = np.load('/users2/local/Venkatesh/Generated_Data/activity_stats/empirical_activity_native_BCed_theta.npz')['bced_activity']

# %%

# %%
# from mne.stats import (spatio_temporal_cluster_1samp_test,
#                        summarize_clusters_stc)
# fsave_vertices = [np.arange(20484)]
# from tqdm import tqdm
# def synthetic_data(trial):

#     cycles = 1
#     resolution = 10
#     np.random.seed(trial)

#     pre = np.random.randint(1,10)
#     post = 40 - resolution - pre
    


#     length = np.pi * 1 * cycles
#     wave = np.pad (np.sin(np.arange(0, length, length / resolution)), (pre, post)) *0.5
#     # noise = np.random.normal(0,1,resolution+pre+post)

#     target_snr_db = 20

#     sig_avg_watts = np.mean(wave)
#     sig_avg_db = 10 * np.log10(sig_avg_watts)

#     noise_avg_db = sig_avg_db - target_snr_db
#     noise_avg_watts = 10 ** (noise_avg_db / 10)

#     mean_noise = 0
#     noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(wave))

#     wave_noised = wave + noise_volts

#     # plt.plot( wave_noised )
#     return wave_noised

# subjects = 25
# events = 30
# vertices = 20484

# syn_data = list()

# for i in tqdm(range(events)):
#     vertices_level = list()
#     for v in range(vertices):
#         vertices_level.append(synthetic_data(i + v))

#     syn_data.append(vertices_level)


# %%
n_subjects = 25
n_events = 30
n_roi = 20484
n_surrogate = 19

second_in_sample = 63
total_no_of_events = '30_events'
samp_freq = 125
#%%
def baseline_correction(empirical_signal):
    empirical_activity_bc = list()

    for sub in tqdm(range(n_subjects)):
        event_level = list()
        
        for event in range(n_events):
            roi_level = list()
            
            for roi in range(n_roi):
                signal_sliced_bc = mne.baseline.rescale(
                        empirical_signal[sub, :, event, roi],
                        times=np.array(list(range(second_in_sample))) / samp_freq,
                        baseline=(None, 0.2),
                        mode="zscore",
                        verbose=False,
                    ) 

                plt.plot(signal_sliced_bc)


                plt.show()
                
                roi_level.append(signal_sliced_bc)
            
            event_level.append(roi_level)
        
        empirical_activity_bc.append(event_level)

    return empirical_activity_bc


bced_activity = baseline_correction(empirical_activity)

#%%
n_times = 37

def first_level_stats(empirical_one_band, band):
    # Step 2 : Test for effect of events

    tvalues_step2 = list()
    
    for sub in tqdm(range(n_subjects)):
        sub_wise_t = list()

        for roi in range(n_roi):
            sub_wise_t_t = list()

            for time in range(n_times):

                data = empirical_one_band[sub, :, roi, time]

                t = mne.stats.ttest_1samp_no_p(data)

                sub_wise_t_t.append(t)
        
            sub_wise_t.append(sub_wise_t_t)
            # print(roi)
        tvalues_step2.append(sub_wise_t)
        
    return tvalues_step2
    # Step 3 : Second level Model

def second_level_stats(test_stats):
    secondlevel_t = list()
    secondlevel_p = list()

    for time in tqdm(range(n_times)):
        roi_wise_t = list()
        roi_wise_p = list()

        for roi in range(n_roi):
            test_stats_sliced = test_stats[:, roi, time]
            t, p = scipy.stats.ttest_1samp(test_stats_sliced, popmean = 0)
            
            roi_wise_t.append(t)
            roi_wise_p.append(p)

        secondlevel_t.append(roi_wise_t)
        secondlevel_p.append(roi_wise_p)

    return secondlevel_t, secondlevel_p

  
#%%

#sub x events x time x vertices
pre_stim = 26
empirical_activity_sliced = np.array(bced_activity)[:, :, :, pre_stim : ]
# print(np.shape(empirical_activity_sliced))
first_level_t = first_level_stats(empirical_activity_sliced, 'theta')
# %%
# np.savez_compressed("/users2/local/Venkatesh/Generated_Data/activity_stats/first_level_test_stats.npz",first_level_t=first_level_t)
first_level_t = np.load("/users2/local/Venkatesh/Generated_Data/activity_stats/first_level_test_stats.npz")['first_level_t']
# %%

final_t, final_p = second_level_stats(np.array(first_level_t))

# %%
# from nilearn import datasets

# fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
# _100ms_in_samples= 12
# for i in range(_100ms_in_samples, 2 * _100ms_in_samples):
#     lh_one_t = np.array(final_t)[i,10242:]
#     lh_one_p = np.array(final_p)[i,10242:]
#     pval = 0.001 
#     df = 25 - 1  # degrees of freedom for the test
#     thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
    


#     plot = plotting.plot_surf_stat_map(
#         surf_mesh = fsaverage.infl_left, stat_map= lh_one_t, bg_map=fsaverage.sulc_left, threshold=thresh, views='lateral',
#         title =f'{(i/samp_freq)*1000} ms PO' )
#     plt.show()


# %%
pre_stim = 26

empirical_activity_strong_isc = empirical_activity[:,pre_stim:,:,:]

np.mean(empirical_activity_strong_isc, axis = (0, 2)).shape

# %%
final_t

# %%
