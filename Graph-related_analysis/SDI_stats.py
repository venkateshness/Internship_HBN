#%%
import nilearn
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import wilcoxon, mannwhitneyu, rankdata, ttest_1samp
from tqdm import tqdm
from statsmodels.stats.multitest import fdrcorrection

from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image, plotting
import mne

total_no_of_events = '30_events'
sdi = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_new/SDI/{total_no_of_events}.npz')

surrogate_SDI = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_new/SDI/surrogate_SDI_baseline_vanilla.npz')
empirical_SDI = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_new/SDI/SDI_baseline_vanilla.npz')



def stats(band, event_group):
    clusters = np.load(
    f"/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events_{total_no_of_events}.npz"
    )
    a = list()

    a.append(clusters['0'])
    a.append(clusters['1'])
    a.append(clusters['2'])

    index = list()
    for i in sorted(np.hstack(a)):

        if i in a[0]:
            index.append(0)
        if i in a[1]:
            index.append(1)
        if i in a[2]:
            index.append(2)

    empirical_one_band = empirical_SDI[f'{band}'][np.where(np.array(index) ==event_group)[0]]
    surrogate_one_band = surrogate_SDI[f'{band}'][np.where(np.array(index) ==event_group)[0]]


    n_subjects = 25
    n_events = len(np.where(np.array(index) ==event_group)[0])
    n_roi = 360
    n_surrogate = 19

    test_stats = list()

    for subject in tqdm(range(n_subjects)):
        event_level_p = list()


        for event in range(n_events):
            roi_level_p = list()

            
            for roi in range(n_roi):

                data_empirical = empirical_one_band[event, subject, roi]
                data_surrogate = surrogate_one_band[event, :, subject, roi]
                
                stat_test = sum(rankdata(np.abs(data_empirical - data_surrogate))*np.sign(data_empirical - data_surrogate))
                stat_test_normalized = stat_test / n_surrogate
                

                roi_level_p.append(stat_test_normalized)
            
            event_level_p.append(roi_level_p)
        
        test_stats.append(event_level_p)

    # Step 2 : Test for effect of events


    pvalues_step2 = list()
    tvalues_step2 = list()

    for sub in range(n_subjects):
        sub_wise_p = list()
        sub_wise_t = list()

        for roi in range(n_roi):
            data = np.array(test_stats)[sub, :, roi]

            t, p = ttest_1samp(data, popmean = 0)
            if t == np.inf:
                t=0
            if t == -np.inf:
                t=0

            if p == np.inf:
                p=1
            if p == -np.inf:
                p=1

            sub_wise_p.append(p)
            sub_wise_t.append(t)
        
        pvalues_step2.append(sub_wise_p)
        tvalues_step2.append(sub_wise_t)



    # Step 3 : Second level Model


    secondlevel_t, secondlevel_p, _ = mne.stats.permutation_t_test(np.array(np.array(tvalues_step2)))


    path_Glasser = "/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz"

    mnitemp = fetch_icbm152_2009()


    for i in range(1):

        thresholded_tvals = (secondlevel_p < 0.05) *  np.array(secondlevel_t) #(secondlevel_p < 0.05) * 
        print(np.max(thresholded_tvals))
        print(np.min(thresholded_tvals))
        # print(np.max(thresholded_tvals), np.min(thresholded_tvals))
        signal = np.expand_dims(thresholded_tvals, axis =(1,2))

        U0_brain = signals_to_img_labels(signal, path_Glasser, mnitemp["mask"])

        plot = plotting.plot_img_on_surf(
            U0_brain, title = f'2nd level; {band}; BL; tvalues; perm-corrected; event_G : {event_group+1}', threshold= 0.1, vmax = 10)

        # U0_brain.to_filename(f'/homes/v20subra/S4B2/Graph-related_analysis/2nd_level_maps/PO/2nd_level_map_perm_corrected_stat_{band}.nii.gz')
        plt.show()



for i in range(3):
    stats('alpha', i)
    stats('theta', i)
    stats('low_beta',i)
    stats('high_beta', i)

# %%
from nilearn import image
from nilearn.datasets import fetch_icbm152_2009
from nilearn.regions import signals_to_img_labels
from nilearn import image, plotting

mnitemp = fetch_icbm152_2009()

path_Glasser = "/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz"
band =['theta']

for i in band:
    img = image.load_img(f'/homes/v20subra/S4B2/Graph-related_analysis/2nd_level_maps_thresholded_at_0.05/PO-BL/2nd_level_map_perm_corrected_stat_{i}.nii.gz')

    fig,_ = plotting.plot_img_on_surf(img, threshold = 0.1, vmax = 10)
    fig.savefig('/homes/v20subra/S4B2/Graph-related_analysis/2nd_level_maps_thresholded_at_0.05/PO-BL/theta.svg', dpi=700)

# %%
