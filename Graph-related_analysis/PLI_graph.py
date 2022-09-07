#%%
import numpy as np
import mne_connectivity
import os
from nilearn import plotting

#%%
regions = 360

total_subjects = ['NDARAD481FXF','NDARBK669XJQ',
'NDARCD401HGZ','NDARDX770PJK',
'NDAREC182WW2','NDARGY054ENV',
'NDARHP176DPE','NDARLB017MBJ',
'NDARMR242UKQ','NDARNT042GRA',
'NDARRA733VWX','NDARRD720XZK',
'NDARTR840XP1','NDARUJ646APQ',
'NDARVN646NZP','NDARWJ087HKJ',
'NDARXB704HFD','NDARXJ468UGL',
'NDARXJ696AMX','NDARXU679ZE8',
'NDARXY337ZH9','NDARYM257RR6',
'NDARYY218AGA','NDARYZ408VWW','NDARZB377WZJ']

subjects_data_available_for =list()

for i in range(1,25):
     if (os.path.isfile(f'/users2/local/Venkatesh/HBN/CPAC_preprocessed/sub-{total_subjects[i]}_ses-1/functional_to_standard/_scan_rest/_selector_CSF-2mmE-M_aC-CSF+WM-2mm-DPC5_M-SDB_P-2_BP-B0.01-T0.1_C-S-1+2-FD-J0.5/bandpassed_demeaned_filtered_antswarp.nii.gz')):
         subjects_data_available_for.append(total_subjects[i])

idx_for_the_existing_subjects = np.argwhere(np.isin(total_subjects, subjects_data_available_for)).ravel()
#%%
def pli_graph(band, fmin, fmax):
    stc = np.load("/users2/local/Venkatesh/Generated_Data/25_subjects_new/rs_STC_envelope.npz")[f"{band}"]
    stc_filtered = stc[idx_for_the_existing_subjects]
    weights = list()

    for subject in range(len(stc_filtered)):

        pli = mne_connectivity.phase_slope_index(np.expand_dims(stc_filtered[subject],0), sfreq= 125,fmin = fmin, fmax = fmax)

        pli_reshaped = np.reshape(pli.get_data(), (regions, regions))
        pli_reshaped_full = pli_reshaped + pli_reshaped.T
        weights.append(pli_reshaped_full)
    return weights

theta_graph = pli_graph('theta', 4, 8)
alpha_graph = pli_graph('alpha', 8, 13)
low_beta_graph = pli_graph('low_beta', 13, 20)
high_beta_graph = pli_graph('high_beta', 20, 30)

# %%


# %%
