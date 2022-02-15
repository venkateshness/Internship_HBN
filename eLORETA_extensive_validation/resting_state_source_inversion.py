#%%
#Packages Importing
from asyncio import events
from curses.ascii import ETB
from unicodedata import name
import mne
import pathlib
from mne.externals.pymatreader import read_mat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d

from mne.minimum_norm import read_inverse_operator, compute_source_psd_epochs
from mne.datasets import sample



from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs

import os.path as op
# Data-loading

def fire_2(raw,events,fwd_model):

    ##################
    #epochs###########
    ##################
    
    epochs = mne.Epochs(raw, events, [20,30,90], tmin=0, tmax=20,preload=True,baseline=(0,None))
    epochs_resampled = epochs.resample(250)# Downsampling to 250Hz
    print(np.shape(epochs_resampled.load_data())) # Sanity Check


    ##################
    ###Noise Covariance
    ##################
    rand = np.random.randint(1,5000,size=500)
    np.random.seed(55)
    cov = mne.EpochsArray(epochs_resampled['20'][0].get_data()[:,:,rand],info=raw.info)

    covariance = mne.compute_covariance(
    cov, method='auto')

    ##################
    ###Source Inversion - forward modeling
    #################
    

    ###Inverse operator
    inverse_operator = make_inverse_operator(raw.info, fwd_model, covariance)


    #PSD at source for the occipital electrodes
    method ='eLORETA'
    snr = 3.
    lambda2 = 1. / snr ** 2
    evoked  = epochs_resampled['20'][1:].average()
    data_path = sample.data_path()
    label_name ='Aud-rh.label' # Have to use 2 labels at the same, but will deal with this later
    fname_label = data_path + '/MEG/sample/labels/%s' % label_name
    label_name2 = 'Aud-lh.label'
    fname_label2 = data_path + '/MEG/sample/labels/%s' % label_name2
    label = mne.read_label(fname_label)
    label2 = mne.read_label(fname_label2)
    bihemi = mne.BiHemiLabel(label,label2)

    stcs = compute_source_psd_epochs(epochs_resampled['20'][1:], inverse_operator, lambda2=lambda2,
                                     method=method, fmin=0, fmax=40, label=bihemi,
                                     verbose=True)

    # stcs2 = compute_source_psd_epochs(epochs_resampled['30'][1:], inverse_operator, lambda2=lambda2,
    #                                 method=method, fmin=0, fmax=40, label=bihemi,
    #                                 verbose=True)
    # print(len(stcs))
    # print(len(stcs2))
    # stcs_averaged_eyes_open = np.sum(stcs)/len(stcs)
    # stcs_averaged_eyes_closed = np.sum(stcs2)/len(stcs2)
    rstate_stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2, method=method)

    
    #a = stats.ttest_rel(np.average(stcs_averaged_eyes_closed.data[:,160:262],axis=1),np.average(stcs_averaged_eyes_open.data[:,160:262],axis=1))
    return rstate_stc 
    #""" stcs_averaged_eyes_open, stcs_averaged_eyes_closed """





fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)


subject = 'fsaverage' # Subject ID for the MRI-head transformation
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
source_space = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif') 
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')


from scipy import stats

subjects = ['NDARCD401HGZ','NDARDX770PJK', 'NDARGY054ENV', 'NDARMR242UKQ', 'NDARRD720XZK', 
'NDARTR840XP1', 'NDARXJ696AMX', 'NDARYY218AGA', 'NDARZP564MHU']
vals = list()
eyes_open = dict()
eyes_closed = dict()

#%%


for i in range(9):
    raw,events = mne.io.read_raw_fif(f'/users/local/Venkatesh/Generated_Data/importing/resting_state/{subjects[i]}/raw.fif'),np.load(f'/users/local/Venkatesh/Generated_Data/importing/resting_state/{subjects[i]}/events.npz')['resting_state_events']

    if i==0:
        fwd_model = mne.make_forward_solution(raw.info, trans=trans, src=source_space, bem=bem, eeg=True, mindist=5.0)
    function_call = fire_2(raw,events,fwd_model)
    vals.append(function_call)




print(vals)

#%%
# np.save('eyes_closed_20o_twosecond_randomly_reg',eyes_closed)
# np.save('eyes_open_20o_twosecond_randomly_reg',eyes_open)
import numpy as np
with np.load(f"/homes/v20subra/S4B2/GSP/hcp/atlas.npz") as dobj:
    atlas = dict(**dobj)

# %%
def averaging_by_parcellation(sub):
    l =list()
    for i in list(set(atlas['labels_L']))[:-1]:
        l.append(np.mean(sub.data[10242:][np.where(i== atlas['labels_L'])],axis=0))

    for i in list(set(atlas['labels_R']))[:-1]:
        l.append(np.mean(sub.data[:10242][np.where(i== atlas['labels_R'])],axis=0))
    return l
# %%

rstate_parcellated = [np.array(averaging_by_parcellation(vals[0])),np.array(averaging_by_parcellation(vals[1])), 
       np.array(averaging_by_parcellation(vals[2])), np.array(averaging_by_parcellation(vals[3])), 
       np.array(averaging_by_parcellation(vals[4])), np.array(averaging_by_parcellation(vals[5])),
       np.array(averaging_by_parcellation(vals[6])), np.array(averaging_by_parcellation(vals[7])), 
       np.array(averaging_by_parcellation(vals[8]))]




# %%
np.savez_compressed('/users/local/Venkatesh/Generated_Data/noise_baseline_properly-done_eloreta/rstate_source_space_parcellated',rstate_source_space_parcellated=rstate_parcellated)
# %%


#############################
#high db vs low db auditory estimation
#############################




