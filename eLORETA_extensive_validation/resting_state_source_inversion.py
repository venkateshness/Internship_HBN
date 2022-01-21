#%%
#Packages Importing
from asyncio import events
from curses.ascii import ETB
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
    epochs_resampled = epochs.resample(250)[3:5] # Downsampling to 250Hz
    print(np.shape(epochs_resampled.load_data())) # Sanity Check


    ##################
    ###Noise Covariance
    ##################
    covariance = mne.compute_covariance(
    epochs_resampled['20'][0], method=['shrunk', 'empirical'])

    ##################
    ###Source Inversion - forward modeling
    #################
    

    ###Inverse operator
    inverse_operator = make_inverse_operator(raw.info, fwd_model, covariance)


    #PSD at source for the occipital electrodes
    method ='eLORETA'
    snr = 3.
    lambda2 = 1. / snr ** 2

    data_path = sample.data_path()
    label_name ='Vis-rh.label' # Have to use 2 labels at the same, but will deal with this later
    fname_label = data_path + '/MEG/sample/labels/%s' % label_name
    label_name2 = 'Vis-lh.label'
    fname_label2 = data_path + '/MEG/sample/labels/%s' % label_name2
    label = mne.read_label(fname_label)
    label2 = mne.read_label(fname_label2)
    bihemi = mne.BiHemiLabel(label,label2)

    stcs = compute_source_psd_epochs(epochs_resampled['20'], inverse_operator, lambda2=lambda2,
                                    method=method, fmin=0, fmax=40, label=bihemi,
                                    verbose=True)

    stcs2 = compute_source_psd_epochs(epochs_resampled['30'], inverse_operator, lambda2=lambda2,
                                    method=method, fmin=0, fmax=40, label=bihemi,
                                    verbose=True)
    print(stcs)
    stcs_averaged_eyes_open = stcs
    stcs_averaged_eyes_closed = stcs2
    
    a = stats.ttest_rel(np.average(stcs_averaged_eyes_closed.data[:,160:262],axis=1),np.average(stcs_averaged_eyes_open.data[:,160:262],axis=1))
    return a, stcs_averaged_eyes_open, stcs_averaged_eyes_closed




def fire(subject):

    def csv_to_raw_mne(path_to_file,path_to_montage_ses,fs,path_to_events,montage = 'GSN-HydroCel-129'):
        ''' Load csv files of data, chan locations and events and return a raw mne instance'''
        data = np.loadtxt(path_to_file,delimiter =',')
        chans = pd.read_csv(path_to_montage_ses,sep = ',',header = None)
        
        ch_names = list(chans.values[1:,0])
    #print(type(ch_names))
        ch_names_appended = list(np.append(ch_names,'stim_channel'))

        types = ['eeg']*(len(ch_names_appended)-1)
        types.append('stim')
        data2 = np.zeros([1,len(data[0])]) #len(raw.times)
        data_appended = np.append(data,data2,axis = 0)

        #print(np.shape(data_appended))
    #print(len(types))

    #types
        info = mne.create_info(ch_names_appended,sfreq = fs,ch_types = types)
    #raw=mne.io.RawArray(data, info)

    #mne.find_events(raw,stim_channel='stim')
        raw = mne.io.RawArray(data_appended, info)
        
        # set standard montage
        if montage:
            raw.set_montage(montage)
            raw.set_eeg_reference(projection=True) 
            raw.apply_proj()
        # events array shape must be (n_events,3)The first column specifies the sample number of each event,
        # the second column is ignored, and the third column provides the event value.
        # If events already exist in the Raw instance at the given sample numbers, the event values will be added together.

        if path_to_events:
            # parse events file
            raw_events = pd.read_csv(path_to_events, sep = r'\s*,\s*', header = None, engine = 'python')
            values = raw_events[0].to_list()
        
            # identify markers start and stop recording
            idx = [i for i, e in enumerate(values) if e == 'break cnt']

            samples = raw_events[1][idx[0] + 1:idx[1]].to_numpy(dtype = int)
            # slicing until '-1' means that we will not know about the last state. Hence removed.
            event_values = raw_events[0][idx[0] + 1:idx[1]].to_numpy(dtype = int)

            
            # append a last value for end of paradigm
            ## I think 1 acts as an explicit EOF, but having this slicing until '-1' as indicated
            # in the previous comment would not let us know the last state
            # event_values = np.append(event_values, 1)

            # Creating an array of len(samples)-1 would not have the sufficient length to add the 
            # sample's last row.
            events = np.zeros((len(samples), 3))
            
            events = events.astype('int')
            events[:, 0] = samples
            events[:, 2] = event_values
            
            # Appending one row of 'ones'. Will be easier to stop parsing once we hit 1
            events_final = np.append(events,np.ones((1, 3)),axis = 0).astype('int')
            raw = exclude_channels_from_raw(raw, ch_list)
            raw.add_events(events[:-1], stim_channel = 'stim_channel',replace = False)

        return raw,events_final

    def exclude_channels_from_raw(raw,ch_to_exclude):
        '''Return a raw structure where ch_to_exclude are removed'''
        idx_keep = mne.pick_channels(raw.ch_names,include = raw.ch_names,exclude = ch_to_exclude)
        raw.pick_channels([raw.ch_names[pick] for pick in idx_keep])
        return raw
    #subject = subjects[i]
    path_to_file = '/users/local/Venkatesh/HBN/%s/EEG/preprocessed/csv_format/RestingState_data.csv'%subject
    path_to_events = '/users/local/Venkatesh/HBN/%s/EEG/preprocessed/csv_format/RestingState_event.csv'%subject
    path_to_montage_glob = '/users/local/Venkatesh/HBN/GSN_HydroCel_129_hbn.sfp'
    path_to_montage_ses = '/users/local/Venkatesh/HBN/%s/EEG/preprocessed/csv_format/RestingState_chanlocs.csv'%subject
    fs = 500
    chans_glob = mne.channels.read_custom_montage(fname = '/users/local/Venkatesh/HBN/GSN_HydroCel_129_hbn.sfp') # read_montage is deprecated
    # channels to exclude because noisy (Nentwich paper)
    ch_list=['E1', 'E8', 'E14', 'E17', 'E21', 'E25', 'E32', 'E38', 'E43', 'E44', 'E48', 'E49', 'E56', 'E57', 'E63', 'E64', 'E69', 'E73', 'E74', 'E81', 'E82', 'E88', 'E89', 'E94', 'E95', 'E99', 'E100', 'E107', 'E113', 'E114', 'E119', 'E120', 'E121', 'E125', 'E126', 'E127', 'E128']


    raw, events = csv_to_raw_mne(path_to_file,path_to_montage_ses,fs,path_to_events,montage = 'GSN-HydroCel-129')

    return raw,events

fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)


subject = 'fsaverage' # Subject ID for the MRI-head transformation
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
source_space = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif') 
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')


from scipy import stats

subjects = ['NDARCD401HGZ','NDARDX770PJK', 'NDARGY054ENV', 'NDARMR242UKQ', 'NDARRD720XZK', 'NDARTR840XP1', 'NDARXJ696AMX', 'NDARYY218AGA', 'NDARZP564MHU']
vals = list()
eyes_open = dict()
eyes_closed = dict()

for i in range(9):
    raw,events = fire(subjects[i])
    if i==0:
        fwd_model = mne.make_forward_solution(raw.info, trans=trans, src=source_space, bem=bem, eeg=True, mindist=5.0)
    function_call = fire_2(raw,events,fwd_model)
    vals.append(function_call[0])
    eyes_open[i] = function_call[1]
    eyes_closed[i] = function_call[2]



print(vals)

#%%
np.save('eyes_closed_20',eyes_closed)
np.save('eyes_open_20',eyes_open)


# %%


# %%
