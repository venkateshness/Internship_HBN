#%%
#Packages Importing
from asyncio import events
from curses.ascii import ETB
from unicodedata import name
from graphql import Source
import mne
import pathlib
from mne.externals.pymatreader import read_mat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd
from mpl_toolkits.mplot3d import axes3d

from mne.minimum_norm import read_inverse_operator, compute_source_psd_epochs
from mne.datasets import sample
from imp import reload 


import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from scipy import signal
from scipy.signal import butter, lfilter
import scipy



from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs

import os.path as op
# Data-loading
def noise_covariance(subject):
    raw_resting_state,events_resting_state = mne.io.read_raw_fif(f'/users2/local/Venkatesh/Generated_Data/importing/resting_state/{subject}/raw.fif'),np.load(f'/users2/local/Venkatesh/Generated_Data/importing/resting_state/{subject}/events.npz')['resting_state_events']

    epochs = mne.Epochs(raw_resting_state, events_resting_state, [20,30,90], tmin=0, tmax=20,preload=True,baseline=(0,None))
    epochs_resampled = epochs.resample(250)# Downsampling to 250Hz
    print(np.shape(epochs_resampled.load_data())) # Sanity Check


    ##################
    ###Noise Covariance
    ##################
    np.random.seed(55)

    rand = np.random.randint(1,5000,size=500)
    cov = mne.EpochsArray(epochs_resampled['20'][0].get_data()[:,:,rand],info=raw_resting_state.info)

    covariance = mne.compute_covariance(
    cov, method='auto')
    return covariance

def fire_2(raw,fwd_model,subject,epochs):

    ##################
    #epochs###########
    ##################
    
    

    ##################
    ###Source Inversion - forward modeling
    #################
    
    covariance = noise_covariance(subject)
    ###Inverse operator
    inverse_operator = make_inverse_operator(raw.info, fwd_model, covariance)
    
    
    return Source_PSD.psd(epochs,inverse_operator)



fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)


subject = 'fsaverage' # Subject ID for the MRI-head transformation
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
source_space = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif') 
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')


from scipy import stats

subjects = ['NDARCD401HGZ','NDARDX770PJK', 'NDAREZ098ZPE', 'NDARGY054ENV', 'NDARMR242UKQ', 
'NDARRD720XZK', 'NDARTR840XP1', 'NDARXJ696AMX', 'NDARYY218AGA', 'NDARZP564MHU']


'''Resting state''' 
subject_list = ['NDARAD481FXF', 'NDARAV945MCQ', 'NDARBK669XJQ', 'NDARCD401HGZ', 'NDARDX770PJK', 'NDAREC182WW2',
 'NDAREZ098ZPE', 'NDARFB107PVH', 'NDARGY054ENV', 'NDARHF023VG3', 'NDARHP176DPE', 'NDARJP133YL3', 'NDARKH741PL8', 'NDARKW999WZD',
 'NDARLB017MBJ', 'NDARMA875ARE', 'NDARMR242UKQ', 'NDARNE511XHU', 'NDARNT042GRA', 'NDARPE596LZL', 'NDARPR768KT4', 'NDARRA733VWX',
 'NDARRD720XZK', 'NDARRN619WHY', 'NDARTR840XP1', 'NDARUJ646APQ', 'NDARVN646NZP', 'NDARWJ087HKJ', 'NDARWV470ATB', 'NDARXB704HFD',
 'NDARXJ468UGL', 'NDARXJ696AMX', 'NDARXU679ZE8', 'NDARXY337ZH9', 'NDARYM257RR6', 'NDARYX530MZU', 'NDARYY218AGA', 'NDARYZ408VWW',
 'NDARZB377WZJ', 'NDARZF288FB7', 'NDARZJ414CAA', 'NDARZT772PU4']


data_present = list()
subjects = list()
##################################
# dataset files check#############@


# for i in range(1, len(subject_list)+1):
#     path_to_file = '/users/local/Venkatesh/HBN/%s/RestingState_data.csv' % subject_list[i-1]
#     path_to_file_video = '/users/local/Venkatesh/HBN/%s/Video3_event.csv' % subject_list[i-1]

#     if os.path.isfile(path_to_file):
        
#         rs_data_present.append(subject_list[i-1])
#     if os.path.isfile(path_to_file_video):
#         subjects.append(subject_list[i-1])
#     print(len(rs_data_present))
#     print(len(subjects))

data_present = list()
for i in range(1, len(subject_list)+1):
    path_to_file = '/users2/local/Venkatesh/HBN/%s/RestingState_data.csv' % subject_list[i-1]
    path_to_file_video = '/users2/local/Venkatesh/HBN/%s/Video3_event.csv' % subject_list[i-1]

    if (os.path.isfile(path_to_file) and os.path.isfile(path_to_file_video)):
        data_present.append (subject_list[i-1])

#%%
import mne
import os
os.chdir('/homes/v20subra/S4B2/')

vals = list()
eyes_open = dict()
eyes_closed = dict()


from Modular_Scripts import epochs_slicing,Source_PSD
vals = dict()
eyes_open = dict()
eyes_closed = dict()

for i in range(len(data_present)):

    raw_video,events_video = mne.io.read_raw_fif(f'/users2/local/Venkatesh/Generated_Data/importing/video-watching/{data_present[i]}/raw.fif'),np.load(f'/users2/local/Venkatesh/Generated_Data/importing/video-watching/{data_present[i]}/events.npz')['video_watching_events']
    epochs = epochs_slicing.epochs(raw_video,events_video,[83,103,9999], tmin=0, tmax=170, fs = 500, epochs_to_slice='83')
    info_d = mne.create_info(raw_video.info['ch_names'],sfreq=125,ch_types = 'eeg')
    ep = mne.EpochsArray(epochs,mne.create_info(raw_video.info['ch_names'],sfreq=500,ch_types = 'eeg')).resample(125)
    raw = mne.io.RawArray(ep.get_data().reshape(91,21250),info_d)
    
   
    
    if i==0:
        fwd_model = mne.make_forward_solution(raw_video.info, trans=trans, src=source_space, bem=bem, eeg=True, mindist=5.0)
    
    function_call = fire_2(raw,fwd_model,data_present[i],ep)


    vals[i] = function_call
    del ep,raw,raw_video,events_video,epochs,info_d

#%%

import numpy as np
with np.load(f"/homes/v20subra/S4B2/GSP/hcp/atlas.npz") as dobj:
    atlas = dict(**dobj)

def averaging_by_parcellation(sub):
    l =list()
    for i in list(set(atlas['labels_R']))[:-1]:
        l.append(np.mean(sub.data[10242:][np.where(i== atlas['labels_R'])],axis=0))

    for i in list(set(atlas['labels_L']))[:-1]:
        l.append(np.mean(sub.data[:10242][np.where(i== atlas['labels_L'])],axis=0))
    print(np.shape(l))
    return l
video_watching_bundle_STC = list()
for i in range(len(data_present)):
    video_watching_bundle_STC.append(np.array(averaging_by_parcellation(vals[i][0])))

# %%
# np.savez_compressed('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/video_watching_bundle_STC_parcellated',video_watching_bundle_STC_parcellated =video_watching_bundle_STC)
video_watching_bundle_STC = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/video_watching_bundle_STC_parcellated.npz')['video_watching_bundle_STC_parcellated']
# %%

# %%
#index_roi = [37,38,39,217,218,219]

##############################################
###RMS VOLUME#################################
##############################################


import librosa
import numpy as np
import matplotlib.pyplot as plt
samples, sample_rate = librosa.load('/homes/v20subra/S4B2/Despicable Me-HQ.wav',sr=None)
samples_normed = (samples - np.average(samples))/np.std(samples)
rms = librosa.feature.rms(y=samples_normed,hop_length=386,frame_length=1000)

# %%

import seaborn as sns
sns.set_theme()
fig=plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)


ax1.plot(samples_normed.T)
ax1.set_xticks([])


ax2.plot(rms.T)
#ax2.set_xticklabels([0,96,97.5,99,100.5,102])
ax2.set_xlabel('time(s)')
fig.suptitle('The waveform (top) and the RMS time-series with frame_length = 1000 samples')
fig.text(0.04, 0.3, 'RMS', va='top', rotation='vertical')

plt.show()





# %%
#####################################
#####Utility function################
#####################################

def mean_std(freq,ax):
    if ax>2:
        d = np.average(np.array(np.abs(freq)),axis=2)[:,1:]
    else: d = np.abs(freq[1:,:])
    mean_t = np.mean(d,axis=0)
    std_t = 2 * np.std(d,axis=0)
    top = mean_t + std_t
    bottom = mean_t - std_t
    
    return mean_t,std_t,top,bottom
 

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def stat_fun(x):
    """Return sum of squares."""
    return np.sum(x ** 2, axis=0)

#%%
################################################################
#####Alpha transform for different brain regions################
################################################################
index_roi = [27,206,23,202,124,303,174,353] 

fig = plt.figure(constrained_layout=True,figsize=(25,25))
subfigs = fig.subfigures(13, 2)
subfigs =np.delete(subfigs,[-1])

for outerind, subfig in enumerate(subfigs.flat):
    subfig.suptitle(f'Subject {outerind+1}')
    axs = subfig.subplots(2, 1)
    for innerind, ax in enumerate(axs.flat):
        if innerind == 0:
            ax.plot(rms[:,19825-25:19825+125].T)
            ax.set_yticks([])
            ax.axvline(x=25,color='r',linestyle='--')
            ax.set_xticks([])
            ax.set_xticks(ticks=np.arange(0,151,25))
            ax.set_xticklabels(np.arange(-200,1001,200))
            ax.set_ylabel("RMS")
        else:
            bandpassed = butter_bandpass_filter(video_watching_bundle_STC[outerind], lowcut = 8, highcut = 13,fs=125)
            hilberted = scipy.signal.hilbert(bandpassed, N=None, axis=- 1)

            ax.plot( (np.average(np.abs(hilberted[:,19825-25:19825+125])[index_roi,:],axis=0).T))
            ax.set_yticks([])
            ax.axvline(x=25,color='r',linestyle='--')
            ax.set_xticks([])
            ax.set_xticks(ticks=np.arange(0,151,25))
            ax.set_xticklabels(np.arange(-200,1001,200))
plt.ylabel("Envelope power")
plt.xlabel('time (in ms)')
fig.suptitle("RMS, up top and the hilbert envelope at the bottom (np.abs(hilberted_signal)) for the Peak around 158th second")
plt.show()
#fig.savefig('/homes/v20subra/S4B2/eLORETA_extensive_validation/25_subjects/rms_hilberted_signal_158s.jpg')

# %%
bandpassed = butter_bandpass_filter(video_watching_bundle_STC[:], lowcut = 8, highcut = 13,fs=125)
hilberted = scipy.signal.hilbert(bandpassed, N=None, axis=- 1)
fig = plt.figure(constrained_layout=True,figsize=(25,25))

subfigs = fig.subfigures(1, 2)
visual = [0,180]
motor = [7,186]
auditory =[23,23+179]
dmn = [32,32+179]

rois = {'VISUAL - V1':visual,'MOTOR - Primary Motor':motor,'AUDITORY - A1':auditory,'DMN - v23ab':dmn}
time = [list(range(12184-250,12184+125)),list(range(19825-250,19825+125))]

for outerind, subfig in enumerate(subfigs.flat):
    if outerind == 0:
        subfig.suptitle(f'Peak around 97th second')
    else:
        subfig.suptitle(f'Peak around 158th second')

    axs = subfig.subplots(5, 1)
    for innerind, ax in enumerate(axs.flat):
        if innerind == 0:
            ax.plot(rms[:,time[outerind][140:]].T)
            ax.set_yticks([])
            ax.axvline(x=110,color='r',linestyle='--')
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_ylabel("RMS")
            if outerind==0:
                ax.axvspan(210-140, 227-140, alpha=0.2, color='green')
            if outerind==1:
                ax.axvspan(217-140, 222-140, alpha=0.2, color='green')
        if innerind>0:
            
            hilberted_avg = np.average(np.abs(hilberted)[:,list(rois.values())[innerind-1],:][:,:,time[outerind]],axis=1)

            hilberted_avg_normalised = list()
            for i in range(25):
                if outerind==0:
                    hilberted_avg_normalised.append( (hilberted_avg[i,140:] - np.average(hilberted_avg[:,:70]))/(np.average(hilberted_avg[:,:70])))
                if outerind==1:
                    hilberted_avg_normalised.append( (hilberted_avg[i,140:] - np.average(hilberted_avg[:,:77]))/(np.average(hilberted_avg[:,:77])))
                
            sem = scipy.stats.sem(np.array(hilberted_avg_normalised))
            averaged = np.average(hilberted_avg_normalised,axis=0)

            ax.plot(averaged)
            ax.fill_between(range(235),averaged-sem,averaged+sem, color='b', alpha=.25)
            ax.set_xticks([])
            ax.axvline(x=110,color='r',linestyle='--')
            if outerind==0:
                ax.axvspan(210-140, 227-140, alpha=0.2, color='green')
                ax.axvspan(0, 210-140, alpha=0.2, color='red')

            if outerind==1:
                ax.axvspan(217-140, 222-140, alpha=0.2, color='green')
                ax.axvspan(0, 217-140, alpha=0.2, color='red')

            ax.legend([list(rois.keys())[innerind-1]])
    ax.set_xticks(ticks=np.arange(-10,235,20))
    ax.set_xticklabels(np.arange(-900,1001,150))
    ax.set_xlabel('time(ms)')
fig.suptitle("The envelope alpha signal for various regions for two periods")
plt.show()
#%%
fig.savefig('/homes/v20subra/S4B2/eLORETA_extensive_validation/25_subjects/rms_hilberted_signal_different_regions_two_periods.jpg')

# %%

#######################################################
#####Exporting data band-wise transform################
#######################################################
bands = dict()

def filter_and_store(low,high,band):
    bandpassed = butter_bandpass_filter(video_watching_bundle_STC, lowcut = low, highcut = high,fs=125)
    hilberted = scipy.signal.hilbert(bandpassed, N=None, axis=- 1)
    bands[band] = np.abs(hilberted)

filter_and_store(8,13,'alpha')
filter_and_store(13,20,'lower_beta')
filter_and_store(20,30,'higher_beta')
filter_and_store(4,8,'theta')

# %%
np.savez_compressed('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/envelope_signal_bandpassed_low_high_beta',**bands)

# %%
####################################
#####Sanity check###################
####################################
np.load('/users/local/Venkatesh/Generated_Data/eLORETA_extensive_validation/envelope_signal_bandpassed.npz', mmap_mode='r')


# %%
np.shape(bands['theta'])
# %%


np.shape(hilberted_avg_normalised)


# %%
import os
os.chdir('/homes/v20subra/S4B2')
from Modular_Scripts import epochs_slicing
raw_video,events_video = mne.io.read_raw_fif(f'/users2/local/Venkatesh/Generated_Data/importing/video-watching/{data_present[0]}/raw.fif'),np.load(f'/users2/local/Venkatesh/Generated_Data/importing/video-watching/{data_present[0]}/events.npz')['video_watching_events']
epochs = epochs_slicing.epochs(raw_video,events_video,[83,103,9999], tmin=0, tmax=170, fs = 500, epochs_to_slice='83')

# %%

len(raw_video.info.ch_names)
# %%

# %%
plt.plot((rms[:,12184-10:12184+125]).T)
# %%
12184-250
# %%
12174-11934
# %%
# %%

# %%
plt.plot((rms[:,19825-33:19825-28]).T)
plt.axvline(x=82)
# %%
plt.plot((rms[:,12184-40:12184+125]).T)

# %%
np.where ( (rms[:,time[outerind][140:]].T) == np.max(rms[:,time[outerind][140:]].T))
# %%
235-110-30
# %%
len(np.arange(-900,1001,150))
# %%
len(np.arange(-10,235,20))
# %%
19825-28

# %%
375-125-40
# %%
375-125-28

# %%
