#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mne
import pathlib
from mne.externals.pymatreader import read_mat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d


# In[2]:


from warnings import simplefilter 
simplefilter(action='ignore', category=DeprecationWarning)


# In[3]:


import os
os.chdir(r'/usr/slurm/venkatesh/HBN/')
subjs = sorted(os.listdir())[1:-2]
#cd


# In[4]:


cd


# # Importing the data

# In[ ]:


def csv_to_raw_mne(path_to_file,path_to_montage_ses,fs,path_to_events,filename,montage = 'GSN-HydroCel-129'):
    ''' Load csv files of data, chan locations and events and return a raw mne instance'''
    data = np.loadtxt(path_to_file,delimiter =',')
    chans = pd.read_csv(path_to_montage_ses,sep = ',',header = None)
    ch_list=['E1', 'E8', 'E14', 'E17', 'E21', 'E25', 'E32', 'E38', 'E43', 'E44', 'E48', 'E49', 'E56', 'E57', 'E63', 'E64', 'E69', 'E73', 'E74', 'E81', 'E82', 'E88', 'E89', 'E94', 'E95', 'E99', 'E100', 'E107', 'E113', 'E114', 'E119', 'E120', 'E121', 'E125', 'E126', 'E127', 'E128']
    ch_names = list(chans.values[1:,0])
#print(type(ch_names))
    #ch_names_appended = list(np.append(ch_names,'stim_channel'))
    #print(len(data[0]))
    types = ['eeg']*(len(ch_names))
    #types.append('stim')
    #data2 = np.zeros([1,len(data[0])]) #len(raw.times)
    #data_appended = np.append(data,data2,axis = 0)

    #print(np.shape(data_appended))
#print(len(types))

#types
    info = mne.create_info(ch_names,sfreq = fs,ch_types = types)
#raw=mne.io.RawArray(data, info)

#mne.find_events(raw,stim_channel='stim')
    raw = mne.io.RawArray(data, info)
    
    # set standard montage
    if montage:
        raw.set_montage(montage)

    # events array shape must be (n_events,3)The first column specifies the sample number of each event,
    # the second column is ignored, and the third column provides the event value.
    # If events already exist in the Raw instance at the given sample numbers, the event values will be added together.

    if path_to_events:
        # parse events file
        raw_events = pd.read_csv(path_to_events, sep = r'\s*,\s*', header = None, engine = 'python')
        values = raw_events[0].to_list()
        
        # identify markers start and stop recording
        idx = [i for i, e in enumerate(values) if e == 'break cnt']
        
         
        if filename == 'NDARDX770PJK':
           
            values.extend(["break cnt"])
            
            idx = [i for i, e in enumerate(values) if e == 'break cnt']
        
        samples = raw_events[1][1:idx[0]].to_numpy(dtype = int)
        # slicing until '-1' means that we will not know about the last state. Hence removed.
        event_values = raw_events[0][1:idx[0]].to_numpy(dtype = int)

        
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
        
    return raw,events_final

def exclude_channels_from_raw(raw,ch_to_exclude):
    '''Return a raw structure where ch_to_exclude are removed'''
    idx_keep = mne.pick_channels(raw.ch_names,include = raw.ch_names,exclude = ch_to_exclude)
    raw.pick_channels([raw.ch_names[pick] for pick in idx_keep])
    return raw



def preparation(filename):
    path_to_file = '/usr/slurm/venkatesh/HBN/%s/EEG/preprocessed/csv_format/Video3_data.csv'% filename
    path_to_events = '/usr/slurm/venkatesh/HBN/%s/EEG/preprocessed/csv_format/Video3_event.csv' %filename
    path_to_montage_glob = '/S4B2/GSN_HydroCel_129_hbn.sfp'
    path_to_montage_ses = '/usr/slurm/venkatesh/HBN/%s/EEG/preprocessed/csv_format/Video3_chanlocs.csv' %filename
    fs = 500
    chans_glob = mne.channels.read_custom_montage(fname = 'S4B2/GSN_HydroCel_129_hbn.sfp') # read_montage is deprecated
# channels to exclude because noisy (Nentwich paper)


    raw, events = csv_to_raw_mne(path_to_file,path_to_montage_ses,fs,path_to_events,filename,montage = 'GSN-HydroCel-129')
    #raw.add_events(events, stim_channel = 'stim_channel',replace = False)
    return raw,events


sub1_raw,sub1_events = preparation(subjs[0])
sub2_raw,sub2_events = preparation(subjs[1])
sub3_raw,sub3_events = preparation(subjs[2])
sub4_raw,sub4_events = preparation(subjs[3])
sub5_raw,sub5_events = preparation(subjs[4])
sub6_raw,sub6_events = preparation(subjs[5])
sub7_raw,sub7_events = preparation(subjs[6])
sub8_raw,sub8_events = preparation(subjs[7])
sub9_raw,sub9_events = preparation(subjs[8])
sub10_raw,sub10_events = preparation(subjs[9])


# # Picking Event of interest

# In[6]:


def epochs(subject_raw,subject_events):

    epochs = mne.Epochs(subject_raw, subject_events, [83,103,9999], tmin=0, tmax=170,preload=True,baseline=(0,None))
    epochs_resampled = epochs#.resample(250) # Downsampling to 250Hz
    
    return epochs_resampled['83']


# # CCA

# In[7]:


import numpy as np
from scipy.linalg import eigh
from timeit import default_timer


def train_cca(data):
    """Run Correlated Component Analysis on your training data.
        Parameters:
        ----------
        data : dict
            Dictionary with keys are names of conditions and values are numpy
            arrays structured like (subjects, channels, samples).
            The number of channels must be the same between all conditions!
        Returns:
        -------
        W : np.array
            Columns are spatial filters. They are sorted in descending order, it means that first column-vector maximize
            correlation the most.
        ISC : np.array
            Inter-subject correlation sorted in descending order
    """

    start = default_timer()

    C = len(data.keys())
    print(f'train_cca - calculations started. There are {C} conditions')

    gamma = 0.1
    Rw, Rb = 0, 0
    for cond in data.values():
        N, D, T, = cond.shape
        print(f'Condition has {N} subjects, {D} sensors and {T} samples')
        cond = cond.reshape(D * N, T)

        # Rij
        Rij = np.swapaxes(np.reshape(np.cov(cond), (N, D, N, D)), 1, 2)
        
        # Rw
        Rw = Rw + np.mean([Rij[i, i, :, :]
                           for i in range(0, N)], axis=0)
        
        # Rb
        Rb = Rb + np.mean([Rij[i, j, :, :]
                           for i in range(0, N)
                           for j in range(0, N) if i != j], axis=0)
        covariance = np.cov(cond)
    # Divide by number of condition
    Rw, Rb = Rw/C, Rb/C

    # Regularization
    Rw_reg = (1 - gamma) * Rw + gamma * np.mean(eigh(Rw)[0]) * np.identity(Rw.shape[0])

    # ISCs and Ws
    [ISC, W] = eigh(Rb, Rw_reg) #Eigen values, W matrix
    
    # Make descending order
    ISC, W = ISC[::-1], W[:, ::-1] 
    #print(ISC[0])
    stop = default_timer()

    print(f'Elapsed time: {round(stop - start)} seconds.')
    return W, ISC


def apply_cca(X, W, fs):
    """Applying precomputed spatial filters to your data.
        Parameters:
        ----------
        X : ndarray
            3-D numpy array structured like (subject, channel, sample)
        W : ndarray
            Spatial filters.
        fs : int
            Frequency sampling.
        Returns:
        -------
        ISC : ndarray
            Inter-subject correlations values are sorted in descending order.
        ISC_persecond : ndarray
            Inter-subject correlations values per second where first row is the most correlated.
        ISC_bysubject : ndarray
            Description goes here.
        A : ndarray
            Scalp projections of ISC.
    """

    start = default_timer()
    print('apply_cca - calculations started')

    N, D, T = X.shape
    # gamma = 0.1
    window_sec = 5
    X = X.reshape(D * N, T)
    
    # Rij
    Rij = np.swapaxes(np.reshape(np.cov(X), (N, D, N, D)), 1, 2)
    print(Rij.shape)
    # Rw
    Rw = np.mean([Rij[i, i, :, :]
                  for i in range(0, N)], axis=0)
    # Rw_reg = (1 - gamma) * Rw + gamma * np.mean(eigh(Rw)[0]) * np.identity(Rw.shape[0])

    # Rb
    Rb = np.mean([Rij[i, j, :, :]
                  for i in range(0, N)
                  for j in range(0, N) if i != j], axis=0)

    # ISCs
    ISC = np.sort(np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W))[::-1]

    # Scalp projections
    A = np.linalg.solve(Rw @ W, np.transpose(W) @ Rw @ W)  # a, b. 
    
    # ISC by subject
    print('by subject is calculating')
    ISC_bysubject = np.empty((D, N))

    for subj_k in range(0, N):
        Rw, Rb = 0, 0
        Rw = np.mean([Rw + 1 / (N - 1) * (Rij[subj_k, subj_k, :, :] + Rij[subj_l, subj_l, :, :])
                      for subj_l in range(0, N) if subj_k != subj_l], axis=0)
        Rb = np.mean([Rb + 1 / (N - 1) * (Rij[subj_k, subj_l, :, :] + Rij[subj_l, subj_k, :, :])
                      for subj_l in range(0, N) if subj_k != subj_l], axis=0)

        ISC_bysubject[:, subj_k] = np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W)

    # ISC per second
    print('by persecond is calculating')
    ISC_persecond = np.empty((D, int(T / fs) ))
    window_i = 0

    for t in range(0, T, fs):

        Xt = X[:, t:t+window_sec*fs] #[subj 1, subj 2........subj 10]
       
        # the covariance
        Rij = np.cov(Xt) #910, 910for all the subjects 
        # <----10 subjects---->
        #  [sub1, sub2... sub10 ] sub 1
        #  [sub1, sub2... sub10 ] sub 2
        #  [sub1, sub2... sub10 ] ..
        #  [sub1, sub2... sub10 ] ..
        #   [sub1, sub2... sub10 ] sub 10
        
        
        
        Rw = np.mean([Rij[i:i + D, i:i + D] # Correlation diagonally (itself)
                      for i in range(0, D * N, D)], axis=0) 
        
        Rb = np.mean([Rij[i:i + D, j:j + D]
                      for i in range(0, D * N, D)
                      for j in range(0, D * N, D) if i != j], axis=0) #Correlation with other subjects
        
        ISC_persecond[:, window_i] = np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W)
        window_i += 1
        
    stop = default_timer()
    print(f'Elapsed time: {round(stop - start)} seconds.')

    return ISC, ISC_persecond, ISC_bysubject, A


# # Preparation to perform CCA

# In[8]:


dic = dict()

dic['condition1'] = np.append(sub1_raw.get_data()[:91,516:85516].reshape(1,91,85000),sub2_raw.get_data()[:91,516:85516].reshape(1,91,85000) ,axis=0)

#for i in range(2,10):
#    dic['condition1'] = np.append(dic['condition1'],sub[i].get_data()[:,:86030].reshape(1,92,86030) ,axis=0)

dic['condition1'] = np.append(dic['condition1'],sub3_raw.get_data()[:91,516:85516].reshape(1,91,85000) ,axis=0)
dic['condition1'] = np.append(dic['condition1'],sub4_raw.get_data()[:91,516:85516].reshape(1,91,85000) ,axis=0)
dic['condition1'] = np.append(dic['condition1'],sub5_raw.get_data()[:91,516:85516].reshape(1,91,85000) ,axis=0)
dic['condition1'] = np.append(dic['condition1'],sub6_raw.get_data()[:91,516:85516].reshape(1,91,85000) ,axis=0)
dic['condition1'] = np.append(dic['condition1'],sub7_raw.get_data()[:91,516:85516].reshape(1,91,85000) ,axis=0)
dic['condition1'] = np.append(dic['condition1'],sub8_raw.get_data()[:91,516:85516].reshape(1,91,85000) ,axis=0)
dic['condition1'] = np.append(dic['condition1'],sub9_raw.get_data()[:91,516:85516].reshape(1,91,85000) ,axis=0)
dic['condition1'] = np.append(dic['condition1'],sub10_raw.get_data()[:91,516:85516].reshape(1,91,85000) ,axis=0)    


# In[9]:


[W,ISC] = train_cca(dic)
#np.shape( sub1_raw.get_data() )

isc_results = dict()
for cond_key, cond_values in dic.items():
    isc_results[str(cond_key)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'], apply_cca(cond_values, W, 500)))


# In[10]:


vals = np.reshape( ((np.random.rand(100,10) * np.random.rand(100,10)).T), [10,1,100])


#print(f'Elapsed time: {round(stop - start)} seconds.')


# # Intra-subject Study Results

# In[11]:


plt.matshow( (isc_results['condition1']['ISC_bysubject']).T)
plt.title('Intra Subject Study')
plt.ylabel('Subjects')
plt.xlabel('Components')

cb =plt.colorbar()
cb.ax.set_title('ISC')


# # ISC results

# In[11]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

def plot_isc(isc_all):
    # plot ISC as a bar chart
    plt.figure()
    comp1 = [cond['ISC'][0] for cond in isc_all.values()]
    comp2 = [cond['ISC'][1] for cond in isc_all.values()]
    comp3 = [cond['ISC'][2] for cond in isc_all.values()]
    barWidth = 0.2
    r1 = np.arange(len(comp1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    plt.bar(r1, comp1, color='gray', width=barWidth, edgecolor='white', label='Comp1')
    plt.bar(r2, comp2, color='green', width=barWidth, edgecolor='white', label='Comp2')
    plt.bar(r3, comp3, color='blue', width=barWidth, edgecolor='white', label='Comp3')
    plt.xticks([r + barWidth for r in range(len(comp1))], isc_all.keys())
    plt.ylabel('ISC', fontweight='bold')
    plt.title('ISC for each condition')
    plt.legend()
    plt.show()

    # plot ISC_persecond
    for cond in isc_all.values():
        for comp_i in range(0, 3):
            plt.subplot(3, 1, comp_i+1)
            plt.plot(cond['ISC_persecond'][comp_i])
            #plt.plot ((np.array(vals)[:,comp_i,:]).T)
            #peaks = scipy.signal.find_peaks(isc_results['condition1']['ISC_persecond'][comp_i],distance=15)
            plt.subplots_adjust(hspace=1)

            plt.plot(isc_results['condition1']['ISC_persecond'][comp_i])
            #plt.plot(peaks[0],isc_results['condition1']['ISC_persecond'][comp_i][peaks[0]],marker='o', ls="")

            
            plt.xlabel('Time (s)')
            plt.ylabel('ISC')
            

            #plt.title('ISC per second for each condition')


    
plot_isc(isc_results)


# # Noise Floor

# In[8]:


#a = list(range(100))
import random

from tqdm.notebook import tqdm

def shuffle(a):
    
    for i in (range(10)):
        for j in range(91):
            np.random.seed(i)
            
            chunked = chunks(a['condition1'][i][j][:85000])
            np.random.shuffle(chunked[0])
            chunked = np.reshape(chunked,[85000,])
            
    return a

def chunks(chunk):
    chunked = chunk[:85000]
    chunked= chunked.reshape(1,34,2500)
    return chunked

valstest = []
for i in tqdm(range(1000)):
    shuffled = shuffle(dic)
    isc_resultstest_ = dict()
    print('step {}'.format(i))
    for cond_key, cond_values in shuffled.items():
        isc_resultstest_[str(cond_key)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'], apply_cca(cond_values, W, 500)))
        #print(np.mean(isc_results['condition1']['ISC_persecond'][0]))
        valstest.append(isc_resultstest_['condition1']['ISC_persecond'])
        #print(np.shape(isc_resultstest_['condition1']['ISC_persecond']))
#np.append(sub1_raw.get_data()[:91,516:85779].reshape(1,91,85263),sub2_raw.get_data()[:91,516:85779].reshape(1,91,85263) ,axis=0)


# In[7]:


len(valstest)


# ## Results

# In[91]:


import seaborn as sns
sns.set_theme()

significance = np.where(np.max(np.array(valstest)[:,0,:],axis=0)<isc_results['condition1']['ISC_persecond'][0])


plt.plot(isc_results['condition1']['ISC_persecond'][0])
plt.plot (np.max(np.array(valstest)[:,0,:],axis=0).T,color='grey')
plt.plot(np.reshape(significance,(21,)),isc_results['condition1']['ISC_persecond'][0][significance],marker='o', ls="",color='red')
#duration = (np.sum(isc_results['condition1']['ISC_persecond'][0]>np.max(vals,axis=0)[0])/170)
#plt.title('First Component, %.2f perc of duration above the significance' %(duration*100))
plt.title('First component with 5-seconds block')
plt.xlabel('time (s)')
plt.ylabel('ISC')


# # Source Inversion

# ## Source Inversion on ISC

# In[11]:


#mne.viz.plot_topomap(isc_results['condition1']['A'][2,:],pos=sub1_raw.info,vmin=-0.8472,vmax=0.21,ch_type='eeg',cmap='inferno')
(isc_results['condition1']['ISC_persecond'][0])[102]


# In[12]:


epochs_ISC = mne.EpochsArray(np.reshape(isc_results['condition1']['A'][:],[1,91,91]),mne.create_info(sub1_raw.info['ch_names'],sfreq=1,ch_types = 'eeg'))
#np.shape(isc_results['condition1']['A'][:])


# In[13]:


cov_isc =mne.compute_covariance(epochs_ISC)


# In[14]:


from mne.datasets import fetch_fsaverage

import os.path as op
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)


subject = 'fsaverage' # Subject ID for the MRI-head transformation
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
source_space = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif') 
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')


# In[15]:


fwd_model_isc = mne.make_forward_solution(sub1_raw.info, trans=trans, src=source_space, bem=bem, eeg=True, mindist=5.0)


# In[16]:


from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs

inverse_operator_isc = make_inverse_operator(sub1_raw.info, fwd_model_isc, cov_isc)


# In[ ]:





# In[ ]:





# In[17]:


method = "eLORETA"
snr = 3.
lambda2 = 1. / snr ** 2


epochs_ISC_2 = mne.EpochsArray(np.reshape(isc_results['condition1']['A'][0],[1,91,1]),mne.create_info(sub1_raw.info['ch_names'],sfreq=1,ch_types = 'eeg'))

# epochs['30'].average() = Averaged evoked response for the event 30
stc_isc = apply_inverse_epochs(epochs_ISC_2, inverse_operator_isc, lambda2,
                             method=method, pick_ori=None, verbose=True)
stc_isc


# In[ ]:


stc_isc


# In[105]:


from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(stc_isc[0].data)

plt.title('Distribution of the eLORETA source inversion on spatial filter')
scaled = scaler.transform(stc_isc[0].data)
plt.hist(scaled,bins=100)


# In[111]:


index = np.where( np.logical_or(scaled<-0.5, scaled>0.5))[0]
scaled[list(set(list(range(20484)))-set(index))] =0


# In[1]:


import brainspace.mesh

mesh = brainspace.mesh.mesh_io.read_surface('S4B2/brainnotation/tpl-fsaverage_den-10k_hemi-L_pial.surf.gii')
mesh2 = brainspace.mesh.mesh_io.read_surface('S4B2/brainnotation/tpl-fsaverage_den-10k_hemi-R_pial.surf.gii')


from surfplot import Plot
get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('gui', 'qt')

p = Plot(mesh,mesh2, zoom=1.2, views='lateral')

p.add_layer(scaled, color_range=None,cmap = 'seismic')
fig = p.build()
plt.title('1st spatial filter using eLORETA (after the threshold on range)')
fig.show()


# In[24]:


#src = mne.setup_source_space('fsaverage', spacing='oct6',
#                             add_dist=False, subjects_dir=subjects_dir)#write_source_spaces('fsaverage-new-ico5-src.fif', src)
#mne.write_source_spaces('fsaverage-new-oct6-src.fif', src)


# In[37]:


stc_isc[0].data.min()


# In[111]:


vertno_max, time_max = stc_isc[0].get_peak()
get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('gui', 'qt')
from pyvistaqt import BackgroundPlotter
fig, ax = plt.subplots()
ax.plot(1e3 * stc_isc[0].times, stc_isc[0].data.T)
ax.set(xlabel='time (ms)', ylabel='%s value' % method)
brain = stc_isc[0].plot(subjects_dir=subjects_dir, initial_time=time_max,time_viewer=False,backend='pyvista')
#surfer_kwargs = dict(
 #   hemi='both', subjects_dir=subjects_dir,
  #  clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
   # initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10)
#brain = stc.plot(**surfer_kwargs)
#brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',scale_factor=0.6, alpha=0.5)
#brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='red',scale_factor=0.6, alpha=0.5)

brain.add_text(0.1, 0.9, 'eLORETA for the first component', 'title' ,font_size=14)


# # Source Estimation on ISC with time-series

# ## Picking necessary chunks of time-series data

# In[11]:


#[29,33],[67,71],[77,83],[130,135],[155,164]]]

#np.max(valstest,axis=0)

#significance
indexes = np.hstack([np.arange(159*500-1000,159*500+1060)])#,np.arange(67*500,72*500),np.arange(77*500,84*500),np.arange(130*500,136*500),np.arange(155*500,165*500)])
indexes2 = np.hstack([np.arange(100*500-1000,100*500+1060)])
#np.where(isc_results['condition1']['ISC_persecond'][0] == isc_results['condition1']['ISC_persecond'][0].max())


# In[ ]:





# In[12]:


#isc_results['condition1']['A'][0]
#sub1_trial = isc_results['condition1']['A'][0] @ sub1_raw.get_data()
import tqdm
#sub1_trial = np.zeros([91,85001])
#a = [sub1_trial[:,i] (sub1_raw.get_data()[:,i]*isc_results['condition1']['A'][0]) for i in tqdm.tqdm(range(86040))]
 #sub1_trial[:,i] = (sub1_raw.get_data()[:,i]*isc_results['condition1']['A'][0])   

def element_wise_multi(epochs):
    subjects = isc_results['condition1']['A'][0].T @ epochs
    return subjects

def spatial_filter(epochs):
    subjects = np.multiply(epochs.T,isc_results['condition1']['A'][0])
    print(np.shape(subjects))
    subjects_final = np.reshape(subjects,[1,91,2060])
    return subjects_final
#np.shape(element_wise_multi(epochs1_ISC_ts.get_data()[:,:,indexes]))

#np.shape(isc_results['condition1']['A'][0],sub1_raw.get_data())


# In[13]:


np.shape( spatial_filter(np.reshape( epochs1_ISC_ts.get_data()[:,:,indexes], [91,2060])) )


# In[14]:


epochs1_ISC_ts = epochs(sub1_raw,sub1_events)
epochs2_ISC_ts = epochs(sub2_raw,sub2_events)
epochs3_ISC_ts = epochs(sub3_raw,sub3_events)
epochs4_ISC_ts = epochs(sub4_raw,sub4_events)
epochs5_ISC_ts = epochs(sub5_raw,sub5_events)
epochs6_ISC_ts = epochs(sub6_raw,sub6_events)
epochs7_ISC_ts = epochs(sub7_raw,sub7_events)
epochs8_ISC_ts = epochs(sub8_raw,sub8_events)
epochs9_ISC_ts = epochs(sub9_raw,sub9_events)
epochs10_ISC_ts = epochs(sub10_raw,sub10_events)


# In[ ]:





# In[24]:


epochs_averaged_ISC_ts = np.average(epochs1_ISC_ts.get_data()[:,:,indexes]+
epochs2_ISC_ts.get_data()[:,:,indexes]+
epochs3_ISC_ts.get_data()[:,:,indexes]+
epochs4_ISC_ts.get_data()[:,:,indexes]+
epochs5_ISC_ts.get_data()[:,:,indexes]+
epochs6_ISC_ts.get_data()[:,:,indexes]+
epochs7_ISC_ts.get_data()[:,:,indexes]+
epochs8_ISC_ts.get_data()[:,:,indexes]+
epochs9_ISC_ts.get_data()[:,:,indexes]+
epochs10_ISC_ts.get_data()[:,:,indexes],axis=0)


# In[ ]:





# In[14]:


#np.shape(epochs_averaged)
epochs_ready_ISC_ts = np.reshape(epochs_averaged_ISC_ts,[1,91,2060])


# In[15]:


events = np.zeros([2,3],dtype=int)
#def ISCed_inversion(subject,events):
 #   e = epochs(subject,events)
events[0][0] = 1
events[0][2] = 83
def get_raw_epochs_data(epochs):

    info = mne.create_info(sub1_raw.info['ch_names'],sfreq=500,ch_types = 'eeg')

    raw = mne.io.RawArray(epochs.reshape(91,2060),info)
    ep = mne.EpochsArray(epochs.reshape(1,91,2060),info)
    raw.set_montage('GSN-HydroCel-129')
    return info, raw, ep
def get_raw_epochs_data_elem(epochs):

    info = mne.create_info(['E1'],sfreq=500,ch_types = 'eeg')

    raw = mne.io.RawArray(epochs.reshape(1,2060),info)
    ep = mne.EpochsArray(epochs.reshape(1,1,2060),info)
    raw.set_montage('GSN-HydroCel-129')
    return info, raw, ep


# In[16]:


_,raw,ep = get_raw_epochs_data(epochs10_ISC_ts.get_data()[:,:,indexes])
ep.get_data()
#len(sub1_raw.info['ch_names'])


# In[17]:


#mne.make_fixed_length_events(raw,start=0,duration = 0.5)
#raw.set_montage('GSN-HydroCel-129')


# ## Modeling begins

# In[17]:


from mne.datasets import fetch_fsaverage

import os.path as op
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)


subject = 'fsaverage' # Subject ID for the MRI-head transformation
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
source_space = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif') 

bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')


# In[15]:


cov_isc_ts =mne.compute_covariance(ep)


# In[18]:


from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs

fwd_model = mne.make_forward_solution(sub1_raw.info, trans=trans, src=source_space, bem=bem, eeg=True, mindist=5.0)
inverse_operator = make_inverse_operator(raw.info, fwd_model, cov_isc_ts)


# In[23]:


method = "eLORETA"
snr = 3.
lambda2 = 1. / snr ** 2

# epochs['30'].average() = Averaged evoked response for the event 30
#stc = apply_inverse_epochs(epochs_resampled, inverse_operator, lambda2,
 #                            method=method, pick_ori=None, verbose=True)
#    return stc


# In[ ]:


vertno_max, time_max = stc[0].get_peak()
get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('gui', 'qt')

fig, ax = plt.subplots()
ax.plot(1e3 * stc[0].times, stc[0].data.T)
ax.set(xlabel='time (ms)', ylabel='%s value' % method)
brain = stc[0].plot(subjects_dir=subjects_dir, initial_time=time_max, time_unit='s',hemi='split',views=['lat'], transparent=True)
#surfer_kwargs = dict(
 #   hemi='both', subjects_dir=subjects_dir,
  #  clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
   # initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10)
#brain = stc.plot(**surfer_kwargs)
#brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',scale_factor=0.6, alpha=0.5)
#brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='red',scale_factor=0.6, alpha=0.5)

brain.add_text(0.1, 0.9, 'eLORETA for the peak of the peak', 'title' ,font_size=14)


# ## Computing Source PSD 

# In[128]:


data_path = sample.data_path()
label_name ='Vis-rh.label'
fname_label = data_path + '/MEG/sample/labels/%s' % label_name
label_name2 = 'Vis-lh.label'
fname_label2 = data_path + '/MEG/sample/labels/%s' % label_name2
label = mne.read_label(fname_label)
label2 = mne.read_label(fname_label2)
mne.BiHemiLabel(label,label2)


# In[19]:


def comp_source_psd(raw):
 
    noise_cov = mne.compute_raw_covariance(raw)
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        raw.info, forward=fwd_model, noise_cov=noise_cov, verbose=True)

    stc_psd, sensor_psd = mne.minimum_norm.compute_source_psd(
        raw, inverse_operator, lambda2=lambda2,method=method,fmax=40,
        dB=False, return_sensor=True, verbose=True)
    


    freq_bands = dict(
        delta=(2, 4), theta=(5, 7), alpha=(8, 13), beta=(15, 29), gamma=(30, 40))
    topos = dict(vv=dict(), opm=dict())
    stcs_dict = dict(vv=dict(), opm=dict())
    
    
    topo_norm = sensor_psd.data.sum(axis=1, keepdims=True)
    stc_norm = stc_psd.sum() 

    for band, limits in freq_bands.items():
        data = sensor_psd.copy().crop(*limits).data.sum(axis=1, keepdims=True)
        topos[band] = mne.EvokedArray(
            100 * data / topo_norm, sensor_psd.info)
        stcs_dict[band] =             100 * stc_psd.copy().crop(*limits).sum() / stc_norm.data
    
    return stcs_dict


# In[20]:


#mne.read_labels_from_annot(subject, parc='aparc2008µ')


# In[ ]:





# In[18]:


#mne.read_labels_from_annot(subject, parc='aparc.a2009s') # 13k
#mne.read_labels_from_annot(subject, parc='HCPMMP1') # 13k
#mne.read_labels_from_annot(subject, parc='rh.cortex.label',surf_name='pial') 
stc1 = comp_source_psd(raw1)


# ## High ISC
# 

# In[21]:



#epochs1_ISC_ts_filtered_high = element_wise_multi(epochs1_ISC_ts.get_data()[:,:,indexes])
# call element_wise_multi for the spatially filtered

epochs1_ISC_ts_filtered_high = spatial_filter(np.reshape( epochs1_ISC_ts.get_data()[:,:,indexes], [91,2060]))
epochs2_ISC_ts_filtered_high = spatial_filter(np.reshape( epochs2_ISC_ts.get_data()[:,:,indexes], [91,2060]))
epochs3_ISC_ts_filtered_high = spatial_filter(np.reshape( epochs3_ISC_ts.get_data()[:,:,indexes], [91,2060]))
epochs4_ISC_ts_filtered_high = spatial_filter(np.reshape( epochs4_ISC_ts.get_data()[:,:,indexes], [91,2060]))
epochs5_ISC_ts_filtered_high = spatial_filter(np.reshape( epochs5_ISC_ts.get_data()[:,:,indexes], [91,2060]))
epochs6_ISC_ts_filtered_high = spatial_filter(np.reshape( epochs6_ISC_ts.get_data()[:,:,indexes], [91,2060]))
epochs7_ISC_ts_filtered_high = spatial_filter(np.reshape( epochs7_ISC_ts.get_data()[:,:,indexes], [91,2060]))
epochs8_ISC_ts_filtered_high = spatial_filter(np.reshape( epochs8_ISC_ts.get_data()[:,:,indexes], [91,2060]))
epochs9_ISC_ts_filtered_high = spatial_filter(np.reshape( epochs9_ISC_ts.get_data()[:,:,indexes], [91,2060]))
epochs10_ISC_ts_filtered_high = spatial_filter(np.reshape( epochs10_ISC_ts.get_data()[:,:,indexes], [91,2060]))
#(epochs10_ISC_ts.get_data()[:,:,indexes])




info1,raw1,ep1 = get_raw_epochs_data(epochs1_ISC_ts_filtered_high)
info2,raw2,ep2 = get_raw_epochs_data(epochs2_ISC_ts_filtered_high)
info3,raw3,ep3 = get_raw_epochs_data(epochs3_ISC_ts_filtered_high)
info4,raw4,ep4 = get_raw_epochs_data(epochs4_ISC_ts_filtered_high)
info5,raw5,ep5 = get_raw_epochs_data(epochs5_ISC_ts_filtered_high)
info6,raw6,ep6 = get_raw_epochs_data(epochs6_ISC_ts_filtered_high)
info7,raw7,ep7 = get_raw_epochs_data(epochs7_ISC_ts_filtered_high)
info8,raw8,ep8 = get_raw_epochs_data(epochs8_ISC_ts_filtered_high)
info9,raw9,ep9 = get_raw_epochs_data(epochs9_ISC_ts_filtered_high)
info10,raw10,ep10 = get_raw_epochs_data(epochs10_ISC_ts_filtered_high)




epochs1_ISC_ts_filtered_high_elem = element_wise_multi(( epochs1_ISC_ts.get_data()[:,:,indexes]))
epochs2_ISC_ts_filtered_high_elem = element_wise_multi(( epochs2_ISC_ts.get_data()[:,:,indexes]))
epochs3_ISC_ts_filtered_high_elem = element_wise_multi(( epochs3_ISC_ts.get_data()[:,:,indexes]))
epochs4_ISC_ts_filtered_high_elem = element_wise_multi(( epochs4_ISC_ts.get_data()[:,:,indexes]))
epochs5_ISC_ts_filtered_high_elem = element_wise_multi(( epochs5_ISC_ts.get_data()[:,:,indexes]))
epochs6_ISC_ts_filtered_high_elem = element_wise_multi(( epochs6_ISC_ts.get_data()[:,:,indexes]))
epochs7_ISC_ts_filtered_high_elem = element_wise_multi(( epochs7_ISC_ts.get_data()[:,:,indexes]))
epochs8_ISC_ts_filtered_high_elem = element_wise_multi(( epochs8_ISC_ts.get_data()[:,:,indexes]))
epochs9_ISC_ts_filtered_high_elem = element_wise_multi(( epochs9_ISC_ts.get_data()[:,:,indexes]))
epochs10_ISC_ts_filtered_high_elem = element_wise_multi(( epochs10_ISC_ts.get_data()[:,:,indexes]))
#(epochs10_ISC_ts.get_data()[:,:,indexes])




info1,raw1_elem,ep1_elem = get_raw_epochs_data_elem(epochs1_ISC_ts_filtered_high_elem)
info2,raw2_elem,ep2_elem = get_raw_epochs_data_elem(epochs2_ISC_ts_filtered_high_elem)
info3,raw3_elem,ep3_elem = get_raw_epochs_data_elem(epochs3_ISC_ts_filtered_high_elem)
info4,raw4_elem,ep4_elem = get_raw_epochs_data_elem(epochs4_ISC_ts_filtered_high_elem)
info5,raw5_elem,ep5_elem = get_raw_epochs_data_elem(epochs5_ISC_ts_filtered_high_elem)
info6,raw6_elem,ep6_elem = get_raw_epochs_data_elem(epochs6_ISC_ts_filtered_high_elem)
info7,raw7_elem,ep7_elem = get_raw_epochs_data_elem(epochs7_ISC_ts_filtered_high_elem)
info8,raw8_elem,ep8_elem = get_raw_epochs_data_elem(epochs8_ISC_ts_filtered_high_elem)
info9,raw9_elem,ep9_elem = get_raw_epochs_data_elem(epochs9_ISC_ts_filtered_high_elem)
info10,raw10_elem,ep10_elem = get_raw_epochs_data_elem(epochs10_ISC_ts_filtered_high_elem)







# In[24]:



# for the top and bottom 10%ile calculation for the source estimation of non-spatially filtered EEG
stc1_high = comp_source_psd(raw1)
stc2_high = comp_source_psd(raw2)
stc3_high = comp_source_psd(raw3)
stc4_high = comp_source_psd(raw4)
stc5_high = comp_source_psd(raw5)
stc6_high = comp_source_psd(raw6)
stc7_high = comp_source_psd(raw7)
stc8_high = comp_source_psd(raw8)
stc9_high = comp_source_psd(raw9)
stc10_high = comp_source_psd(raw10)


# In[67]:


ep1.load_data().get_data()
#scipy.signal.get_window('hamming', 100)
#        delta=(2, 4), theta=(5, 7), alpha=(8, 13), beta=(15, 29), gamma=(30, 40))
#delta = 1,2
#theta = 3,4
#alpha = 


# In[25]:


stc1_high


# In[26]:


#
epochs1_ISC_ts_filtered_low = spatial_filter(np.reshape( epochs1_ISC_ts.get_data()[:,:,indexes2], [91,2060]))
epochs2_ISC_ts_filtered_low = spatial_filter(np.reshape( epochs2_ISC_ts.get_data()[:,:,indexes2], [91,2060]))
epochs3_ISC_ts_filtered_low = spatial_filter(np.reshape( epochs3_ISC_ts.get_data()[:,:,indexes2], [91,2060]))
epochs4_ISC_ts_filtered_low = spatial_filter(np.reshape( epochs4_ISC_ts.get_data()[:,:,indexes2], [91,2060]))
epochs5_ISC_ts_filtered_low = spatial_filter(np.reshape( epochs5_ISC_ts.get_data()[:,:,indexes2], [91,2060]))
epochs6_ISC_ts_filtered_low = spatial_filter(np.reshape( epochs6_ISC_ts.get_data()[:,:,indexes2], [91,2060]))
epochs7_ISC_ts_filtered_low = spatial_filter(np.reshape( epochs7_ISC_ts.get_data()[:,:,indexes2], [91,2060]))
epochs8_ISC_ts_filtered_low = spatial_filter(np.reshape( epochs8_ISC_ts.get_data()[:,:,indexes2], [91,2060]))
epochs9_ISC_ts_filtered_low = spatial_filter(np.reshape( epochs9_ISC_ts.get_data()[:,:,indexes2], [91,2060]))
epochs10_ISC_ts_filtered_low = spatial_filter(np.reshape( epochs10_ISC_ts.get_data()[:,:,indexes2], [91,2060]))





info1_low,raw1_low,ep1_low = get_raw_epochs_data(epochs1_ISC_ts_filtered_low)
info2_low,raw2_low,ep2_low = get_raw_epochs_data(epochs2_ISC_ts_filtered_low)
info3_low,raw3_low,ep3_low = get_raw_epochs_data(epochs3_ISC_ts_filtered_low)
info4_low,raw4_low,ep4_low = get_raw_epochs_data(epochs4_ISC_ts_filtered_low)
info5_low,raw5_low,ep5_low = get_raw_epochs_data(epochs5_ISC_ts_filtered_low)
info6_low,raw6_low,ep6_low = get_raw_epochs_data(epochs6_ISC_ts_filtered_low)
info7_low,raw7_low,ep7_low = get_raw_epochs_data(epochs7_ISC_ts_filtered_low)
info8_low,raw8_low,ep8_low = get_raw_epochs_data(epochs8_ISC_ts_filtered_low)
info9_low,raw9_low,ep9_low = get_raw_epochs_data(epochs9_ISC_ts_filtered_low)
info10_low,raw10_low,ep10_low = get_raw_epochs_data(epochs10_ISC_ts_filtered_low)




epochs1_ISC_ts_filtered_low_elem = element_wise_multi(( epochs1_ISC_ts.get_data()[:,:,indexes2]))
epochs2_ISC_ts_filtered_low_elem = element_wise_multi(( epochs2_ISC_ts.get_data()[:,:,indexes2]))
epochs3_ISC_ts_filtered_low_elem = element_wise_multi(( epochs3_ISC_ts.get_data()[:,:,indexes2]))
epochs4_ISC_ts_filtered_low_elem = element_wise_multi(( epochs4_ISC_ts.get_data()[:,:,indexes2]))
epochs5_ISC_ts_filtered_low_elem = element_wise_multi(( epochs5_ISC_ts.get_data()[:,:,indexes2]))
epochs6_ISC_ts_filtered_low_elem = element_wise_multi(( epochs6_ISC_ts.get_data()[:,:,indexes2]))
epochs7_ISC_ts_filtered_low_elem = element_wise_multi(( epochs7_ISC_ts.get_data()[:,:,indexes2]))
epochs8_ISC_ts_filtered_low_elem = element_wise_multi(( epochs8_ISC_ts.get_data()[:,:,indexes2]))
epochs9_ISC_ts_filtered_low_elem = element_wise_multi(( epochs9_ISC_ts.get_data()[:,:,indexes2]))
epochs10_ISC_ts_filtered_low_elem = element_wise_multi(( epochs10_ISC_ts.get_data()[:,:,indexes2]))
#(epochs10_ISC_ts.get_data()[:,:,indexes])




info1,raw1_low_elem,ep1_low_elem = get_raw_epochs_data_elem(epochs1_ISC_ts_filtered_low_elem)
info2,raw2_low_elem,ep2_low_elem = get_raw_epochs_data_elem(epochs2_ISC_ts_filtered_low_elem)
info3,raw3_low_elem,ep3_low_elem = get_raw_epochs_data_elem(epochs3_ISC_ts_filtered_low_elem)
info4,raw4_low_elem,ep4_low_elem = get_raw_epochs_data_elem(epochs4_ISC_ts_filtered_low_elem)
info5,raw5_low_elem,ep5_low_elem = get_raw_epochs_data_elem(epochs5_ISC_ts_filtered_low_elem)
info6,raw6_low_elem,ep6_low_elem = get_raw_epochs_data_elem(epochs6_ISC_ts_filtered_low_elem)
info7,raw7_low_elem,ep7_low_elem = get_raw_epochs_data_elem(epochs7_ISC_ts_filtered_low_elem)
info8,raw8_low_elem,ep8_low_elem = get_raw_epochs_data_elem(epochs8_ISC_ts_filtered_low_elem)
info9,raw9_low_elem,ep9_low_elem = get_raw_epochs_data_elem(epochs9_ISC_ts_filtered_low_elem)
info10,raw10_low_elem,ep10_low_elem = get_raw_epochs_data_elem(epochs10_ISC_ts_filtered_low_elem)


freqs = list()
def freq(epochs):
    return mne.time_frequency.psd_welch(epochs,fmax=40,n_fft=2000)



# for the top and bottom 10%ile calculation for the source estimation of non-spatially filtered EEG
stc1_low = comp_source_psd(raw1_low)
stc2_low = comp_source_psd(raw2_low)
stc3_low = comp_source_psd(raw3_low)
stc4_low = comp_source_psd(raw4_low)
stc5_low = comp_source_psd(raw5_low)
stc6_low = comp_source_psd(raw6_low)
stc7_low = comp_source_psd(raw7_low)
stc8_low = comp_source_psd(raw8_low)
stc9_low = comp_source_psd(raw9_low)
stc10_low = comp_source_psd(raw10_low)

freqs.append(freq(ep1_elem)[0] - freq(ep1_low_elem)[0])
freqs.append(freq(ep2_elem)[0] - freq(ep2_low_elem)[0])
freqs.append(freq(ep3_elem)[0] - freq(ep3_low_elem)[0])
freqs.append(freq(ep4_elem)[0] - freq(ep4_low_elem)[0])
freqs.append(freq(ep5_elem)[0] - freq(ep5_low_elem)[0])
freqs.append(freq(ep6_elem)[0] - freq(ep6_low_elem)[0])
freqs.append(freq(ep7_elem)[0] - freq(ep7_low_elem)[0])
freqs.append(freq(ep8_elem)[0] - freq(ep8_low_elem)[0])
freqs.append(freq(ep9_elem)[0] - freq(ep9_low_elem)[0])
freqs.append(freq(ep10_elem)[0] - freq(ep10_low_elem)[0])


# In[86]:


freqs


# In[29]:



import seaborn as sns
sns.set_theme()
plt.plot(freq(ep1)[1],np.mean(freqs,axis=0).reshape(161,))
plt.title('Power difference of the EEG time-series filtered by first component (averaged)')
plt.xlabel('Hz')
plt.ylabel('Power difference in volts^2')

#np.mean(freqs,axis=0).reshape(161,)


# In[200]:


freqs1 = np.reshape(freqs,[10,161])
plt.plot(freq(ep1)[1],(freqs1.T))
plt.title('Power difference of the EEG time-series filtered by first component (all subjects)')
plt.xlabel('Hz')
plt.ylabel('Power difference in volts^2')


# In[31]:


#import scipy
#scipy.stats.ttest_1samp((np.reshape(np.mean(np.reshape(freqs,[10,161])[:,alpha],axis=0),[57,1])),popmean=0)
def ttest(s):
    alpha = np.where((np.logical_and(freq(ep1)[1]>=s[0], freq(ep1)[1]<=s[1])))
    _,a,_ = mne.stats.permutation_t_test(np.reshape(np.mean(np.reshape(freqs,[10,161])[:,alpha],axis=0),[len(alpha[0]),1]),n_permutations=1000)
    return a
#(np.mean(np.reshape(freqs,[10,161])[:,alpha],axis=0),)


# In[121]:


get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('gui', 'qt')

to_not_miss = d[band[4]]
l = ttest(to_not_miss)
plt.xlabel('p-value = {}'.format(l))
    
    #plt.plot(freq(ep1)[1][alpha],np.mean(freqs,axis=0).reshape(161,)[alpha])
alpha = np.where((np.logical_and(freq(ep1)[1]>=to_not_miss[0], freq(ep1)[1]<=to_not_miss[1])))

freqs1 = np.reshape(freqs,[10,161])
freqs2 = (np.reshape(freqs1[:,alpha],[10,len(alpha[0])]))
freq_mean = np.mean(freqs2,axis=0)

plt.boxplot(freq_mean)
    
plt.title('{} band'.format(band[4]))
plt.ylabel('Power difference (volts)^2 ')
plt.tight_layout()

plt.figure()

#plt.savefig('{} band.jpg'.format(band[1]))


# In[89]:


d[band[2]]


# In[224]:




a = 2  # number of rows
b = 5  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(14,10))
band = ['theta','alpha','beta','gamma','delta']
for i in range(5):
    plt.subplot(a, b, c)
    l = ttest(d[band[i]])
    plt.xlabel('p-value = {}'.format(l))
    
    #plt.plot(freq(ep1)[1][alpha],np.mean(freqs,axis=0).reshape(161,)[alpha])
    alpha = np.where((np.logical_and(freq(ep1)[1]>=d[band[i]][0], freq(ep1)[1]<=d[band[i]][1])))

    freqs1 = np.reshape(freqs,[10,161])
    freqs2 = (np.reshape(freqs1[:,alpha],[10,len(alpha[0])]))
    freq_mean = np.mean(freqs2,axis=0)
    plt.boxplot(freq_mean)
    
    plt.title('{} band'.format(band[i]))
    plt.ylabel('Power difference (volts)^2 ')

    c = c + 1
    
plt.tight_layout()
plt.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('gui', 'qt')
def plot_band(kind, band):
    """Plot activity within a frequency band on the subject's brain."""
    #title = "%s\n(%d Hz)" % (freq_bands[band])
    topos[band].plot_topomap(
         scalings=1., cbar_fmt='%0.1f', vmin=0, cmap='inferno')
    brain = stcs_dict[band].plot(
        subject=subject, subjects_dir=subjects_dir, views='cau', hemi='lh',
         colormap='inferno',backend='matplotlib',
        time_viewer=False, show_traces=False,
        clim=dict(kind='percent', lims=(70, 85, 99)), smoothing_steps=10)
    brain.show_view(dict(azimuth=0, elevation=0), roll=0)

fig_theta, brain_theta = plot_band('vv', 'alpha')


# In[37]:


def high_minus_low(high,low,band):
    return high[band] - low[band]


# In[ ]:





# ## Low ISC

# In[38]:



def difference_and_t_test(band):
    alpha_difference_1 = high_minus_low(stc1_high,stc1_low,band)
    alpha_difference_2 = high_minus_low(stc2_high,stc2_low,band)
    alpha_difference_3 = high_minus_low(stc3_high,stc3_low,band)
    alpha_difference_4 = high_minus_low(stc4_high,stc4_low,band)
    alpha_difference_5 = high_minus_low(stc5_high,stc5_low,band)
    alpha_difference_6 = high_minus_low(stc6_high,stc6_low,band)
    alpha_difference_7 = high_minus_low(stc7_high,stc7_low,band)
    alpha_difference_8 = high_minus_low(stc8_high,stc8_low,band)
    alpha_difference_9 = high_minus_low(stc9_high,stc9_low,band)
    alpha_difference_10 = high_minus_low(stc10_high,stc10_low,band)
    averaged_alpha =(alpha_difference_1+alpha_difference_2+alpha_difference_3+alpha_difference_4+alpha_difference_5          +alpha_difference_6+alpha_difference_7+alpha_difference_8+alpha_difference_9+alpha_difference_10)/10
    a,b,c = mne.stats.permutation_t_test(averaged_alpha.data,n_permutations=1000)
    alpha_stacked = (np.hstack((alpha_difference_1.data,alpha_difference_2.data,                   alpha_difference_3.data,alpha_difference_4.data,                   alpha_difference_5.data,alpha_difference_6.data,                   alpha_difference_7.data,alpha_difference_8.data,                   alpha_difference_9.data,alpha_difference_10.data)))
    return averaged_alpha,alpha_stacked,b


# In[82]:


averaged,to_test,b = difference_and_t_test('gamma')
#plt.hist(averaged.data)


# In[83]:


indices_for_slicing_beyond_percentiles = np.where( np.logical_or( (averaged.data <np.percentile(averaged.data,10)), (averaged.data>np.percentile(averaged.data,90))))

averaged.data[list(set(list(range(20484)))-set(indices_for_slicing_beyond_percentiles[0]))] =0


# In[271]:


get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('gui', 'qt')
def plot(views,hemi,string):
    
    brain = averaged.plot(subjects_dir=subjects_dir, time_unit='s',hemi=hemi,background='grey',views=views
                          ,backend='matplotlib',colormap='seismic',clim=dict(kind='value', lims=[-averaged.data.max(), np.median(averaged.data), averaged.data.max()]))
    plt.title('Condition -"High- Low"')
    #brain.savefig('S4B2/results_high_minus_low')
    return None


# In[272]:


plot('lat','lh','Lateral Left')
plot('lat','rh','Lateral Right')
plot('cau','lh','Caudal Left')
plot('cau','rh','Caudal Right')
plot('med','lh','Medial Left')
plot('med','rh','Medial Right')


# In[122]:



l = difference_and_t_test(band[4])
plt.xlabel('p-value = {}'.format(l[2]))


plt.boxplot(l[0].data)

plt.ylabel('Power difference in micro volts')
plt.title('{} band (source level)'.format(band[4]))
plt.figure()
plt.tight_layout()
plt.show(block=False)
#plt.savefig('{} band(source level).jpg'.format(band[0]))
#plt.close()


# In[111]:


d =dict(delta=(2, 4), theta=(5, 7), alpha=(8, 13), beta=(15, 29), gamma=(30, 40))

l = ttest(d[band[1]])
plt.xlabel('p-value = {}'.format(l))
    #plt.plot(freq(ep1)[1][alpha],np.mean(freqs,axis=0).reshape(161,)[alpha])
    
alpha = np.where((np.logical_and(freq(ep1)[1]>=d[band[1]][0], freq(ep1)[1]<=d[band[1]][1])))

freqs1 = np.reshape(freqs,[10,161])
freqs2 = (np.reshape(freqs1[:,alpha],[10,len(alpha[0])]))
freq_mean = np.mean(freqs2,axis=0)
plt.boxplot(freq_mean)
plt.tight_layout()
plt.title('{} band'.format(band[1]))
plt.ylabel('Power difference (volts)^2 ')


# In[34]:


a = 2  # number of rows
b = 3  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(14,10))
band = ['theta','alpha','beta','gamma','delta']
for i in range(5):
    plt.subplot(a, b, c)
    l = difference_and_t_test(band[i])
    plt.xlabel('p-value = {}'.format(l[2]))
    print(l[0])
    plt.boxplot(l[0].data)
    plt.title('{} band'.format(band[i]))
    plt.ylabel('Power difference in micro volts')

    c = c + 1
    
plt.tight_layout()
plt.show()


# In[61]:


#np.shape(epochs_averaged)
#epochs_ready_ISC_ts_low = np.reshape(epochs_averaged_ISC_ts_low,[1,91,2060])
b


# In[48]:


events = np.zeros([2,3],dtype=int)
#def ISCed_inversion(subject,events):
 #   e = epochs(subject,events)
events[0][0] = 1
events[0][2] = 83


info_low = mne.create_info(sub1_raw.info['ch_names'],sfreq=500,ch_types = 'eeg')

raw_low = mne.io.RawArray(epochs_averaged_ISC_ts_low,info)
ep_low = mne.EpochsArray(epochs_ready_ISC_ts_low,info_low)



# In[ ]:





# In[49]:


#mne.make_fixed_length_events(raw,start=0,duration = 0.5)
raw_low.set_montage('GSN-HydroCel-129')


# In[55]:


stc_psd_low,sensor_psd_low = comp_source_psd(raw_low)


# In[68]:



freq_bands_2 = dict(
        delta=(2, 4), theta=(5, 7), alpha=(8, 13), beta=(15, 29), gamma=(30, 45))
topos_2 = dict(vv=dict(), opm=dict())
stcs_dict_2 = dict(vv=dict(), opm=dict())
    
    
topo_norm_2 = sensor_psd_low.data.sum(axis=1, keepdims=True)
stc_norm_2 = stc_psd_low.sum() 

for band, limits in freq_bands_2.items():
        data = sensor_psd_low.copy().crop(*limits).data.sum(axis=1, keepdims=True)
        topos_2[band] = mne.EvokedArray(
            100 * data / topo_norm_2, sensor_psd_low.info)
        stcs_dict_2[band] =             100 * stc_psd_low.copy().crop(*limits).sum() / stc_norm_2.data


# In[77]:


get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('gui', 'qt')
def plot_band2(kind, band):
    """Plot activity within a frequency band on the subject's brain."""
    #title = "%s\n(%d Hz)" % (freq_bands[band])
    topos_2[band].plot_topomap(
         scalings=1., cbar_fmt='%0.1f',vmin=4.434287102645273e-10, cmap='inferno')
    
    brain = stcs_dict_2[band].plot(
        subject=subject, subjects_dir=subjects_dir, views='cau', hemi='lh',
         colormap='inferno',backend='matplotlib',
        time_viewer=False, show_traces=False,
        clim=dict(kind='percent', lims=(70, 85, 99)), smoothing_steps=10)
    brain.show_view(dict(azimuth=0, elevation=0), roll=0)

fig_theta, brain_theta = plot_band2('vv', 'alpha')


# In[ ]:





# In[29]:


mins =list()
maxs = list()
for band, limits in freq_bands.items():
        
        mins.append(np.min(topos[band].data))
        maxs.append(np.max(topos[band].data))
print(np.where(mins==np.min(mins)))
print(np.where(maxs==np.max(maxs)))
#topos


# In[ ]:





# In[ ]:





# # Independent Source Inversion

# In[12]:


indexes = np.hstack([np.arange(159*500-50,159*500+50)])#,np.arange(67*500,72*500),np.arange(77*500,84*500),np.arange(130*500,136*500),np.arange(155*500,165*500)])
indexes_low = np.hstack([np.arange(100*500-50,100*500+50)])


# In[13]:



def ISCed_inversion(indexed_subject_data):
    #e = epochs(subject,events)
    
    ep = mne.EpochsArray(indexed_subject_data,mne.create_info(sub1_raw.info['ch_names'],sfreq=500,ch_types = 'eeg'))
    raw = mne.io.RawArray(ep.get_data().reshape(91,100),sub1_raw.info)


    cov_ =mne.compute_covariance(ep)


    
    inverse_operator = make_inverse_operator(sub1_raw.info, fwd_model, cov_)


    method = "eLORETA"
    snr = 3.
    lambda2 = 1. / snr ** 2


    stc = apply_inverse_epochs(ep, inverse_operator, lambda2,
                             method=method, pick_ori=None, verbose=True)
    
    
    return stc


# In[22]:


np.shape(epochs1_ISC_ts.get_data()[:,:,indexes])


# In[92]:


#isc_results['condition1']['A'][0]
#sub1_trial = isc_results['condition1']['A'][0] @ sub1_raw.get_data()
import tqdm
#sub1_trial = np.zeros([91,85001])
#a = [sub1_trial[:,i] (sub1_raw.get_data()[:,i]*isc_results['condition1']['A'][0]) for i in tqdm.tqdm(range(86040))]
 #sub1_trial[:,i] = (sub1_raw.get_data()[:,i]*isc_results['condition1']['A'][0])   

def element_wise_multi(epochs):
    subjects = isc_results['condition1']['A'][0] @ epochs
    return subjects

(element_wise_multi(epochs5_ISC_ts.get_data()[:,:,indexes]))

#np.shape(isc_results['condition1']['A'][0],sub1_raw.get_data())


# In[17]:


def epochs_filter(epochs1_ISC_ts,epochs2_ISC_ts,epochs3_ISC_ts,epochs4_ISC_ts,epochs5_ISC_ts,epochs6_ISC_ts,                 epochs7_ISC_ts,epochs8_ISC_ts,epochs9_ISC_ts,epochs10_ISC_ts,indexes):
    epochs1_ISC_ts_filtered = element_wise_multi(epochs1_ISC_ts.get_data()[:,:,indexes].reshape([91,100]))
    epochs2_ISC_ts_filtered = element_wise_multi(epochs2_ISC_ts.get_data()[:,:,indexes].reshape([91,100]))
    epochs3_ISC_ts_filtered = element_wise_multi(epochs3_ISC_ts.get_data()[:,:,indexes].reshape([91,100]))
    epochs4_ISC_ts_filtered = element_wise_multi(epochs4_ISC_ts.get_data()[:,:,indexes].reshape([91,100]))
    epochs5_ISC_ts_filtered = element_wise_multi(epochs5_ISC_ts.get_data()[:,:,indexes].reshape([91,100]))
    epochs6_ISC_ts_filtered = element_wise_multi(epochs6_ISC_ts.get_data()[:,:,indexes].reshape([91,100]))
    epochs7_ISC_ts_filtered = element_wise_multi(epochs7_ISC_ts.get_data()[:,:,indexes].reshape([91,100]))
    epochs8_ISC_ts_filtered = element_wise_multi(epochs8_ISC_ts.get_data()[:,:,indexes].reshape([91,100]))
    epochs9_ISC_ts_filtered = element_wise_multi(epochs9_ISC_ts.get_data()[:,:,indexes].reshape([91,100]))
    epochs10_ISC_ts_filtered = element_wise_multi(epochs10_ISC_ts.get_data()[:,:,indexes].reshape([91,100]))
    
    return epochs1_ISC_ts_filtered,epochs2_ISC_ts_filtered,            epochs3_ISC_ts_filtered,epochs4_ISC_ts_filtered,            epochs5_ISC_ts_filtered,epochs6_ISC_ts_filtered,            epochs7_ISC_ts_filtered,epochs8_ISC_ts_filtered,            epochs9_ISC_ts_filtered,epochs10_ISC_ts_filtered


# In[206]:


fwd_model = mne.make_forward_solution(sub1_raw.info, trans=trans, src=source_space, bem=bem, eeg=True, mindist=5.0)

def inversion(epochs1_ISC_ts_filtered,epochs2_ISC_ts_filtered,epochs3_ISC_ts_filtered,epochs4_ISC_ts_filtered,             epochs5_ISC_ts_filtered,epochs6_ISC_ts_filtered,epochs7_ISC_ts_filtered,epochs8_ISC_ts_filtered,             epochs9_ISC_ts_filtered,epochs10_ISC_ts_filtered):
    
    sub1_ISC_trial = ISCed_inversion(epochs1_ISC_ts_filtered.reshape([1,91,100]))
    sub2_ISC_trial = ISCed_inversion(epochs2_ISC_ts_filtered.reshape([1,91,100]))
    sub3_ISC_trial = ISCed_inversion(epochs3_ISC_ts_filtered.reshape([1,91,100]))
    sub4_ISC_trial = ISCed_inversion(epochs4_ISC_ts_filtered.reshape([1,91,100]))
    sub5_ISC_trial = ISCed_inversion(epochs5_ISC_ts_filtered.reshape([1,91,100]))
    sub6_ISC_trial = ISCed_inversion(epochs6_ISC_ts_filtered.reshape([1,91,100]))
    sub7_ISC_trial = ISCed_inversion(epochs7_ISC_ts_filtered.reshape([1,91,100]))
    sub8_ISC_trial = ISCed_inversion(epochs8_ISC_ts_filtered.reshape([1,91,100]))
    sub9_ISC_trial = ISCed_inversion(epochs9_ISC_ts_filtered.reshape([1,91,100]))
    sub10_ISC_trial = ISCed_inversion(epochs10_ISC_ts_filtered.reshape([1,91,100]))
    averaged_stc = (sub1_ISC_trial[0]+sub2_ISC_trial[0]+sub3_ISC_trial[0]+sub4_ISC_trial[0]+sub5_ISC_trial[0]+sub6_ISC_trial[0]+sub7_ISC_trial[0]+sub8_ISC_trial[0]+sub9_ISC_trial[0]+sub10_ISC_trial[0])/10
    
    return averaged_stc
            


# In[207]:


epochs1_ISC_ts_filtered,epochs2_ISC_ts_filtered,            epochs3_ISC_ts_filtered,epochs4_ISC_ts_filtered,            epochs5_ISC_ts_filtered,epochs6_ISC_ts_filtered,            epochs7_ISC_ts_filtered,epochs8_ISC_ts_filtered,            epochs9_ISC_ts_filtered,epochs10_ISC_ts_filtered = epochs_filter(epochs1_ISC_ts,epochs2_ISC_ts,epochs3_ISC_ts,epochs4_ISC_ts,epochs5_ISC_ts,epochs6_ISC_ts,                 epochs7_ISC_ts,epochs8_ISC_ts,epochs9_ISC_ts,epochs10_ISC_ts,indexes)


epochs1_ISC_ts_filtered_low,epochs2_ISC_ts_filtered_low,            epochs3_ISC_ts_filtered_low,epochs4_ISC_ts_filtered_low,            epochs5_ISC_ts_filtered_low,epochs6_ISC_ts_filtered_low,            epochs7_ISC_ts_filtered_low,epochs8_ISC_ts_filtered_low,            epochs9_ISC_ts_filtered_low,epochs10_ISC_ts_filtered_low = epochs_filter(epochs1_ISC_ts,epochs2_ISC_ts,epochs3_ISC_ts,epochs4_ISC_ts,epochs5_ISC_ts,epochs6_ISC_ts,                 epochs7_ISC_ts,epochs8_ISC_ts,epochs9_ISC_ts,epochs10_ISC_ts,indexes_low)


# In[208]:


#fwd_model = mne.make_forward_solution(sub1_raw.info, trans=trans, src=source_space, bem=bem, eeg=True, mindist=5.0)
avg_ISC_trial = inversion(epochs1_ISC_ts_filtered,epochs2_ISC_ts_filtered,epochs3_ISC_ts_filtered,epochs4_ISC_ts_filtered,             epochs5_ISC_ts_filtered,epochs6_ISC_ts_filtered,epochs7_ISC_ts_filtered,epochs8_ISC_ts_filtered,             epochs9_ISC_ts_filtered,epochs10_ISC_ts_filtered)


avg_trial_low = inversion(epochs1_ISC_ts_filtered_low,epochs2_ISC_ts_filtered_low,            epochs3_ISC_ts_filtered_low,epochs4_ISC_ts_filtered_low,            epochs5_ISC_ts_filtered_low,epochs6_ISC_ts_filtered_low,            epochs7_ISC_ts_filtered_low,epochs8_ISC_ts_filtered_low,            epochs9_ISC_ts_filtered_low,epochs10_ISC_ts_filtered_low)


# In[84]:





# In[64]:




vertno_max, time_max = averaged_stc.get_peak()
get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('gui', 'qt')

fig, ax = plt.subplots()
ax.plot(1e3 * averaged_stc.times, averaged_stc.data.T)
method = "eLORETA"
ax.set(xlabel='time (ms)', ylabel='%s value' % method)




views = 'lat'
brain = averaged_stc.plot(subjects_dir=subjects_dir, initial_time=time_max, time_unit='s',hemi='lh',background='grey',views=views
                          ,backend='matplotlib', colormap='inferno'
                          ,title='eLORETA for the peak of the peak ')
plt.title('eLORETA for the peak of the peak (Lateral)' )
#surfer_kwargs = dict(
 #   hemi='both', subjects_dir=subjects_dir,
  #  clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
   # initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10)
#brain = stc.plot(**surfer_kwargs)
#brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',scale_factor=0.6, alpha=0.5)
#brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='red',scale_factor=0.6, alpha=0.5)

#brain.title(0.1, 0.9, 'eLORETA for the peak of the peak (%s)' % views ,font_size=14)


# In[59]:


def plot(views,hemi,string):
    
    brain = averaged_stc.plot(subjects_dir=subjects_dir, initial_time=time_max, time_unit='s',hemi=hemi,background='grey',views=views
                          ,backend='matplotlib', colormap='inferno'
                          ,title='eLORETA for the peak of the peak ')
    plt.title('eLORETA (Filtered by 1st comp) for the peak of the peak (%s)' %string)
    brain.savefig('S4B2/Results(Filtered by first component)/%s.jpg' %string)
    return None


# In[112]:


plot('lat','lh','Lateral Left')
plot('lat','rh','Lateral Right')


plot('med','lh','Medial Left')
plot('med','rh','Medial Right')

plot('ros','lh','Rostral Left')
plot('ros','rh','Rostral Right')

plot('cau','lh','Caudal Left')
plot('cau','rh','Caudal Right')

plot('dor','lh','Dorsal Left')
plot('dor','rh','Dorsal Right')


# In[26]:


#def psd_ISC(indexed_subject_data):    
freq_bands = dict(
        delta=(2, 4), theta=(5, 7), alpha=(8, 13), beta=(15, 29), gamma=(30, 45))
topos = dict(vv=dict(), opm=dict())
stcs_dict = dict(vv=dict(), opm=dict())

ep = mne.EvokedArray(e1.get_data()[:,:,indexes].reshape(91,100),mne.create_info(sub1_raw.info['ch_names'],sfreq=500,ch_types = 'eeg'))
raw = mne.io.RawArray(ep.data.reshape(91,100),sub1_raw.info)

    
noise_cov = mne.compute_raw_covariance(raw)
inverse_operator = mne.minimum_norm.make_inverse_operator(
        raw.info, forward=fwd_model, noise_cov=noise_cov, verbose=True)
    

method = "eLORETA"
snr = 3.
lambda2 = 1. / snr ** 2

    
stc_psd, sensor_psd = mne.minimum_norm.compute_source_psd(
        raw, inverse_operator, lambda2=lambda2,method=method,fmax=40,
        dB=False, return_sensor=True, verbose=True)


# In[28]:


#psd_ISC(e1.get_data()[:,:,indexes])
ep.data.shape


# # Statistics at source

# In[30]:



adjacency = mne.spatial_src_adjacency(inverse_operator['src'])


# In[31]:


n_vertices, n_times =20484, 1


# In[32]:


adjacency


# In[ ]:





# In[33]:


vertices = [s['vertno'] for s in inverse_operator['src']]
vertices


# In[ ]:





# In[26]:


spatiotemp = np.append(np.reshape(sub1_ISC_trial[0].data,[1,100,20484]),np.reshape(sub2_ISC_trial[0].data,[1,100,20484]),axis=0)


spatiotemp = np.append(spatiotemp,np.reshape(sub3_ISC_trial[0].data,[1,100,20484]),axis=0)

spatiotemp = np.append(spatiotemp,np.reshape(sub4_ISC_trial[0].data,[1,100,20484]),axis=0)
spatiotemp = np.append(spatiotemp,np.reshape(sub5_ISC_trial[0].data,[1,100,20484]),axis=0)
spatiotemp = np.append(spatiotemp,np.reshape(sub6_ISC_trial[0].data,[1,100,20484]),axis=0)
spatiotemp = np.append(spatiotemp,np.reshape(sub7_ISC_trial[0].data,[1,100,20484]),axis=0)
spatiotemp = np.append(spatiotemp,np.reshape(sub8_ISC_trial[0].data,[1,100,20484]),axis=0)
spatiotemp = np.append(spatiotemp,np.reshape(sub9_ISC_trial[0].data,[1,100,20484]),axis=0)
spatiotemp = np.append(spatiotemp,np.reshape(sub10_ISC_trial[0].data,[1,100,20484]),axis=0)


# In[27]:


spatiotemp2 = np.append(np.reshape(sub1_ISC_trial_low[0].data,[1,100,20484]),np.reshape(sub2_ISC_trial_low[0].data,[1,100,20484]),axis=0)


spatiotemp2 = np.append(spatiotemp2,np.reshape(sub3_ISC_trial_low[0].data,[1,100,20484]),axis=0)

spatiotemp2 = np.append(spatiotemp2,np.reshape(sub4_ISC_trial_low[0].data,[1,100,20484]),axis=0)
spatiotemp2 = np.append(spatiotemp2,np.reshape(sub5_ISC_trial_low[0].data,[1,100,20484]),axis=0)
spatiotemp2 = np.append(spatiotemp2,np.reshape(sub6_ISC_trial_low[0].data,[1,100,20484]),axis=0)
spatiotemp2 = np.append(spatiotemp2,np.reshape(sub7_ISC_trial_low[0].data,[1,100,20484]),axis=0)
spatiotemp2 = np.append(spatiotemp2,np.reshape(sub8_ISC_trial_low[0].data,[1,100,20484]),axis=0)
spatiotemp2 = np.append(spatiotemp2,np.reshape(sub9_ISC_trial_low[0].data,[1,100,20484]),axis=0)
spatiotemp2 = np.append(spatiotemp2,np.reshape(sub10_ISC_trial_low[0].data,[1,100,20484]),axis=0)


# In[57]:


a =mne.time_frequency.psd_welch(ep1,fmax=40)


# In[59]:


plt.plot(a[1],a[0].reshape(21,))


# In[46]:


from scipy import stats as stats
from mne.stats import (spatio_temporal_cluster_1samp_test,summarize_clusters_stc)

n_subjects = 10
p_threshold = 0.001
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu =     spatio_temporal_cluster_1samp_test((np.reshape(to_test.data,[10,1,20484])), adjacency=adjacency, n_jobs=1,
                                       threshold=t_threshold, buffer_size=None,n_permutations=1000,
                                       verbose=True)



good_cluster_inds = np.where(cluster_p_values < 0.05)[0]


# In[83]:


cluster_p_values


# In[197]:


#tstep = averaged.tstep *1000

stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep,p_thresh=0.001,
                                             vertices=vertices,
                                             subject='fsaverage')


# In[ ]:





# In[ ]:





# In[9]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[85]:


import epochs_slicing 
from imp import reload 

reload(epochs_slicing)


# In[90]:


epochs_slicing.epochs(sub1_raw,sub1_events,[83,103,9999],0,170,500,'83')


# In[ ]:





# In[14]:


import nilearn.surface
nilearn.surface.load_surf_data('fsaverage5.gii')


# In[8]:


import numpy as np
np.shape(labels)


# In[15]:


aud_label = [label for label in labels if label.name == 'R_10d_ROI-rh'][0]


# In[ ]:





# In[ ]:





# In[18]:


import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters
from pygsp import plotting as gsp_plt
from nilearn import image, plotting, datasets

rs = np.random.RandomState(42)  # Reproducible results.
W = rs.uniform(size=(30, 30))  # Full graph.
#print(rs.__doc__)
print (W.shape)
W[W < 0.93] = 0  # Sparse graph.
W = W + W.T  # Symmetric graph.
np.fill_diagonal(W, 0)  # No self-loops.
G_simple = graphs.Graph(W)
print('{} nodes, {} edges'.format(G_simple.N, G_simple.Ne))


G_simple.set_coordinates('ring2D')
G_simple.plot()
plotting.plot_matrix(W, colorbar=True)

#print(G.L)


# In[20]:


from pathlib import Path
from scipy import io as sio
from pygsp import graphs

path_Glasser='Glasser_masker.nii.gz'
res_path=''

# Load structural connectivity matrix
connectivity = sio.loadmat('S4B2/GSP/SC_avg56.mat')['SC_avg56']
connectivity.shape
coordinates = sio.loadmat('S4B2/GSP/Glasser360_2mm_codebook.mat')['codeBook'] # coordinates in brain space


G_Comb = graphs.Graph(connectivity,gtype='HCP subject',lap_type='combinatorial',coords=coordinates)# combinatorial laplacian
G=graphs.Graph(connectivity,gtype='HCP subject',lap_type='normalized',coords=coordinates) #
G_RandW=graphs.Graph(connectivity,gtype='HCP subject',lap_type='normalized',coords=coordinates) #
print(G.is_connected())


G.set_coordinates('spring')
#G.plot()   #edges > 10^4 not shown
D=np.array(G.dw)
D.shape


# In[21]:


import numpy as np
with np.load(f"S4B2/GSP/hcp/atlas.npz") as dobj:
    atlas = dict(**dobj)


# In[22]:


l =list()
for i in list(set(atlas['labels_L']))[:-1]:
    l.append(np.mean(stc_isc[0].data[10242:][np.where(i== atlas['labels_L'])]))

for i in list(set(atlas['labels_R']))[:-1]:
    l.append(np.mean(stc_isc[0].data[:10242][np.where(i== atlas['labels_R'])]))
    


# In[26]:


G.compute_fourier_basis()


# In[33]:


from data import sns_plot


sns_plot.plot(range(len(G.gft(np.array(l)))),G.gft(np.array(l)),xlabel='Graph freq',ylabel='gPSD',title='GFTed spatial filter')


# In[32]:


a=0
for i in range(1,len(labels)):
    a+= len(labels[i].values)
a


# In[33]:


G_Comb.set_coordinates()
G_Comb.plot()


# In[38]:


from nilearn.regions import signals_to_img_labels  
# load nilearn label masker for inverse transform
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image, plotting
from nilearn import datasets
from os.path import join as opj


path_Glasser = 'S4B2/GSP/Glasser_masker.nii.gz'


mnitemp = fetch_icbm152_2009()
mask_mni=image.load_img(mnitemp['mask'])
glasser_atlas=image.load_img(path_Glasser)


#print(NiftiMasker.__doc__)


fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(10,10))

signal=[]
U0_brain=[]
signal=np.expand_dims(np.array(G.gft(np.array(l))), axis=0) # add dimension 1 to signal array
U0_brain = signals_to_img_labels(signal,path_Glasser,mnitemp['mask'])
plotting.plot_glass_brain(U0_brain,title='GFTed spatial filter',colorbar=True,plot_abs=False,cmap='spring',display_mode='xz',figure=fig,axes=ax)


# In[ ]:





# In[90]:


from data import surface_plot
from imp import reload 
reload(surface_plot)

v = np.zeros([20484,])
v[:10242][np.where(atlas['labels_L']==-1)] =1
v[10242:][np.where(atlas['labels_R']==-1)] =1

surface_plot.plot(v,'Where in the vertices spaces -1 located at','viridis',None, c_range = [0,1])


# In[79]:


np.sum(v)


# In[62]:





# In[ ]:




