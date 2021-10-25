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


# In[9]:




# In[10]:





# # Importing & Parsing the csv

# In[11]:



path_to_file = '/homes/v20subra/S4B2/filtered_ipynbs/NDARBF805EHN/EEG/preprocessed/csv_format/RestingState_data.csv'
path_to_events = '/homes/v20subra/S4B2/filtered_ipynbs/NDARBF805EHN/EEG/preprocessed/csv_format/RestingState_event.csv'
path_to_montage_glob = '/users/local/Venkatesh/HBN/GSN_HydroCel_129_hbn.sfp'
path_to_montage_ses = '/homes/v20subra/S4B2/filtered_ipynbs/NDARBF805EHN/EEG/preprocessed/csv_format/RestingState_chanlocs.csv'
fs = 500
chans_glob = mne.channels.read_custom_montage(fname = '/users/local/Venkatesh/HBN/GSN_HydroCel_129_hbn.sfp') # read_montage is deprecated
# channels to exclude because noisy (Nentwich paper)
ch_list=['E1', 'E8', 'E14', 'E17', 'E21', 'E25', 'E32', 'E38', 'E43', 'E44', 'E48', 'E49', 'E56', 'E57', 'E63', 'E64', 'E69', 'E73', 'E74', 'E81', 'E82', 'E88', 'E89', 'E94', 'E95', 'E99', 'E100', 'E107', 'E113', 'E114', 'E119', 'E120', 'E121', 'E125', 'E126', 'E127', 'E128']


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
        
    return raw,events_final

def exclude_channels_from_raw(raw,ch_to_exclude):
    '''Return a raw structure where ch_to_exclude are removed'''
    idx_keep = mne.pick_channels(raw.ch_names,include = raw.ch_names,exclude = ch_to_exclude)
    raw.pick_channels([raw.ch_names[pick] for pick in idx_keep])
    return raw


raw, events = csv_to_raw_mne(path_to_file,path_to_montage_ses,fs,path_to_events,montage = 'GSN-HydroCel-129')


# # Adding events into the raw structure

# In[12]:


raw.add_events(events[:-1], stim_channel = 'stim_channel',replace = False)


# In[13]:


raw['stim_channel']#sanity check, it produces 2 arrays.. the last one is just the time slots. 3.52e+02 = 352seconds


# In[14]:


events


# # Plotting the Electrodes

# In[15]:



# kind = kinda standard which has 3D coordinates for 128 electrodes and 3 default things
#montage_plot = mne.channels.make_standard_montage(kind= "GSN-HydroCel-129")  
# Note: By default, the 3d plots displayed here does not show the 3rd axis, thus require a
# a package called qt, can be called with %matplotlib qt

raw.plot_sensors(show_names=True)


# ## Raw data in time domain

# In[8]:


from plotly.graph_objs import Layout, YAxis, Scatter, Annotation, Annotations, Data, Figure, Marker, Font
from plotly import tools
from plotly import graph_objects
import chart_studio.plotly as py
import matplotlib.pyplot as plt

n_channels = 20
start, stop = raw.time_as_index([0, 5])
picks = mne.pick_channels(raw.ch_names, include=raw.ch_names[:n_channels], exclude=[])


data, times = raw[picks[:n_channels], start:stop]
ch_names = [raw.info['ch_names'][p] for p in picks[:n_channels]]
#ch_names


step = 1. / n_channels
kwargs = dict(domain=[1 - step, 1], showticklabels=False, zeroline=False, showgrid=False)

# create objects for layout and traces
layout = Layout(yaxis=YAxis(kwargs), showlegend=False)
traces = [Scatter(x=times, y=data.T[:, 0])]

# loop over the channels
for ii in range(1, n_channels):
        kwargs.update(domain=[1 - (ii + 1) * step, 1 - ii * step])
        layout.update({'yaxis%d' % (ii + 1): YAxis(kwargs), 'showlegend': False})
        traces.append(Scatter(x=times, y=data.T[:, ii], yaxis='y%d' % (ii + 1)))

# add channel names using Annotations
annotations = Annotations([Annotation(x=-0.06, y=0, xref='paper', yref='y%d' % (ii + 1),
                                      text=ch_name, font=Font(size=9), showarrow=False)
                          for ii, ch_name in enumerate(ch_names)])
layout.update(annotations=annotations)

# set the size of the figure and plot it
layout.update(autosize=False, width=1000, height=600)
fig = Figure(data=Data(traces), layout=layout)
fig.show()
fig.write_html("raw_time_domain.html")


# In[ ]:





# # Epoch (ing) the raw data

# In[16]:


epochs = mne.Epochs(raw, events, [20,30,90], tmin=0, tmax=20,preload=True,baseline=(0,None))
epochs_resampled = epochs.resample(250) # Downsampling to 250Hz
np.shape(epochs_resampled.load_data()) # Sanity Check


# # Topo Plot

# In[10]:


layout = mne.find_layout(epochs.info)
epochs.average().plot_topo(layout=layout)


# # Plotting events

# In[17]:


mne.viz.plot_events(events[:-1], sfreq=raw.info['sfreq'])
#The last event 20 falls on 174120 and the last sample is 176386. That's just 4 seconds before the end of the EEG


# # Raw PSD 

# In[18]:


mne.viz.plot_raw_psd(raw,tmax=40,fmax=40,picks=['E22','E20','E23'])


# # Covariance

# In[19]:


# Plotting covariance
covariance = mne.compute_covariance(
    epochs_resampled, tmax=0., method=['shrunk', 'empirical'])


# # Topomap PSD

# In[22]:


#epochs
mne.viz.plot_epochs_psd_topomap(epochs['20']) # Eyes open


# # PSD plot for the electrodes located in Occipital lobe

# In[15]:


##for i in range(0,5): #0 is 90 = beginning of EEG, so skipped
    #plt.title('event ={}. Note: open = 20'.format(events[i][2])) this works nicely if plotted through qt
  ##  print("\t\t\tEVENT IN THE GRAPH BELOW IS = {}".format(events[i][2]))
    ##print("\t\t\tNOTE, EYES OPEN = 20 & EYES CLOSE = 30")
    ##print("\t\t\tOnset at {}s".format(events[i][0]/500))
    ##mne.viz.plot_raw_psd(raw,tmin= events[i][0]/500,tmax=events[i+1][0]/500,fmax=40,picks=['E70','E75','E83'])

    
# Types of waves: https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/brain-waves
    


# In[16]:


#%matplotlib inline
#for i in range(1,5): #0 is 90 = beginning of EEG, so skipped
    #plt.title('event ={}. Note: open = 20'.format(events[i][2])) this works nicely if plotted through qt
    #print("\t\t\tEVENT IN THE GRAPH BELOW IS = {}".format(events[i][2]))
    #print("\t\t\tNOTE, EYES OPEN = 20 & EYES CLOSE = 30")
    #mne.viz.plot_raw_psd(raw,tmin= events[i][0]/500,tmax=events[i+1][0]/500,fmax=40,picks=['E34','E33'])


# # Source Reconstruction

# ### Setting up BEM model

# In[23]:


from mne.datasets import fetch_fsaverage

import os.path as op
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)


subject = 'fsaverage' # Subject ID for the MRI-head transformation
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
source_space = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif') 
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')


# ### Forward Solution

# In[24]:


fwd_model = mne.make_forward_solution(raw.info, trans=trans, src=source_space, bem=bem, eeg=True, mindist=5.0)


# ### Inverse operator with the known forward operator

# In[25]:


from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs

inverse_operator = make_inverse_operator(raw.info, fwd_model, covariance)


# ### Applying Inverse

# In[26]:


method = "eLORETA"
snr = 3.
lambda2 = 1. / snr ** 2

# epochs['30'].average() = Averaged evoked response for the event 30
stc = apply_inverse_epochs(epochs_resampled['30'], inverse_operator, lambda2,
                             method=method, pick_ori=None, verbose=True)


# In[ ]:

np.savez_compressed('/users/local/Venkatesh/Generated_Data/rs_stc',stc=stc)



# In[13]:


vertno_max, time_max = stc[0].get_peak()
get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('gui', 'qt # Solves the black-screen on mayavi front. This 5 letters, 2 words took me hours and hours to figure out :D')

fig, ax = plt.subplots()
ax.plot(1e3 * stc[0].times, stc[0].data[::100, :].T)
ax.set(xlabel='time (ms)', ylabel='%s value' % method)
brain = stc[0].plot(subjects_dir=subjects_dir, initial_time=time_max, time_unit='s',hemi='both')
#surfer_kwargs = dict(
 #   hemi='both', subjects_dir=subjects_dir,
  #  clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
   # initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10)
#brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',scale_factor=0.6, alpha=0.5)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='red',scale_factor=0.6, alpha=0.5)

brain.add_text(0.1, 0.9, 'eLORETA', 'title' ,font_size=14)


# # Computing PSD at source level (Label = Occipital Area)

# In[27]:


from mne.minimum_norm import read_inverse_operator, compute_source_psd_epochs
from mne.datasets import sample


# In[31]:



data_path = sample.data_path()
label_name ='Vis-rh.label' 
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


# In[ ]:

print(stcs)



# In[ ]:





# In[ ]:





# In[30]:


freq_bands = dict(
    delta=(2, 4), theta=(5, 7), alpha=(8, 13), beta=(15, 29), gamma=(30, 45))
topos = dict(vv=dict(), opm=dict())
stcs_dict = dict(vv=dict(), opm=dict())


noise_cov = mne.compute_raw_covariance(raw)
inverse_operator = mne.minimum_norm.make_inverse_operator(
        raw.info, forward=fwd_model, noise_cov=noise_cov, verbose=True)

stc_psd, sensor_psd = mne.minimum_norm.compute_source_psd(
        raw, inverse_operator, lambda2=lambda2,
        dB=False, return_sensor=True, verbose=True)

topo_norm = sensor_psd.data.sum(axis=1, keepdims=True)
stc_norm = stc_psd.sum()  # same operation on MNE object, sum across freqs
    # Normalize each source point by the total power across freqs


# In[31]:


for band, limits in freq_bands.items():
        data = sensor_psd.copy().crop(*limits).data.sum(axis=1, keepdims=True)
        topos[band] = mne.EvokedArray(
            100 * data / topo_norm, sensor_psd.info)
        stcs_dict[band] =             100 * stc_psd.copy().crop(*limits).sum() / stc_norm.data


# In[32]:


def plot_band(kind, band):
    """Plot activity within a frequency band on the subject's brain."""
    #title = "%s\n(%d Hz)" % (freq_bands[band])
    topos[band].plot_topomap(
         scalings=1., cbar_fmt='%0.1f', vmin=0, cmap='inferno')
    brain = stcs_dict[band].plot(
        subject=subject, subjects_dir=subjects_dir, views='cau', hemi='both',
         colormap='inferno',
        time_viewer=False, show_traces=False,
        clim=dict(kind='percent', lims=(70, 85, 99)), smoothing_steps=10)
    brain.show_view(dict(azimuth=0, elevation=0), roll=0)

fig_theta, brain_theta = plot_band('vv', 'alpha')


# In[65]:


get_ipython().run_line_magic('matplotlib', 'notebook')

topos['alpha'].plot_topomap(scalings=1., cbar_fmt='%0.1f', title='Alpha', vmin=0, cmap='inferno',outlines='skirt')


# # PSD for the two events

# In[36]:


stcs_averaged_eyes_open = np.sum(stcs)/5
stcs_averaged_eyes_closed = np.sum(stcs2)/5


# In[37]:


import seaborn as sns
sns.set_theme()

m,s,top,bottom = mean_std(stcs_averaged_eyes_closed)

plt.plot(stcs_averaged_eyes_closed.times,m, linewidth=2) 
plt.fill_between(stcs_averaged_eyes_closed.times,bottom, top, color='b', alpha=.1)

plt.axis([0,40,0e-9,2.8e-9])

plt.xlabel('Frequency (Hz)')
plt.ylabel('V^2/hz (PSD)')
plt.title('Source PSD for eyes closed')
plt.show()


# In[38]:

def mean_std(data1):

    m1 = np.mean(data1.data.T,axis=1)
    s = 2 * np.std(data1.data.T,axis=1)

    top = m+s
    bottom = m-s
    return m1,s,top,bottom
stcs_to_plot = [stcs_averaged_eyes_closed,stcs_averaged_eyes_open]

for i in range(1):
    
    fig1 = plt.gcf()

    m,s,top,bottom= mean_std(stcs_to_plot[i])
    plt.plot(stcs_to_plot[i].times,m, linewidth=2,color='r',label='Eyes-closed') 
    plt.fill_between(stcs_to_plot[i].times,bottom, top, color='r', alpha=.1)

    m,s,top,bottom= mean_std(stcs_to_plot[1])
    plt.plot(stcs_to_plot[1].times,m, linewidth=2,color='b',label='Eyes-open')
    plt.fill_between(stcs_to_plot[1].times,bottom, top, color='b', alpha=.1)
    plt.legend()

    plt.axis([0,40,0e-9,2.8e-9])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('V^2/hz (PSD)')
    plt.title('Source PSD for the events')
    plt.show()
    fig1.savefig('PSD.png',dpi=700)

# # Null Hypothesis test

# In[40]:




# In[]
from scipy import stats
import scipy
#a = []
a = stats.ttest_rel(np.average(stcs_averaged_eyes_closed.data[:,160:262],axis=1),np.average(stcs_averaged_eyes_open.data[:,160:262],axis=1))
#np.shape(stcs_averaged_eyes_open.data)
a

#8-13

#260*40/801 

#160
#261
#801/5


# %%
