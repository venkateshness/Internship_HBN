
import mne
import pathlib
from mne.externals.pymatreader import read_mat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d

##
# In[3]:


from warnings import simplefilter 
simplefilter(action='ignore', category=DeprecationWarning)


# In[4]:


import os
os.chdir(r'/usr/slurm/venkatesh/HBN/')
subjs = sorted(os.listdir())[1:-3]
#cd


# In[5]:


cd


# In[6]:


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


# # Resting State <a id='resting-state'></a>

# In[6]:


resting_state_sub1_raw,resting_state_sub1_events = preparation_resting_state('task1/NDARBF805EHN')
#resting_state_sub1_events


# ### Raw structure

# In[7]:


resting_state_sub1_raw.info


# ### Plotting the Electrodes

# In[8]:



# kind = kinda standard which has 3D coordinates for 128 electrodes and 3 default things
#montage_plot = mne.channels.make_standard_montage(kind= "GSN-HydroCel-129")  
# Note: By default, the 3d plots displayed here does not show the 3rd axis, thus require a
# a package called qt, can be called with 
get_ipython().run_line_magic('matplotlib', 'qt')

resting_state_sub1_raw.plot_sensors(show_names=True)


# ### Raw data plot in Time domain

# In[2]:


from data import raw_time_domain
from imp import reload 

reload(raw_time_domain)
raw_time_domain.time_domain_plot(resting_state_sub1_raw).show()


# ### Epoch (ing) the raw data

# In[10]:


epochs = mne.Epochs(resting_state_sub1_raw, resting_state_sub1_events, [20,30,90], tmin=0, tmax=20,preload=True,baseline=(0,None))
epochs_resampled = epochs.resample(250) # Downsampling to 250Hz
np.shape(epochs_resampled.load_data()) # Sanity Check


# ### Topo Plot

# In[11]:


layout = mne.find_layout(epochs.info)
get_ipython().run_line_magic('matplotlib', 'inline')
epochs.average().plot_topo(layout=layout)


# ### Plotting events

# In[12]:


mne.viz.plot_events(resting_state_sub1_events[:-1], sfreq=resting_state_sub1_raw.info['sfreq'])
#The last event 20 falls on 174120 and the last sample is 176386. That's just 4 seconds before the end of the EEG


# ### Raw PSD 

# In[13]:


mne.viz.plot_raw_psd(resting_state_sub1_raw,tmax=40,fmax=40,picks=['E22','E20','E23'])


# ### Topomap PSD

# In[15]:


#epochs
mne.viz.plot_epochs_psd_topomap(epochs['20']) # Eyes open


# ### PSD plot for the electrodes located in Occipital lobe

# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
for i in range(0,5): #0 is 90 = beginning of EEG, so skipped
    #plt.title('event ={}. Note: open = 20'.format(events[i][2])) this works nicely if plotted through qt
    print("\t\t\tEVENT IN THE GRAPH BELOW IS = {}".format(resting_state_sub1_events[i][2]))
    print("\t\t\tNOTE, EYES OPEN = 20 & EYES CLOSE = 30")
    print("\t\t\tOnset at {}s".format(resting_state_sub1_events[i][0]/500))
    mne.viz.plot_raw_psd(resting_state_sub1_raw,tmin= resting_state_sub1_events[i][0]/500,tmax=resting_state_sub1_events[i+1][0]/500,fmax=40,picks=['E70','E75','E83'])

    
# Types of waves: https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/brain-waves
    
#%matplotlib inline
#for i in range(1,5): #0 is 90 = beginning of EEG, so skipped
    #plt.title('event ={}. Note: open = 20'.format(events[i][2])) this works nicely if plotted through qt
    #print("\t\t\tEVENT IN THE GRAPH BELOW IS = {}".format(events[i][2]))
    #print("\t\t\tNOTE, EYES OPEN = 20 & EYES CLOSE = 30")
    #mne.viz.plot_raw_psd(raw,tmin= events[i][0]/500,tmax=events[i+1][0]/500,fmax=40,picks=['E34','E33'])


# ### Source Reconstruction

# In[17]:



from data import source_inversion,fwd_model
reload(source_inversion)
#reload(fwd_model)

#epochs_ISC = mne.EpochsArray(np.reshape(isc_results['condition1']['A'][:],[1,91,91]),mne.create_info(sub1_raw.info['ch_names'],sfreq=1,ch_types = 'eeg'))
forward_model = fwd_model.fwd(resting_state_sub1_raw)

resting_state_cov, resting_state_inverse_operator = source_inversion.inversion(resting_state_sub1_raw,epochs_resampled,forward_model,0,['shrunk', 'empirical'])


# In[19]:


method = "eLORETA"
snr = 3.
lambda2 = 1. / snr ** 2

# epochs['30'].average() = Averaged evoked response for the event 30

'''Memory gets exhausted if entire 352s of data is used'''
#stc = apply_inverse_epochs(epochs_resampled['20'], resting_state_inverse_operator, lambda2,
 #                            method=method, pick_ori=None, verbose=True)


# ### Source PSD epochs for Occipital region

# In[20]:


#stc2=stc[0].in_label(mne.Label(inverse_operator['src'][0]['vertno'], hemi='lh') +
    #              mne.Label(inverse_operator['src'][1]['vertno'], hemi='rh'))
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, compute_source_psd_epochs

data_path = sample.data_path()
label_name ='Vis-rh.label' # Have to use 2 labels at the same, but will deal with this later
fname_label = data_path + '/MEG/sample/labels/%s' % label_name
label_name2 = 'Vis-lh.label'
fname_label2 = data_path + '/MEG/sample/labels/%s' % label_name2
label = mne.read_label(fname_label)
label2 = mne.read_label(fname_label2)
bihemi = mne.BiHemiLabel(label,label2)

stcs = compute_source_psd_epochs(epochs_resampled['20'], resting_state_inverse_operator, lambda2=lambda2,
                                 method=method, fmin=0, fmax=40, label=bihemi,
                                  verbose=True)

stcs2 = compute_source_psd_epochs(epochs_resampled['30'], resting_state_inverse_operator, lambda2=lambda2,
                                 method=method, fmin=0, fmax=40, label=bihemi,
                                  verbose=True)


# ### PSD at source

# In[31]:


from data import compute_psd_at_source
reload(compute_psd_at_source)
compute_psd_at_source.compute(resting_state_sub1_raw,forward_model)


# In[32]:


get_ipython().run_line_magic('matplotlib', 'notebook')

topos['alpha'].plot_topomap(scalings=1., cbar_fmt='%0.1f', title='Alpha', vmin=0, cmap='inferno',outlines='skirt')


# In[ ]:





# ### Source PSD for eyes closed

# In[38]:


stcs_averaged_eyes_open = np.sum(stcs)/5
stcs_averaged_eyes_closed = np.sum(stcs2)/5

m = np.mean(stcs_averaged_eyes_closed.data.T,axis=1)
s = 2 * np.std(stcs_averaged_eyes_closed.data.T,axis=1)

top = m+s
bottom = m-s

from data import sns_plot
reload(sns_plot)


sns_plot.plot(x = stcs_averaged_eyes_closed.times, y = m, xlabel = 'Frequency (Hz)',
 ylabel = 'V^2/hz (PSD)',
 title = 'Source PSD for eyes closed',
 color='b',\
 top = top, bottom = bottom,axis=[0,40,0e-9,2.8e-9],fill_between=True)


# ### Source PSD for eyes opened

# In[47]:


m1 = np.mean(stcs_averaged_eyes_open.data.T,axis=1)
s = 2 * np.std(stcs_averaged_eyes_open.data.T,axis=1)

top = m1+s
bottom = m1-s

sns_plot.plot(x = stcs_averaged_eyes_open.times, y = m1,
 xlabel = 'Frequency (Hz)',
 ylabel = 'V^2/hz (PSD)',
 title = 'Source PSD for eyes opened',
 color='b',
 top = top, bottom = bottom,axis=[0,40,0e-9,2.8e-9],fill_between=True)


# ### Null Hypothesis T-test

# In[55]:


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


# # 2. Video-Watching state (Inter-subject Correlation Study)  <a id='ISC'></a>

# In[7]:


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


# In[11]:


# Import
from data import epochs_slicing 
from data import CCA
from data import plot_matplotlib
from data import source_inversion, surface_plot, fwd_model
from imp import reload 


reload(epochs_slicing)
reload(CCA)
reload(source_inversion)
reload(surface_plot)
reload(fwd_model)


# In[12]:


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


# In[13]:


[W,ISC] = CCA.train_cca(dic)
#np.shape( sub1_raw.get_data() )

isc_results = dict()
for cond_key, cond_values in dic.items():
    isc_results[str(cond_key)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'], CCA.apply_cca(cond_values, W, 500)))


# In[14]:


plt.matshow( (isc_results['condition1']['ISC_bysubject']).T)
plt.title('Intra Subject Study')
plt.ylabel('Subjects')
plt.xlabel('Components')

cb =plt.colorbar()
cb.ax.set_title('ISC')


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


# ### Noise Floor

# In[81]:


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
    chunked= chunked.reshape(1,34,2500) #5s chunk
    return chunked

valstest = []
for i in tqdm(range(1000)):
    shuffled = shuffle(dic)
    isc_resultstest_ = dict()
    print('step {}'.format(i))
    for cond_key, cond_values in shuffled.items():
        isc_resultstest_[str(cond_key)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'], CCA.apply_cca(cond_values, W, 500)))
        #print(np.mean(isc_results['condition1']['ISC_persecond'][0]))
        valstest.append(isc_resultstest_['condition1']['ISC_persecond'])
        #print(np.shape(isc_resultstest_['condition1']['ISC_persecond']))
#np.append(sub1_raw.get_data()[:91,516:85779].reshape(1,91,85263),sub2_raw.get_data()[:91,516:85779].reshape(1,91,85263) ,axis=0)


# In[104]:


#significance = np.where(np.max(np.array(valstest)[:,0,:],axis=0)<isc_results['condition1']['ISC_persecond'][0])
#np.savez('noise_floor.npz', a=valstest)

np.shape(np.load('noise_floor.npz')['a'])


# In[77]:


get_ipython().run_line_magic('matplotlib', 'qt')
reload(sns_plot)
sns_plot.plot(range(1,171),isc_results['condition1']['ISC_persecond'][0],xlabel = 'time (s)', ylabel = 'ISC',                                title = 'First component with 5-seconds block',                                color='r',fill_between=False)


# In[126]:


reload(sns_plot) # 'Reload' updates immediately the pointer in memory of the edited script

significance = np.where(np.max(np.array(valstest)[:,0,:],axis=0)<isc_results['condition1']['ISC_persecond'][0])

sns_plot.plot(range(1,171),isc_results['condition1']['ISC_persecond'][0])
sns_plot.plot(range(1,171),np.max(np.array(valstest)[:,0,:],axis=0).T,color='grey')
sns_plot.plot(np.reshape(significance,(21,)),isc_results['condition1']['ISC_persecond'][0][significance],
              marker='o', ls="",color='red',
             title='First component with 5-seconds block',
             xlabel ='time (s)',
             ylabel='ISC')


# ### Source Inversion on ISC component

# In[13]:


reload(source_inversion)
#reload(fwd_model)

epochs_ISC = mne.EpochsArray(np.reshape(isc_results['condition1']['A'][:],[1,91,91]),mne.create_info(sub1_raw.info['ch_names'],sfreq=1,ch_types = 'eeg'))
forward_model = fwd_model.fwd(sub1_raw)

cov, inverse_operator = source_inversion.inversion(sub1_raw,epochs_ISC,forward_model,tmax=None,method='empirical')


# In[250]:


method = "eLORETA"
snr = 3.
lambda2 = 1. / snr ** 2
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs


epochs_ISC_2 = mne.EpochsArray(np.reshape(isc_results['condition1']['A'][0],[1,91,1]),mne.create_info(sub1_raw.info['ch_names'],sfreq=1,ch_types = 'eeg'))

# epochs['30'].average() = Averaged evoked response for the event 30
stc_isc = apply_inverse_epochs(epochs_ISC_2, inverse_operator, lambda2,
                             method=method, pick_ori=None, verbose=True)
stc_isc


# In[149]:


#averaged,to_test,b = difference_and_t_test('theta')
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(stc_isc[0].data)
plt.title('Distribution of the eLORETA source inversion on spatial filter')
scaled = scaler.transform(stc_isc[0].data)
plt.hist(scaled,bins=100)


# In[150]:


index = np.where( np.logical_or(scaled<-0.5, scaled>0.5))[0]
scaled[list(set(list(range(20484)))-set(index))] =0


# In[155]:



reload(surface_plot)

get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('gui', 'qt')

surface_plot.plot(mesh, mesh2, scaled, '1st spatial filter using eLORETA (after the threshold on range)')


# In[11]:


#to find the peak
#np.where((isc_results['condition1']['ISC_persecond'][0]) == np.max(isc_results['condition1']['ISC_persecond'][0]))

# for the 7second -> 7*500
indexes = np.hstack([np.arange(159*500-50,159*500+50)])


# ### Source Inversion on Specific chunk of Raw EEG (100ms window around the peak of the peak)

# In[15]:


def epochs(title,raw_bundle,events_bundle):
    for i in range(1, 11):
        globals()[f"epochs{i}_ISC"+title] = epochs_slicing.epochs(raw_bundle[i-1],events_bundle[i-1],[83,103,9999], tmin=0, tmax=170, fs = 500, epochs_to_slice='83')


# In[16]:


reload(epochs_slicing)

raw_bundle = [sub1_raw,sub2_raw,sub3_raw,sub4_raw,sub5_raw,sub6_raw,sub7_raw,sub8_raw,sub9_raw,sub10_raw]
events_bundle = [sub1_events,sub2_events,sub3_events,sub4_events,sub5_events,sub6_events,sub7_events,sub8_events,sub9_events,sub10_events]

epochs("_g",raw_bundle,events_bundle)


# In[17]:


forward_model = fwd_model.fwd(sub1_raw)
from data import apply_inverse_epochs
reload(apply_inverse_epochs)


def source_inversion_bundle(ep,inverse_operator):
    stc1 = apply_inverse_epochs.apply(ep,inverse_operator)
    
    return stc1

def indexing_epochs(epochs,index,downsample=False):
    indexed_epochs = epochs.get_data()[:,:,index]
    
    if downsample:
        
        info_d = mne.create_info(sub1_raw.info['ch_names'],sfreq=250,ch_types = 'eeg')
        ep = mne.EpochsArray(indexed_epochs,mne.create_info(sub1_raw.info['ch_names'],sfreq=500,ch_types = 'eeg'))
        ep = ep.resample(250)
        
        raw = mne.io.RawArray(ep.get_data().reshape(91,int(len(index)/2)),info_d)
        #raw.set_eeg_reference('average', projection=True)
    else:
        ep = mne.EpochsArray(indexed_epochs,mne.create_info(sub1_raw.info['ch_names'],sfreq=500,ch_types = 'eeg'))
        raw = mne.io.RawArray(ep.get_data().reshape(91,len(index)),sub1_raw.info)
        #raw.set_eeg_reference('average', projection=True)

    _,inverse_operator = source_inversion.inversion(raw,ep,forward_model,None,'empirical')
    
    return source_inversion_bundle(ep,inverse_operator)


# In[18]:


#from data import apply_inverse_epochs
#reload(apply_inverse_epochs)
#stc1 = apply_inverse_epochs.apply(ep,inverse_operator)

stc1 =indexing_epochs(epochs1_ISC_ts,False,indexes)
stc2 =indexing_epochs(epochs2_ISC_ts,False,indexes)
stc3 =indexing_epochs(epochs3_ISC_ts,False,indexes)
stc4 =indexing_epochs(epochs4_ISC_ts,False,indexes)
stc5 =indexing_epochs(epochs5_ISC_ts,False,indexes)
stc6 =indexing_epochs(epochs6_ISC_ts,False,indexes)
stc7 =indexing_epochs(epochs7_ISC_ts,False,indexes)
stc8 =indexing_epochs(epochs8_ISC_ts,False,indexes)
stc9 =indexing_epochs(epochs9_ISC_ts,False,indexes)
stc10 =indexing_epochs(epochs10_ISC_ts,False,indexes)


# In[33]:


def average_source_inversion(*stcs):
    average = np.sum(stcs)/len(stcs)
    return average


# In[34]:


average_stc = average_source_inversion(stc1,stc2,stc3,stc4,stc5,stc6,stc7,stc8,stc9,stc10)


# In[74]:


reload(surface_plot)
surface_plot.plot(average_stc.data,'trial','gnuplot')


# ### Specific chunk of filtered EEG(by first component and +/- 100ms around the peak)

# In[ ]:





# ### GSP - High and Low's spectra on Graph

# In[19]:


indexes_high = np.hstack([np.arange(159*500-250,159*500+250)])#,np.arange(67*500,72*500),np.arange(77*500,84*500),np.arange(130*500,136*500),np.arange(155*500,165*500)])
indexes_low = np.hstack([np.arange(100*500-250,100*500+250)])


# In[20]:


src_high1 = indexing_epochs(epochs1_ISC_g,indexes_high,False)
src_high2 = indexing_epochs(epochs2_ISC_g,indexes_high,False)
src_high3 = indexing_epochs(epochs3_ISC_g,indexes_high,False)
src_high4 = indexing_epochs(epochs4_ISC_g,indexes_high,False)
src_high5 = indexing_epochs(epochs5_ISC_g,indexes_high,False)
src_high6 = indexing_epochs(epochs6_ISC_g,indexes_high,False)
src_high7 = indexing_epochs(epochs7_ISC_g,indexes_high,False)
src_high8 = indexing_epochs(epochs8_ISC_g,indexes_high,False)
src_high9 = indexing_epochs(epochs9_ISC_g,indexes_high,False)
src_high10 = indexing_epochs(epochs10_ISC_g,indexes_high,False)


# In[16]:


src_high1[0].data #= indexing_epochs(epochs1_ISC_g,indexes_high,False)


# In[21]:



src_low1 = indexing_epochs(epochs1_ISC_g,indexes_low,False)
src_low2 = indexing_epochs(epochs2_ISC_g,indexes_low,False)
src_low3 = indexing_epochs(epochs3_ISC_g,indexes_low,False)
src_low4 = indexing_epochs(epochs4_ISC_g,indexes_low,False)
src_low5 = indexing_epochs(epochs5_ISC_g,indexes_low,False)
src_low6 = indexing_epochs(epochs6_ISC_g,indexes_low,False)
src_low7 = indexing_epochs(epochs7_ISC_g,indexes_low,False)
src_low8 = indexing_epochs(epochs8_ISC_g,indexes_low,False)
src_low9 = indexing_epochs(epochs9_ISC_g,indexes_low,False)
src_low10 = indexing_epochs(epochs10_ISC_g,indexes_low,False)


# In[22]:


src_low1[0].data #= indexing_epochs(epochs1_ISC_g,indexes_high,False)


# In[23]:


import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters
from pygsp import plotting as gsp_plt
from nilearn import image, plotting, datasets


# In[24]:


from pathlib import Path
from scipy import io as sio
from pygsp import graphs

path_Glasser='S4B2/GSP/Glasser_masker.nii.gz'
res_path=''

# Load structural connectivity matrix
connectivity = sio.loadmat('S4B2/GSP/SC_avg56.mat')['SC_avg56']
connectivity.shape
coordinates = sio.loadmat('S4B2/GSP/Glasser360_2mm_codebook.mat')['codeBook'] # coordinates in brain space


#G_Comb = graphs.Graph(connectivity,gtype='HCP subject',lap_type='combinatorial',coords=coordinates)# combinatorial laplacian
G=graphs.Graph(connectivity,gtype='HCP subject',lap_type='normalized',coords=coordinates) #
#G_RandW=graphs.Graph(connectivity,gtype='HCP subject',lap_type='normalized',coords=coordinates) #
print(G.is_connected())


G.set_coordinates('spring')
#G.plot()   #edges > 10^4 not shown
D=np.array(G.dw)
D.shape


# In[25]:


G.compute_fourier_basis()


# In[26]:


import numpy as np
with np.load(f"S4B2/GSP/hcp/atlas.npz") as dobj:
    atlas = dict(**dobj)


# In[27]:


def averaging_by_parcellation(sub):
    l =list()
    for i in list(set(atlas['labels_L']))[:-1]:
        l.append(np.mean(sub[0].data[10242:][np.where(i== atlas['labels_L'])],axis=0))

    for i in list(set(atlas['labels_R']))[:-1]:
        l.append(np.mean(sub[0].data[:10242][np.where(i== atlas['labels_R'])],axis=0))
    return l


# In[28]:


low = [G.gft(np.array(averaging_by_parcellation(src_low1))),G.gft(np.array(averaging_by_parcellation(src_low2))), 
       G.gft(np.array(averaging_by_parcellation(src_low3))), G.gft(np.array(averaging_by_parcellation(src_low4))), 
       G.gft(np.array(averaging_by_parcellation(src_low5))), G.gft(np.array(averaging_by_parcellation(src_low6))),
       G.gft(np.array(averaging_by_parcellation(src_low7))), G.gft(np.array(averaging_by_parcellation(src_low8))), 
       G.gft(np.array(averaging_by_parcellation(src_low9))), G.gft(np.array(averaging_by_parcellation(src_low10)))]


# In[29]:


high = [G.gft(np.array(averaging_by_parcellation(src_high1))),G.gft(np.array(averaging_by_parcellation(src_high2))), 
       G.gft(np.array(averaging_by_parcellation(src_high3))), G.gft(np.array(averaging_by_parcellation(src_high4))), 
       G.gft(np.array(averaging_by_parcellation(src_high5))), G.gft(np.array(averaging_by_parcellation(src_high6))),
       G.gft(np.array(averaging_by_parcellation(src_high7))), G.gft(np.array(averaging_by_parcellation(src_high8))), 
       G.gft(np.array(averaging_by_parcellation(src_high9))), G.gft(np.array(averaging_by_parcellation(src_high10)))]




# ### Subject-wise Spectra, while Time being variability

# In[30]:


def mean_std(freq,ax):
    if ax>2:
        d = np.average(np.array(np.abs(freq)),axis=2)[:,1:]
    else: d = np.abs(freq[1:,:])
    mean_t = np.mean(d,axis=0)
    std_t = 2 * np.std(d,axis=0)
    top = mean_t + std_t
    bottom = mean_t - std_t
    
    return mean_t,std_t,top,bottom


# In[ ]:





# ### Power distribution finding

# In[31]:


values,_,_,_ = mean_std(np.array(low),3)
np.sum(values)/2


# In[32]:


np.sum(values[:78])


# In[33]:


G.e[79]


# ### Dichotomy 

# In[34]:


#1
l = np.where(G.e<=0.82)[0][1:]
h = np.where(G.e>0.82)[0]


# In[35]:


def filters(isc,band,length):
    indicator = np.ones([1,length])
    cll =list() 
    cll.append(np.matmul(indicator,np.abs(np.array(isc)[0,band,:]))) # 1 x length & length x time
    for i in range(1,10):
        cll.append(np.matmul(indicator,np.abs(np.array(isc)[i,band,:])))
    cll = np.reshape(cll,[10,500])
    return cll


# In[ ]:





# In[36]:


#plt.plot(np.average(cll,axis=0).T)
np.shape(np.abs(np.array(low)[0,h,:]))


# In[46]:


np.savez_compressed('data.npz',mean_t1=mean_t1, mean_t2=mean_t2,mean_std=mean_std )


# In[44]:


get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('gui', 'qt')


#def lowISC_high_ISC(*typ):
a = 1  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter
plt.figure(figsize=(15,15))
typ = {'High ISC':high,'Low ISC':low}
for i in range(2):
        
        plt.subplot(a, b, c)
        cll1 = filters(typ[list(typ.keys())[i]],l,len(l))
        cll2 = filters(typ[list(typ.keys())[i]],h,len(h))
        mean_t1,std_t1, top1, bottom1= mean_std(cll1,2)
        mean_t2,std_t2, top2, bottom2= mean_std(cll2,2)

        
        plt.legend()
        plt.plot(range(500),mean_t1,color='b')
        plt.fill_between(range(500),bottom1,top1, color='b', alpha=.1,label='Low')
        plt.plot(range(500),mean_t2,color='r')
        plt.fill_between(range(500),bottom2, top2, color='r', alpha=.1,label='High')
        plt.ylabel('gPSDs sliced using Eigen values')
        plt.xlabel('Time (s)',fontsize=10)
        plt.title(list(typ.keys())[i])
        plt.xticks(ticks=[0,125,250,375,500],labels=["-0.5","-0.25","0","0.25","0.5"],rotation='horizontal')
        plt.axvline(x=250, linestyle = '--', color='g')
        
        #plt.ylabel('log (gPSD)')
        c = c + 1
plt.suptitle('Dichotomized the eigen values(at 0.8) such that the power distribution is same & sliced the PSD using the same [Low freq = blue]')
plt.show()

# ideas:
#1. Sub-wise plot
#2. Freq-wise plot
#3. High - Low "dicotomized plot" and compare high - low heatmap


# ### Frequency-wise
# 

# In[44]:


get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('gui', 'qt')


#def lowISC_high_ISC(*typ):
a = 1  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter
plt.figure(figsize=(15,15))
typ = {'High ISC':high,'Low ISC':low}
freq = [l,h]
title = ['Low Frequency','High Frequency']
for i in range(2):
        
        plt.subplot(a, b, c)
        cll1 = filters(typ[list(typ.keys())[0]],freq[i],len(freq[i]))
        cll2 = filters(typ[list(typ.keys())[1]],freq[i],len(freq[i]))
        mean_t1,std_t1, top1, bottom1= mean_std(cll1,2)
        mean_t2,std_t2, top2, bottom2= mean_std(cll2,2)

        plt.legend()
        plt.plot(range(500),mean_t1,color='b')
        plt.fill_between(range(500),bottom1,top1, color='b', alpha=.1,label='High ISC')
        plt.plot(range(500),mean_t2,color='r')
        plt.fill_between(range(500),bottom2, top2, color='r', alpha=.1,label='Low ISC')
        #plt.title('Graph PSD for the conditions (CI = Subjects. Time = 0.6 - 0.7s, 50 samples)')
        plt.ylabel('gPSDs sliced using Eigen values')
        plt.xlabel('Time (s)',fontsize=10)
        plt.title(title[i])
        plt.xticks(ticks=[0,125,250,375,500],labels=["-0.5","-0.25","0","0.25","0.5"],rotation='horizontal')
        plt.axvline(x=250, linestyle = '--', color='g')
    
        c = c + 1
plt.suptitle('Dichotomized the eigen values(at 0.8) such that the power distribution is same & sliced the PSD using the same [blue = High ISC]')
plt.show()


# ### Subject-wise

# In[118]:


get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('gui', 'qt')


def filters_subj(isc,band,length):
    indicator = np.ones([1,length])
    cll =list() 
    cll.append(np.matmul(indicator,np.abs(np.array(isc)[band,:])))
    
    cll = np.reshape(cll,[1,500])
    d = np.abs(freq[1:,:])
    #mean_t = np.mean(d,axis=0)
    #std_t = 2 * np.std(d,axis=0)
    #top = mean_t + std_t
    #bottom = mean_t - std_t
    
    return cll

#def lowISC_high_ISC(*typ):
a = 2  # number of rows
b = 5  # number of columns
c = 1  # initialize plot counter
plt.figure(figsize=(10,10))
typ = {'High ISC':high,'Low ISC':low}
freq = [l,h]
for i in range(10):
        plt.subplot(a, b, c)
        cll1 = filters_subj(typ[list(typ.keys())[0]][i],freq[0],len(freq[0]))
        cll2 = filters_subj(typ[list(typ.keys())[0]][i],freq[1],len(freq[1]))
        
        mean_t1,std_t1, top1, bottom1= mean_std(cll1,2)
        mean_t2,std_t2, top2, bottom2= mean_std(cll2,2)
        
        
        plt.legend()
        plt.plot(range(500),mean_t1,color='b')
        plt.fill_between(range(500),bottom1,top1, color='b', alpha=.1,label='Low')
        plt.plot(range(500),mean_t2,color='r')
        plt.fill_between(range(500),bottom2, top2, color='r', alpha=.1,label='High')
        #plt.title('Graph PSD for the conditions (CI = Subjects. Time = 0.6 - 0.7s, 50 samples)')
        plt.xticks(ticks=[0,125,250,375,500],labels=["-0.5","-0.25","0","0.25","0.5"],rotation='horizontal')
        plt.axvline(x=250, linestyle = '--', color='g')
        

        plt.ylabel('gPSDs sliced using Eigen values')
        plt.xlabel('Time (s)',fontsize=10)
        plt.title('Low freq (high ISC = blue)')
        #plt.ylabel('log (gPSD)')
        c = c + 1
plt.suptitle('Dichotomized the eigen values(at 0.8) such that the power distribution is same & sliced the PSD using the same [Low freq = blue]')
plt.show()


# In[52]:


np.savez_compressed('data.npz',mean_t1=mean_t1, mean_t2=mean_t2,mean_std=mean_std )high_isc = [(np.array(averaging_by_parcellation(src_high1))),(np.array(averaging_by_parcellation(src_high2))), 
       (np.array(averaging_by_parcellation(src_high3))), (np.array(averaging_by_parcellation(src_high4))), 
       (np.array(averaging_by_parcellation(src_high5))), (np.array(averaging_by_parcellation(src_high6))),
       (np.array(averaging_by_parcellation(src_high7))), (np.array(averaging_by_parcellation(src_high8))), 
       (np.array(averaging_by_parcellation(src_high9))), (np.array(averaging_by_parcellation(src_high10)))]

low_isc = [(np.array(averaging_by_parcellation(src_low1))),(np.array(averaging_by_parcellation(src_low2))), 
       (np.array(averaging_by_parcellation(src_low3))), (np.array(averaging_by_parcellation(src_low4))), 
       (np.array(averaging_by_parcellation(src_low5))), (np.array(averaging_by_parcellation(src_low6))),
       (np.array(averaging_by_parcellation(src_low7))), (np.array(averaging_by_parcellation(src_low8))), 
       (np.array(averaging_by_parcellation(src_low9))), (np.array(averaging_by_parcellation(src_low10)))]


diff = np.array(high_isc) - np.array(low_isc)



# In[129]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('gui', 'qt')
import matplotlib
def spectrogram(diff,title,start1,end1,div,start2,end2):
    plt.figure(figsize=(10,10))
    cmap_reversed = matplotlib.cm.get_cmap('Spectral').reversed()

    sns.heatmap((diff[2]),cmap=cmap_reversed) 
    plt.ylabel('Graph Freqs')
    plt.xlabel('Time (s)')
    plt.xticks(ticks=[0,125,250,375,500],labels=["-0.5","-0.25","0","0.25","0.5"],rotation='horizontal')
    plt.yticks(ticks=np.arange(start1,end1,div),labels=np.arange(start2,end2,div),rotation='horizontal')
    plt.axvline(x=250, linestyle = '--', color='b')
    #plt.axvline(x=132, linestyle = '--', color='b')
    plt.title(title)
    plt.tight_layout()
    plt.show()
#spectrogram(differenced_low,'Spectrogram for 1-50 freqs (averaged thru subjs)',1,50,2,1,50)# (differenced high with low & averaged through subjects )
#spectrogram(differenced_medium,'Spectrogram for 50-200 freqs (averaged thru subjs)',1,150,5,50,200)# (differenced high with low & averaged through subjects )
#spectrogram(differenced_high,'Spectrogram for 200-360 freqs (averaged thru subjs)',1,160,5,200,360)# (differenced high with low & averaged through subjects )
spectrogram(diff,'Time-Series Activations for all subjects (averaged) after differencing ISCs (high - low)',1,360,10,1,360)# (differenced high with low & averaged through subjects )

#spectrogram(high_isc,'Time-Series Activations for all subjects (averaged) for High ISC',1,360,10,1,360)# (differenced high with low & averaged through subjects )


# avg across subj & std across subj (low, medium, high)


# In[ ]:


from shutil import make_archive
make_archive('data', 'zip', '/home/v20subra/HBN/S4B2')


# In[80]:


get_ipython().system('pwd')


# In[ ]:





# In[ ]:




