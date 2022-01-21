# In[1]:
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import ttest_rel



# In[2]:

noise = np.load('/users/local/Venkatesh/Generated_Data/noise_floor.npz')['a']

isc_result = np.load('/users/local/Venkatesh/Generated_Data/CCA_ISC.npz')['CCA_ISC']

significance = np.where(np.max(np.array(noise)[:,0,:],axis=0)<isc_result)

# %%
import os
os.chdir('/homes/v20subra/S4B2/')

print(np.shape(noise))
# %%
from Modular_Scripts import sns_plot

significance = np.where(np.max(np.array(noise)[:,0,:],axis=0)<isc_result[0])

fig, ax = plt.subplots()
#ax.plot(range(20))



ax.plot(range(1,171),isc_result[0])
#sns_plot.plot(range(1,171),np.max(np.array(noise)[:,0,:],axis=0).T,color='grey')
ax.plot(np.reshape(significance,(21,)),isc_result[0][significance],
              marker='o', ls="",color='red',markersize=4)
#ax.plot(noise[:,0,:].T,c='grey')
ax.axvspan(158.5, 159.5, alpha=0.5, color='darkgreen',label = 'High ISC',lw=0.5)
ax.axvspan(99.5, 100.5, alpha=0.5, color='lightsalmon',label = 'Low ISC',lw=0.5)
ax.legend()
ax.fill_between(range(170),np.max(np.array(noise)[:,0,:],axis=0).T,np.min(np.array(noise)[:,0,:],axis=0).T,color ='grey',alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('ISC Coefficient')
plt.title('First ISC component with noise floor from 5-s bootstrapping window')
fig.savefig('bootstrapping.png',dpi=500)
plt.show()
plt.tight_layout()
# %%
high = np.load('/users/local/Venkatesh/Generated_Data/high_isc_averaged_with_cov.npz')['high_isc_averaged']
low = np.load('/users/local/Venkatesh/Generated_Data/low_isc_averaged_with_cov.npz')['low_isc_averaged']
differenced =high-low

activation = np.average(high,axis=(0,2))
activation_un = np.average(differenced,axis=(0,2))
to_plot_new = np.zeros(shape=(360,))
activation[activation <np.percentile(activation,90)]= 0
to_plot_new = activation
to_plot_new



# %%
np.shape(np.array(noise)[:,0,:])

# %%
low_ISC = np.load('/users/local/Venkatesh/Generated_Data/low_isc_averaged_with_cov.npz')['low_isc_averaged']
#np.shape(low_ISC)
high_ISC = np.load('/users/local/Venkatesh/Generated_Data/high_isc_averaged_with_cov.npz')['high_isc_averaged']
differenced = high_ISC - low_ISC
# %%

from scipy import stats
test = list()
tvals = list()
for i in range(360):
    #test.append(stats.ttest_1samp(np.hstack(differenced[:,i,:]),popmean=False)[1])
    #tvals.append(stats.ttest_1samp(np.hstack(differenced[:,i,:]),popmean=False)[0])
    test.append(stats.ttest_rel(np.hstack(high_ISC[:,i,:]),np.hstack(low_ISC[:,i,:]))[1])
    tvals.append(stats.ttest_rel(np.hstack(high_ISC[:,i,:]),np.hstack(low_ISC[:,i,:]))[0])
    #print(np.shape(np.hstack(differenced[:,i,:])))
    
# %%
#np.array(tvals)[np.where(np.array(test)<0.005)]
# %%
np.where(np.array(test)<0.0001)
# %%
rois = np.load('/homes/v20subra/S4B2/GSP/hcp/regions.npy')
#rois[np.where(np.array(test)<0.0001)]

# %%
sum(np.array(test)[np.where(np.array(test)<0.0001)]>0)

# %%
to_plot = np.zeros(shape=(360,))
to_plot[np.where(np.array(test)<0.0001)] = np.array(tvals)[np.where(np.array(test)<0.0001)]

np.array(tvals)[np.where(np.array(test)<0.0001)]
# %%
from nilearn.regions import signals_to_img_labels  
# load nilearn label masker for inverse transform
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image, plotting
from nilearn import datasets
from os.path import join as opj
import matplotlib.pyplot as plt

path_Glasser = '/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz'


mnitemp = fetch_icbm152_2009()
mask_mni=image.load_img(mnitemp['mask'])
glasser_atlas=image.load_img(path_Glasser)


#print(NiftiMasker.__doc__)
def brain_plot(data, title, expand):

    signal=[]
    U0_brain=[]
    if expand:
        signal=np.expand_dims(data, axis=0) # add dimension 1 to signal array
    else:
        signal=data # add dimension 1 to signal array
    U0_brain = signals_to_img_labels(signal,path_Glasser,mnitemp['mask'])
    plotting.plot_glass_brain(U0_brain,title=title,colorbar=True,plot_abs=False,cmap='seismic',display_mode='lzr',symmetric_cbar=False,figure=fig,axes=g2)
    U0_brain.to_filename('eloreta_differenced.nii.gz')

brain_plot(np.reshape(activation_un,(1,360)),'ff',expand=False)

# %%
cd /homes/v20subra/S4B2
# %%
high = np.load('/users/local/Venkatesh/Generated_Data/high_isc_averaged_with_cov.npz')['high_isc_averaged']
low = np.load('/users/local/Venkatesh/Generated_Data/low_isc_averaged_with_cov.npz')['low_isc_averaged']

import seaborn as sns
import matplotlib.ticker as ticker

import matplotlib

fig=plt.figure(figsize = (15, 15))
grid = fig.add_gridspec(2,2, wspace =0.2, hspace = 0.3)
#grid = gridspec.GridSpec(2,2, wspace =0.3, hspace = 0.8)
g1 = fig.add_subplot(grid[0, :-1])
g2 = fig.add_subplot(grid[0, -1])
g3 = fig.add_subplot(grid[1, :])
#g4 = fig.add_subplot(grid[1, -1])


def activation_time_series(diff,title,start1,end1,div,start2,end2,operation):
    
    

    #fig, ax = plt.subplots()

    cmap_reversed = matplotlib.cm.get_cmap('Spectral').reversed()
    if operation == 'std':
        svm = sns.heatmap(np.std(diff,axis=0),cmap=cmap_reversed,ax=g1) 
    else:
        svm = sns.heatmap(np.average(diff,axis=0),cmap=cmap_reversed,ax=g1) 
    g1.set_ylabel('ROI Parcels')
    g1.set_xlabel('Time (s)')
    xticks= [0,125,250,375,500]
    g1.set_xticks([])
    g1.set_xticks(xticks)
    g1.set_xticklabels(labels=[-0.5,-0.25,0,0.25,0.5],rotation='horizontal')

    #g1.set_yticklabels(ticks=np.arange(start1,end1,div),labels=np.arange(start2,end2,div),rotation='horizontal')
    g1.axvline(x=250, linestyle = '--', color='b')
    #plt.axvline(x=132, linestyle = '--', color='b')
    #g1.set_title(title)
    #plt.tight_layout()
    #plt.show()


    g1.text(0.5,-0.2, "(a)", size=12, ha="center", 
         transform=g1.transAxes)
    #figure = svm.get_figure()    
    #figure.savefig('timeseries.jpg', dpi=500)

    #fig.savefig('timeseries.png',dpi=500)
#spectrogram(differenced_low,'Spectrogram for 1-50 freqs (averaged thru subjs)',1,50,2,1,50)# (differenced high with low & averaged through subjects )
#spectrogram(differenced_medium,'Spectrogram for 50-200 freqs (averaged thru subjs)',1,150,5,50,200)# (differenced high with low & averaged through subjects )
#spectrogram(differenced_high,'Spectrogram for 200-360 freqs (averaged thru subjs)',1,160,5,200,360)# (differenced high with low & averaged through subjects )
activation_time_series(high,None,1,360,10,1,360,'AVG')# (differenced high with low & averaged through subjects )

#spectrogram(high_isc,'Time-Series Activations for all subjects (averaged) for High ISC',1,360,10,1,360)# (differenced high with low & averaged through subjects )



signal=[]
U0_brain=[]

signal=np.reshape(to_plot_new,(1,360)) # add dimension 1 to signal array
U0_brain = signals_to_img_labels(signal,path_Glasser,mnitemp['mask'])
plotting.plot_glass_brain(U0_brain,colorbar=True,plot_abs=False,cmap='YlOrRd',display_mode='lzr',symmetric_cbar=False,figure=fig,axes=g2)
g2.text(0.5,-0.2, "(b)", size=12, ha="center", 
         transform=g2.transAxes)
#U0_brain.to_filename('d'.nii.gz')


# Corrcoef on eloreta activation

import numpy as np

#low_isc = np.load('S4B2/Generated_Data/low_isc_averaged_with_cov.npz')['low_isc_averaged']
#high_isc['high_isc_averaged'] low_isc
high_isc = np.load('/users/local/Venkatesh/Generated_Data/high_isc_averaged_with_cov.npz')['high_isc_averaged']
plot_high = dict() 
for m in range(360):
    hm = np.triu(np.corrcoef(np.array(high_isc)[:,m,:])) #high_isc['high_isc_averaged']
    
    iscs = list()
    for i in range(9):
         #iscs.append(hm[i][i+1:])
         #plot_high[m] = np.hstack(iscs)
         iscs.append(np.sum(hm[i][i+1:])/len((hm[i][i+1:])))
    plot_high[m] = iscs
    #plot_high.append(sum(iscs)/ sum(np.arange(10)))

low_isc = np.load('/users/local/Venkatesh/Generated_Data/low_isc_averaged_with_cov.npz')['low_isc_averaged']
plot_low = dict()
for m in range(360):
    hm = np.triu(np.corrcoef(np.array(low_isc)[:,m,:])) #corrcoef and take only the upper half of the matrix

    iscs = list() #to store row-wise results
    for i in range(9): # Loop through each of the 9 rows
        #iscs.append(hm[i][i+1:]) #
        #plot_low[m] = np.hstack(iscs) # store ROI-wise
        iscs.append(np.sum(hm[i][i+1:])/len((hm[i][i+1:]))) #
    plot_low[m] = iscs # store ROI-wise
    
from scipy import stats
ttest = list()
tvalues = list()

for i in range(360):
    ttest.append(stats.ttest_rel(plot_high[i],plot_low[i])[1])
    tvalues.append(stats.ttest_rel(plot_high[i],plot_low[i])[0])

zeroed_for_rois = np.zeros(shape=(1,360))
zeroed_for_rois [:,np.where(np.array(ttest) < 0.05)] = np.array(tvalues)[np.where(np.array(ttest) < 0.05)]


zeroed_for_rois_pvalue = np.zeros(shape=(1,360))
zeroed_for_rois_pvalue [:,np.where(np.array(ttest) < 0.05)] = np.array(ttest)[np.where(np.array(ttest) < 0.05)]



data = zeroed_for_rois

signal=[]
U0_brain=[]
signal=data # add dimension 1 to signal array
U0_brain = signals_to_img_labels(signal,path_Glasser,mnitemp['mask'])
plotting.plot_glass_brain(U0_brain,colorbar=True,plot_abs=False,cmap='seismic',display_mode='lzr',symmetric_cbar=True,figure=fig,axes=g3)
g3.text(0.5,0, "(c)", size=12, ha="center", 
         transform=g3.transAxes)



fig.suptitle('CCA', size=20)

#fig.tight_layout()

# avg across subj & std across subj (low, medium, high)
# %%
fig.savefig('/homes/v20subra/S4B2/noise_baseline/cca_grid.jpg',dpi=300)


# %%
sum(isc_result[0]<=0.10)/170
# %%
# %%
np.array(tvalues)[np.where(np.array(ttest)<=0.05)]
# %%
# %%

# %%
stats.ttest_rel(plot_high[i],plot_low[i])
# %%
# %%
activation = np.average(differenced,axis=(0,2))
to_plot_new = np.zeros(shape=(360,))
activation[activation <np.percentile(activation,90)]= 0
for i in range(20):

    #print(np.max(activation))
    print(rois[np.where(activation == np.max(activation))])
    activation[np.where(activation == np.max(activation))] = 0
    print(np.array(activation)[np.where(activation == np.max(activation))])
# %%
np.array(tvalues)[np.array(ttest)<0.05]
# %%
rois[np.array(ttest)<0.05]


# %%

sum(activation>0)
# %%

# %%
