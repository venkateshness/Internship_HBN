#%%
from cProfile import label
from logging import error
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
from pygsp import graphs, filters
from pygsp import plotting as gsp_plt
from nilearn import image, plotting, datasets

from pathlib import Path
from scipy import io as sio
from pygsp import graphs
from seaborn.utils import axis_ticklabels_overlap
from scipy.sparse import csr_matrix
import scipy
import matplotlib
import seaborn as sns

sns.set_theme()
############################################################
##########Getting the Graph ready###########################
############################################################ 
def graph_setup(thresholding, percentage):
    path_Glasser='/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz'
    res_path=''

    # Load structural connectivity matrix
    connectivity = sio.loadmat('/homes/v20subra/S4B2/GSP/SC_avg56.mat')['SC_avg56']
    connectivity.shape
    coordinates = sio.loadmat('/homes/v20subra/S4B2/GSP/Glasser360_2mm_codebook.mat')['codeBook'] # coordinates in brain space

    G=graphs.Graph(connectivity,gtype='HCP subject',lap_type='combinatorial',coords=coordinates) 
    G.set_coordinates('spring')
    print('{} nodes, {} edges'.format(G.N, G.Ne))

    if thresholding:
        weights = csr_matrix(G.W).toarray()
        weights[weights<np.percentile(weights,percentage)] =0

        G = graphs.Graph(weights)
        print(G.is_connected())
        print('{} nodes, {} edges'.format(G.N, G.Ne))

    return G


G = graph_setup(False,90)
###############################
####Decomposing into eigenmodes
###############################
G.compute_fourier_basis()

#%%

###############################
####Get the Glasser atlas
###############################

with np.load(f"/homes/v20subra/S4B2/GSP/hcp/atlas.npz") as dobj:
    atlas = dict(**dobj)


###############################
####Loading the eloreta activation files
###############################
time_series = np.load('/users2/local/Venkatesh/Generated_Data/noise_baseline_properly-done_eloreta/SI_full.npz')['high_isc_averaged']
#low = np.load('/users/local/Venkatesh/Generated_Data/noise_baseline_properly-done_eloreta/low_isc.npz')['low_isc_averaged']

######GFTing them
time_series_gft = [G.gft(np.array(time_series[0])),G.gft(np.array(time_series[1])), 
       G.gft(np.array(time_series[2])), G.gft(np.array(time_series[3])), 
       G.gft(np.array(time_series[4])), G.gft(np.array(time_series[5])),
       G.gft(np.array(time_series[6])), G.gft(np.array(time_series[7])), 
       G.gft(np.array(time_series[8])), G.gft(np.array(time_series[9]))]


differenced = np.array(time_series_gft) 

#Trichotomizing sequentially the freqs
differenced_low_freq = differenced[:,1:51,:]
differenced_medium_freq = differenced[:,51:200,:]
differenced_high_freq = differenced[:,200:,:]

np.shape(differenced_low_freq)
#%%
####################
#resting state data
####################
rstate = np.load('/users2/local/Venkatesh/Generated_Data/noise_baseline_properly-done_eloreta/rstate_source_space_parcellated.npz')['rstate_source_space_parcellated']


rstate_gft = [G.gft(np.array(rstate[0])),G.gft(np.array(rstate[1])), 
       G.gft(np.array(rstate[2])), G.gft(np.array(rstate[3])), 
       G.gft(np.array(rstate[4])), G.gft(np.array(rstate[5])),
       G.gft(np.array(rstate[6])), G.gft(np.array(rstate[7])), 
       G.gft(np.array(rstate[8]))]


#%%
#####################
### Mean std calculation
####################


def mean_std(freq,ax):
    if ax>2:
        d = np.average(np.array(np.abs(freq)),axis=2)[:,1:]
    else: d = np.abs(freq[1:,:])
    mean_t = np.mean(d,axis=0)
    std_t = 2 * np.std(d,axis=0)
    top = mean_t + std_t
    bottom = mean_t - std_t
    
    return mean_t,std_t,top,bottom
#%%
values,_,_,_ = mean_std(np.array(rstate_gft),3)
print('divided power: ', np.sum(values)/2)
print('sum of power is: ',np.sum(values[:93])) #3%=88;7=92;10=92;0=93
print('the eigenvalue is:',G.e[94])#3%=89

l = np.where(G.e<=7.36)[0][1:]#3%=4.89;7=6.34;6.67
h = np.where(G.e>7.36)[0]

#%%
####################################
####indicator function##############

def filters(isc,band,length):
    indicator = np.ones([1,length])
    cll =list() 
    cll.append(np.matmul(indicator,np.abs(np.array(isc)[0,band,:]))) # 1 x length & length x time
    for i in range(1,10):
        cll.append(np.matmul(indicator,np.abs(np.array(isc)[i,band,:])))
    cll = np.reshape(cll,[10,21250])
    return cll


####################################
#%%

global_mean_ts = np.mean(time_series_gft,axis=2)

time_series_gft_chunked = [np.sum(global_mean_ts[:,:50],axis=1), np.sum(global_mean_ts[:,50:200],axis=1),np.sum(global_mean_ts[:,200:],axis=1)]

#low_GFT_power_chunked = [np.sum(global_mean_low[:,:50],axis=1), np.sum(global_mean_low[:,50:200],axis=1),np.sum(global_mean_low[:,200:],axis=1)]

print(np.shape(time_series_gft_chunked))



#######################################
#%%



time_series_gft_chunked_sem = [scipy.stats.sem(time_series_gft_chunked[0]),
                scipy.stats.sem(time_series_gft_chunked[1]),
                scipy.stats.sem(time_series_gft_chunked[2])
                ]
# std_err_low = [scipy.stats.sem(low_GFT_power_chunked[0]),
#                 scipy.stats.sem(low_GFT_power_chunked[1]),
#                 scipy.stats.sem(low_GFT_power_chunked[2])
#                 ]




#%%

# labels = ['Low', 'Med', 'High']
# import pandas as pd
# data= pd.DataFrame({'labels':labels,'gPSD_low':low_GFT_power_chunked,'gPSD_high':high_GFT_power_chunked},index=None)


from nilearn.regions import signals_to_img_labels  
# load nilearn label masker for inverse transform
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image, plotting
from nilearn import datasets
from os.path import join as opj

path_Glasser='/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz'
mnitemp = fetch_icbm152_2009()
mask_mni=image.load_img(mnitemp['mask'])
glasser_atlas=image.load_img(path_Glasser)



print('ttest')
print('for low freq       :',scipy.stats.mstats.ttest_rel(time_series_gft_chunked[0],time_series_gft_chunked[1]))
# print('for medium         :',scipy.stats.mstats.ttest_rel(high_GFT_power_chunked[1],low_GFT_power_chunked[1]))
# print('for high           :',scipy.stats.mstats.ttest_rel(high_GFT_power_chunked[2],low_GFT_power_chunked[2]))

# %%

# data = pd.DataFrame({'labels':labels,'gPSD':std_err_low})

# data2 = pd.DataFrame({'labels':labels,'gPSD':std_err_high})
# %%
# data_fin = data.append(data2,ignore_index=True)
# data_fin['cond'] = ['Low_ISC','Low_ISC','Low_ISC','High_ISC','High_ISC','High_ISC']
# %%

import seaborn as sns 
import pandas as pd
fig=plt.figure(figsize = (17, 17))
import seaborn
seaborn.despine(left=True, bottom=True, right=True)

plt.rc('font', family='serif')
grid = fig.add_gridspec(6,6, wspace =1.3, hspace = 1.2)

#grid = gridspec.GridSpec(2,2, wspace =0.3, hspace = 0.8)
g1 = fig.add_subplot(grid[0:2, :2])
g2 = fig.add_subplot(grid[0:2, 2:4])
g3 = fig.add_subplot(grid[0:2, 4:])

g4 = fig.add_subplot(grid[2:4, :2])
g5 = fig.add_subplot(grid[2:, 2:])
g6 = fig.add_subplot(grid[4:, :2])

import matplotlib
def heatmap(diff,title,start1,end1,div,start2,end2,operation,ylabel,axes):
    

    #fig, ax = plt.subplots()

    cmap_reversed = matplotlib.cm.get_cmap('Spectral').reversed()
    if operation == 'std':
        svm = sns.heatmap(np.std(diff,axis=0),cmap=cmap_reversed,ax=axes) 
    else:
        svm = sns.heatmap(np.average(diff,axis=0),cmap=cmap_reversed,ax=axes) 
    axes.set_ylabel('Graph Frequencies')
    axes.set_xlabel('Time (s)')
    
    xticks= np.arange(1,21250,2125)
    axes.set_xticks([])
    axes.set_xticks(xticks)
    axes.set_xticklabels(labels=np.arange(1,170,17),rotation='vertical')
    yticks= np.arange(start1,end1,div)
    axes.set_yticks([])
    axes.set_yticks(yticks)
    axes.set_yticklabels(labels=np.arange(start2,end2,div),rotation='horizontal')
    
    
    axes.yaxis.set_tick_params(rotation=360)
    axes.axvline(x=160*125, linestyle = '--', color='b')
    axes.set_title(title,pad=10)
    

    axes.text(0.5,-0.22, "(a)", size=12, ha="center", 
         transform=axes.transAxes)
heatmap(differenced_low_freq,'Spectrogram after differencing power',1,50,2,1,50,'AVG','gFreqs',axes=g1)
heatmap(differenced_medium_freq,'Spectrogram after differencing power',1,150,5,50,200,'AVG','gFreqs',axes=g2)
heatmap(differenced_high_freq,'Spectrogram after differencing power',1,160,5,200,360,'AVG','gFreqs',axes=g3)




#def lowISC_high_ISC(*typ):
a = 1  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter
plt.figure(figsize=(15,15))
typ = {'High ISC':time_series_gft}
for i in range(2):
        
        mean_t1,std_t1, top1, bottom1= mean_std(time_series_gft,3)
        #mean_t2,std_t2, top2, bottom2= mean_std(low_gft,3)
        
        #plt.legend()
        g5.plot(range(359),mean_t1[:],color='r')
        g5.fill_between(range(359),bottom1[:],top1[:], color='r', alpha=.1,label='High ISC')
        # g5.plot(range(359),mean_t2[:],color='b')
        # g5.fill_between(range(359),bottom2[:], top2[:], color='b', alpha=.1,label='Low ISC')
        g5.set_ylabel('gPSDs')
        g5.set_xlabel('Eigen values')
        g5.set_title('Graph PSD for both conditions (after temporal avg) (c)')
        xticks= np.arange(0,360,71)

        g5.set_xticks([])
        g5.set_xticks(xticks)
#        g2.set_xticklabels(labels=[-0.5,-0.25,0,0.25,0.5],rotation='horizontal')
        g5.set_xticklabels(labels=np.round(G.e[np.arange(1,360,71)],decimals=2),rotation='horizontal')
        #plt.axvline(x=250, linestyle = '--', color='g')
        g5.text(0.5,-0.10, "(c)", size=12, ha="center", 
        transform=g5.transAxes)
        #plt.ylabel('log (gPSD)')
#plt.suptitle('Dichotomized the eigen values(at 1.02) such that the power distribution is same & sliced the PSD using the same [Low freq = blue] Note: used np.abs while using indicator')
g5.legend(['High ISC','Low ISC'])



#def lowISC_high_ISC(*typ):
a = 1  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter
typ = {'High ISC':time_series_gft}


#plt.subplot(a, b, c)
cll2 = filters(typ[list(typ.keys())[0]],h,len(h))
mean_t1,std_t1, top1, bottom1= mean_std(cll1,2)
mean_t2,std_t2, top2, bottom2= mean_std(cll2,2)


g4.legend()
g4.plot(range(21250),mean_t1,color='b')
g4.fill_between(range(21250),bottom1,top1, color='b', alpha=.1,label='Low frequency')
g4.plot(range(21250),mean_t2,color='r')
g4.fill_between(range(21250),bottom2, top2, color='r', alpha=.1,label='High frequency')

g4.set_ylabel('gPSDs sliced using Eigen values')
g4.set_xlabel('Time (s)',fontsize=10)
# xticks= [0,125,250,375,500]

# g4.set_xticks(xticks)
# g4.set_xticklabels(labels=[-0.5,-0.25,0,0.25,0.5],rotation='horizontal')
g4.axvline(x=250, linestyle = '--', color='g')
g4.set_title('gPSD time-series for High ISC', pad=10)
g4.legend()

g4.text(0.5,-0.20, "(b)", size=12, ha="center", 
         transform=g4.transAxes)




labels = ['Low', 'Med', 'High']
global_mean_low = mean_std(time_series_gft,ax=3)[0]
# global_mean_high = mean_std(high_gft,ax=3)[0]

# gPSD_high = [np.sum(global_mean_high[:50]), np.sum(global_mean_high[50:200]),np.sum(global_mean_high[200:])]
gPSD_low = [np.sum(global_mean_low[:50]), np.sum(global_mean_low[50:200]),np.sum(global_mean_low[200:])]

data= pd.DataFrame({'labels':labels,'gPSD_low':gPSD_low},index=None)

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

# rects1 = g6.bar(x - width/2, gPSD_high, width, label='High ISC',yerr=std_err_high, align='center', alpha=0.5, ecolor='black', capsize=10,color='red')
rects2 = g6.bar(x + width/2, gPSD_low, width, yerr=time_series_gft_chunked_sem, align='center', alpha=0.5, ecolor='black', capsize=10,color='blue')

ylab = "gPSD"
# Add some text for labels, title and custom x-axis tick labels, etc.
g6.set_ylabel(ylab)
g6.set_title('Freq-wise power grouping (errorbar = SEM)',pad=10)#(after trichotomizing) while SEM being error bars
g6.set_xticks(x)
g6.set_xticklabels(labels)
#order = ['low ISC','high ISC'] 
from statannot import add_stat_annotation
# add_stat_annotation(g6,data=data_fin, y='gPSD', x ='labels', hue='cond',
#                     box_pairs=[(("Low", "Low_ISC"), ("Low", "High_ISC")),
#                                 (("Med", "Low_ISC"), ("Med", "High_ISC")),
#                                 (("High", "Low_ISC"), ("High", "High_ISC"))],
                        
#                                  perform_stat_test=False,  #2.71e-05,1.70e-09,3.33e-14, #0.28,0.47,0.013, #3.00e-05,3.08e-18,2.53e-23 #0.047,9.3e-10,1.03e-08
#                     line_offset_to_box=0.95, line_offset=0.5, line_height=0.05, text_format='simple', loc='inside', verbose=2)
g6.set_xlabel('Graph Frequency bands')

g6.legend(bbox_to_anchor=(0.65, 0.55), bbox_transform=g6.transAxes)
g6.text(0.5,-0.25, "(d)", size=12, ha="center", 
         transform=g6.transAxes)

#signal=[]
#U0_brain=[]
#signal=np.expand_dims(np.array(G.U[:, -3]), axis=0) # add dimension 1 to signal array
#U0_brain = signals_to_img_labels(signal,path_Glasser,mnitemp['mask'])
#plotting.plot_glass_brain(U0_brain,colorbar=True,plot_abs=False,cmap='spring',display_mode='lzr',axes=g5,title='eigenvector 358 (e)')
#g5.text(0.5,-0.25, "(e)", size=12, ha="center", 
#         transform=g5.transAxes)
fig.suptitle('Noise-baseline-corrected eLORETA signal with no graph thresholding', size=20)
#fig.savefig('/homes/v20subra/S4B2/graph_PhD/Results_thresholding/no_percentile_thresholding.png',dpi=500)

# %%



isc_result = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects/sourceCCA_ISC.npz')['sourceISC']
v = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects/noise_floor_source.npz')['sourceCCA']

significance = np.array(np.where(np.max(np.array(v)[0,:],axis=0)<isc_result[0]))




fig = plt.figure(figsize = (10,10))
ax1 = plt.subplot(411, frameon=False)
ax2 = plt.subplot(412, frameon=False)
ax3 = plt.subplot(413, frameon=False)
ax4 = plt.subplot(414, frameon=False)

ax1.plot(np.average(np.average(np.abs(differenced_low_freq),axis=0),axis=0),color='g',label='Low')[0]
ax1.legend(loc="upper left")

ax2.plot(np.average(np.average(np.abs(differenced_medium_freq),axis=0),axis=0),color='b',label='Med')[0]
ax2.legend(loc="upper left")

ax3.plot(np.average(np.average(np.abs(differenced_high_freq),axis=0),axis=0),color='r',label='high')[0]
ax3.legend(loc="upper left")

# xticks= np.arange(1,21250,125)
#ax1.legend((a1), ('Low', 'Med','High'), loc='lower left', shadow=True)

xticks= np.arange(1,21250,2125)


ax4.plot(range(1,171),isc_result[0], label='Source-level estimation')
ax4.fill_between(range(1,171),np.max(np.array(v)[:,0,:],axis=0).T,np.min(np.array(v)[:,0,:],axis=0).T,color ='grey',alpha=0.8)
ax4.plot(significance,isc_result[0][significance],
              marker='o', ls="",color='red',markersize=4)
ax4.legend(loc="upper left")
def set_ticks(ax):
    ax.set_xticks([])
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels=np.arange(1,170,17),rotation='vertical')
set_ticks(ax1)
set_ticks(ax2)
set_ticks(ax3)
ax4.set_xticks([])
ax4.set_xticks(np.arange(1,170,17))
ax4.set_xticklabels(labels=np.arange(1,170,17),rotation='vertical')


ax1.get_shared_x_axes().join(ax1, ax2,ax3,ax4)
ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])


fig.text(0.5, 0.04, 'time (s)', ha='center')
fig.text(0.04, 0.25, 'ISC coefficients', rotation='vertical')
fig.text(0.04, 0.65, 'graph PSD',  rotation='vertical')
plt.suptitle('Band-wise averaged power spectrum for video-watching after applying abs')
plt.show()
fig.savefig('/homes/v20subra/S4B2/graph_PhD/video-watching-graph_avged_abs.jpeg')
# %%

# %%
np.shape(np.average(np.array(rstate_gft),axis=2))
# %%
np.array(time_series_gft)[0,0,0]

# %%

# %%
np.average(np.abs(differenced_low_freq),axis=0).T
# %%
np.shape(np.sum(np.average(differenced_low_freq,axis=0),axis=0))
# %%
np.average(np.average(differenced_medium_freq,axis=0),axis=0)
# %%
