#!/usr/bin/env python
# coding: utf-8

# In[1]:


from logging import error
import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters
from pygsp import plotting as gsp_plt
from nilearn import image, plotting, datasets


# In[2]:


from pathlib import Path
from scipy import io as sio
from pygsp import graphs
from seaborn.utils import axis_ticklabels_overlap

path_Glasser='/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz'
res_path=''

# Load structural connectivity matrix
connectivity = sio.loadmat('/homes/v20subra/S4B2/GSP/SC_avg56.mat')['SC_avg56']
connectivity.shape
coordinates = sio.loadmat('/homes/v20subra/S4B2/GSP/Glasser360_2mm_codebook.mat')['codeBook'] # coordinates in brain space


#G_Comb = graphs.Graph(connectivity,gtype='HCP subject',lap_type='combinatorial',coords=coordinates)# combinatorial laplacian
G=graphs.Graph(connectivity,gtype='HCP subject',lap_type='combinatorial',coords=coordinates) #
#G_RandW=graphs.Graph(connectivity,gtype='HCP subject',lap_type='normalized',coords=coordinates) #
print(G.is_connected())


G.set_coordinates('spring')
#G.plot()   #edges > 10^4 not shown
D=np.array(G.dw)
D.shape

# In[3]:


G.compute_fourier_basis()


# In[4]:


import numpy as np
with np.load(f"/homes/v20subra/S4B2/GSP/hcp/atlas.npz") as dobj:
    atlas = dict(**dobj)


# In[5]:


high = np.load('/users/local/Venkatesh/Generated_Data/high_isc_averaged_with_cov.npz')['high_isc_averaged']
low = np.load('/users/local/Venkatesh/Generated_Data/low_isc_averaged_with_cov.npz')['low_isc_averaged']
np.shape(low)


# In[6]:


np.shape(low[0])


# In[7]:


low_gft = [G.gft(np.array(low[0])),G.gft(np.array(low[1])), 
       G.gft(np.array(low[2])), G.gft(np.array(low[3])), 
       G.gft(np.array(low[4])), G.gft(np.array(low[5])),
       G.gft(np.array(low[6])), G.gft(np.array(low[7])), 
       G.gft(np.array(low[8])), G.gft(np.array(low[9]))]


# In[8]:


high_gft = [G.gft(np.array(high[0])),G.gft(np.array(high[1])), 
       G.gft(np.array(high[2])), G.gft(np.array(high[3])), 
       G.gft(np.array(high[4])), G.gft(np.array(high[5])),
       G.gft(np.array(high[6])), G.gft(np.array(high[7])), 
       G.gft(np.array(high[8])), G.gft(np.array(high[9]))]

differenced = np.array(high_gft) - np.array(low_gft)


# In[9]:


differenced_low_freq = differenced[:,1:51,:]
differenced_medium_freq = differenced[:,51:200,:]
differenced_high_freq = differenced[:,200:,:]


# In[10]:


#np.shape(np.std(diff,axis=0))


# In[ ]:



import matplotlib
import seaborn as sns
def heatmap(diff,title,start1,end1,div,start2,end2,operation,ylabel):
    cmap_reversed = matplotlib.cm.get_cmap('Spectral').reversed()
    fig1 = plt.gcf()
    if operation == 'std':
        sns.heatmap(np.std(diff,axis=0),cmap = cmap_reversed)
    else:
        sns.heatmap(np.average(diff,axis=0),cmap = cmap_reversed)
    plt.ylabel(ylabel)
    plt.xlabel('Time (s)')
    plt.xticks(ticks=[0,125,250,375,500],labels=["-0.5","-0.25","0","0.25","0.5"],rotation='horizontal')
    plt.yticks(ticks=np.arange(start1,end1,div),labels=np.arange(start2,end2,div),rotation='horizontal')
    plt.axvline(x=250, linestyle = '--', color='b')
    #plt.axvline(x=132, linestyle = '--', color='b')
    plt.title(title)
    plt.tight_layout()
    plt.show(block=False)
    fig1.savefig('spectrogram.jpeg')
heatmap(differenced_low_freq,'Spectrogram for 2-50 freqs (averaged thru subjs)',2,51,2,2,51,'AVG','gFreqs')# (differenced high with low & averaged through subjects )
#heatmap(differenced_medium_freq,'Spectrogram for 50-200 freqs (averaged thru subjs)',1,150,5,50,200,'AVG','gFreqs')# (differenced high with low & averaged through subjects )
#heatmap(differenced_high_freq,'Spectrogram for 200-360 freqs (averaged thru subjs)',1,160,5,200,360,'AVG','gFreqs')# (differenced high with low & averaged through subjects )


#heatmap(differenced_low_freq,'Spectrogram for 1-50 freqs (std thru subjs)',1,50,2,1,50,'std','gFreqs')# (differenced high with low & averaged through subjects )
#heatmap(differenced_medium_freq,'Spectrogram for 50-200 freqs (std thru subjs)',1,150,5,50,200,'std','gFreqs')# (differenced high with low & averaged through subjects )
#heatmap(differenced_high_freq,'Spectrogram for 200-360 freqs (std thru subjs)',1,160,5,200,360,'AVG','gFreqs')# (differenced high with low & averaged through subjects )


# ### Subject-wise Spectra, while Time being variability


# ### Subject-wise Spectra, while Time being variability

# In[44]:


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

# In[63]:


values,_,_,_ = mean_std(np.array(low_gft),3)
np.sum(values)/2


# In[64]:


np.sum(values[:177])


# In[65]:


G.e[178]


# ### Dichotomy 

# In[49]:


#1
l = np.where(G.e<=11.32)[0][1:]
h = np.where(G.e>11.32)[0]


# In[65]:


def filters(isc,band,length):
    indicator = np.ones([1,length])
    cll =list() 
    cll.append(np.matmul(indicator,np.abs(np.array(isc)[0,band,:]))) # 1 x length & length x time
    for i in range(1,10):
        cll.append(np.matmul(indicator,np.abs(np.array(isc)[i,band,:])))
    cll = np.reshape(cll,[10,500])
    return cll


# In[61]:





# In[66]:




#def lowISC_high_ISC(*typ):
a = 1  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter
plt.figure(figsize=(15,15))
typ = {'High ISC':high_gft,'Low ISC':low_gft}
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
plt.suptitle('Dichotomized the eigen values such that the power distribution is same & sliced the PSD using the same [Low freq = blue] Note: used np.abs while using indicator')
plt.show()

# ideas:
#1. Sub-wise plot
#2. Freq-wise plot
#3. High - Low "dicotomized plot" and compare high - low heatmap


# ### Frequency-wise
# In[]


b = np.arange(0,360)
b


# In[54]:




#def lowISC_high_ISC(*typ):
a = 1  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter
plt.figure(figsize=(15,15))
typ = {'High ISC':high_gft,'Low ISC':low_gft}
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
plt.suptitle('Dichotomized the eigen values such that the power distribution is same & sliced the PSD using the same [blue = High ISC]')
plt.show()


# ### Subject-wise

# In[ ]:

""" 
get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('gui', 'qt') """


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
a = 1  # number of rows
b = 2  # number of columns
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


""" np.savez_compressed('data.npz',mean_t1=mean_t1, mean_t2=mean_t2,mean_std=mean_std )


high_isc = [(np.array(averaging_by_parcellation(src_high1))),(np.array(averaging_by_parcellation(src_high2))), 
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
 """


# In[5]:





# In[13]:

print(G.e[0])

# In[ ]:


# In[71]:




#def lowISC_high_ISC(*typ):
a = 1  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter
plt.figure(figsize=(15,15))
typ = {'High ISC':high_gft,'Low ISC':low_gft}
for i in range(2):
        
        mean_t1,std_t1, top1, bottom1= mean_std(high_gft,3)
        mean_t2,std_t2, top2, bottom2= mean_std(low_gft,3)
        
        plt.legend()
        plt.plot(range(359),mean_t1[:],color='r')
        plt.fill_between(range(359),bottom1[:],top1[:], color='r', alpha=.1,label='High ISC')
        plt.plot(range(359),mean_t2[:],color='b')
        plt.fill_between(range(359),bottom2[:], top2[:], color='b', alpha=.1,label='Low ISC')
        plt.ylabel('gPSDs')
        plt.xlabel('Eigen values')
        plt.title('Graph PSD for both the conditions while subject being the variability for the low freq')
        plt.xticks(ticks=np.arange(1,360,44),labels=np.round(G.e[np.arange(1,360,44)],decimals=2),rotation='horizontal')
        #plt.axvline(x=250, linestyle = '--', color='g')
        
        #plt.ylabel('log (gPSD)')
#plt.suptitle('Dichotomized the eigen values(at 1.02) such that the power distribution is same & sliced the PSD using the same [Low freq = blue] Note: used np.abs while using indicator')
plt.show()

# ideas:
#1. Sub-wise plot
#2. Freq-wise plot
#3. High - Low "dicotomized plot" and compare high - low heatmap


# In[79]:




#def lowISC_high_ISC(*typ):
a = 1  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter
plt.figure(figsize=(15,15))
typ = {'High ISC':high_gft,'Low ISC':low_gft}
for i in range(2):
        
        mean_t1,std_t1, top1, bottom1= mean_std(high_gft,3)
        mean_t2,std_t2, top2, bottom2= mean_std(low_gft,3)
        
        plt.legend()
        plt.plot(range(177,359),mean_t1[177:],color='r')
        plt.fill_between(range(177,359),bottom1[177:],top1[177:], color='r', alpha=.1,label='High ISC')
        plt.plot(range(177,359),mean_t2[177:],color='b')
        plt.fill_between(range(177,359),bottom2[177:], top2[177:], color='b', alpha=.1,label='Low ISC')
        plt.ylabel('gPSDs')
        plt.xlabel('Eigen values')
        plt.title('Graph PSD for both the conditions while subject being the variability for the high freq')
        plt.xticks(ticks=np.arange(177,360,44),labels=[11.24, 13.4 , 15.7 , 18.32, 28.18],rotation='horizontal')
        #plt.axvline(x=250, linestyle = '--', color='g')
        
        #plt.ylabel('log (gPSD)')
#plt.suptitle('Dichotomized the eigen values(at 1.02) such that the power distribution is same & sliced the PSD using the same [Low freq = blue] Note: used np.abs while using indicator')
plt.show()

# ideas:
#1. Sub-wise plot
#2. Freq-wise plot
#3. High - Low "dicotomized plot" and compare high - low heatmap


# In[77]:


np.round(G.e[np.arange(177,360,44)],2)

# In[ ]:

import scipy
std_err = scipy.stats.sem(np.mean(high_gft,axis=2))
std_err2 = scipy.stats.sem(np.mean(low_gft,axis=2))


# %%


global_mean = mean_std(high_gft,ax=3)[0]


global_mean2 = mean_std(low_gft,ax=3)[0]


print(np.shape(global_mean))
print([np.sum(global_mean[:50]),np.sum(global_mean2[:50])])

print([np.sum(global_mean[50:200]),np.sum(global_mean2[50:200])])

# %%
np.sum(std_err2[100:250])

np.sum(std_err[100:250])
# %%
np.sum(std_err2[250:])

np.sum(std_err[250:])
# %%


# %%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

labels = ['Low', 'Med', 'High']
high_GFT = [np.sum(global_mean[:50]), np.sum(global_mean[50:200]),np.sum(global_mean[200:])]
low_GFT = [np.sum(global_mean2[:50]), np.sum(global_mean2[50:200]),np.sum(global_mean2[200:])]
error = [sum(std_err[:50]),sum(std_err[50:200]),sum(std_err[200:])]
error2 = [sum(std_err2[:50]),sum(std_err2[50:200]),sum(std_err2[200:])]
import pandas as pd
data= pd.DataFrame({'labels':labels,'gPSD_low':low_GFT,'gPSD_high':high_GFT},index=None)

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, high_GFT, width, label='High ISC',yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
rects2 = ax.bar(x + width/2, low_GFT, width, label='Low ISC',yerr=error2, align='center', alpha=0.5, ecolor='black', capsize=10)

ylab = "gPSD"
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel(ylab)
ax.set_title('Global Average of the gPSD freq-wise (after trichotomizing) while SEM being error bars')
ax.set_xticks(x)
ax.set_xticklabels(labels)


ax.legend()


fig.tight_layout()

plt.show()


# %%

box_pairs=[(("Low", "Low_ISC"), ("High", "High_ISC"))],
len(box_pairs)
#font-size = 14
#line_width = 2
#subplot-grid
#


# %%
from nilearn.regions import signals_to_img_labels  
# load nilearn label masker for inverse transform
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image, plotting
from nilearn import datasets
from os.path import join as opj



G.compute_fourier_basis()

mnitemp = fetch_icbm152_2009()
mask_mni=image.load_img(mnitemp['mask'])
glasser_atlas=image.load_img(path_Glasser)


#print(NiftiMasker.__doc__)

signal=[]
U0_brain=[]
signal=np.expand_dims(np.array(G.U[:, 357]), axis=0) # add dimension 1 to signal array
U0_brain = signals_to_img_labels(signal,path_Glasser,mnitemp['mask'])
plotting.plot_glass_brain(U0_brain,title=f'Eigenvector {358}',colorbar=True,plot_abs=False,cmap='spring',display_mode='lzr')
plt.savefig("eigvector358.png")

# %%

U0_brain.to_filename('6th eigen vector.nii.gz')
# %%
# %%

high_gft[0]

# %%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

labels = ['Low', 'Med', 'High']
high_GFT = [np.sum(std_err[:50]), np.sum(std_err[50:200]),np.sum(std_err[200:])]
low_GFT = [np.sum(std_err2[:50]), np.sum(std_err2[50:200]),np.sum(std_err2[200:])]


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, high_GFT, width, label='High ISC', align='center', alpha=0.5, ecolor='black', capsize=10)
rects2 = ax.bar(x + width/2, low_GFT, width, label='Low ISC', align='center', alpha=0.5, ecolor='black', capsize=10)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('SEM')
ax.set_title('SEM of the gPSD freq-wise (after trichotomizing)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


fig.tight_layout()

plt.show()
# %%


print('for low freq       :',scipy.stats.mstats.ttest_rel(global_mean[1:51],global_mean2[1:51]))
print('for medium         :',scipy.stats.mstats.ttest_rel(global_mean[51:200],global_mean2[51:200]))
print('for high           :',scipy.stats.mstats.ttest_rel(global_mean[200:],global_mean2[200:]))

# %%
low_group = np.hstack([global_mean[:50],global_mean2[:50]])
med_group = np.hstack([global_mean[50:200],global_mean2[50:200]])
high_group = np.hstack([global_mean[200:],global_mean2[200:]])



scipy.stats.f_oneway(low_group,med_group,high_group)

# %%

# %%


# %%
# %%
#hue = low, high
#y=gpsd
#x = low med high

# %%
import pandas as pd
labels = ['Low', 'Med', 'High']

data = pd.DataFrame({'labels':labels,'gPSD':low_GFT})

data2 = pd.DataFrame({'labels':labels,'gPSD':high_GFT})
# %%
data_fin = data.append(data2,ignore_index=True)
data_fin['cond'] = ['Low_ISC','Low_ISC','Low_ISC','High_ISC','High_ISC','High_ISC']
# %%

data_fin

# %%
np.shape(low_gft)
# %%



# %%

import seaborn as sns 
import pandas as pd
fig=plt.figure(figsize = (17, 17))
import seaborn
seaborn.despine(left=True, bottom=True, right=True)

plt.rc('font', family='serif')
grid = fig.add_gridspec(6,5, wspace =1.3, hspace = 1.2)

#grid = gridspec.GridSpec(2,2, wspace =0.3, hspace = 0.8)
g1 = fig.add_subplot(grid[0:2, :2])
g2 = fig.add_subplot(grid[0:4, 2:])
g3 = fig.add_subplot(grid[2:4, :2])

g4 = fig.add_subplot(grid[4:, :2])
g5 = fig.add_subplot(grid[4:, 2:])
import matplotlib
def heatmap(diff,title,start1,end1,div,start2,end2,operation,ylabel):
    

    #fig, ax = plt.subplots()

    cmap_reversed = matplotlib.cm.get_cmap('Spectral').reversed()
    if operation == 'std':
        svm = sns.heatmap(np.std(diff,axis=0),cmap=cmap_reversed,ax=g1) 
    else:
        svm = sns.heatmap(np.average(diff,axis=0),cmap=cmap_reversed,ax=g1) 
    g1.set_ylabel('Graph Frequencies')
    g1.set_xlabel('Time (s)')
    xticks= [0,125,250,375,500]
    g1.set_xticks([])
    g1.set_xticks(xticks)
    g1.set_xticklabels(labels=[-0.5,-0.25,0,0.25,0.5],rotation='horizontal')
    yticks= np.arange(1,51,3)
    g1.set_yticks([])
    g1.set_yticks(yticks)
    g1.set_yticklabels(labels=np.arange(1,51,3),rotation='horizontal')
    
    
    g1.yaxis.set_tick_params(rotation=360)
    g1.axvline(x=250, linestyle = '--', color='b')
    g1.set_title(title,pad=10)
    #ax.tight_layout()
    #plt.show()


    g1.text(0.5,-0.22, "(a)", size=12, ha="center", 
         transform=g1.transAxes)
heatmap(differenced_low_freq,'Spectrogram for 1-50 frequencies',1,50,2,1,50,'AVG','gFreqs')# (differenced high with low & averaged through subjects )
#heatmap(differenced_medium_freq,'Spectrogram for 50-200 freqs (averaged thru subjs)',1,150,5,50,200,'AVG','gFreqs')# (differenced high with low & averaged through subjects )
#heatmap(differenced_high_freq,'Spectrogram for 200-360 freqs (averaged thru subjs)',1,160,5,200,360,'AVG','gFreqs')# (differenced high with low & averaged through subjects )




#def lowISC_high_ISC(*typ):
a = 1  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter
plt.figure(figsize=(15,15))
typ = {'High ISC':high_gft,'Low ISC':low_gft}
for i in range(2):
        
        mean_t1,std_t1, top1, bottom1= mean_std(high_gft,3)
        mean_t2,std_t2, top2, bottom2= mean_std(low_gft,3)
        
        #plt.legend()
        g2.plot(range(359),mean_t1[:],color='r')
        g2.fill_between(range(359),bottom1[:],top1[:], color='r', alpha=.1,label='High ISC')
        g2.plot(range(359),mean_t2[:],color='b')
        g2.fill_between(range(359),bottom2[:], top2[:], color='b', alpha=.1,label='Low ISC')
        g2.set_ylabel('gPSDs')
        g2.set_xlabel('Eigen values')
        #g2.set_title('Graph PSD for both conditions')
        xticks= np.arange(0,360,71)

        g2.set_xticks([])
        g2.set_xticks(xticks)
#        g2.set_xticklabels(labels=[-0.5,-0.25,0,0.25,0.5],rotation='horizontal')
        g2.set_xticklabels(labels=np.round(G.e[np.arange(1,360,71)],decimals=2),rotation='horizontal')
        #plt.axvline(x=250, linestyle = '--', color='g')
        g2.text(0.5,-0.10, "(c)", size=12, ha="center", 
        transform=g2.transAxes)
        #plt.ylabel('log (gPSD)')
#plt.suptitle('Dichotomized the eigen values(at 1.02) such that the power distribution is same & sliced the PSD using the same [Low freq = blue] Note: used np.abs while using indicator')
g2.legend(['High ISC','Low ISC'])



#def lowISC_high_ISC(*typ):
a = 1  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter
typ = {'High ISC':high_gft,'Low ISC':low_gft}


#plt.subplot(a, b, c)
cll1 = filters(typ[list(typ.keys())[0]],l,len(l))
cll2 = filters(typ[list(typ.keys())[0]],h,len(h))
mean_t1,std_t1, top1, bottom1= mean_std(cll1,2)
mean_t2,std_t2, top2, bottom2= mean_std(cll2,2)


g3.legend()
g3.plot(range(500),mean_t1,color='b')
g3.fill_between(range(500),bottom1,top1, color='b', alpha=.1,label='Low frequency')
g3.plot(range(500),mean_t2,color='r')
g3.fill_between(range(500),bottom2, top2, color='r', alpha=.1,label='High frequency')

g3.set_ylabel('gPSDs sliced using Eigen values')
g3.set_xlabel('Time (s)',fontsize=10)
xticks= [0,125,250,375,500]

g3.set_xticks(xticks)
g3.set_xticklabels(labels=[-0.5,-0.25,0,0.25,0.5],rotation='horizontal')
g3.axvline(x=250, linestyle = '--', color='g')
#g3.set_title('Graph PSD time-series', pad=10)
g3.legend()

g3.text(0.5,-0.20, "(b)", size=12, ha="center", 
         transform=g3.transAxes)




labels = ['Low', 'Med', 'High']
gPSD_high = [np.sum(global_mean[:50]), np.sum(global_mean[50:200]),np.sum(global_mean[200:])]
gPSD_low = [np.sum(global_mean2[:50]), np.sum(global_mean2[50:200]),np.sum(global_mean2[200:])]
error = [sum(std_err[:50]),sum(std_err[50:200]),sum(std_err[200:])]
error2 = [sum(std_err2[:50]),sum(std_err2[50:200]),sum(std_err2[200:])]

data= pd.DataFrame({'labels':labels,'gPSD_high':gPSD_high,'gPSD_low':gPSD_low},index=None)

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

rects1 = g4.bar(x - width/2, high_GFT, width, label='High ISC',yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
rects2 = g4.bar(x + width/2, low_GFT, width, label='Low ISC',yerr=error2, align='center', alpha=0.5, ecolor='black', capsize=10)

ylab = "gPSD"
# Add some text for labels, title and custom x-axis tick labels, etc.
g4.set_ylabel(ylab)
#g4.set_title('Frequency group-wise gPSD following smoothing',pad=10)#(after trichotomizing) while SEM being error bars
g4.set_xticks(x)
g4.set_xticklabels(labels)
order = ['low ISC','high ISC'] 
from statannot import add_stat_annotation
add_stat_annotation(g4,data=data_fin, y='gPSD', x ='labels', hue='cond',
                    box_pairs=[(("Med", "Low_ISC"), ("Med", "High_ISC")),
                                (("High", "Low_ISC"), ("High", "High_ISC"))],
                                 perform_stat_test=False, pvalues=[9.306376939220152e-10,1.0396018685855328e-08], #2.71e-05,1.70e-09,3.33e-14, #0.28,0.47,0.013, #3.00e-05,3.08e-18,2.53e-23 #0.047,9.3e-10,1.03e-08
                    line_offset_to_box=0.20, line_offset=0.1, line_height=0.05, text_format='simple', loc='inside', verbose=2)
g4.set_xlabel('Graph Frequency bands')

g4.legend(bbox_to_anchor=(0.35, 0.8), bbox_transform=g4.transAxes)
g4.text(0.5,-0.25, "(d)", size=12, ha="center", 
         transform=g4.transAxes)

signal=[]
U0_brain=[]
signal=np.expand_dims(np.array(G.U[:, -3]), axis=0) # add dimension 1 to signal array
U0_brain = signals_to_img_labels(signal,path_Glasser,mnitemp['mask'])
plotting.plot_glass_brain(U0_brain,colorbar=True,plot_abs=False,cmap='spring',display_mode='lzr',axes=g5)
g5.text(0.5,-0.25, "(e)", size=12, ha="center", 
         transform=g5.transAxes)

fig.suptitle('Graph', size=20)
#fig.savefig('/homes/v20subra/S4B2/Pub-quality Figures/Graph_grid.png',dpi=500)


# %%
fig.savefig('/homes/v20subra/S4B2/noise_baseline/Graph_grid.png',dpi=100)
# %%
np.average(differenced,axis=0)[6]
# %%
signl = np.ones(shape=(1,360))
signl[:,(np.where(G.U[:,6]<=0)[0])] = 0
# %%
signl
# %%
np.where(G.U[:,6]<=0)[0]

# %%
print(np.shape(differenced_high_freq))
# %%


fig2=plt.figure(figsize = (10,5.1))
import seaborn
seaborn.despine(left=True, bottom=True, right=True)

plt.rc('font', family='serif')
grid2 = fig2.add_gridspec(2,4, wspace =2.7)

#grid = gridspec.GridSpec(2,2, wspace =0.3, hspace = 0.8)
g11 = fig2.add_subplot(grid2[0:, :2])
g22 = fig2.add_subplot(grid2[0:, 2:])

fig2.tight_layout()

import matplotlib
def heatmap(diff,title,start1,end1,div,start2,end2,operation,ylabel):
    

    #fig, ax = plt.subplots()

    cmap_reversed = matplotlib.cm.get_cmap('Spectral').reversed()
    if operation == 'std':
        svm = sns.heatmap(np.std(diff,axis=0),cmap=cmap_reversed,ax=g11) 
    else:
        svm = sns.heatmap(np.average(diff,axis=0),cmap=cmap_reversed,ax=g11) 
    g11.set_ylabel('Graph Frequencies')
    g11.set_xlabel('Time (s)',fontsize=10)
    xticks= [0,125,250,375,500]
    g11.set_xticks([])
    g11.set_xticks(xticks)
    g11.set_xticklabels(labels=[-0.5,-0.25,0,0.25,0.5],rotation='horizontal')
    yticks= np.arange(2,51,3)
    g11.set_yticks([])
    g11.set_yticks(yticks)
    g11.set_yticklabels(labels=np.arange(2,51,3),rotation='horizontal')
    
    
    g11.yaxis.set_tick_params(rotation=360)
    g11.axvline(x=250, linestyle = '--', color='b')
    g11.set_title(title,pad=10)
    #ax.tight_layout()
    #plt.show()


    g11.text(0.5,-0.15, "(a)", size=10, ha="center", 
         transform=g11.transAxes)
heatmap(differenced_low_freq,'Spectrogram for 2-50 frequencies',1,50,2,1,50,'AVG','gFreqs')# (differenced high with low & averaged through subjects )
#heatmap(differenced_medium_freq,'Spectrogram for 50-200 freqs (averaged thru subjs)',1,150,5,50,200,'AVG','gFreqs')# (differenced high with low & averaged through subjects )
#heatmap(differenced_high_freq,'Spectrogram for 200-360 freqs (averaged thru subjs)',1,160,5,200,360,'AVG','gFreqs')# (differenced high with low & averaged through subjects )



#def lowISC_high_ISC(*typ):
a = 1  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter
typ = {'High ISC':high_gft,'Low ISC':low_gft}


#plt.subplot(a, b, c)
cll1 = filters(typ[list(typ.keys())[0]],l,len(l))
cll2 = filters(typ[list(typ.keys())[0]],h,len(h))
mean_t1,std_t1, top1, bottom1= mean_std(cll1,2)
mean_t2,std_t2, top2, bottom2= mean_std(cll2,2)


g22.legend()
g22.plot(range(500),mean_t1,color='b')
g22.fill_between(range(500),bottom1,top1, color='b', alpha=.1,label='Low frequency')
g22.plot(range(500),mean_t2,color='r')
g22.fill_between(range(500),bottom2, top2, color='r', alpha=.1,label='High frequency')

g22.set_ylabel('gPSDs sliced using Eigen values')
g22.set_xlabel('Time (s)',fontsize=10)
xticks= [0,125,250,375,500]

g22.set_xticks(xticks)
g22.set_xticklabels(labels=[-0.5,-0.25,0,0.25,0.5],rotation='horizontal')
g22.axvline(x=250, linestyle = '--', color='g')
g22.set_title('Graph PSD time-series', pad=10)
g22.legend()

g22.text(0.5,-0.15, "(b)", size=10, ha="center", 
         transform=g22.transAxes)
fig2.savefig('graph_pres1.png')
# %%
fig2=plt.figure(figsize = (5,5))
import seaborn
#seaborn.despine(left=True, bottom=True, right=True)

plt.rc('font', family='serif')
grid2 = fig2.add_gridspec(1,2)

#grid = gridspec.GridSpec(2,2, wspace =0.3, hspace = 0.8)
g11 = fig2.add_subplot(grid2[0, 0])
g22 = fig2.add_subplot(grid2[0, 1])
# %%

new_fig = plt.figure(figsize = (10,5))

new_grid = new_fig.add_gridspec(1,2)

G1 = new_fig.add_subplot(new_grid[0,0])

G2 = new_fig.add_subplot(new_grid[0,1])
# %%

fig2=plt.figure(figsize = (10,5))
import seaborn
#seaborn.despine(left=True, bottom=True, right=True)

plt.rc('font', family='serif')
grid2 = fig2.add_gridspec(1,2)

#grid = gridspec.GridSpec(2,2, wspace =0.3, hspace = 0.8)
g11 = fig2.add_subplot(grid2[0, 0:])


labels = ['Low', 'Med', 'High']
gPSD_high = [np.sum(global_mean[:50]), np.sum(global_mean[50:200]),np.sum(global_mean[200:])]
gPSD_low = [np.sum(global_mean2[:50]), np.sum(global_mean2[50:200]),np.sum(global_mean2[200:])]
error = [sum(std_err[:50]),sum(std_err[50:200]),sum(std_err[200:])]
error2 = [sum(std_err2[:50]),sum(std_err2[50:200]),sum(std_err2[200:])]

data= pd.DataFrame({'labels':labels,'gPSD_high':gPSD_high,'gPSD_low':gPSD_low},index=None)

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

rects1 = g11.bar(x - width/2, high_GFT, width, label='High ISC',yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
rects2 = g11.bar(x + width/2, low_GFT, width, label='Low ISC',yerr=error2, align='center', alpha=0.5, ecolor='black', capsize=10)

ylab = "gPSD"
# Add some text for labels, title and custom x-axis tick labels, etc.
g11.set_ylabel(ylab)
g11.set_title('Frequency group-wise gPSD following smoothing',pad=10)#(after trichotomizing) while SEM being error bars
g11.set_xticks(x)
g11.set_xticklabels(labels)
order = ['low ISC','high ISC'] 
from statannot import add_stat_annotation
add_stat_annotation(g11,data=data_fin, y='gPSD', x ='labels', hue='cond',
                    box_pairs=[(("Low", "Low_ISC"), ("Low", "High_ISC")),
                                (("Med", "Low_ISC"), ("Med", "High_ISC")),
                                (("High", "Low_ISC"), ("High", "High_ISC"))],
                                 perform_stat_test=False, pvalues=[0.052,9.306376939220152e-10,1.0396018685855328e-08], #2.71e-05,1.70e-09,3.33e-14, #0.28,0.47,0.013, #3.00e-05,3.08e-18,2.53e-23 #0.047,9.3e-10,1.03e-08
                    line_offset_to_box=0.20, line_offset=0.1, line_height=0.05, text_format='simple', loc='inside', verbose=2)
g11.set_xlabel('Graph Frequency bands',size=10)

g11.legend(bbox_to_anchor=(0.35, 0.8), bbox_transform=g11.transAxes)
g11.text(0.5,-0.15, "(d)", size=10, ha="center", 
         transform=g11.transAxes)
fig2.savefig("graph_pres3.png")

# %%
