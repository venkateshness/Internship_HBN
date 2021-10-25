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
    plt.figure()
heatmap(differenced_low_freq,'Spectrogram for 1-50 freqs (averaged thru subjs)',1,50,2,1,50,'AVG','gFreqs')# (differenced high with low & averaged through subjects )
heatmap(differenced_medium_freq,'Spectrogram for 50-200 freqs (averaged thru subjs)',1,150,5,50,200,'AVG','gFreqs')# (differenced high with low & averaged through subjects )
heatmap(differenced_high_freq,'Spectrogram for 200-360 freqs (averaged thru subjs)',1,160,5,200,360,'AVG','gFreqs')# (differenced high with low & averaged through subjects )


heatmap(differenced_low_freq,'Spectrogram for 1-50 freqs (std thru subjs)',1,50,2,1,50,'std','gFreqs')# (differenced high with low & averaged through subjects )
heatmap(differenced_medium_freq,'Spectrogram for 50-200 freqs (std thru subjs)',1,150,5,50,200,'std','gFreqs')# (differenced high with low & averaged through subjects )
heatmap(differenced_high_freq,'Spectrogram for 200-360 freqs (std thru subjs)',1,160,5,200,360,'AVG','gFreqs')# (differenced high with low & averaged through subjects )


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
l = np.where(G.e<=11.3275)[0][1:]
h = np.where(G.e>11.3275)[0]


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
plt.suptitle('Dichotomized the eigen values(at 1.02) such that the power distribution is same & sliced the PSD using the same [Low freq = blue] Note: used np.abs while using indicator')
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
plt.suptitle('Dichotomized the eigen values(at 1.02) such that the power distribution is same & sliced the PSD using the same [blue = High ISC]')
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
        plt.plot(range(177),mean_t1[:177],color='r')
        plt.fill_between(range(177),bottom1[:177],top1[:177], color='r', alpha=.1,label='High ISC')
        plt.plot(range(177),mean_t2[:177],color='b')
        plt.fill_between(range(177),bottom2[:177], top2[:177], color='b', alpha=.1,label='Low ISC')
        plt.ylabel('gPSDs')
        plt.xlabel('Eigen values')
        plt.title('Graph PSD for both the conditions while subject being the variability for the low freq')
        plt.xticks(ticks=[0,25,50,75,100,125,150,175],labels=[ 0.04,3.58, 4.85, 6.29, 7.57, 8.96, 9.9,11.17],rotation='horizontal')
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


global_mean = np.mean((np.mean(high_gft,axis=2)),axis=0)

global_mean2 = np.mean((np.mean(low_gft,axis=2)),axis=0)

print(np.shape(global_mean))
print([np.sum(std_err[:100]),np.sum(std_err2[:100])])


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
men_means = [np.sum(global_mean[:50]), np.sum(global_mean[50:200]),np.sum(global_mean[200:])]
women_means = [np.sum(global_mean2[:50]), np.sum(global_mean2[50:200]),np.sum(global_mean2[200:])]
error = [sum(std_err[:50]),sum(std_err[50:200]),sum(std_err[200:])]


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='High ISC',yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
rects2 = ax.bar(x + width/2, women_means, width, label='Low ISC',yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('gPSD')
ax.set_title('Global Average of the gPSD freq-wise (after trichotomizing) while SEM being error bars')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()


# %%

std_err[:50]
# %%


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
signal=np.expand_dims(np.array(G.U[:, 5]), axis=0) # add dimension 1 to signal array
U0_brain = signals_to_img_labels(signal,path_Glasser,mnitemp['mask'])
plotting.plot_glass_brain(U0_brain,title=f'Vector {6}',colorbar=True,plot_abs=False,cmap='spring',display_mode='lzr')


# %%

U0_brain.to_filename('6th eigen vector.nii.gz')
# %%
