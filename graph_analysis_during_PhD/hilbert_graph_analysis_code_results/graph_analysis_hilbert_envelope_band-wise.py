#%%
from cProfile import label
from email import header
from fileinput import filename
from logging import error
from operator import index
from turtle import color
from matplotlib import axis
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
from scipy import stats
import pandas as pd

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

    G=graphs.Graph(connectivity,gtype='HCP subject',lap_type='normalized',coords=coordinates) 
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

envelope_signal_bandpassed = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects/envelope_signal_bandpassed.npz', mmap_mode='r')

alpha = envelope_signal_bandpassed['alpha']
beta = envelope_signal_bandpassed['beta']
theta = envelope_signal_bandpassed['theta']

envelope_signal_bandpassed_gft_total = dict()

def GFT(band,which_band):
    envelope_signal_bandpassed_gft_total[which_band] = [G.gft(np.array(band[0])),G.gft(np.array(band[1])), 
        G.gft(np.array(band[2])), G.gft(np.array(band[3])), 
        G.gft(np.array(band[4])), G.gft(np.array(band[5])),
        G.gft(np.array(band[6])), G.gft(np.array(band[7])), 
        G.gft(np.array(band[8])),  G.gft(np.array(band[9])),
        G.gft(np.array(band[10])),  G.gft(np.array(band[11])),
        G.gft(np.array(band[12])),  G.gft(np.array(band[13])),
        G.gft(np.array(band[14])),  G.gft(np.array(band[15])),
        G.gft(np.array(band[16])),  G.gft(np.array(band[17])),
        G.gft(np.array(band[18])),  G.gft(np.array(band[19])),
        G.gft(np.array(band[20])),  G.gft(np.array(band[21])),
        G.gft(np.array(band[22])),  G.gft(np.array(band[23])),
        G.gft(np.array(band[24]))]    
GFT(theta,'theta')
GFT(alpha,'alpha')
GFT(beta,'beta')
#%%




#1x1

# envelope_signal_bandpassed_gft_total_downsampling = dict()

# def downsampling(which_band):
#     to_slice =list(np.arange(0,21251,125))

#     averaged_slices = list()
#     for i in range(len(to_slice)-1):
#         print(to_slice[i+1])
#         averaged_slices.append(np.average(np.array(envelope_signal_bandpassed_gft_total[which_band])[:,:,to_slice[i]:to_slice[i+1]],axis=2))

#     stage1 = np.swapaxes(averaged_slices,0,1)
#     envelope_signal_bandpassed_gft_total_downsampling[which_band] = np.swapaxes(stage1,1,2)
    
    

# downsampling('theta')
# downsampling('alpha')
# downsampling('beta')

#%%
# %%
#########################################
######Utility function###################
#########################################
def mean_std(freq,ax):
    if ax>2:
        d = np.average(np.array(np.abs(freq)),axis=2)[:,1:]
    else: d = np.abs(freq[1:,:])
    mean_t = np.mean(d,axis=0)
    std_t = 2 * np.std(d,axis=0)
    top = mean_t + std_t
    bottom = mean_t - std_t
    
    return mean_t,std_t,top,bottom

values,_,_,_ = mean_std(np.array(envelope_signal_bandpassed_gft_total['beta']),3)
print('divided power: ', np.sum(values)/2)
print('sum of power is: ',np.sum(values[:87])) #


def split(which_gft):
    which_gft = np.array(which_gft)
    print("the size before splitting",np.shape(which_gft))
    # which_gft_low_freq = which_gft[:,1:88,:]
    # #which_gft_medium_freq = which_gft[:,52:201,:]
    # which_gft_high_freq = which_gft[:,88:,:]
    which_gft_whole_freq = which_gft[:,1:,:]
    print("the low-freq size after splitting",np.shape(which_gft_whole_freq))

    return which_gft_whole_freq



#%%

isc_result = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects/sourceCCA_ISC.npz')['sourceISC']
noise_floor_source = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects/noise_floor_source.npz')['sourceCCA']

print(np.shape(noise_floor_source))

significance = np.array(np.where(np.max(np.array(noise_floor_source)[:,0,:],axis=0)<isc_result[0]))

def axvspan(axes,fs,c,alpha):
    # axes.axvspan(154*fs, 162*fs, alpha=alpha, color=c)
    # axes.axvspan(84*fs, 87*fs, alpha=alpha, color=c)
    # axes.axvspan(102*fs, 107*fs, alpha=alpha, color=c)
    # axes.axvspan(113*fs, 115*fs, alpha=alpha, color=c)
    # axes.axvspan(119*fs, 123*fs, alpha=alpha, color=c)
    # axes.axvspan(48*fs, 49*fs, alpha=alpha, color=c)

    # axes.axvspan(53*fs, 58*fs, alpha=alpha, color=c)
    # axes.axvspan(131*fs, 135*fs, alpha=alpha, color=c)
    # axes.axvspan(16*fs, 17*fs, alpha=alpha, color=c)
    # axes.axvspan(74*fs, 75*fs, alpha=alpha, color=c)
    # axes.axvspan(5*fs, 13*fs, alpha=alpha, color='r')
    # axes.axvspan(40*fs, 46*fs, alpha=alpha, color='r')
    # axes.axvspan(164*fs, 170*fs, alpha=alpha, color='r')
    # axes.axvspan(53*fs, 59*fs, alpha=alpha, color='b')
    # axes.axvspan(84*fs, 87*fs, alpha=alpha, color='b')
    # axes.axvspan(102*fs, 105*fs, alpha=alpha, color='b')
    axes.axvspan(0*fs, 0*fs, alpha=alpha, color='b')

    # items_weak = np.hstack([np.arange(5*125,13*125),np.arange(40*125,46*125),np.arange(164*125,170*125)])
    # items_strong = np.hstack([np.arange(53*125,59*125),np.arange(84*125,87*125),np.arange(102*125,105*125),np.arange(154*125,162*125)])




def plot(kwargs,title,filename):
    axes =[411,412,413,414]
    fig = plt.figure(figsize = (35,25))
    ax1 = plt.subplot(axes[0], frameon=False)
    ax2 = plt.subplot(axes[1], frameon=False)
    ax3 = plt.subplot(axes[2], frameon=False)
    ax4 = plt.subplot(axes[3], frameon=False)

    ax1.plot(np.sum(np.average(np.abs(kwargs[0]),axis=0),axis=0), marker = 'x', markeredgecolor = 'black', markevery=significance[0],color='g',label='Low (2 to 50 freqs)')[0]
    axvspan(ax1,125, c = 'lavender', alpha = 0.8)
    ax1.legend(loc="upper left")

    ax2.plot(np.sum(np.average(np.abs(kwargs[1]),axis=0),axis=0), marker = 'x', markeredgecolor = 'black', markevery=significance[0],color='b',label='Med (51 to 200)')[0]
    axvspan(ax2,125, c = 'lavender', alpha = 0.8)
    ax2.legend(loc="upper left")

    # ax3.plot(np.sum(np.average(np.abs(kwargs[2]),axis=0),axis=0), marker = 'x', markeredgecolor = 'black', markevery=significance[0],color='r',label='high (200 & above)')[0]
    # axvspan(ax3,125, c = 'lavender', alpha = 0.8)
    # ax3.legend(loc="upper left")

    # xticks= np.arange(1,21250,125)
    #ax1.legend((a1), ('Low', 'Med','High'), loc='lower left', shadow=True)

    xticks= np.arange(1,21250,2125)


    ax4.plot(range(170),isc_result[0], label='Source-level estimation')
    ax4.fill_between(range(170),np.max(np.array(noise_floor_source)[:,0,:],axis=0).T,np.min(np.array(noise_floor_source)[:,0,:],axis=0).T,color ='grey',alpha=0.8)
    ax4.plot(significance,isc_result[0][significance],
                marker='o', ls="",color='red',markersize=4)
    ax4.legend(loc="upper left")

  

    # ax1.get_shared_x_axes().join(ax1, ax2,ax3,ax4)
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    
    fig.text(0.5, 0.04, 'time (s)', ha='center')
    fig.text(0.08, 0.15, 'ISC coefficients', rotation='vertical')
    fig.text(0.08, 0.65, 'graph PSD',  rotation='vertical')
    plt.suptitle(title)
    plt.show()
    # fig.savefig(f'/homes/v20subra/S4B2/graph_analysis_during_PhD/hilbert_graph_analysis_code_results/{filename}')

plot(split(envelope_signal_bandpassed_gft_total['beta']),title='The GFT of the Alpha band (8 - 13 Hz) envelope signal',filename='AlphaBand')


def slicing(what_to_slice,where_to_slice):
    array_to_append = list()
    array_to_append.append ( what_to_slice[:,where_to_slice] )
    return array_to_append


# %%

# items_weak = np.hstack([np.arange(1*125,6*125),np.arange(20*125,26*125),np.arange(33*125,39*125),np.arange(62*125,67*125),
#                         np.arange(70*125,81*125),np.arange(88*125,95*125),np.arange(129*125,141*125),np.arange(148*125,154*125)])

# items_strong = np.hstack([np.arange(6*125,9*125),np.arange(11*125,17*125), np.arange(18*125-62,18*125+63),np.arange(27*125,30*125),np.arange(40*125,44*125),np.arange(46*125,51*125),np.arange(53*125-62,53*125+63),
# np.arange(56*125,61*125),np.arange(68*125-62,68*125+63),np.arange(85*125-62,85*125+63),np.arange(99*125-62,99*125+63),np.arange(102*125,110*125),np.arange(111*125,113*125),np.arange(114*125-62,114*125+63),
# np.arange(118*125,122*125),np.arange(123*125-62,123*125+63),np.arange(127*125-62,127*125+63),np.arange(144*125,147*125),np.arange(155*125,158*125),np.arange(159*125-62,159*125+63),np.arange(163*125-62,163*125+63),np.arange(159*125,161*125)])


#####################
#######1st component
#####################
items_weak = np.hstack([np.arange(0*125,5*125),np.arange(33*125,38*125),np.arange(62*125,66*125),
                        np.arange(70*125,74*125),np.arange(88*125,95*125),np.arange(129*125,133*125),np.arange(148*125,153*125)])


items_strong = np.hstack([np.arange(7*125,9*125),np.arange(13*125,17*125), np.arange(27*125-62,30*125+63),np.arange(40*125,44*125),np.arange(58*125-62,58*125+63),
np.arange(85*125-62,85*125+63),np.arange(99*125-62,99*125+63),np.arange(104*125,110*125),np.arange(118*125,124*125),np.arange(155*125,158*125),np.arange(127*125-62,127*125+63),np.arange(144*125-62,144*125+63)])

print(len(items_strong)/125)
print(len(items_weak)/125)


#Step 1
full_band = split(np.abs(envelope_signal_bandpassed_gft_total['beta']))

population_data = dict()

def slice_and_sum(freqs,items_strong,items_weak,which_freq):

    summed_strong_time = np.average(slicing(freqs,items_strong),axis=2)
    summed_weak_time =  np.average(slicing(freqs,items_weak),axis=2)
    print(np.shape(summed_weak_time))

    summed_strong_subjs =  np.mean(summed_strong_time,axis=1)
    summed_weak_subjs =  np.mean(summed_weak_time,axis=1)
    population_data[f'{which_freq}+weak'] = summed_weak_time
    population_data[f'{which_freq}+strong'] = summed_strong_time

    return summed_strong_subjs, summed_weak_subjs,scipy.stats.ttest_rel(summed_strong_time,summed_weak_time,axis=1)



#Step 2
summed_full_band_freqs = np.linalg.norm(full_band,axis=1)
#summed_med_freqs = np.linalg.norm(med,axis=1)
# summed_high_freqs = np.linalg.norm(high,axis=1)

print("indicator",np.shape(summed_full_band_freqs))

labels = ['Wide-band']
to_plot_strong_ISC_full_band, to_plot_weak_ISC_full_band, ttest_full_band = slice_and_sum(summed_full_band_freqs,items_strong,items_weak,'full_band')
#to_plot_strong_ISC_med_freqs, to_plot_weak_ISC_med_freqs, ttest_med_freqs = slice_and_sum(summed_med_freqs,items_strong,items_weak)
# to_plot_strong_ISC_high_freqs, to_plot_weak_ISC_high_freqs, ttest_high_freqs = slice_and_sum(summed_high_freqs,items_strong,items_weak,'high_freq')

to_plot_strong_ISC = [to_plot_strong_ISC_full_band[0]]
to_plot_weak_ISC = [to_plot_weak_ISC_full_band[0]]

print(ttest_full_band)


to_plot_strong_ISC
#%%


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


def stats_SEM(freqs,items):

    return scipy.stats.sem(np.average(np.array(slicing(freqs,items)),axis=2).T)#/np.sqrt(25)

std_err_weak = np.hstack([stats_SEM(summed_full_band_freqs,items_weak)])

std_err_strong = np.hstack([stats_SEM(summed_full_band_freqs,items_strong)])

print(std_err_strong)
print(std_err_weak)

rects1 = plt.bar(x - width/2, np.hstack(to_plot_strong_ISC), width, label='Strong ISC', yerr = std_err_strong, align='center', alpha=0.5, ecolor='black', capsize=10)
rects2 = plt.bar(x + width/2, np.hstack(to_plot_weak_ISC), width, label='Weak ISC', yerr = std_err_weak, align='center', alpha=0.5, ecolor='black', capsize=10)

plt.legend()
plt.ylabel("gPSD")
plt.title('Accumulated graph power for the envelope beta-band signal')
plt.xticks(x,labels)
plt.show()

# %%

# sig_combined = list(np.logical_or( set ( np.array(np.where(np.max(np.array(noise_floor_source)[:,0,:],axis=0)<isc_result[0]))[0]), 
# set ( np.array(np.where(np.max(np.array(noise_floor_source)[:,1,:],axis=0)<isc_result[1]))[0])))

sig_combined =sorted(list(set(np.where(np.max(np.array(noise_floor_source)[:,0,:],axis=0)<isc_result[0])[0]).union (set(np.where(np.max(np.array(noise_floor_source)[:,1,:],axis=0)<isc_result[1])[0]))))
len(sig_combined)

# %%
# np.shape(np.max(noise_floor_source[:,0,:],axis=0))
# # %%
# len(sig_1C[0])
# # %%
plt.figure(figsize=(25,15))

sig_1C = np.array(np.where(np.max(np.array(noise_floor_source)[:,0,:],axis=0)<isc_result[0]))
plt.plot(isc_result[0])
plt.plot(sig_1C,isc_result[0][sig_1C],
                marker='o', ls="",color='red',markersize=4)
sig_2C = np.array(np.where(np.max(np.array(noise_floor_source)[:,1,:],axis=0)<isc_result[1]))
plt.plot(isc_result[1])
plt.plot(sig_2C,isc_result[1][sig_2C],
                marker='o', ls="",color='green',markersize=4)

# #0 to 5,
items_weak = np.hstack([np.arange(0*125,6*125),np.arange(20*125,26*125),np.arange(33*125,39*125),np.arange(62*125,67*125),
                        np.arange(70*125,82*125),np.arange(88*125,95*125),np.arange(129*125,141*125),np.arange(148*125,154*125)])
print(len(items_weak)/125)

plt.axvspan(0, 5, alpha=0.1, color='b')#5
plt.axvspan(20, 25, alpha=0.1, color='b')#5
plt.axvspan(33, 38, alpha=0.1, color='b')#9
plt.axvspan(62, 66, alpha=0.1, color='b')#3
plt.axvspan(70, 82, alpha=0.1, color='b')#13

plt.axvspan(88, 95, alpha=0.1, color='b')#8

plt.axvspan(129, 141, alpha=0.1, color='b')#12

plt.axvspan(148, 153, alpha=0.1, color='b')#5

# # %%
# plt.figure(figsize=(25,15))

# plt.plot(isc_result[0])
# plt.axvspan(0, 5, alpha=0.1, color='b')#5
# plt.axvspan(33, 37, alpha=0.1, color='b')#9
# plt.axvspan(62, 65, alpha=0.1, color='b')#3
# plt.axvspan(70, 73, alpha=0.1, color='b')#13

# plt.axvspan(88, 95, alpha=0.1, color='b')#8

# plt.axvspan(129, 133, alpha=0.1, color='b')#12

# plt.axvspan(148, 153, alpha=0.1, color='b')#5


# # %%

# # r = set(np.where(isc_result[0]>np.max (noise_floor_source[:, 0, :], axis=(0)))[0]).union(set(np.where(isc_result[1]>np.max (noise_floor_source[:, 1, :], axis=(0)))[0]))
# # r2 =r.union(set(np.where(isc_result[2]>np.max (noise_floor_source[:, 2, :], axis=(0)))[0]))
# # r2


# # %%


# def mean_std_fresh(subjects,ax):
   
#     d = np.abs(subjects[:,:])
#     mean_t = np.mean(d,axis=0)
#     std_t = 2 * np.std(d,axis=0)
#     top = mean_t + std_t
#     bottom = mean_t - std_t
    
#     return mean_t,std_t,top,bottom
# whole_band = np.abs(envelope_signal_bandpassed_gft_total['beta'])


# for_strong = np.mean(whole_band[:,:,items_strong],axis=(2))
# for_weak = np.mean(whole_band[:,:,items_weak],axis=(2))


# mean_t1,std_t1, top1, bottom1= mean_std_fresh(for_strong,2)
# mean_t2,std_t2, top2, bottom2= mean_std_fresh(for_weak,2)


# plt.plot(range(360),np.log(mean_t1),color='g')
# plt.fill_between(range(360),np.log(bottom1),np.log(top1), color='g', alpha=.1,label='Strong')
# plt.plot(range(360),np.log(mean_t2),color='r')
# plt.fill_between(range(360),np.log(bottom2), np.log(top2), color='r', alpha=.1,label='Weak')
# plt.legend()
# plt.xlabel("Graph Frequencies")
# plt.ylabel("Graph power in log")
# plt.title("The Graph power of the Alpha envelope signal for two periods")
# # %%
# len(items_strong)/125

# %%

# %%
fig = plt.figure(figsize =(10, 5))
 
# Creating axes instance
ax = fig.add_axes([0, 1, 2, 3])

pop = [population_data['low_freq+strong'].T,population_data['low_freq+weak'].T,population_data['high_freq+strong'].T,population_data['high_freq+weak'].T]
ax.boxplot(np.squeeze(pop).T,labels=['Strong ISC, Low Freq','Weak ISC, Low Freq','Strong ISC, High Freq','Weak ISC, High Freq'])
plt.title('Boxplot for the different conditions & different frequencies')
# %%
np.shape(np.squeeze(pop).T)
# %%
# plt.plot(population_data['low_freq+strong'].T)
# plt.plot(population_data['low_freq+weak'].T)

# %%


def squeeze(data_to_squeeze):
    return population_data[data_to_squeeze].reshape(-1, )

df = pd.DataFrame(index=None)
df['low_freqs'] = squeeze('low_freq+strong')
df['low_freqw'] = squeeze('low_freq+weak')

df['high_freqs'] = squeeze('high_freq+strong')
df['high_freqw'] = squeeze('high_freq+weak')
df['subjects']= list(np.arange(1,26))
# %%
import scipy.stats as stats
# %%
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# formula = smf.mixedlm('subjects ~ C(low_freqs) + C(low_freqw) : C(high_freqs) + C(high_freqw):C(low_freqs) + C(low_freqw)+C(high_freqs) + C(high_freqw)',df, groups=df['subjects'])
#formula = smf.mixedlm('subjects ~ C(low_freqs,low_freqw) : C(high_freqs,high_freqw):C(low_freqs,low_freqw,high_freqs,high_freqw)',df, groups=df['subjects'])

# lm = smf.ols('subjects~high_freqs+high_freqw+low_freqs+low_freqw', data =df).fit()
# sm.stats.anova_lm(lm)


# %%
df2 = pd.DataFrame()
df2['subjects'] = [*range(1, 26), *range(1, 26),*range(1, 26),*range(1, 26)]

new_df=df2.sort_values(by=['subjects'],ascending=True)


new_df
lis2 =list()
for i in range(25):
    lis2.append([['low']*2])
    lis2.append([['high']*2])
np.hstack(lis2)


lis = list()
for i in range(50):
    lis.append(['strong','weak'])

new_df['frequencies']= np.array(np.hstack(lis2))[0]
new_df['isc_condition'] = np.array(np.hstack(lis))
df.columns = ['low_stro', 'low_weak', 'hig_stron', 'hig_weak', 'subjects']

new_df['values']= np.hstack(df[['low_stro', 'low_weak', 'hig_stron' ,'hig_weak']].values)



np.hstack(df[['low_stro', 'low_weak', 'hig_stron' ,'hig_weak']].values)
# %%
new_df.to_csv('for_spss_anova_theta.csv')
# %%

lm = smf.ols('values~C(frequencies)*C(isc_condition)', data =new_df).fit()
sm.stats.anova_lm(lm)

# %%
