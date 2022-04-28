#%%
from cProfile import label
from collections import defaultdict
from tkinter import Y
from turtle import color, pos, shape
from click import style
from cv2 import accumulate, norm
import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters
from pygsp import plotting as gsp_plt
from nilearn import image, plotting, datasets


from pathlib import Path
from scipy import io as sio
from pygsp import graphs

import scipy
import torch
import pickle
import seaborn as sns
import pandas as pd
from statannot import add_stat_annotation
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import ptitprince as pt

#%%

sns.set_theme()
############################################################
##########Getting the Graph ready###########################
############################################################ 
def graph_setup(unthresholding, percentage,weights):
    path_Glasser='/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz'
    res_path=''


    # Load structural connectivity matrix
    # connectivity = sio.loadmat('/homes/v20subra/S4B2/GSP/SC_avg56.mat')['SC_avg56']
 
    coordinates = sio.loadmat('/homes/v20subra/S4B2/GSP/Glasser360_2mm_codebook.mat')['codeBook'] 

    G=graphs.Graph(weights,gtype='HCP subject',lap_type='combinatorial',coords=coordinates) 
    G.set_coordinates('spring')
    print('{} nodes, {} edges'.format(G.N, G.Ne))

    if unthresholding:
        pickle_file = '/homes/v20subra/S4B2/GSP/MMP_RSFC_brain_graph_fullgraph.pkl'

        with open(pickle_file, 'rb') as f:
                    [connectivity]= pickle.load(f)
        np.fill_diagonal(connectivity,0)

        G = graphs.Graph(connectivity)
        print(G.is_connected())
        print('{} nodes, {} edges'.format(G.N, G.Ne))

    return G

def NNgraph():
    
  
    pickle_file = '/homes/v20subra/S4B2/GSP/MMP_RSFC_brain_graph_fullgraph.pkl'

    with open(pickle_file, 'rb') as f:
                [connectivity]= pickle.load(f)
    np.fill_diagonal(connectivity,0)
    
    graph = torch.from_numpy(connectivity)
    knn_graph = torch.zeros(graph.shape)
    for i in range(knn_graph.shape[0]):
        graph[i,i] = 0
        best_k = torch.sort(graph[i,:])[1][-8:]
        knn_graph[i, best_k] = 1
        knn_graph[best_k, i] = 1
        
    degree = torch.diag(torch.pow(knn_graph.sum(dim = 0), -0.5))

    weight_matrix_after_NN = torch.matmul(degree, torch.matmul(knn_graph, degree))
    return weight_matrix_after_NN


G = graph_setup(False,66,NNgraph())
G.compute_fourier_basis()

#%%
envelope_signal_bandpassed = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/envelope_signal_bandpassed_low_high_beta.npz', mmap_mode='r')

alpha = envelope_signal_bandpassed['alpha']
low_beta = envelope_signal_bandpassed['lower_beta']
high_beta = envelope_signal_bandpassed['higher_beta']
theta = envelope_signal_bandpassed['theta']
# %%
def slicing(what_to_slice,where_to_slice,axis):
        array_to_append = list()
        if axis >2:
            array_to_append.append ( what_to_slice[:,:,where_to_slice] )
        else:
            array_to_append.append ( what_to_slice[:,where_to_slice] )
        return array_to_append


def slicing_freqs(freqs,indices_pre_strong,indices_post_strong):

    # print("indices length for Pre-Strong ISC is: ", len(indices_pre_strong)/125)
    items_pre_strong = np.squeeze(slicing(freqs,indices_pre_strong,axis=3))

    # print("indices length for Post-Strong ISC is: ", len(indices_post_strong)/125)
    items_post_strong = np.squeeze(slicing(freqs,indices_post_strong,axis=3))

    return items_pre_strong,items_post_strong


def sum_freqs(freqs,axis):
        # print("before summing the freqs",np.shape(freqs))
        # print("after summing:",np.shape(np.sum(np.sqrt(np.power(freqs,2)),axis=axis)))

    return np.sum(np.sqrt(np.power(freqs,2)),axis=axis)#L2 norm


def stats_SEM(freqs):
        # print("the shape before:",np.shape(freqs))
        return scipy.stats.sem(freqs,axis=1)#/np.sqrt(25)


def baseline(pre,post):
    # print(np.shape(pre))
    # print("mean baseline setup",np.mean(pre[:-65].T))
    # print("after setup",np.shape(np.array((post.T - np.mean(pre.T))/np.mean(pre.T))))
    return np.array((post.T - np.mean(pre.T))/np.mean(pre.T))

def averaging_time(freqs,axis=-1):
    # print("before average",np.shape(freqs))
    # print("after",np.shape(np.average(freqs.T,axis=-1)))
    return np.average(freqs.T,axis=axis)


def accumulate_freqs_and_plot(freq_pre_low,freq_post_low,freq_pre_med,freq_post_med,freq_pre_high,freq_post_high,label,color,band,index):
        fig = plt.figure(figsize=(45,25))
        a = 3  # number of rows
        b = 1  # number of columns
        c = 1  # initialize plot counter
        dic_append_everything = defaultdict(dict)
        for i in range(3):
            # plt.subplot(a,b,c)
            if i ==0:
                freq = np.concatenate([baseline(freq_pre_low,freq_pre_low),baseline(freq_pre_low,freq_post_low)])
            elif i ==1:
                freq = np.concatenate([baseline(freq_pre_med,freq_pre_med),baseline(freq_pre_med,freq_post_med)])
            else:
                freq = np.concatenate([baseline(freq_pre_high,freq_pre_high),baseline(freq_pre_high,freq_post_high)])
            dic_append_everything[i]= freq
            
        return dic_append_everything
        #     plt.plot(np.mean(freq,axis=1),label=label,color=color)
        #     plt.fill_between(range(376),np.mean(freq,axis=1)-stats_SEM(freq),np.mean(freq,axis=1)+stats_SEM(freq),alpha=0.2)
        #     plt.axvline(125,color='black',linestyle='--')
        #     plt.axvspan(xmin=0,xmax=113,color='r',alpha=0.2,label ='Baseline')
        #     plt.legend()
        #     plt.xticks(ticks=np.arange(0,376,62.5),labels= np.arange(-1000,2500,500))
        #     c+=1
        # plt.suptitle(f'ERD for the {band}; time = around 8s')
        # plt.xlabel('time (ms)')
        # plt.ylabel('Relative Power Difference')
        # plt.show()
        # fig.savefig(f'/homes/v20subra/S4B2/Graph-related_analysis/ERD/{band}',bbox_inches='tight')

def master(band,label_band):
    GFTed_cortical_signal = [G.gft(np.array(band[i])) for i in range(25)]
    GFTed_cortical_signal_low_freq = np.array(GFTed_cortical_signal)[:,1:51,:]
    GFTed_cortical_signal_medium_freq = np.array(GFTed_cortical_signal)[:,51:200,:]
    GFTed_cortical_signal_high_freq = np.array(GFTed_cortical_signal)[:,200:,:]
    index = [8,56,68,74,86,132,162]
    dic_accumulated = defaultdict(dict)


    for i in range(len(index)):
        indices_pre_strong = np.hstack([
        np.arange(index[i]-1*125,index[i]-1*125+113)])
        print("the pre-indices length",np.shape(indices_pre_strong))
        print(index[i]*125-13,(index[i]+1)*125)
        indices_post_strong =  np.hstack([
        np.arange(index[i]*125-13,(index[i]+2)*125)])
        print("the post-indices length",np.shape(indices_post_strong))


    

        low_freq_pre, low_freq_post = slicing_freqs(GFTed_cortical_signal_low_freq,indices_pre_strong,indices_post_strong)
        med_freq_pre, med_freq_post = slicing_freqs(GFTed_cortical_signal_medium_freq,indices_pre_strong,indices_post_strong)
        high_freq_pre, high_freq_post = slicing_freqs(GFTed_cortical_signal_high_freq,indices_pre_strong,indices_post_strong)


        low_freq_pre_f_summed, low_freq_post_f_summed = sum_freqs(low_freq_pre,axis=1),sum_freqs(low_freq_post,axis=1)
        med_freq_pre_f_summed, med_freq_post_f_summed = sum_freqs(med_freq_pre,axis=1),sum_freqs(med_freq_post,axis=1)
        high_freq_pre_f_summed, high_freq_post_f_summed = sum_freqs(high_freq_pre,axis=1),sum_freqs(high_freq_post,axis=1)


        

        # print(np.shape(low_freq_pre_f_summed[0,:-12]))
        # a = 5  # number of rows
        # b = 5  # number of columns
        # c = 1  # initialize plot counter
        # for i in range(25):
            # plt.subplot(a,b,c)
        dic_accumulated[f'{index[i]}'] =accumulate_freqs_and_plot(low_freq_pre_f_summed,low_freq_post_f_summed,med_freq_pre_f_summed,med_freq_post_f_summed,high_freq_pre_f_summed,high_freq_post_f_summed,'Low Freqs','g',label_band,index[i])
        # accumulate_freqs_and_plot(med_freq_pre_f_summed,med_freq_post_f_summed,'Med Freqs','g')
    # accumulate_freqs_and_plot(high_freq_pre_f_summed,high_freq_post_f_summed,'High Freqs','g')
    # plt.plot(np.concatenate([baseline(med_freq_pre_f_summed[i],med_freq_pre_f_summed[i]),baseline(med_freq_pre_f_summed[i],med_freq_post_f_summed[i])])*100,label='gMedium Freq')
        # plt.plot(np.concatenate([baseline(high_freq_pre_f_summed[i],high_freq_pre_f_summed[i]),baseline(high_freq_pre_f_summed[i],high_freq_post_f_summed[i])])*100,label='gHigh Freq')
    # plt.plot(mean)
    # sem = stats_SEM(np.concatenate([baseline(low_freq_pre_f_summed,low_freq_pre_f_summed),baseline(low_freq_pre_f_summed,low_freq_post_f_summed)]))
    # # plt.fill_between(range(len(indices_post_strong)+len(indices_pre_strong)),mean+sem,mean-sem)
    # plt.plot(np.concatenate([baseline(low_freq_pre_f_summed,low_freq_pre_f_summed),baseline(low_freq_pre_f_summed,low_freq_post_f_summed)]))
   
    #     # plt.xticklabels()
    #     c+=1
    # plt.suptitle(f"Time period = around 8s; ERD for the {label_band} band")
    # fig.supxlabel('time (ms)')
    # fig.supylabel('Relative graph power (in %)')


    # fig.savefig(f'/homes/v20subra/S4B2/Graph-related_analysis/ERD/{label_band}',bbox_inches='tight')

    # print(np.shape(baseline(low_freq_pre,low_freq_pre).T))
    low_freq_pre_t_averaged_f_summed, low_freq_post_t_averaged_f_summed = averaging_time(baseline(low_freq_pre_f_summed,low_freq_pre_f_summed)),averaging_time(baseline(low_freq_pre_f_summed,low_freq_post_f_summed))
    med_freq_pre_t_averaged_f_summed, med_freq_post_t_averaged_f_summed = averaging_time(baseline(med_freq_pre_f_summed,med_freq_pre_f_summed)),averaging_time(baseline(med_freq_pre_f_summed,med_freq_post_f_summed))
    high_freq_pre_t_averaged_f_summed, high_freq_post_t_averaged_f_summed = averaging_time(baseline(high_freq_pre_f_summed,high_freq_pre_f_summed)),averaging_time(baseline(high_freq_pre_f_summed,high_freq_post_f_summed))

    
  


    # print("the size before ttest",np.shape(high_freq_pre_t_averaged_f_summed))
    # print("the size before ttest",np.shape(high_freq_post_t_averaged_f_summed))

    # print("high" ,ttest(high_freq_pre_t_averaged_f_summed, high_freq_post_t_averaged_f_summed) )
    # print ("medium", ttest(med_freq_pre_t_averaged_f_summed, med_freq_post_t_averaged_f_summed) ) 
    # print ("low:",ttest(low_freq_pre_t_averaged_f_summed, low_freq_post_t_averaged_f_summed))

    # pvalues = [ttest(low_freq_pre_t_averaged_f_summed, low_freq_post_t_averaged_f_summed),
    # ttest(med_freq_pre_t_averaged_f_summed, med_freq_post_t_averaged_f_summed),
    # ttest(high_freq_pre_t_averaged_f_summed, high_freq_post_t_averaged_f_summed) ]


   


    # mpl.rcParams['font.family'] = 'Arial'

    # plt.rc('font', family='serif')
    # plt.rc('xtick', labelsize='x-small')
    # plt.rc('ytick', labelsize='x-small')

    # fig,ax = plt.subplots(figsize=(10,10))
    # labels = ['Low', 'Med', 'High']

    # x = np.arange(len(labels))  # the label locations
    # width=0.35

    # low = stats_SEM(np.squeeze(smoothness_roughness_time_series_dict['theta']))
    # med = stats_SEM(np.squeeze(smoothness_roughness_time_series_dict['low_beta']))
    # hig = stats_SEM(np.squeeze(smoothness_roughness_time_series_dict['high_beta']))

    # error_alpha = stats_SEM(np.squeeze(smoothness_roughness_time_series_dict['alpha']))

    # pre_strong = [low_freq_pre_t_averaged_f_summed, med_freq_pre_t_averaged_f_summed, high_freq_pre_t_averaged_f_summed]
    # post_strong = [low_freq_post_t_averaged_f_summed, med_freq_post_t_averaged_f_summed, high_freq_post_t_averaged_f_summed]

    # # post_strong = [np.average(np.squeeze(smoothness_roughness_time_series_dict['theta']),axis=1)[1],
    # #             np.average(np.squeeze(smoothness_roughness_time_series_dict['alpha']),axis=1)[1],
    # #             np.average(np.squeeze(smoothness_roughness_time_series_dict['low_beta']),axis=1)[1],
    # #             np.average(np.squeeze(smoothness_roughness_time_series_dict['high_beta']),axis=1)[1]]

    # ax.violinplot(positions= x - width/2, dataset = pre_strong,widths=width,showextrema=True,showmeans=True )
    # ax.violinplot(positions=x + width/2, dataset =post_strong,widths=width,showextrema=True,showmeans=True)
    # ax.legend(['Pre','post'])
    # data = pd.DataFrame({'labels':labels,'graphPower':np.sum(pre_strong,axis=1)})

    # data2 = pd.DataFrame({'labels':labels,'graphPower':np.sum(post_strong,axis=1)})
    # data_fin = data.append(data2,ignore_index=True)
    # data_fin['cond'] = ['Pre','Pre','Pre','Post','Post','Post']
    # pvalues_slicing =[pvalues[i][1] for i in range(3)]
    # add_stat_annotation(ax,data=data_fin, y='graphPower', x ='labels', hue='cond',
    #                     box_pairs=[(("Low", "Post"), ("Low", "Pre")),
    #                     (("Med", "Post"), ("Med", "Pre")),
    #                     (("High", "Post"), ("High", "Pre"))],
    #                                 perform_stat_test=False, pvalues=pvalues_slicing,
    # line_offset_to_box=0.25, line_offset=0.1, line_height=0.05, text_format='star', loc='outside', verbose=2)

    # plt.tight_layout()
    # plt.legend()
    # plt.xticks(x,labels)
    # plt.ylabel('gPower')
    # plt.xlabel('gFrequency bands')
    # # plt.title('Theta')
    # plt.show()
    return dic_accumulated
dic = master(alpha,'Theta')

df = pd.DataFrame(columns=['gPower','gFreqs'])
to_df = defaultdict(dict)
def ttest(band1, band2):
        return scipy.stats.ttest_rel(band1,band2)
def freq_plot(which_freq,env_band):
    fig =plt.figure(figsize=(45,25))

    total = (dic['8'][which_freq] +dic['56'][which_freq] +dic['68'][which_freq] + dic['74'][which_freq] + dic['86'][which_freq] + dic['132'][which_freq] + dic['162'][which_freq])/7
    pre_total = np.mean(total[:125,:],axis=0)
    post_total = np.mean(total[125:,:],axis=0)
    print(ttest(pre_total,post_total))
    plt.plot( np.mean(total,axis=1),color='r')
    plt.fill_between(range(251+125),np.mean(total,axis=1)-stats_SEM(total),np.mean(total,axis=1)+stats_SEM(total),alpha=0.2)
    plt.axvline(125)
    # df['gPower']=df['gPower'].append( pd.Series( np.concatenate(([pre_total,post_total]))))
    # df['group']=df['group'].append(pd.Series([['pre-']*25,['post-']*25]))
    # df['Env. bands']=df['Env. bands'].append(pd.Series([[f'{env_band}']*50]))
    dic2 = defaultdict(dict)
    dic2['gPower'] = np.squeeze(np.concatenate([pre_total,post_total]).T)
    dic2['stim_group'] =np.squeeze(np.concatenate([['pre-']*25,['post-']*25]).T)
    dic2['gFreqs'] = np.squeeze(np.concatenate([[f'{env_band}']*50])).T
    return dic2

sns.set(style="whitegrid",font_scale=2)

env_bands = ['Low','Med','High']
for i in range(3):

        the_returned=freq_plot(which_freq=i,env_band=env_bands[i])
        
        df = pd.concat([pd.DataFrame(the_returned),df],ignore_index=True)
        print(df.head())


# for i in dic.keys():
#     for j in dic[i].keys():
#         print(i,j)
# 
width = 0.35
ort = "h"; pal = "blue_red_r"
f, ax = plt.subplots(figsize=(12, 15))

pt.RainCloud(hue="stim_group",y="gPower",x="gFreqs",palette = ['C0','C1'],data=df, width_viol = .7,
            ax = ax, orient = ort , alpha = .45, dodge = True)
# %%
print(df.head())
# low_freq_pre_summed, low_freq_post_summed = sum_freqs(low_freq_pre,axis=1),sum_freqs(low_freq_pre,axis=1)

# plt.plot(np.average(np.concatenate([low_freq_pre_summed,low_freq_post_summed],axis=-1).T,axis=1)[:])
# # %%
# def load_smoothness(band):
#     smoothness_roughness_time_series = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/smoothness_time_series{band}.npz')['smoothness_time_series']
#     return np.squeeze(smoothness_roughness_time_series)
# smoothness = load_smoothness('theta')

# # %%
# smoothness_concatenated = np.concatenate ([np.squeeze(slicing(smoothness.T,indices_pre_strong,axis=2)),
#  np.squeeze(slicing(smoothness.T,indices_post_strong,axis=2))],axis=-1)[:,800:1100]

# # %%

# def normalisation(data):
    
#     normalised =(data - np.min(data)/(np.max(data) - np.min(data)))
#     return normalised
# from sklearn.preprocessing import StandardScaler
# b = np.concatenate([low_freq_pre_summed,low_freq_post_summed],axis=-1)[:,800:1100]
# a = normalisation(b)#StandardScaler().fit(b).transform(b)
# print(a)
# # %%
# plt.plot(np.mean(a,axis=0))
# sem = scipy.stats.sem(a,axis=0)
# print(sem)
# plt.fill_between(range(300),np.mean(a,axis=0)-sem,np.mean(a,axis=0)+sem,color='r',alpha=0.2)
# plt.axvspan(75,200,alpha=0.2)
# # plt.axvline(6*125)
# # %%
# sc_scaled = normalisation(smoothness_concatenated)#StandardScaler().fit(smoothness_concatenated.T).transform(smoothness_concatenated.T)

# # %%
# plt.plot(np.mean(sc_scaled,axis=0))
# sem_sc_scaled = scipy.stats.sem(sc_scaled)
# plt.fill_between(range(300),np.mean(sc_scaled,axis=0)-sem_sc_scaled,np.mean(sc_scaled,axis=0)+sem_sc_scaled,color='r',alpha=0.2)# %%
# plt.axvspan(75,200,alpha=0.2)
# # plt.axvline(6*125)

# # %%


isc_results_source = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/sourceCCA_ISC_8s_window.npz')['sourceISC']
noise_SI = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/noise_floor_8s_window.npz')['isc_noise_floored']
comp=0
significance = np.array(np.where(np.max(np.array(noise_SI)[:,0,:],axis=0)<isc_results_source[0]))


# %%

plt.plot(isc_results_source[0])
plt.plot(significance,isc_results_source[comp][significance],
                marker='o', ls="",color='red',markersize=4)


# %%
significance
# %%
len(np.arange(7*125+13,7*125+113))
# %%
np.arange(0,376,62.5)
# %%
np.arange(-1000,2500,500)
# %%
