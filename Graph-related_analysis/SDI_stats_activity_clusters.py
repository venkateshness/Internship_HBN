#%%
import numpy as np
import mne
from tqdm import tqdm
from scipy import stats


empirical_activity = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_activation_sliced_for_all_subjects_all_bands.npz')['theta']
# bced_activity = np.load('/users2/local/Venkatesh/Generated_Data/activity_stats/empirical_activity_native_BCed_theta.npz')['bced_activity']
np.shape(empirical_activity)
# %%

n_subjects = 25
n_roi = 20484
n_times = 38

def first_level_stats(empirical_one_band):
    # Step 2 : Test for effect of events

    tvalues_step2 = list()
    
    for sub in tqdm(range(n_subjects)):
        sub_wise_t = list()

        for roi in range(n_roi):
            sub_wise_t_t = list()

            for time in range(n_times):

                data = empirical_one_band[sub, :, roi, time]

                t = mne.stats.ttest_1samp_no_p(data)

                sub_wise_t_t.append(t)
        
            sub_wise_t.append(sub_wise_t_t)

        tvalues_step2.append(sub_wise_t)
        
    return tvalues_step2
# %%
first_lev = first_level_stats(np.array(bced_activity)[:,:,:,25:])
first_lev = np.swapaxes(first_lev, 1,2)
#%%
first_lev = np.load('/users2/local/Venkatesh/Generated_Data/activity_stats/first_level_stats.npz')['first_lev']


ss = mne.setup_source_space('fsaverage', spacing='ico5', add_dist=None)
adj = mne.spatial_src_adjacency(ss)

n_observations = 25
pval = 0.001
df = n_observations - 1  # degrees of freedom for the test 
thresh = stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution 

t_obs, cluster_masks, pvals_clus, _ = clus =mne.stats.spatio_temporal_cluster_1samp_test(np.array(first_lev), adjacency=adj, out_type = 'mask', tail = 0, max_step=4, threshold = thresh, n_permutations=5000)

#%%

empirical_activity_space_averaged_strong_isc = np.mean(empirical_activity, axis = (0,2))[25:]

# %%
# np.savez_compressed('/users2/local/Venkatesh/Generated_Data/activity_stats/2nd_level_spatiotemporaltest.npz', clus = clus)
t_obs, cluster_masks, pvals_clus, _ = clus = np.load('/users2/local/Venkatesh/Generated_Data/activity_stats/2nd_level_spatiotemporaltest.npz',allow_pickle=True)['clus']

significant_indices = np.where(pvals_clus<0.05)[0]
# %%
cluster1 = (np.array(cluster_masks)[significant_indices])[0]
cluster2 = (np.array(cluster_masks)[significant_indices])[1]
cluster3 = (np.array(cluster_masks)[significant_indices])[2]
cluster4 = (np.array(cluster_masks)[significant_indices])[3]

# cluster = cluster1 + cluster2 + cluster3 + cluster4
cluster = np.logical_or.reduce( (cluster1, cluster2, cluster3, cluster4))

# %%
stat_significant_activity = empirical_activity_space_averaged_strong_isc * cluster

# %%

def temporally_average(activity):
    steps = [[np.arange(0,10)], [np.arange(10,20)], 
    [np.arange(20,30)], [np.arange(30,38)] ]
    averaged_activity = list()

    for i in range(len(steps)):
    
        temporal_mask = np.logical_or.reduce(cluster[steps[i]])
    
        averaged_activity.append(np.mean(activity[steps[i]], axis = 0) * temporal_mask)

    return averaged_activity

averaged_activity = temporally_average(t_obs)


# %%
np.savez_compressed('averaged_activity', averaged_activity = averaged_activity)

# np.savez_compressed('stat_significant_activity',stat_significant_activity=stat_significant_activity)
# %%
# import matplotlib.pyplot as plt
# plt.style.use('fivethirtyeight')
# fig, ax = plt.subplots()

# fig.canvas.draw()

# plt.plot(np.sum(cluster1, axis = 1))
# # plt.plot(np.sum(cluster2, axis = 1))
# plt.plot(np.sum(cluster3, axis = 1))
# # plt.plot(np.sum(cluster4, axis = 1))
# labels = [item.get_text() for item in ax.get_xticklabels()]
# labels = [0, 0, 80, 160, 240, 320]
# ax.set_xticklabels(labels)

# plt.xlabel('time (ms)')
# plt.ylabel('# significant vertices' )
# plt.title('sig vertices over time')
# %%
pvals_clus[np.where(pvals_clus<0.05)[0]]

# %%
np.where((cluster1*1)==1)[1].min()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Create some mock data

fig, ax1 = plt.subplots(figsize = (25,15))

color = 'tab:red'
ax1.set_xlabel('time (ms)')
ax1.set_ylabel('# vertices primary clu', color=color)
ax1.plot(np.sum(cluster1, axis = 1), color='#880808')
ax1.plot(np.sum(cluster3, axis = 1), color='#AA4A44')

plt.style.use("fivethirtyeight")

ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('# vertices secondary', color='#0000FF')  # we already handled the x-label with ax1
ax2.plot(np.sum(cluster2, axis = 1), color="#0000FF")
ax2.plot(np.sum(cluster4, axis = 1), color='#7393B3')

ax2.tick_params(axis='y', labelcolor=color)
labels = [item.get_text() for item in ax1.get_xticklabels()]
labels = [0, 0, 40, 80, 120, 160, 200, 240, 280, 320]
ax2.set_xticklabels(labels)
fig.suptitle('sig vertices over time')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.savefig('/homes/v20subra/S4B2/Results_paper/2nd_figure/evolution_cluster.svg')
plt.show()
# %%
