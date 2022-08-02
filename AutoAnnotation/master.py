#%%
from cProfile import label
from email.mime import audio
from re import A
from cv2 import kmeans, threshold
import numpy as np
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import pandas as pd
framewise_probs = np.load("/homes/v20subra/S4B2/AutoAnnotation/framewise_probs.npz")["framewise_probs"]

isc_result = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/sourceCCA_ISC_8s_window.npz')['sourceISC']#CCA on the eLORETA signal
noise_floor_source = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/noise_floor_8s_window.npz')['isc_noise_floored']#Noise floor

n_comps = 3

significance_sampled = list()
significance = np.where(np.max(np.array(noise_floor_source)[:,:3,:],axis=0)<isc_result[:3])[1]
unique_samples = np.unique(significance)
significance_sampled.append(1)
significance_sampled.append(unique_samples[np.where(np.diff(unique_samples)>1)[0]+1])


_500_ms_in_samples = 50
sample_sorted = np.unique(np.array(sorted( np.hstack(significance_sampled) )))
#######################################################################################################################################
# 1. Sound Event Detection
def pre_stimulus_average(array_to_average_on, timeframe, times, ):
    """Temporal average of the predicted probability for the pre-stimulus period
    Args:
        array_to_average_on (array): Predicted prob/RMS of volume change for the entire video frames. Dim = Frames x n_classes
        timeframe (int): Onset point. Range = -500 ms to 0ms
    
    Returns:
        Averaged probablities. Dim = 1 x n_classes
    """
    return np.average(array_to_average_on[ (timeframe * times - _500_ms_in_samples) : (timeframe * times) ], axis = 0)

def post_stimulus_average(array_to_average_on, timeframe, times):
    """Temporal average of the predicted probability for the post-stimulus period
    Args:
        array_to_average_on (array): Predicted prob/RMS volume change for the entire video frames. Dim = Frames x n_classes
        timeframe (int): Onset point.  Range = 0ms to +500ms
    Returns:
        Averaged probs. Dim =  1 x n_classes
    """
    return np.average(array_to_average_on[ (timeframe * times) : (timeframe * times + _500_ms_in_samples) ], axis=0)

def entire_stimulus_average(array_to_average_on, timeframe, times):
    """Temporal average of the predicted probability for the entire event duration: -500ms to +500 ms
    Args:
        array_to_average_on (array): Predicted prob/RMS volume change for the entire video frames. Dim = Frames x n_classes
        timeframe (int): Onset point.  Range = -500ms to +500ms
    Returns:
        Averaged probs. Dim =  1 x n_classes
    """
    return np.average(array_to_average_on[ (timeframe * times - _500_ms_in_samples)  : ( timeframe * times + _500_ms_in_samples) ],axis = 0)

audiotagging_entire = list()
audiotagging_pre = list()
audiotagging_post = list()


ix_to_lb = {label for i, label in enumerate(labels)}

def df_indexing(label, averaged_probs):
    """Zip the averaged predictions with the multi-level columns for a creation of the dataframe
    Args:
        label (string): label for the outer-level column
        array (_type_): _description_
    Returns:
        DataFrame with multi-level column 
    """
    mux = pd.MultiIndex.from_product([[label], sorted(ix_to_lb) ])
    df  = pd.DataFrame(averaged_probs, columns = mux )
    return df

for i in sample_sorted:
    audiotagging_entire.append(entire_stimulus_average(framewise_probs, i+1, 100))
    df_entire = df_indexing('Entire_stimulus', audiotagging_entire)

    audiotagging_pre.append(pre_stimulus_average(framewise_probs, i+1, 100))
    df_pre = df_indexing('Pre_stimulus', audiotagging_pre)

    audiotagging_post.append(post_stimulus_average(framewise_probs, i+1, 100))
    df_post = df_indexing('Post_stimulus', audiotagging_post)

# Concatenating all three stimulus periods
df_sed = pd.concat([df_pre, df_post, df_entire], axis=1)

##################################################################################################################################
#%%
# 2. Scene Detection
from scenedetect import detect, ContentDetector
scene_list = detect('/homes/v20subra/S4B2/3Source_Inversion_full_stack/Videos/DM2_video.mp4', ContentDetector())
frame_timestamp = list()
for i, scene in enumerate(scene_list):
    frame_timestamp.append([scene[0].get_frames(), scene[1].get_frames()])


unique_frame_timestamp = np.unique(frame_timestamp)

scene_detection_pre = list()
scene_detection_post = list()
scene_detection_entire = list()
offset = list()

frames_in_samples_pre = 12 #25 fps; 12 for 500 ms
frames_in_samples_post = 13 #25 fps; 13 for 500 ms
_1s_in_frames = 25
for i in sample_sorted * _1s_in_frames:

    scene_detection_entire.append(len(set(np.arange(i - frames_in_samples_pre, i + frames_in_samples_post)).intersection(unique_frame_timestamp)))
    scene_detection_pre.append(len(set(np.arange(i - frames_in_samples_pre, i)).intersection(unique_frame_timestamp)))
    scene_detection_post.append(len(set(np.arange(i, i + frames_in_samples_post)).intersection(unique_frame_timestamp)))
    
    if len(set(np.arange(i-frames_in_samples_pre, i + frames_in_samples_post)).intersection(unique_frame_timestamp))>0:
        relative_scence_change_from_onset = unique_frame_timestamp - i
        print(np.abs(relative_scence_change_from_onset))
        offset.append(relative_scence_change_from_onset[np.abs(relative_scence_change_from_onset).argmin()])
        # print(offset)
    else:
        offset.append(0)

df_scene_change = pd.DataFrame({ "Offset(in frames)":offset})
# pd.concat([df_sed, df_scene_change],axis=1)
#%%
##################################################################################################################################
# 3. RMS volume change


import librosa
import numpy as np
import matplotlib.pyplot as plt
samples, sample_rate = librosa.load('/homes/v20subra/S4B2/Despicable Me-HQ.wav',sr=None)
samples_normed = (samples - np.average(samples))/np.std(samples)
rms = librosa.feature.rms(y=samples_normed,hop_length=384,frame_length=1000)

fs_of_EEG = 125
_500_ms_in_samples = 62

rms_pre_stim = list()
rms_post_stim = list()
rms_entire_event = list()

for i in sample_sorted:
    rms_pre_stim.append(pre_stimulus_average(rms.T, i, fs_of_EEG))
    rms_post_stim.append(post_stimulus_average(rms.T, i, fs_of_EEG))
    rms_entire_event.append(entire_stimulus_average(rms.T, i, fs_of_EEG))

df_rms = pd.DataFrame({'RMS_pre-stim':np.hstack(rms_pre_stim),'RMS_post_stim':np.hstack(rms_post_stim),'RMS_entire_event':np.hstack(rms_entire_event)})

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

df_rms_scaled = pd.DataFrame(scale.fit_transform(df_rms))
df_rms_scaled.columns = ['RMS_pre-stim','RMS_post_stim','RMS_entire_event']
the_df = pd.concat([df_sed, df_scene_change, df_rms_scaled], axis = 1)
the_df_raw_rms = pd.concat([df_sed, df_scene_change, df_rms], axis = 1)

from sklearn.feature_selection import VarianceThreshold
def variance_thresholding(dataset):
    threshold = 0.03
    VT = VarianceThreshold(threshold=threshold)
    the_df_transformed = pd.DataFrame(VT.fit_transform(dataset))
    the_df_transformed.columns = dataset.columns[VT.variances_>threshold]
    return the_df_transformed



import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from nilearn.plotting import plot_matrix

the_df_transformed = variance_thresholding(the_df)

# plot_matrix(cosine_similarity(the_df_transformed), labels = range(38), reorder=True)
# plt.title('Cosine-similarity for the annotation matrix')
# plt.xlabel('Events')
# plt.ylabel('Events')
# plt.show()


# from sklearn.metrics import pairwise_distances
# cosine_distance = np.exp(- 1./(2 * 1) * pairwise_distances(the_df, metric='cosine'))
# cosine_distance
# from sklearn.cluster import AffinityPropagation

# af = AffinityPropagation(random_state=4,affinity='precomputed').fit(A)
# af.labels_



from sklearn.decomposition import PCA
import seaborn as sns
sns.set_theme()

def pca(dataset):
    pca = PCA(n_components=2)
    plot= pca.fit_transform(dataset)
    # print(np.shape(plot))
    # for_interpretation = ['Scene Change (pre-stim)','RMS group1', 'Scene Change(Post-stim)','RMS G2', 'RMS G3']
    
    # plt.scatter(plot[i,0], plot[i,1])

    # plt.title('PCA-fyed on the averaged events given clusters')
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.show()
    return plot, pca
# %%


from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm
from scipy.spatial import ConvexHull


range_n_clusters = [3]


def clustering(which_cluster,cluster_labels=None, range_n_clusters = range_n_clusters ):

    if which_cluster=='sub_cluster':
        # VT = VarianceThreshold(threshold=threshold)
        the_df_transformed_single_cluster = variance_thresholding(the_df_transformed.iloc[np.where(cluster_labels==0)])
        # the_df_transformed_single_cluster = the_df_transformed_single_cluster.drop(['Offset(in frames)'],axis=1)
        # the_df_transformed_single_cluster = pd.DataFrame(VT.fit_transform(the_df_transformed.iloc[np.where(cluster_labels==1)]))
        # the_df_transformed_single_cluster.columns = the_df_transformed.columns[VT.variances_>threshold]
        # print(the_df_transformed_single_cluster)


        dataset = the_df_transformed_single_cluster
        plot  = pca(dataset)[0]
        
    else:
        dataset = the_df_transformed
        plot = pca(dataset)[0]
        

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(dataset) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(dataset)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(dataset, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(dataset, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(plot[:,0], plot[:,1], s=30, lw=0,  c=colors, edgecolor="k")
        # ax2.scatter(plot[32,0], plot[32,1], s=30, lw=0,  edgecolor="r")

        # plt.title('PCA-fyed on the averaged events given clusters')
        # plt.xlabel('PC1')
        # plt.ylabel('PC2')
        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers


        z = plot[:,0]
        y = plot[:,1]
        for i, txt in enumerate(range(len(cluster_labels))):
            ax2.annotate(txt, (z[i], y[i]),c='r')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            f"Silhouette analysis for KMeans clustering on similarity matrix with n_clusters = %d; sil score = {silhouette_avg}"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    # plt.legend()
    plt.show()
    if which_cluster == 'sub_cluster':
        return cluster_labels, centers, the_df_transformed_single_cluster
    else:
        return cluster_labels, centers


cluster_labels_full, center_full = clustering(which_cluster='',range_n_clusters=range_n_clusters)
centroids_for_full = pd.DataFrame(center_full)
centroids_for_full.columns = the_df_transformed.columns
for i in range(3):
    centroids_for_full.iloc[i,-3:] =np.mean(the_df_raw_rms.iloc[np.where(cluster_labels_full==i)[0], -3: ],axis=0).values


range_n_clusters = [3]
cluster_labels_sub, center_sub, the_df_transformed_single_cluster = clustering(which_cluster='sub_cluster',cluster_labels=cluster_labels_full,range_n_clusters=range_n_clusters)


centroids_for_sub = pd.DataFrame(center_sub)
centroids_for_sub.columns = the_df_transformed_single_cluster.columns

for i in range(3):
    centroids_for_sub.iloc[i,-3:] =np.mean(the_df_raw_rms.iloc[np.where(cluster_labels_sub==i)[0], -3: ],axis=0).values

# %%
dic_of_groups = {}
for i in range(3):
    indices_full = sample_sorted[np.where(cluster_labels_full==i)[0]]
    if i==0:
        for j in range(3):
            indices_sub = indices_full[np.where(cluster_labels_sub==j)]
            dic_of_groups[str(j)]= indices_sub
    else:
        dic_of_groups[str(i+2)]= indices_full
dic_of_groups
# %%

np.savez(file='dict_of_clustered_events',**dic_of_groups)

# %%
the_df_transformed.iloc[np.where(cluster_labels_full == 2)]
# %%
