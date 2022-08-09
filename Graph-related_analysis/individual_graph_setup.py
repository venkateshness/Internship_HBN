#%%
path_Glasser = '/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz'

from nilearn.regions import signals_to_img_labels  
# load nilearn label masker for inverse transform
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image, plotting
from nilearn import datasets
from os.path import join as opj
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os
import numpy as np
from nilearn.connectome import ConnectivityMeasure


subjects = ['NDARAD481FXF','NDARBK669XJQ',
'NDARCD401HGZ','NDARDX770PJK',
'NDAREC182WW2','NDARGY054ENV',
'NDARHP176DPE','NDARLB017MBJ',
'NDARMR242UKQ','NDARNT042GRA',
'NDARRA733VWX','NDARRD720XZK',
'NDARTR840XP1','NDARUJ646APQ',
'NDARVN646NZP','NDARWJ087HKJ',
'NDARXB704HFD','NDARXJ468UGL',
'NDARXJ696AMX','NDARXU679ZE8',
'NDARXY337ZH9','NDARYM257RR6',
'NDARYY218AGA','NDARYZ408VWW','NDARZB377WZJ']

subjects_data_available_for = list()
for i in range(1,25):
     if (os.path.isfile(f'/users2/local/Venkatesh/HBN/CPAC_preprocessed/sub-{subjects[i]}_ses-1/functional_to_standard/_scan_rest/_selector_CSF-2mmE-M_aC-CSF+WM-2mm-DPC5_M-SDB_P-2_BP-B0.01-T0.1_C-S-1+2-FD-J0.5/bandpassed_demeaned_filtered_antswarp.nii.gz')):
         subjects_data_available_for.append(subjects[i])
# %%
mnitemp = fetch_icbm152_2009()
mask_mni=image.load_img(mnitemp['mask'])
glasser_atlas=image.load_img(path_Glasser)


# %%
def parcellation(mask,img):
    # Glasser is a reference map for regions in the brain. It splits the brain into 360 regions.
    glassermasker = NiftiLabelsMasker(labels_img=path_Glasser,mask_img=mask,standardize=True)
    parcellated = glassermasker.fit_transform(img)
    return parcellated

def connectivitymeasure(which_subject_after_parcellation):

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([which_subject_after_parcellation])[0]# 25 indiv 
    np.fill_diagonal(correlation_matrix, 0)

    return correlation_matrix
#%%


img = f'/users2/local/Venkatesh/HBN/CPAC_preprocessed/sub-NDARBK669XJQ_ses-1/functional_to_standard/_scan_rest/_selector_CSF-2mmE-M_aC-CSF+WM-2mm-DPC5_M-SDB_P-2_BP-B0.01-T0.1_C-S-1+2-FD-J0.5/bandpassed_demeaned_filtered_antswarp.nii.gz'
mask = f'/users2/local/Venkatesh/HBN/CPAC_preprocessed/sub-NDARBK669XJQ_ses-1/functional_brain_mask_to_standard/_scan_rest/sub-NDARBK669XJQ_task-rest_bold_calc_resample_volreg_mask_antswarp.nii.gz'
# print('running parcellation')
parcellated = parcellation(mask,img)

# print('running correlation_matrix')
correlation_matrix = connectivitymeasure(parcellated)


# %%

