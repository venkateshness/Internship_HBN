#!/usr/bin/env python
# coding: utf-8


import numpy as np

high_isc = np.load('high_isc_average.npz')
plot = list()
for m in range(360):
    hm = np.triu(np.corrcoef(np.array(high_isc['high_isc'])[:,m,:]))

    iscs = list()
    for i in range(9):
        iscs.append(np.sum(hm[i][i+1:]))
    plot.append(sum(iscs)/ sum(np.arange(10)))



#np.savez('low_isc_averaged',low_isc = low_isc)
import matplotlib.pyplot as plt
from IPython import get_ipython


from nilearn.regions import signals_to_img_labels  
# load nilearn label masker for inverse transform
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image, plotting
from nilearn import datasets
from os.path import join as opj
import matplotlib.pyplot as plt

path_Glasser = 'S4B2/GSP/Glasser_masker.nii.gz'


mnitemp = fetch_icbm152_2009()
mask_mni=image.load_img(mnitemp['mask'])
glasser_atlas=image.load_img(path_Glasser)


#print(NiftiMasker.__doc__)

fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(10,10))

signal=[]
U0_brain=[]
signal=np.expand_dims(np.array(plot), axis=0) # add dimension 1 to signal array
U0_brain = signals_to_img_labels(signal,path_Glasser,mnitemp['mask'])
plotting.plot_glass_brain(U0_brain,title='high ISC',colorbar=True,plot_abs=False,cmap='spring',display_mode='xz',figure=fig,axes=ax)
