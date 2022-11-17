#%%
path_Glasser = "/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz"

from nilearn.regions import signals_to_img_labels

# load nilearn label masker for inverse transform
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image, plotting
from nilearn import datasets
from os.path import join as opj
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np

mnitemp = fetch_icbm152_2009()
mask_mni = image.load_img(mnitemp["mask"])
glasser_atlas = image.load_img(path_Glasser)
zero_array = np.zeros(
    360,
)
zero_array[[24]] = 1
signal = np.expand_dims(zero_array, axis=0)  # add dimension 1 to signal array

U0_brain = signals_to_img_labels(signal, path_Glasser, mnitemp["mask"])
plotting.plot_glass_brain(U0_brain)

#%%
zero_array = np.zeros(
    360,
)
# %%
