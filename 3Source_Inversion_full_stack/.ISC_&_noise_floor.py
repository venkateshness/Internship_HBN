import os
import sys

sys.path.insert(1, "/homes/v20subra/S4B2/")

from Modular_Scripts import epochs_slicing
from Modular_Scripts import CCA
from imp import reload

reload(CCA)

from timeit import default_timer

import numpy as np

video_watching_bundle_STC = np.load(
    "/users2/local/Venkatesh/Generated_Data/parcellation_corrected/video_watching_bundle_STC_parcellated.npz"
)["video_watching_bundle_STC_parcellated"]

sta = default_timer()


from joblib import Parallel, delayed
import random
from tqdm import tqdm


dic = dict()
dic["condition1"] = video_watching_bundle_STC


import multiprocessing

NB_CPU = multiprocessing.cpu_count()
# print(NB_CPU)
W, _ = CCA.train_cca(dic)


def process(i):
    np.random.seed(i)

    for subjects in range(25):
        np.random.seed(subjects)
        rng = np.random.default_rng()

        rng.shuffle(
            np.swapaxes(dic["condition1"][subjects, :, :].reshape(360, 34, 625), 0, 1)
        )

    return CCA.apply_cca(dic["condition1"], W, 125)[1]


isc_noise_floored = Parallel(n_jobs=NB_CPU - 1, max_nbytes=None)(
    delayed(process)(i) for i in tqdm(range(1000))
)
stop = default_timer()

print(np.shape(isc_noise_floored))
print(f"Whole Elapsed time: {round(stop - sta)} seconds.")
np.savez_compressed(
    "/users2/local/Venkatesh/Generated_Data/parcellation_corrected/noise_floor_8s_window",
    isc_noise_floored=isc_noise_floored,
)
