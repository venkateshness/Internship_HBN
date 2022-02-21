import numpy as np
import os

# Import
from Modular_Scripts import epochs_slicing 
from Modular_Scripts import CCA
from imp import reload 
from joblib import Parallel, delayed
import random
from tqdm.notebook import tqdm
from joblib.externals.loky import set_loky_pickler

import multiprocessing

reload(epochs_slicing)
reload(CCA)


def CCAfy(epochs,how_many_channels_for_CCA,need_noise_floor,how_many_subjects,how_many_trials_for_noise_floor):
    dic = dict()
    dic['condition1'] = epochs
    np.shape(dic['condition1'])


    [W,ISC] = CCA.train_cca(dic)

    isc_results = dict()
    for cond_key, cond_values in dic.items():
        isc_results[str(cond_key)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'], CCA.apply_cca(cond_values, W, 125)))


    if need_noise_floor:

        NB_CPU = multiprocessing.cpu_count()
        def process(i):
            np.random.seed(i)
            for subjects in range(how_many_subjects):
                rng = np.random.default_rng()
                np.random.seed(subjects)
                rng.shuffle(
                            np.swapaxes(
                                        dic['condition1'][subjects,:,:].reshape(how_many_channels_for_CCA,34,625)
                            , 0,1))
            return dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'], CCA.apply_cca(dic['condition1'], W, 125)))['ISC_persecond']

        isc_noise_floored= Parallel(n_jobs=NB_CPU-1,max_nbytes=None)(delayed(process)(i) for i in tqdm(range(how_many_trials_for_noise_floor)))
        return isc_results, isc_noise_floored
    return isc_results
