import numpy as np
high = np.load('/users/local/Venkatesh/Generated_Data/noise_baseline_properly-done_eloreta/SI_full.npz')['high_isc_averaged']

# Import
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/homes/v20subra/S4B2/Modular_Scripts/')
import CCA
import plot_matplotlib
import source_inversion, surface_plot, fwd_model
from imp import reload 


reload(CCA)
reload(source_inversion)
reload(surface_plot)
reload(fwd_model)


dic = dict()
dic['condition1'] = np.array(high)
np.shape(dic['condition1'])


[W,ISC] = CCA.train_cca(dic)

isc_results = dict()
for cond_key, cond_values in dic.items():
    isc_results[str(cond_key)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'], CCA.apply_cca(cond_values, W, 125)))

import random
from tqdm.notebook import tqdm

def shuffle(a):
    
    for i in (range(10)):
        for j in range(360):
            np.random.seed(i)
            
            chunked = chunks(a['condition1'][i][j][:21250]) # numpy array
            np.random.shuffle(chunked[0])
            chunked = np.reshape(chunked,[21250,])
            
    return a

def chunks(chunk):
    chunked = chunk[:21250]
    chunked= chunked.reshape(1,34,625) #5s chunk
    return chunked


from joblib import Parallel, delayed

import multiprocessing
NB_CPU = multiprocessing.cpu_count()

valstest = []
def process(i):

    shuffled = shuffle(dic)
    isc_resultstest_ = dict()
    for cond_key, cond_values in shuffled.items():
        isc_resultstest_[str(cond_key)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'], CCA.apply_cca(cond_values, W, 125)))
        valstest.append(isc_resultstest_['condition1']['ISC_persecond'])    
    return valstest

v = Parallel(n_jobs=NB_CPU-1)([delayed(process)(i) for i in tqdm(range(5))])

