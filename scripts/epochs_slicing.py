#Slicing only the epochs wanted
import mne
import pathlib



def epochs(subject_raw,subject_events, epochs_list, tmin, tmax, fs, epochs_to_slice):

    epochs = mne.Epochs(subject_raw, subject_events, epochs_list, tmin=tmin, tmax=tmax,preload=True,baseline=(0,None))
    epochs_resampled = epochs.resample(fs) # Downsampling
    
    return epochs_resampled[epochs_to_slice]
