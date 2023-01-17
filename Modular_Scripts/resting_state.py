# %%
import fwd_model
import source_inversion
import inverse_epochs
import epochs_slicing
import matplotlib.pyplot as plt
import mne
import raw_time_domain
from imp import reload  # force the mouse pointer to take the changes into account
import numpy as np

#'exec(%matplotlib qt)'
#'exec(%gui qt)'
# %%
"""importing the already-exported data"""
resting_state_sub2_raw = mne.io.read_raw(
    "/homes/v20subra/S4B2/Generated_Data/importing/resting_state_sub2_raw.fif"
)
resting_state_sub2_events = np.load(
    "/homes/v20subra/S4B2/Generated_Data/importing/resting_state_sub2_events.npz"
)["sub2"]

# %%

"""EEG Montage plot"""
# montage_plot = mne.channels.make_standard_montage(kind= "GSN-HydroCel-129") # This gives the standard montage plot
resting_state_sub2_raw.plot_sensors(show_names=True)
plt.show()
# %%

"""8s of Raw data time-series after excluding the noisy channels (based on Nentwich paper)"""

reload(raw_time_domain)
raw_time_domain.time_domain_plot(resting_state_sub2_raw).show()

# %%

"""Epoch(ing) raw data"""
epochs = mne.Epochs(
    resting_state_sub2_raw,
    resting_state_sub2_events,
    [20, 30, 90],
    tmin=0,
    tmax=20,
    preload=True,
    baseline=(0, None),
)
epochs_resampled = epochs.resample(250)  # Downsampling to 250Hz
print(np.shape(epochs_resampled.load_data()))  # Sanity Check

# %%
"""Topographical plot"""
layout = mne.find_layout(epochs.info)
epochs.average().plot_topo(layout=layout)


# %%
"""Plotting events"""
# The last event 20 falls on 174120 and the last sample is 176386. That's just 4 seconds before the end of the EEG
mne.viz.plot_events(
    resting_state_sub2_events[:-1], sfreq=resting_state_sub2_raw.info["sfreq"]
)

# %%

"""Raw PSD"""
mne.viz.plot_raw_psd(
    resting_state_sub2_raw, tmax=40, fmax=40, picks=["E22", "E20", "E23"]
)
# %%

"""Topomap PSD"""
# epochs
mne.viz.plot_epochs_psd_topomap(epochs["20"])  # Eyes open
# %%

"""PSD of the Occipital electrodes"""
for i in range(0, 5):  # 0 is 90 = beginning of EEG, so skipped
    # plt.title('event ={}. Note: open = 20'.format(events[i][2])) this works nicely if plotted through qt
    print(
        "\t\t\tEVENT IN THE GRAPH BELOW IS = {}".format(resting_state_sub2_events[i][2])
    )
    print("\t\t\tNOTE, EYES OPEN = 20 & EYES CLOSE = 30")
    print("\t\t\tOnset at {}s".format(resting_state_sub2_events[i][0] / 500))

    # Electrodes in the Occipital lobe are taken
    mne.viz.plot_raw_psd(
        resting_state_sub2_raw,
        tmin=resting_state_sub2_events[i][0] / 500,
        tmax=resting_state_sub2_events[i + 1][0] / 500,
        fmax=40,
        picks=["E70", "E75", "E83"],
    )

# %%

"""Source reconstruction"""


"""Noise Covariance"""
epochs_for_noice_covariance = epochs_slicing.epochs(
    resting_state_sub2_raw,
    resting_state_sub2_events,
    [20, 30, 90],
    tmin=0,
    tmax=0.95,
    fs=500,
    epochs_to_slice="90",
)
noise_covariance = mne.compute_covariance(epochs_for_noice_covariance[0], method="auto")

# fig_cov, fig_spectra = mne.viz.plot_cov(noise_covariance, resting_state_sub2_raw.info)

"""Forward Model & Inverse Operator set up"""

reload(source_inversion)
forward_model = fwd_model.fwd(resting_state_sub2_raw)
resting_state_cov, resting_state_inverse_operator = source_inversion.inversion(
    raw=resting_state_sub2_raw, fwd_model=forward_model, cov=noise_covariance
)


# %%

"""Inversion begins"""
reload(inverse_epochs)
stc = inverse_epochs.apply(
    data=epochs_resampled["20"],
    inverse_operator=resting_state_inverse_operator,
    raw=False,
)
# %%
