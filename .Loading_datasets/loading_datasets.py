import mne
import numpy as np
import pandas as pd
import os


def csv_to_raw_mne(
    path_to_file,
    path_to_montage_ses,
    fs,
    path_to_events,
    filename,
    state,
    montage="GSN-HydroCel-129",
):

    """Load csv files of data, chan locations and events and return a raw mne instance"""
    data = np.loadtxt(path_to_file, delimiter=",")
    chans = pd.read_csv(path_to_montage_ses, sep=",", header=None)
    ch_list = [
        "E1",
        "E8",
        "E14",
        "E17",
        "E21",
        "E25",
        "E32",
        "E38",
        "E43",
        "E44",
        "E48",
        "E49",
        "E56",
        "E57",
        "E63",
        "E64",
        "E69",
        "E73",
        "E74",
        "E81",
        "E82",
        "E88",
        "E89",
        "E94",
        "E95",
        "E99",
        "E100",
        "E107",
        "E113",
        "E114",
        "E119",
        "E120",
        "E121",
        "E125",
        "E126",
        "E127",
        "E128",
    ]
    ch_names = list(chans.values[1:, 0])

    if state == "Rest":
        ch_names_appended = list(np.append(ch_names, "stim_channel"))
        types = ["eeg"] * (len(ch_names_appended) - 1)
        types.append("stim")
        data2 = np.zeros([1, len(data[0])])  # len(raw.times)
        data_appended = np.append(data, data2, axis=0)
        info = mne.create_info(ch_names_appended, sfreq=fs, ch_types=types)
        raw = mne.io.RawArray(data_appended, info)

    else:
        types = ["eeg"] * (len(ch_names))
        info = mne.create_info(ch_names, sfreq=fs, ch_types=types)
        raw = mne.io.RawArray(data, info)

    # set standard montage
    if montage:
        raw.set_montage(montage)
        raw.set_eeg_reference("average", projection=True)
        raw.apply_proj()

    if path_to_events:
        # parse events file
        raw_events = pd.read_csv(
            path_to_events, sep=r"\s*,\s*", header=None, engine="python"
        )
        values = raw_events[0].to_list()

        print(filename)
        if filename == "NDARDX770PJK":
            values.extend(["break cnt"])

        idx = [i for i, e in enumerate(values) if e == "break cnt"]
        if state == "Rest":
            # idx = [i for i, e in enumerate(values) if e == 'break cnt']
            samples = raw_events[1][idx[0] + 1 : idx[1]].to_numpy(dtype=int)
            event_values = raw_events[0][idx[0] + 1 : idx[1]].to_numpy(dtype=int)

        else:
            samples = raw_events[1][1 : idx[0]].to_numpy(dtype=int)
            event_values = raw_events[0][1 : idx[0]].to_numpy(dtype=int)

        events = np.zeros((len(samples), 3))

        events = events.astype("int")
        events[:, 0] = samples
        events[:, 2] = event_values

        # Appending one row of 'ones'. Will be easier to stop parsing once we hit 1
        events_final = np.append(events, np.ones((1, 3)), axis=0).astype("int")
        raw = exclude_channels_from_raw(raw, ch_list)

    return raw, events_final


def exclude_channels_from_raw(raw, ch_to_exclude):
    """Return a raw structure where ch_to_exclude are removed"""
    idx_keep = mne.pick_channels(
        raw.ch_names, include=raw.ch_names, exclude=ch_to_exclude
    )
    raw.pick_channels([raw.ch_names[pick] for pick in idx_keep])
    return raw


def preparation(filename, state):
    path_to_file = "/users/local/Venkatesh/HBN/%s/Video3_data.csv" % filename
    path_to_events = "/users/local/Venkatesh/HBN/%s/Video3_event.csv" % filename
    path_to_montage_glob = "/users/local/Venkatesh/HBN/GSN_HydroCel_129_hbn.sfp"
    path_to_montage_ses = "/users/local/Venkatesh/HBN/%s/Video3_chanlocs.csv" % filename
    fs = 500
    chans_glob = mne.channels.read_custom_montage(
        fname="/users/local/Venkatesh/HBN/GSN_HydroCel_129_hbn.sfp"
    )  # read_montage is deprecated
    # channels to exclude because noisy (Nentwich paper)

    raw, events = csv_to_raw_mne(
        path_to_file,
        path_to_montage_ses,
        fs,
        path_to_events,
        state=state,
        filename=filename,
        montage="GSN-HydroCel-129",
    )
    # raw.add_events(events, stim_channel = 'stim_channel',replace = False)
    return raw, events


def preparation_resting_state(filename, state):
    path_to_file = "/users/local/Venkatesh/HBN/%s/RestingState_data.csv" % filename
    path_to_events = "/users/local/Venkatesh/HBN/%s/RestingState_event.csv" % filename
    path_to_montage_glob = "/users/local/Venkatesh/HBN/GSN_HydroCel_129_hbn.sfp"
    path_to_montage_ses = (
        "/users/local/Venkatesh/HBN/%s/RestingState_chanlocs.csv" % filename
    )
    fs = 500
    chans_glob = mne.channels.read_custom_montage(
        fname="/users/local/Venkatesh/HBN/GSN_HydroCel_129_hbn.sfp"
    )  # read_montage is deprecated

    raw, events = csv_to_raw_mne(
        path_to_file,
        path_to_montage_ses,
        fs,
        path_to_events,
        filename=filename,
        state=state,
        montage="GSN-HydroCel-129",
    )
    # raw.add_events(events, stim_channel = 'stim_channel',replace = False)
    return raw, events


"""Resting state"""
subject_list = [
    "NDARAD481FXF",
    "NDARAV945MCQ",
    "NDARBK669XJQ",
    "NDARCD401HGZ",
    "NDARDX770PJK",
    "NDAREC182WW2",
    "NDAREZ098ZPE",
    "NDARFB107PVH",
    "NDARGY054ENV",
    "NDARHF023VG3",
    "NDARHP176DPE",
    "NDARJP133YL3",
    "NDARKH741PL8",
    "NDARKW999WZD",
    "NDARLB017MBJ",
    "NDARMA875ARE",
    "NDARMR242UKQ",
    "NDARNE511XHU",
    "NDARNT042GRA",
    "NDARPE596LZL",
    "NDARPR768KT4",
    "NDARRA733VWX",
    "NDARRD720XZK",
    "NDARRN619WHY",
    "NDARTR840XP1",
    "NDARUJ646APQ",
    "NDARVN646NZP",
    "NDARWJ087HKJ",
    "NDARWV470ATB",
    "NDARXB704HFD",
    "NDARXJ468UGL",
    "NDARXJ696AMX",
    "NDARXU679ZE8",
    "NDARXY337ZH9",
    "NDARYM257RR6",
    "NDARYX530MZU",
    "NDARYY218AGA",
    "NDARYZ408VWW",
    "NDARZB377WZJ",
    "NDARZF288FB7",
    "NDARZJ414CAA",
    "NDARZT772PU4",
]


rs_data_present = list()
video_data_present = list()
##################################
# dataset files check#############@


for i in range(1, len(subject_list) + 1):
    path_to_file = (
        "/users/local/Venkatesh/HBN/%s/RestingState_data.csv" % subject_list[i - 1]
    )
    path_to_file_video = (
        "/users/local/Venkatesh/HBN/%s/Video3_event.csv" % subject_list[i - 1]
    )

    if os.path.isfile(path_to_file):

        rs_data_present.append(subject_list[i - 1])
    if os.path.isfile(path_to_file_video):
        video_data_present.append(subject_list[i - 1])
    print(len(rs_data_present))
    print(len(video_data_present))

print(len(np.logical_and(set(rs_data_present), set(video_data_present))))


for i in range(1, len(rs_data_present) + 1):
    if not os.path.exists(
        f"/users/local/Venkatesh/Generated_Data/importing/resting_state/{rs_data_present[i-1]}"
    ):
        os.makedirs(
            f"/users/local/Venkatesh/Generated_Data/importing/resting_state/{rs_data_present[i-1]}"
        )

    resting_state_raw, resting_state_events = preparation_resting_state(
        rs_data_present[i - 1], "Rest"
    )
    resting_state_raw.save(
        f"/users/local/Venkatesh/Generated_Data/importing/resting_state/{rs_data_present[i-1]}/raw.fif",
        overwrite=True,
    )
    np.savez_compressed(
        f"/users/local/Venkatesh/Generated_Data/importing/resting_state/{rs_data_present[i-1]}/events.npz",
        resting_state_events=resting_state_events,
    )


"""Video-watching"""

for i in range(1, len(video_data_present) + 1):
    if not os.path.exists(
        f"/users/local/Venkatesh/Generated_Data/importing/video-watching/{video_data_present[i-1]}"
    ):
        os.makedirs(
            f"/users/local/Venkatesh/Generated_Data/importing/video-watching/{video_data_present[i-1]}"
        )

    sub_raw, sub_events = preparation(video_data_present[i - 1], "others")
    sub_raw.save(
        f"/users/local/Venkatesh/Generated_Data/importing/video-watching/{video_data_present[i-1]}/raw.fif",
        overwrite=True,
    )
    np.savez_compressed(
        f"/users/local/Venkatesh/Generated_Data/importing/video-watching/{video_data_present[i-1]}/events.npz",
        video_watching_events=sub_events,
    )
