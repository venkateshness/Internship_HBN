#%%
import numpy as np
import pandas as pd

a = pd.read_csv(
    "/homes/v20subra/S4B2/Eyetracking/NDARAA075AMK_Video-DM_Events.txt", delimiter="\n"
)


# %%
a
# %%

from pliers.stimuli import VideoStim
from pliers.converters import VideoFrameIterator
from pliers.filters import FrameSamplingFilter
from pliers.extractors import GoogleVisionAPIFaceExtractor, merge_results

video = VideoStim(
    "/homes/v20subra/S4B2/3Source_Inversion_full_stack/Videos/DM2_video.mp4"
)

# Convert the VideoStim to a list of ImageStims
conv = VideoFrameIterator()
frames = conv.transform(video)

# Sample 2 frames per second

# Detect faces in all frames
ext = GoogleVisionAPIFaceExtractor()
face_features = ext.transform(frames)

# Merge results from all frames into one big pandas DataFrame
data = merge_results(face_features)
# %%
len(frames)

# %%
4268 / 25
# %%
