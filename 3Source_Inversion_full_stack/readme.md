* The `$eLORETA.py` is a full-stack source inversion, which does the following:
    * Noise Covariance Baseline using subject-specific Resting State eyes-open data
    * Forward Modeling (using Freesurfer & BEM)
    * Inverse Modeling (using eLORETA indeed)
    * Application of Glasser Parcellation on the fsaverage native space
    * Reliability check of the source inversion (look for the code for more details)
    * Bandpass-filtering of cortical activations estimated using eLORETA
    * Hilbert transform on the bands

* The `.ISC_&_noise-floor.py` does the following:
    * Inter-Subject Correlation of cortical activity using estimated using Correlated Components Analysis (CCA) which was introduced by JP. Dmochowski et al. 2012
    * Stats -- Null distribution