# This project has been carried out as part of the final-year internship of the MSc program

# Commonalities:
  * Dataset: 
    * Source: http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/
    * Characteristics: 
        * Number of Electrodes (no. of channels) : 128
        * Sampling Frequency: 500hz
        * Duration: ~352.77 seconds for the resting state and 170 for the video-watching
        * Events (States during the EEG): 90, 20, 30 which represents start of the recording, eyes-open and eyes-closed respectively
        * Subjects: NDARBF805EHN, NDARBG574KF4 (from https://partage.imt.fr/index.php/s/97wDFxzLQ5NgkNJ *not* on HBN link)
  * Resource (Project-related): https://docs.google.com/document/d/1YkX0Mfeq030oCQQDL4xb87XxXjqXyLqncg1FA7nANp0
  * A UE (Useful resources): https://github.com/brain-bzh/health_exg
  * Tool: https://mne.tools/stable/index.html, among other typical data-manipulating packages
  

# Specifics to the project as part-time:
  * Source Signal Reconstruction (Activation region's)
  * Inter-subject Study
  
# Details related to the project as full-time (internship)
 * CCA on the Scalp signals
 * Analysis of the CCA components on Source Space
 * Analysis of High-ISC & Low-ISC period data on Source Space
 * Usage of Structural Connectivity graph, after employing Glasser Parcellation for the Graph-related analysis of source space signals
  

