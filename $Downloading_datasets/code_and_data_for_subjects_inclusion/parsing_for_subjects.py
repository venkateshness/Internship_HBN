import scipy
from scipy import io
import numpy as np
import pandas as pd
import ezodf
import os
import csv

os.chdir("/homes/v20subra/S4B2/$Downloading_datasets/code_and_data_for_subjects_inclusion")

mat = scipy.io.loadmat('subject_list.mat')


df = pd.DataFrame(mat['good_EEG'])
# df1 = pd.DataFrame(mat['not_bad_EEG'])
df_sliced = [df.values[i][0][0] for i in range(len(df))]
# df_sliced2 = [df1.values[i][0][0] for i in range(len(df1))]
# df_total = list(set(df_sliced).union(set(df_sliced2)))


df_SI =pd.read_csv('participants_SI.tsv')
df_RU =pd.read_csv('participants_RU.tsv',sep='\t')
df_CUNY =pd.read_csv('participants_CUNY.tsv',sep='\t')
df_CBIC =pd.read_csv('participants_CBIC.tsv',sep='\t')

subjects_aged = list()
def find_subject_age(which_df,age):
    subjects_aged.append (which_df[which_df['Age']>=age]['participant_id'].values)

age = 5
find_subject_age(df_CBIC,age)
find_subject_age(df_RU,age)
find_subject_age(df_CUNY,age)
find_subject_age(df_SI,age)
subjects_aged = np.hstack(subjects_aged)

print (len(subjects_aged))



def read_ods(filename, sheet_no=0, header=0):
    tab = ezodf.opendoc(filename=filename).sheets[sheet_no]
    return pd.DataFrame({col[header+1].value:[x.value for x in col[header+1:]]
                         for col in tab.columns()})

df_dwi = read_ods('dwi-subject_list.ods',0,0)
df_freesurfer = read_ods('subjects_freesurfer.ods')


df_freesurfer_sliced = [ df_freesurfer[df_freesurfer.columns[0]].values[i][31:-1] for i in range(1,len(df_freesurfer))]
df_freesurfer_sliced.append('NDARAD481FXF')
df_dwi_sliced = [ df_dwi[df_dwi.columns[0]].values[i][31:] for i in range(0,len(df_dwi))]

subjects_aged_sliced = [subjects_aged[i][4:] for i in range(0,len(subjects_aged))]
total_subjects = list(set(df_dwi_sliced).intersection( set(df_sliced).intersection(set(subjects_aged_sliced))))


data_present = list()
for i in range(len(total_subjects)):
    path_to_file = '/users2/local/Venkatesh/HBN/%s/RestingState_data.csv' % total_subjects[i]
    path_to_file_video = '/users2/local/Venkatesh/HBN/%s/Video3_event.csv' % total_subjects[i]

    if (os.path.isfile(path_to_file) and os.path.isfile(path_to_file_video)):
        data_present.append (total_subjects[i])



subjects_sanity_check  = [
 'NDARAD481FXF', 'NDARBK669XJQ','NDARCD401HGZ', 'NDARDX770PJK', 'NDAREC182WW2', 
 'NDARGY054ENV', 'NDARHP176DPE', 'NDARLB017MBJ', 'NDARMR242UKQ','NDARNT042GRA',
 'NDARRA733VWX', 'NDARRD720XZK', 'NDARTR840XP1', 'NDARUJ646APQ', 'NDARVN646NZP', 
 'NDARWJ087HKJ', 'NDARXB704HFD', 'NDARXJ468UGL', 'NDARXJ696AMX',  'NDARXU679ZE8', 
 'NDARXY337ZH9', 'NDARYM257RR6', 'NDARYY218AGA', 'NDARYZ408VWW', 'NDARZB377WZJ']



print(len(set(data_present).intersection(subjects_sanity_check)))