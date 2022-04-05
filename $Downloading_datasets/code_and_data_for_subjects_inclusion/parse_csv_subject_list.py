''' PARSE CSV SUBJECT FILEs '''

import pandas as pd
import numpy as np
import ezodf
import csv

def read_ods(filename, sheet_no=0, header=0):
    tab = ezodf.opendoc(filename=filename).sheets[sheet_no]
    return pd.DataFrame({col[header].value:[x.value for x in col[header+1:]]
                         for col in tab.columns()})

def sub_list_from_tsv(filename, age_min,separator='\t'):
    df=pd.read_csv(filename, sep = separator)
    id_sub=df['participant_id']
    age=df['Age']
    sub_fs_list=[]
    for i in range(len(df.values)):
        if age[i] > age_min:
            val=id_sub[i]
            idx=val.find('N')
            sub_fs_list.append(val[idx:])
    return sub_fs_list

df = read_ods(filename='/home/g20lioi/fixe/datasets/HBN_small/subjects_freesurfer.ods')
df_good= read_ods(filename='/home/g20lioi/fixe/datasets/HBN_small/subjects_good.ods')


eeg_major_list= sub_list_from_tsv('/home/g20lioi/fixe/datasets/HBN_small/EEG/participants.tsv', age_min=18)
mri_cbic_list= sub_list_from_tsv(filename='/home/g20lioi/fixe/datasets/HBN_small/MRI/Site-CBIC/participants.tsv', age_min=1)
mri_ru_list= sub_list_from_tsv(filename='/home/g20lioi/fixe/datasets/HBN_small/MRI/Site-RU/participants.tsv', age_min=1)
mri_cuny_list= sub_list_from_tsv(filename='/home/g20lioi/fixe/datasets/HBN_small/MRI/Site-CUNY/participants.tsv', age_min=1)
#SI list is comma separated
mri_si_list= sub_list_from_tsv(filename='/home/g20lioi/fixe/datasets/HBN_small/MRI/Site-SI/participants.tsv', age_min=1,separator=',')




age_min=1
    
    for i in range(len(df_mri_cbic.values)):
        if df.values[i][4] > age_min:
            val=df.values[i][1]
            idx=val.find('N')
            sub_fs_list.append(val[idx:])


sub_fs_list=[]
for i in range(len(df.values)):
    val=df.values[i][0]
    idx=df.values[i][0].find('N')
    sub_fs_list.append(val[idx:-1])

sub_good_list=[]

for i in range(len(df_good.values)):
    sub_good_list.append(df_good.values[i][0])




common_sub_list= list(set(sub_good_list).intersection(sub_fs_list))

empty=[]
with open('/home/g20lioi/fixe/datasets/HBN_small/subjects_good_withfs.csv', 'w') as f:

 for i in range(len(common_sub_list)):
     # using csv.writer method from CSV package
     write = csv.writer(f)
      
     write.writerow([common_sub_list[i]])
    

common_major_eeg_fs_good = list(set(eeg_major_list).intersection(common_sub_list))

with open('/home/g20lioi/fixe/datasets/HBN_small/subjects_good_withfs_age18.csv', 'w') as f:

 for i in range(len(common_major_eeg_fs_good)):
     # using csv.writer method from CSV package
     write = csv.writer(f)
      
     write.writerow([common_major_eeg_fs_good[i]])



common_all_cbic=list(set(mri_cbic_list).intersection(common_major_eeg_fs_good))


with open('/home/g20lioi/fixe/datasets/HBN_small/MRI/Site-CBIC/subjects_good_withfs_age18.csv', 'w') as f:

 for i in range(len(common_all_cbic)):
     # using csv.writer method from CSV package
     write = csv.writer(f)
      
     write.writerow([common_all_cbic[i]])    



common_all_ru=list(set(mri_ru_list).intersection(common_major_eeg_fs_good))


with open('/home/g20lioi/fixe/datasets/HBN_small/MRI/Site-RU/subjects_good_withfs_age18.csv', 'w') as f:

 for i in range(len(common_all_ru)):
     # using csv.writer method from CSV package
     write = csv.writer(f)
      
     write.writerow([common_all_ru[i]])    



common_all_cuny=list(set(mri_cuny_list).intersection(common_major_eeg_fs_good))


with open('/home/g20lioi/fixe/datasets/HBN_small/MRI/Site-CUNY/subjects_good_withfs_age18.csv', 'w') as f:

 for i in range(len(common_all_cuny)):
     # using csv.writer method from CSV package
     write = csv.writer(f)
      
     write.writerow([common_all_cuny[i]])    


common_all_si=list(set(mri_si_list).intersection(common_major_eeg_fs_good))

    