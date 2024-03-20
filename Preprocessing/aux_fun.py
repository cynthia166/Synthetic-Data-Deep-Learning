
import pandas as pd
from Preprocessing.function_mapping import read_director
import numpy as np
ejemplo_dir = 'input_model_pred_drugs/'
ejemplo_dir_u = 'input_model_pred_drugs_u/'  # Ensure this directory exists
im = ["ATC3_outs_visit_non_filtered.csv","ATC4_outs_visit_non_filtered.csv"]
ficheros = read_director(ejemplo_dir)
ficher_new = [i for i in ficheros if i not in im]

for i in range(len(ficher_new)):
    df1  = pd.read_csv(ejemplo_dir+im[0])
    df2 = pd.read_csv(ejemplo_dir+ficher_new[i])
    result_df = pd.merge(df1,df2 , on=["HADM_ID"], how='inner')
    df2 = df2[df2["HADM_ID"].isin(list(result_df["HADM_ID"]))]
    df2.to_csv(ejemplo_dir_u + ficher_new[i], index=False)
    
    value_to_count = True
    t = (df1.HADM_ID.values ==df2.HADM_ID.values)
    # Count occurrences of the value
    occurrences = np.count_nonzero(t == value_to_count)
    print(occurrences ==df2.shape[0])

for i in im:
    aux  = pd.read_csv(ejemplo_dir+i)
    aux.to_csv(ejemplo_dir_u + i, index=False)
    



'''
import pandas as pd
from function_mapping import read_director
import numpy as np
ejemplo_dir = 'input_model_pred_diagnosis/'
ejemplo_dir_u = 'input_model_pred_diagnosis_u/'  # Ensure this directory exists
ficher_new= ["input_model_pred_diagnosisthreshold_0.999_diagnosis_outs_visit_non_filtered.csv",]
ficheros = read_director(ejemplo_dir)
im = [i for i in ficheros if i not in ficher_new]

for i in range(len(ficher_new)):
    df1  = pd.read_csv(ejemplo_dir+im[0])
    df2 = pd.read_csv(ejemplo_dir+ficher_new[i])
    result_df = pd.merge(df1,df2 , on=["HADM_ID"], how='inner')
    df2 = df2[df2["HADM_ID"].isin(list(result_df["HADM_ID"]))]
    df2.to_csv(ejemplo_dir_u + ficher_new[i], index=False)
    
    value_to_count = True
    t = (df1.HADM_ID.values ==df2.HADM_ID.values)
    # Count occurrences of the value
    occurrences = np.count_nonzero(t == value_to_count)
    print(occurrences ==df2.shape[0])

for i in im:
    aux  = pd.read_csv(ejemplo_dir+i)
    aux.to_csv(ejemplo_dir_u + i, index=False)
    


file_save = "input_model_pred_diagnosis/"
from function_mapping import *
import pandas as pd
import argparse

ficheros = read_director(file_save)
for i in ficheros:
    ccs =pd.read_csv(file_save +i)
    print(i, ccs.shape)'''
