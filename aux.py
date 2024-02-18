

import pandas as pd
from function_mapping import read_director
import numpy as np



#ejemplo_dir = 'input_model_diagnosis_patient/'
ejemplo_dir = 'input_model_pred/'
ficheros = read_director(ejemplo_dir)
for i in ficheros:
    df = pd.read_csv(ejemplo_dir+i)
    print(i, df.shape)

ejemplo_dir_u = 'input_model_patient_drugs/'  # Ensure this directory exists
im = ["ATC3_Patient_non_filtered.csv","ATC4_Patient_non_filtered.csv"]
ficheros = read_director(ejemplo_dir)
ficher_new = [i for i in ficheros if i not in im]

for i in range(len(ficher_new)):
    df1  = pd.read_csv(ejemplo_dir+im[0])
    df2 = pd.read_csv(ejemplo_dir+ficher_new[i])
    result_df = pd.merge(df1,df2 , on=["SUBJECT_ID"], how='left')
    df2 = df2[df2["SUBJECT_ID"].isin(list(result_df["SUBJECT_ID"]))]
    df2.to_csv(ejemplo_dir_u + ficher_new[i], index=False)
    
    value_to_count = True
    t = (df1.SUBJECT_ID.values ==df2.SUBJECT_ID.values)
    # Count occurrences of the value
    occurrences = np.count_nonzero(t == value_to_count)
    print(occurrences ==df2.shape[0])

