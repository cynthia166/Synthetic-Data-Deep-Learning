import pandas as pd

# Assuming df1, df2, df3 are your dataframes
df_drugs =pd.read_csv('./input_model_pred_drugs_u/ATC3_outs_visit_non_filtered.csv')
df_diagnosis = pd.read_csv('./input_model_pred_diagnosis_u/CCS_CODES_diagnosis_outs_visit_non_filtered.csv')
df_procedures = pd.read_csv('./input_model_visit_procedures/CCS CODES_proc_outs_visit_non_filtered.csv')
# Drop the columns categotical
numerical_cols =  ['Age_max', 'LOSRD_sum',
       'L_1s_last', 'LOSRD_avg','L_1s_last_p1']

categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                'MARITAL_STATUS', 'ETHNICITY','GENDER']

size1 = df_drugs.size
size2 = df_diagnosis.size
size3 = df_procedures.size


# Find out which DataFrame is the largest
if size1 >= size2 and size1 >= size3:
    print("df1 is the largest")
elif size2 >= size1 and size2 >= size3:
    print("df2 is the largest")
else:
    print("df3 is the largest")



df_diagnosis = df_diagnosis.drop(columns=categorical_cols)
df_drugs = df_drugs.drop(columns=categorical_cols+numerical_cols)
df_procedures = df_procedures.drop(columns=categorical_cols+numerical_cols)

rename_dict_d = {col: col + '_diagnosis' for col in df_diagnosis.columns if col not in numerical_cols+["SUBJECT_ID"+ "HADM_ID"] }
df_diagnosis.rename(columns=rename_dict_d, inplace=True)

rename_dict = {col: col + '_drugs' for col in df_drugs.columns if col != "SUBJECT_ID" and col != "HADM_ID" }

df_drugs.rename(columns=rename_dict, inplace=True)

rename_dict = {col: col + '_procedures' for col in df_procedures.columns if col != "SUBJECT_ID" and col != "HADM_ID" }
df_procedures.rename(columns=rename_dict, inplace=True)



result = pd.merge(df_diagnosis, df_drugs, on=["SUBJECT_ID","HADM_ID"], how='outer')
result_final = pd.merge(result, df_procedures, on=["SUBJECT_ID","HADM_ID"], how='outer')

# Assuming df is your DataFrame



adm = pd.read_csv('./data/data_preprocess_nonfilteres.csv')

res = pd.merge(adm[categorical_cols+["ADMITTIME","SUBJECT_ID","HADM_ID"]],result_final, on=["SUBJECT_ID","HADM_ID"], how='roght')

# Assuming df is your DataFrame

# Find columns that contain 'unnamed' in their name
cols_to_drop = res.filter(like='unnamed', axis=1).columns

# Drop these columns
res.drop(cols_to_drop, axis=1, inplace=True)