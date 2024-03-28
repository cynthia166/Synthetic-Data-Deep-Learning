# -*- coding: utf-8 -*-
# %%
import subprocess
#######################################################Files#######################################################
#data_preprocess_non_filtered.csv
## File for demographic and admission figures
###input:  
###output:
# %jupyter notebook 4_concat_input_.ipynb


# %% [markdown]
# DARTA_INTERM/"choices_preprocess_threshold_nonfiltered.csv"
# # File for 10 top counts, histogram per patient and admission leveñ
# ##input:  
# ##output: 

# %% [markdown]
# ICD9_CODE_procedures_outs_visit_non_filtered
# #file for LOS and age kernel density plot
# ##input:
# ##output:  not applied
# ######################################################Algorithm####################################################

# %%


# %% [markdown]
# ######################################################Figures#######################################################
# MIMIC-III database schema
# # 
# ##input:  not applied
# ##output:  not applied

# %% [markdown]
#  Entity relationship diagram
# # It is done manually with Mino website
# ##input:  not applied
# ##output:  not applied


# %%
# Demographics and admission variables
## Figure pie charts (Figure 2.3:)
###input:  MIMIC / 'ADMISSIONS.csv.gz' /DARTA_INTERM/ 'data_preprocess_non_filtered.csv'
###output: IMAGES_Demo+'demo_pie.svg'
python vis_.py "demo_pie"


# %%
#  Distribution of counts of ICD-9 ['procedures', 'diagnosis', 'medicament']
##  ['procedures', 'diagnosis', 'medicament'] per patient and admission.
###input:  DARTA_INTERM/"choices_preprocess_threshold_nonfiltered.csv" 
###output: IMAGES_Demo + 'counts_10.svg' /IMAGES_Demo + 'histograns_patient.svg'
python vis_.py "countpr_admi_patient" --type_procedur procedures
choices=['procedures', 'diagnosis', 'medicament']


# %%
# 10 most frequent Procedures, Diagnosis and Medicament
## ['procedures', 'diagnosis', 'medicament']
###input:   DARTA_INTERM/"choices_preprocess_threshold_nonfiltered.csv"
###output: 
python vis_.py "10_most_frequent" --type_procedur procedures
choices=['procedures', 'diagnosis', 'medicament']


# %%
# Kernel density plot, age and lengh of stay
## 
###input:  input_model_pred / 'ICD9_CODE_procedures_outs_visit_non_filtered.csv'
###output: 
python vis_.py "Kernel_density_estimation" 


# %%
# Histograme of thresholds procedures
## 
###input: list IC9-codes, num_bins, threshold_value, cat is the name of plot
###output: IMAGES_Demo+'thresholds_+'+threshold_value+'.svg'
from preprocess_input1 import  cumulative_plot
cumulative_plot(icd9_codes, num_bins,threshold_value,cat)

# %%
#######################################################Data Cleaning######################################################
# Function to clean the data
###input:  raw data /ADMISSIONS.csv.gz /PROCEDUREs.csv.gz /DIAGNOSES_ICD.csv.gz  /type_p = ['procedures', 'diagnosis', 'medicament']
###output: clean data
from data_preprocessing import DataPreprocessor
preprocessor = DataPreprocessor(type_p,doc_path, admissions_path, patients_path, categorical_cols, real, level, numerical_cols, prepomax,name,n, cols_to = None,normalize_matrix=normalize_matrix, log_transformation=log_transformation, encode_categorical=encode_categorical, final_preprocessing=True,proportion = True)   
type_p = ['procedures', 'diagnosis', 'medicament']
data = preprocessor.load_data_clean_data(type_p[0])



# %% [markdown]
# ######################################################Other######################################################

# %% [markdown]
# Preparing the input for the generation of synthetic data
# #  Done manually with Mino website
# ##input:  not applied 
# ##output: not applied
# ######################################################Data reduction-data transformation######################################################
# ######################################################Demographics and Admission Variables .######################################################

# %%
# Demographics and Admission Variables .
## 
###input:  adm = ADMISSIONS.csv.gz / pa = PATIENTS.csv.gz / categorical_cols = ['INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL'..] 
#/level = [Patiemnt, Admission] /cat_considered =     categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
# 'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION', 'MARITAL_STATUS',  'ETHNICITY','GENDER']
# cat_considered = ['ADMITTIME', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'DEATHTIME'] + ['DISCHTIME', 'SUBJECT_ID', 'HADM_ID']
###output: demographics dataframe
from preprocess_class1 import calculate_demographics
demos = calculate_demographics(adm,pa, categorical_cols, level,cat_considered,prod_ipvot=None)
#######################################################Data reduction of ICD-9 Codes Procedures######################################################


# %%
#  CCS codes proceduers
## map ccs codes to ICD-9 codes
###input: ,data/raw/suplement$prref 2015.csv /  data/raw/ PROCEDURE_ICD.csv.gz  
###output:CCS codes 
from preprocess_class1 import ccsc_codes
ccsc_codes(PROCEDURE_ICD.csv.gz )

# %%
# Thresholds
## Thresholds procedure
###input:  PROCEDURE_ICD.csv.gz  /n =[thresholds] , name = "ICD9_CODE"
###output:icd-9 codes simpñified
from preprocess_class1 import funcion_acum
funcion_acum(nuevo_df,n,name)
#######################################################Data reduction of ICD-9 Codes Diagnosis######################################################

# %% [markdown]
#  Same as procedures but with DIAGNOSIS_ICD.csv.gz input

# %% [markdown]
# ######################################################Data reduction of ICD-9 Codes Diagnosis######################################################

# %% [markdown]
#  Same as procedures but with PRESCRIPTION.csv.gz input and name = DRUGS.  / mappin ATC4
# RAW/"suplement/RXCUI2atc4.csv
# RAW/"suplement/ndc2RXCUI.txt

# %%
#######################################################Product from Chapiter 2#######################################################
# Input for clustering_ Admission and Prediction
#python tu_script.py diagnosis --normalize_matrix --log_transformation --encode_categorical
'''Esto significa que estos argumentos serán True por defecto y se establecerán en False cuando 
se proporcionen en la línea de comandos.'''

# %%
subprocess.run(["python", "preprocess_class1.py", "diagnosis"])
subprocess.run(["python", "preprocess_class1.py", "procedures"])
subprocess.run(["python", "preprocess_class1.py", "drug2"])

# %%
#Input not preprocesed for SD
#python tu_script.py diagnosis --normalize_matrix --log_transformation --encode_categorical
subprocess.run(["python", "preprocess_class1.py", "diagnosis", "--normalize_matrix", "--log_transformation", "--encode_categorical"])
subprocess.run(["python", "preprocess_class1.py", "procedures", "--normalize_matrix", "--log_transformation", "--encode_categorical"])
subprocess.run(["python", "preprocess_class1.py", "drug2", "--normalize_matrix", "--log_transformation", "--encode_categorical"])


# %% [markdown]
# input condat in preprocesin carpet
# "concat","entire_ceros","temporal_state"
# python preprocess_input_SDmodel.py "concat"

# %% [markdown]
# run for dataset padded with 0
# python preprocess_input_SDmodel.py "entire_ceros"

# %% [markdown]
# dataset con shape (n_patient, n_features ,ntime_step), statics features
# python preprocess_input_SDmodel.py "temporal_state"
