# -*- coding: utf-8 -*-

# The code snippet you provided is importing various Python libraries and modules that are commonly
# used in data analysis, visualization, and manipulation tasks. Here is a brief explanation of each
# import statement:
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import polars as pl
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import os
import glob
#import psycopg2
import datetime
import sys
from operator import itemgetter, attrgetter, methodcaller
import numpy as np
import itertools
import os.path
import matplotlib.pyplot as plt
import math
from multiprocessing import Pool, cpu_count
from config import *
import os
import gzip



def split(valid_perc,dataset_name,features,attributes,):

    N, T, D = features.shape  
    # further split the training data into train and validation set - same thing done in forecasting task
    N_train = int(N * (1 - valid_perc))
    N_valid = N - N_train

    # Shuffle data
    #np.random.shuffle(full_train_data)

    train_data_features = features[:N_train]
    valid_data_features = features[N_train:]   

    train_data_attributes = attributes[:N_train]
    valid_data_attributes = attributes[N_train:]   
    print("train/valid shapes: ", train_data_features.shape, valid_data_features.shape)    
    SD_DATA_split = "train_sp"
    with gzip.open(SD_DATA_split + dataset_name + 'train_data_features.pkl', 'wb') as f:
        pickle.dump(train_data_features, f)
    with gzip.open(SD_DATA_split+ dataset_name + 'valid_data_features.pkl', 'wb') as f:
        pickle.dump(valid_data_features, f)

    with gzip.open(SD_DATA_split + dataset_name + 'train_data_attributes.pkl', 'wb') as f:
        pickle.dump(train_data_attributes, f)
    with gzip.open(SD_DATA_split+ dataset_name + 'valid_data_attributes.pkl', 'wb') as f:
        pickle.dump(valid_data_attributes, f)


def get_input_time( type_df,arhivo,name,cols_to_drop1,res,keywords,s,status):
     # Obtener una lista de pacientes únicos
    
    if s != 1:
        unique_patients = res['SUBJECT_ID'].unique()
    # Calcular el 20% del total de pacientes únicos
        sample_size = int(s* len(unique_patients))
        # Obtener una muestra aleatoria del 20% de los pacientes únicos
        sample_patients = np.random.choice(unique_patients, size=sample_size, replace=False)
        # Filtrar el DataFrame para incluir solo los registros de los pacientes en la muestra
        sample_df = res[res['SUBJECT_ID'].isin(sample_patients)]
    else:
        sample_df = res 
        sample_patients = "null"
    
    if status == "full_input":
        dfs = sample_df
    else:         
        dfs = obtener_entire(sample_df,type_df)
    
    # Lista de nombres de columnas a buscar
    # Supongamos que type_df es tu DataFrame
      # Filtrar las columnas de type_df que contienen las palabras clave
    static_data_cols = [col for col in dfs.columns if any(keyword in col for keyword in keywords)]
    # Crear un nuevo DataFrame que solo incluye las columnas filtradas
    not_considet_temporal = [
    i for i in static_data_cols if i != 'SUBJECT_ID'
    ]

    static_data = dfs[static_data_cols].groupby('SUBJECT_ID').max().reset_index(drop=True)
    outcomes = dfs[['HOSPITAL_EXPIRE_FLAG','SUBJECT_ID']].groupby('SUBJECT_ID').max().reset_index(drop=True)
    temporal_data = dfs.drop(columns=not_considet_temporal+['HOSPITAL_EXPIRE_FLAG'])
    cols_to_drop = list(temporal_data.filter(like='Unnamed', axis=1).columns) + ['ADMITTIME','HADM_ID']
    temporal_data.drop(cols_to_drop, axis=1, inplace=True)
    

    # Establecer 'SUBJECT_ID' y 'visit_rank' como índices
    temporal_data.set_index(['SUBJECT_ID', 'visit_rank'], inplace=True)
    # Ordenar temporal_data por el índice 'visit_rank' en orden ascendente
    temporal_data.sort_index(level='visit_rank', inplace=True)
    # Ahora se agrupa por 'SUBJECT_ID' y se recoge la información necesaria.
    grouped = temporal_data.groupby(level='SUBJECT_ID')

    # Se obtiene 'observations' como los índices de segundo nivel para cada grupo.
    observation_data = [group.index.get_level_values('visit_rank').tolist() for _, group in grouped]

    # Se obtiene los dataframes temporales por grupo, reseteando el índice 'SUBJECT_ID'.
    temporal_dataframes = [group.reset_index(level=0, drop=True) for _, group in grouped]
    return static_data, temporal_dataframes, observation_data,outcomes,sample_patients

def cols_not_consideres(aux):
    int_index_columns = []
    for col in aux.columns:
        try:
            int(col)  # Intenta convertir el índice de la columna a un entero
            # Si la conversión es exitosa, agrega la columna a la lista
        except ValueError:
            int_index_columns.append(col)
            # Si la conversión falla, ignora la columna

    print(int_index_columns)
    return int_index_columns
def concat_input(df_procedures,df_diagnosis,df_drugs,name_df):
    df_procedures = pd.read_csv(df_procedures)
    df_diagnosis =pd.read_csv(df_diagnosis)
    df_drugs = pd.read_csv(df_drugs)
    categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                    'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                    'MARITAL_STATUS',  'ETHNICITY','GENDER']
    numerical_cols =  ['Age_max', 'LOSRD_sum',
            'LOSRD_avg','L_1s_last_p1']
    # The code `columnas` and `
    # The code `columnas` and `
    columnas = categorical_cols+numerical_cols
    drugs_cols = [col for col in df_drugs.columns if any(item in col for item in columnas)]
    catnum_cols = [col for col in df_procedures.columns if any(item in col for item in columnas)]
    categorical = list(set(catnum_cols+drugs_cols))
   
    df_drugs = df_drugs.drop(columns=drugs_cols)
    df_procedures = df_procedures.drop(columns=catnum_cols)

    #df_diagnosis = df_diagnosis.drop(columns=categorical_cols)


    rename_dict_d = {col: col + '_diagnosis' if col not in ["SUBJECT_ID", "HADM_ID"] else col for col in df_diagnosis.columns if col not in categorical}
    df_diagnosis.rename(columns=rename_dict_d, inplace=True)

    rename_dict = {col: col + '_drugs' for col in df_drugs.columns if col != "SUBJECT_ID" and col != "HADM_ID" }

    df_drugs.rename(columns=rename_dict, inplace=True)

    rename_dict = {col: col + '_procedures' for col in df_procedures.columns if col != "SUBJECT_ID" and col != "HADM_ID" }
    df_procedures.rename(columns=rename_dict, inplace=True)



    result = pd.merge(df_diagnosis, df_drugs, on=["SUBJECT_ID","HADM_ID"], how='outer')
    result_final = pd.merge(result, df_procedures, on=["SUBJECT_ID","HADM_ID"], how='outer')
    cols_to_drop = result_final.filter(like='Unnamed', axis=1).columns

    # Drop these columns
    result_final.drop(cols_to_drop, axis=1, inplace=True)
    res = result_final.fillna(0)
    res.to_csv(DARTA_INTERM_intput+ name_df)

# Assuming df is your DataFrame









    



'''La primeras dor gunciones se corren si debo volver a correl el input 
obtener_added_cols_targer_visitrank_ este es el input del general
obtener_entire rste es el input para time series'''
def concat_input_(df_drugs, df_diagnosis, df_procedures,numerical_cols,categorical_cols,name_df):
    # Drop the columns categotical funcio que concatena los 3 inputs, manteniendo las columasn del mayot
 
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

    rename_dict_d = {col: col + '_diagnosis' if col not in ["SUBJECT_ID", "HADM_ID"] else col for col in df_diagnosis.columns if col not in numerical_cols}
    df_diagnosis.rename(columns=rename_dict_d, inplace=True)

    rename_dict = {col: col + '_drugs' for col in df_drugs.columns if col != "SUBJECT_ID" and col != "HADM_ID" }

    df_drugs.rename(columns=rename_dict, inplace=True)

    rename_dict = {col: col + '_procedures' for col in df_procedures.columns if col != "SUBJECT_ID" and col != "HADM_ID" }
    df_procedures.rename(columns=rename_dict, inplace=True)



    result = pd.merge(df_diagnosis, df_drugs, on=["SUBJECT_ID","HADM_ID"], how='outer')
    result_final = pd.merge(result, df_procedures, on=["SUBJECT_ID","HADM_ID"], how='outer')

    # Assuming df is your DataFrame
    


    adm = pd.read_csv('./data/data_preprocess_nonfilteres.csv')

    res = pd.merge(adm[categorical_cols+["ADMITTIME","SUBJECT_ID","HADM_ID"]],result_final, on=["SUBJECT_ID","HADM_ID"], how='right')

    # Assuming df is your DataFrame

    # Find columns that contain 'unnamed' in their name
    cols_to_drop = res.filter(like='Unnamed', axis=1).columns

    # Drop these columns
    res.drop(cols_to_drop, axis=1, inplace=True)
    res = res.fillna(0)
    res.to_csv(DARTA_INTERM_intput+ name_df)
    return res




def obtener_added_cols_targer_visitrank_(arhivo):
    
    df =  pd.read_csv(arhivo)

    # Limitar las fechas a un rango permitido

    # Ahora deberías poder restar las fechas sin problemas
    '''Funcion que regredsa si es tipo entire datafram donde se agregan ceros para que todos tengas mas visitas dataframe, o lista de listas que es el input de time gan'''
    #adm = pd.read_csv('./data/data_preprocess_nonfilteres.csv')
    ad_f = MIMIC/"ADMISSIONS.csv.gz"
    adm = pd.read_csv(ad_f)
 
    res = pd.merge(adm[["HOSPITAL_EXPIRE_FLAG","SUBJECT_ID","HADM_ID","ADMITTIME"]],df, on=["SUBJECT_ID","HADM_ID"], how='right')

    res['ADMITTIME'] = pd.to_datetime(res['ADMITTIME'])
    #res['DOB'] = pd.to_datetime(res['DOB'], format='%Y-%m-%d %H:%M:%S')
    #res['DOB'] = pd.to_datetime(res['DOB'])
   
    

    res = res.fillna(0)



    print(res.shape, adm.shape, df.shape)
    # Assuming df is your DataFrame

    # Find columns that contain 'unnamed' in their name
    cols_to_drop = res.filter(like='Unnamed', axis=1).columns
    res.drop(cols_to_drop, axis=1, inplace=True)
    print(res.shape, adm.shape, df.shape)

    print(res.isnull().sum().sum())
    res = res.fillna(0)
    print(res.isnull().sum().sum())
       


    res = res.sort_values(by=['SUBJECT_ID', 'ADMITTIME'])

# Agregar una nueva columna 'VISIT_NUMBER' que indica el número de visita para cada 'SUBJECT_ID'
    res['visit_rank'] = res.groupby('SUBJECT_ID').cumcount() + 1
# Crear una nueva columna 'visit_rank' que represente el número de la visita para cada paciente


# Ahora, vamos a separar las visitas en DataFrames individuales y guardarlos en una lista
    
    # Asegúrate de que 'ADMITTIME' es una fecha

    # Ordena los datos por 'SUBJECT_ID' y 'ADMITTIME'
    res = res.sort_values(['SUBJECT_ID', 'ADMITTIME'])
    return res


  

#res['horizons'] = res.groupby('SUBJECT_ID')['ADMITTIME'].diff().fillna(pd.Timedelta(seconds=0))
#res['horizons'] =[int(i) for i in res['horizons'].dt.total_seconds()]
def obtener_entire(res,type_df):
    
    res = res.copy()
    res['visit_rank'] = res['visit_rank'].astype(int)

    temporal_surv = res.copy()
# Supongamos que 'temporal_surv' es tu DataFrame original y tiene una columna 'SUBJECT_ID' y 'visitas'.
# Crear un nuevo DataFrame con todas las combinaciones posibles de 'SUBJECT_ID' y 'visitas'.
    unique_subjects = temporal_surv['SUBJECT_ID'].unique()
    all_visits = range(1, 43)  # Crear una lista de visitas de 1 a 42.

    # Utiliza el producto cartesiano para obtener todas las combinaciones posibles de 'SUBJECT_ID' y 'visitas'.
    combinations = pd.MultiIndex.from_product([unique_subjects, all_visits], names=['SUBJECT_ID', 'visit_rank']).to_frame(index=False)

    # Ahora mergea esto con el DataFrame original. 
    # Esto asegurará que todas las combinaciones de 'SUBJECT_ID' y 'visitas' estén presentes.
    new_temporal_surv = pd.merge(combinations, temporal_surv, on=['SUBJECT_ID', 'visit_rank'], how='left')

    # Rellena los NaNs con ceros para todas las columnas excepto 'visitas' y 'SUBJECT_ID'.
    columns_to_fill = new_temporal_surv.columns.difference(['SUBJECT_ID', 'visit_rank'])
    new_temporal_surv[columns_to_fill] = new_temporal_surv[columns_to_fill].fillna(0)


            


        
    if type_df == "entire":
        df =new_temporal_surv.copy()
        dfs = df.sort_values('SUBJECT_ID')   
    else :
        new_df_list =new_temporal_surv.copy()
        dfs = df.sort_values('SUBJECT_ID')      
        dfs = [group for _, group in df.groupby('SUBJECT_ID')]   
       
    return dfs     
    

def main(task,preprocessing):
    import pickle
    import gzip
    if preprocessing=="True":
       name_df = "raw_input.csv"
    else:
       name_df = "raw_input_non_preprocess.csv"
           
    
    if task =="concat":
        if preprocessing=="True":
            df_procedures = DARTA_INTERM_intput+"CCS CODES_procedures.csv"
            df_diagnosis =DARTA_INTERM_intput+"CCS CODES_diagnosis.csv"
            df_drugs = DARTA_INTERM_intput+"ATC3_drug2.csv"
            df =  concat_input(df_procedures,df_diagnosis,df_drugs,name_df)
        else:
            df_procedures = DARTA_INTERM_intput+"CCS CODES_procedures_non_prep.csv"
            df_diagnosis =DARTA_INTERM_intput+"CCS CODES_diagnosis_non_prepo.csv"
            df_drugs = DARTA_INTERM_intput+"ATC3_drug2_non_prepo.csv"
            df =  concat_input(df_procedures,df_diagnosis,df_drugs,name_df)

                
       
    if task =="entire_ceros":
        
        if preprocessing == "True":
            name_entire = "entire_ceros"
        else:
            name_entire = "entire_ceros_non_preprocess"    
        aux = obtener_added_cols_targer_visitrank_(DARTA_INTERM_intput+ name_df)
        dfs = obtener_entire(aux,"entire")    
       
        dfs.to_csv(DARTA_INTERM_intput+ name_entire+"_.csv")
        
           
           
    if task =="temporal_state":
        if preprocessing == "True":
            name_entire = "entire_ceros"
            dataset_name = 'DATASET_NAME_prepo'
        else:
            name_entire = "entire_ceros_non_preprocess"    
            dataset_name = 'DATASET_NAME_non_prepo'
        type_df = "entire"
        
        arhivo = DARTA_INTERM_intput+ name_df
        s = 1
        name  = "input_generative_g.csv"
        status = "full_input"
        #res = obtener_added_cols_targer_visitrank_(arhivo,name)
        cols_to_drop1 = ['ADMITTIME','HADM_ID']
        try:
            res = pd.read_csv(DARTA_INTERM_intput+ name_entire+".csv")
        except:    
            res = pd.read_parquet(DARTA_INTERM_intput+ name_entire+".parquet", engine='pyarrow')
        keywords = ['INSURANCE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY','GENDER','SUBJECT']
        static_data, temporal_dataframes, observation_data,outcomes,sample_patients = get_input_time( type_df,arhivo,name,cols_to_drop1,res,keywords,s,status)
        
        
        #guardar_data
        if preprocessing == "True":
            with gzip.open(DARTA_INTERM_intput + dataset_name + '_preprocess.pkl', 'wb') as f:
                pickle.dump(temporal_dataframes, f)
            pd.DataFrame(observation_data).to_csv(DARTA_INTERM_intput+"observation_data_preprocess.csv")
            static_data.to_csv(DARTA_INTERM_intput+"static_data_preprocess.csv")
            outcomes.to_csv(DARTA_INTERM_intput+"outcomes_data_preprocess.csv")     
        else:
            with gzip.open(DARTA_INTERM_intput + dataset_name + '_non_preprocess.pkl', 'wb') as f:
             pickle.dump(temporal_dataframes, f)
            pd.DataFrame(observation_data).to_csv(DARTA_INTERM_intput+"observation_data_non_preprocess.csv")
            static_data.to_csv(DARTA_INTERM_intput+"static_data_non_preprocess.csv")
            outcomes.to_csv(DARTA_INTERM_intput+"outcomes_data__non_preprocess.csv")     
        
if __name__ == "__main__":
    import argparse
    import os
    name_entire = "entire_ceros"
    ruta = DARTA_INTERM_intput+ name_entire+"_.parquet"

    if os.path.exists(ruta):
        print("La ruta existe.")
    else:
        print("La ruta no existe.")
    
    
    
    parser = argparse.ArgumentParser(description="Scripti input special SD model")

    # Este argumento es obligatorio, así que no tiene un valor por defecto.
    parser.add_argument("task", type=str, choices=["concat","entire_ceros","temporal_state"],default="temporal_state",
                        help="Tipo de procesamiento a realizar.")
    parser.add_argument("preprocessing", choices=["True","False"],default="True",
                        help="Esta preprocess")
    args = parser.parse_args()
    task = args.task
    preprocessing = args.preprocessing
    #ask = "temporal_state"
    main(task,preprocessing)
####################################Conatenacion primera
#ruta_archivos = 's_data\*.csv.gz'  # Puedes cambiar '*.csv' por la extensión que desees
#save = True #dfale

#mi_objeto = PreprocessInput()
#df_concat_primero = concat_archivo_primeto(procedures,admi,ruta_archivos,save,nom_archivo)



#d1 = '.\s_data\DIAGNOSES_ICD.csv.gz'


#didf_diagnosisa = diagnosis(d1,n,name)

#desconcatenación, y se obtiene threshold y mapping
#d2 = '.\s_data\PROCEDURES_ICD.csv.gz' 
#prod = procedures(d2,n,name)
#prod = limipiar_Codigos(prod)
#caluco de matrix de conteo

#prod_ipvot  = calculate_pivot_df(prod, real, level) #if normalize matrix == True
#prod_ipvot = normalize_count_matrix(prod_ipvot, level, )
#prod_ipvot

#prod_ipvot
#calcul de variable demos
#adm = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/MIMIC/ADMISSIONS.csv.gz"
#pa = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/MIMIC/PATIENTS.csv.gz"


#cat_considered = ['ADMITTIME','ADMISSION_TYPE', 'ADMISSION_LOCATION',
#                'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
#                'MARITAL_STATUS', 'ETHNICITY','DEATHTIME'] + ['DISCHTIME','SUBJECT_ID', 'HADM_ID']
#demos_pivot = calculate_agregacion_cl(adm,pa, categorical_cols, level,cat_considered,prod_ipvot)


#print(demos_pivot.isnull().sum()) #if variable logtransforamtio == true
#demos_pivot = apply_log_transformation(demos_pivot, 'L_1s_last_p1')
#merge demographic df with count matrix
#df_merge  = merge_df(demos_pivot, prod_ipvot)
#df_merge.shape
#if encoding categorical variables=true
#df_merge_e = encoding(df_merge, categorical_cols, 'onehot')
#df_merge_e


# if preprocesamiento final == True
#numerical_cols =  ['Age_max', 'LOSRD_sum',
#    'LOSRD_avg','L_1s_last_p1']
#df_merge_e.columns = df_merge_e.columns.astype(str)
#cols_to_normalize = [col for col in df_merge_e.columns if col not in ['SUBJECT_ID', 'HADM_ID']]



#df_final_prepo = preprocess(df_merge_e, "max", cols_to_normalize)
       
       
#d1 = '..\s_data\PRESCRIPTIONS.csv.gz'
#name1 = "DRUG"
#df_drugs = drugs(d1,name1)

#name_df = "raw_input.csv"
#name_encodeing = "input_onehot_encoding.csv"


#df_final = concat_input(df_drugs, df_diagnosis, df_procedures,numerical_cols,categorical_cols,name_df)
#df_final_encoded = encoding(df_final,categorical_cols)