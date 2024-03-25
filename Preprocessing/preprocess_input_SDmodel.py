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

name_df = "raw_input.csv"
archivo = ["aux/CCS CODESprocedures.csv","aux/ATC3drug2.csv","aux/CCS CODESdiagnosis.csv"]


def get_input_time( type_df,arhivo,name,cols_to_drop1,res,keywords,s):
# Obtener una lista de pacientes únicos
    unique_patients = res['SUBJECT_ID'].unique()

    # Calcular el 20% del total de pacientes únicos
    sample_size = int(s* len(unique_patients))
    # Obtener una muestra aleatoria del 20% de los pacientes únicos
    sample_patients = np.random.choice(unique_patients, size=sample_size, replace=False)
    # Filtrar el DataFrame para incluir solo los registros de los pacientes en la muestra
    sample_df = res[res['SUBJECT_ID'].isin(sample_patients)]

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
def concat_input(archivo,name_df):

    catnum_cols = cols_not_consideres(aux)
    catnum_cols = [i for i in catnum_cols if i not in ['SUBJECT_ID', 'HADM_ID']]
    drugs_cols = ['ADMISSION_TYPE_EMERGENCY', 'ADMISSION_TYPE_Otra',
        'ADMISSION_LOCATION_CLINIC REFERRAL/PREMATURE',
        'ADMISSION_LOCATION_EMERGENCY ROOM ADMIT', 'ADMISSION_LOCATION_Otra',
        'DISCHARGE_LOCATION_HOME', 'DISCHARGE_LOCATION_HOME HEALTH CARE',
        'DISCHARGE_LOCATION_Otra', 'DISCHARGE_LOCATION_SNF',
        'INSURANCE_Medicare', 'INSURANCE_Otra', 'RELIGION_CATHOLIC',
        'RELIGION_Otra', 'RELIGION_Unknown', 'MARITAL_STATUS_DIVORCED',
        'MARITAL_STATUS_LIFE PARTNER', 'MARITAL_STATUS_MARRIED',
        'MARITAL_STATUS_SEPARATED', 'MARITAL_STATUS_SINGLE',
        'MARITAL_STATUS_Unknown', 'MARITAL_STATUS_WIDOWED', 'ETHNICITY_Otra',
        'ETHNICITY_WHITE', 'GENDER_F', 'GENDER_M','Unnamed: 0', 'Age_max', 'LOSRD_sum', 'L_1s_last_p1', 'LOSRD_avg']

    df_procedures = pd.read_csv(archivo[0])
    df_diagnosis =pd.read_csv(archivo[2])
    df_drugs = pd.read_csv(archivo[1])

    df_drugs = df_drugs.drop(columns=drugs_cols)
    df_procedures = df_procedures.drop(columns=catnum_cols)

    #df_diagnosis = df_diagnosis.drop(columns=categorical_cols)


    rename_dict_d = {col: col + '_diagnosis' if col not in ["SUBJECT_ID", "HADM_ID"] else col for col in df_diagnosis.columns if col not in catnum_cols}
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
    result_final.to_csv("aux/"+ name_df)

# Assuming df is your DataFrame









    
numerical_cols =  ['Age_max', 'LOSRD_sum',
       'L_1s_last', 'LOSRD_avg','L_1s_last_p1']

categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                'MARITAL_STATUS', 'ETHNICITY','GENDER']


########onehot encoding y agrupacion de categoria de 80##
name_encodeing = "input_onehot_encoding.csv"
'''La primeras dor gunciones se corren si debo volver a correl el input 
obtener_added_cols_targer_visitrank_ este es el input del general
obtener_entire rste es el input para time series'''
def concat_input(df_drugs, df_diagnosis, df_procedures,numerical_cols,categorical_cols,name_df):
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
    res.to_csv("generative_input/"+ name_df)
    return res




def obtener_added_cols_targer_visitrank_(arhivo,name):
    df =  pd.read_csv(arhivo)

    # Limitar las fechas a un rango permitido

    # Ahora deberías poder restar las fechas sin problemas
    '''Funcion que regredsa si es tipo entire datafram donde se agregan ceros para que todos tengas mas visitas dataframe, o lista de listas que es el input de time gan'''
    #adm = pd.read_csv('./data/data_preprocess_nonfilteres.csv')
    ad_f = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/MIMIC/ADMISSIONS.csv.gz"
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
    res.to_csv("generative_input/"+name)    


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
    
