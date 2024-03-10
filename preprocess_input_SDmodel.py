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







    




def obtener_added_cols_targer_visitrank_(arhivo,name):
    df =  pd.read_csv(arhivo)

    # Limitar las fechas a un rango permitido

    # Ahora deberías poder restar las fechas sin problemas
    '''Funcion que regredsa si es tipo entire datafram donde se agregan ceros para que todos tengas mas visitas dataframe, o lista de listas que es el input de time gan'''
    #adm = pd.read_csv('./data/data_preprocess_nonfilteres.csv')
    ad_f = "ADMISSIONS.csv.gz"
    adm = pd.read_csv(ad_f)
 
    res = pd.merge(adm[["HOSPITAL_EXPIRE_FLAG","SUBJECT_ID","HADM_ID"]],df, on=["SUBJECT_ID","HADM_ID"], how='right')

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
    max_visits = res['visit_rank'].max()
    res['visit_rank'] = res['visit_rank'].astype(int)

    temporal_surv = [res[res['visit_rank'] == i] for i in range(1, max_visits + 1)]
    import pandas as pd
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
    
