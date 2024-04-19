
import sys
import os
current_directory = os.getcwd()
os.chdir("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning")
current_directory = os.getcwd()

print(current_directory)
sys.path.append('preprocessing')
sys.path.append('evaluation')
import config

#from evaluation.resemb.resemblance.metric_stat import *
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, f1_score
from xgboost import XGBClassifier  # Specifically
import time 

import pandas as pd 
import sys
import os
import os
import pickle
import gzip
sys.path.append('generative_models')
import os


def crear_syntheti_visit(df):

    try:
        # Paso 1: Identificar todos los id_paciente únicos
        unique_ids = df[681].unique()

        # Paso 2: Asignar un número máximo de consultas aleatorio para cada id_paciente
        np.random.seed(42)  # Para reproducibilidad de los resultados
        max_consultas = np.random.randint(1, 21, size=len(unique_ids))  # Números entre 1 y 20

        # Crear un diccionario para mapear cada id_paciente con su máximo de consultas
        id_to_max_consultas = dict(zip(unique_ids, max_consultas))

        # Paso 3: Mapear estos máximos al DataFrame original
        df['max_consultas'] = df[681].map(id_to_max_consultas)
    except: 
            # Paso 1: Identificar todos los id_paciente únicos
        unique_ids = df['id_patient'].unique()

        # Paso 2: Asignar un número máximo de consultas aleatorio para cada id_paciente
        np.random.seed(42)  # Para reproducibilidad de los resultados
        max_consultas = np.random.randint(1, 21, size=len(unique_ids))  # Números entre 1 y 20

        # Crear un diccionario para mapear cada id_paciente con su máximo de consultas
        id_to_max_consultas = dict(zip(unique_ids, max_consultas))

        # Paso 3: Mapear estos máximos al DataFrame original
        df['max_consultas'] = df['id_patient'].map(id_to_max_consultas)
    

    return df 


def concat_attributes(a,b):

    # Redimensiona 'b' para que tenga la misma forma que 'a' en los dos primeros ejes
    #b is attributes
    #a = np.random.rand(100, 244, 5)
    #b = np.random.rand(100, 2)

    b = np.repeat(b[:, :, np.newaxis], a.shape[2], axis=2)
    print(b.shape)
    print(a.shape)
    # Ahora 'b' es un array de forma (100, 244, 2)
    print(b.shape)  # Impri   me: (100, 244, 2)

    # Concatena 'a' y 'b' a lo largo del tercer eje
    c = np.concatenate((a, b), axis=1)

# Ahora 'c' es un array de forma (100, 244, 7)
    print(c.shape) 
    return c    


def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)
    
def get_valid_train_synthetic (path_features, attributes_path_train, features_path_train, features_path_valid, attributes_path_valid,synthetic_path_attributes,synthetic_path_features):
    attributes= load_data(path_features+attributes_path_train)
    features =load_data(path_features+features_path_train)

    features_valid = load_data(path_features+features_path_valid)
    attributes_valid=load_data(path_features+attributes_path_valid)


    synthetic_attributes = load_data(path_features+synthetic_path_attributes)
    synthetic_features=load_data(path_features+synthetic_path_features)

    config_w = {
            "preprocessed": "True",
            "max_sequence_len": 666,
            "sample_len": 111,
            "batch_size": 100,
            "epochs": 60
        }
    #wandb.init(project='SD_generation',config=config_w)


    features_v = features[:attributes_valid.shape[0]]
    attributes_v = attributes[:attributes_valid.shape[0]]   

        # Imprimir la ruta del directorio actual
    N, T, D = features.shape  


    print(features.shape)
    print(attributes.shape)
    print(synthetic_features.shape)
    print(synthetic_attributes.shape)
    print(attributes_valid.shape)

    #revisar que este productp sea cohere con primer debe ser 0
    total_features_synthethic = concat_attributes(synthetic_features, synthetic_attributes)
    total_fetura_valid = concat_attributes(features_valid, attributes_valid)
    total_features_train = concat_attributes(features_v, attributes_v)


    #aqui tengo que agregar una columnas que sea id_paciente y otra visit_rank

    print(total_features_synthethic.shape)
    print(total_fetura_valid.shape)
    print(total_features_train.shape)
    return total_features_synthethic, total_fetura_valid,total_features_train,attributes



def obtener_dataframe_inicial_denumpyarrray(total_features_synthethic, total_fetura_valid,total_features_train ):
    #funcion que crea dataframe a partir de numpy array
    rango_visitas = np.arange(1, 43)
    visit_rank = np.tile(rango_visitas, (total_features_synthethic.shape[0], 1, 1))
    #

    # Crear un array que vaya de 1 a total_features_synthethic.shape[0] + 1
    id_paciente = np.arange(1, total_features_synthethic.shape[0] + 1)
    # Darle la forma (total_features_synthethic.shape[0], 1, 1)
    id_paciente = id_paciente.reshape(-1, 1, 1)
    # Repetir este array para cada visita
    id_paciente = np.tile(id_paciente, (1, 1, 42))
    print(id_paciente.shape)  # Debería imprimir: (4496, 1, 42)

    #concatenar id_paciente y visit_rank
    total_features_synthethic = np.concatenate((total_features_synthethic, visit_rank, id_paciente), axis=1)
    total_fetura_valid = np.concatenate((total_fetura_valid, visit_rank,id_paciente), axis=1)
    total_features_train = np.concatenate((total_features_train, visit_rank,id_paciente), axis=1)
    print(total_features_synthethic.shape)
    print(total_fetura_valid.shape)
    print(total_features_train.shape)

    # Supongamos que df es tu DataFrame y 'visit_rank' es la última columna

    # Calcular el porcentaje de valores no ceros en cada fila
    lista_dfs_synthetic = []
    lista_dfs_train = []
    lista_dfs_valid = []
    for i in range(total_fetura_valid.shape[0]):
        # Convertir el subarray en un DataFrame y agregarlo a la lista
        df = pd.DataFrame(total_fetura_valid[i])
        lista_dfs_valid.append(df.T)
    for i in range(total_features_train.shape[0]):    
        df1 = pd.DataFrame(total_features_train[i])
        lista_dfs_train.append(df1.T)
    for i in range(total_features_synthethic.shape[0]):    
        df2 = pd.DataFrame(total_features_synthethic[i])
        lista_dfs_synthetic.append(df2.T)

    total_fetura_valid = pd.concat(lista_dfs_valid, ignore_index=True)
    total_features_train = pd.concat(lista_dfs_train, ignore_index=True)
    total_features_synthethic = pd.concat(lista_dfs_synthetic, ignore_index=True)


    print(total_features_synthethic.shape)
    print(total_fetura_valid.shape)
    print(total_features_train.shape)
    
    return total_features_synthethic,total_fetura_valid,total_features_train

def add_names_numpy_array(total_fetura_valid,total_cols):
    
    mapeo_nombres = {total_fetura_valid.columns[i]: total_cols[i] for i in range(len(total_cols))}
    total_fetura_valid = total_fetura_valid.rename(columns=mapeo_nombres)
    return total_fetura_valid   

def visitas_readmission(df):
    try:
        condiciones = [
            df['max_consultas'] == df['visit_rank'],
            df['max_consultas'] < df['visit_rank'],
            df['max_consultas'] > df['visit_rank']
        ]
        valores = [0, 1, -1]

        # Crear la nueva columna usando np.select
        df['readmission'] = np.select(condiciones, valores)
    except:
        condiciones = [
            df['max_consultas'] == df[680],
            df['max_consultas'] < df[680],
            df['max_consultas'] > df[680]
        ]
        valores = [0, 1, -1]

        # Crear la nueva columna usando np.select
        df['readmission'] = np.select(condiciones, valores)
    return df

def obtain_readmission_realdata(total_fetura_valid):
    # Ordenando el DataFrame por 'id_' y 'visit_rank'
    total_fetura_valid = total_fetura_valid.sort_values(by=['id_patient', 'visit_rank'])

    # Crear una nueva columna 'readmission'
    # Comparamos si el siguiente 'visit_rank' es mayor que el actual para el mismo 'id_'
    total_fetura_valid['readmission'] = total_fetura_valid.groupby('id_patient')['visit_rank'].shift(-1).notna().astype(int)  
    return  total_fetura_valid

def preprocess_data(total_cols,total_features_synthethic,total_cols1,total_fetura_valid,total_features_train):

   
    #crea un random visitas maximas
    total_features_synthethic = crear_syntheti_visit(total_features_synthethic)
    #agrega los nomrbres de columnas
    total_features_synthethic = add_names_numpy_array(total_features_synthethic,total_cols)
    #crea readmission columnas
    total_features_synthethic = visitas_readmission(total_features_synthethic)
    
    
    
    total_features_synthethic = total_features_synthethic[total_features_synthethic['readmission']!=-1]
    total_fetura_valid = total_fetura_valid.drop(columns=[300])
    total_features_train = total_features_train.drop(columns=[300])

    total_fetura_valid = total_fetura_valid[(total_fetura_valid[1]!=0.0)]
    total_features_train = total_features_train[(total_features_train[1]!=0.0)]

    total_fetura_valid = add_names_numpy_array(total_fetura_valid,total_cols1)
    total_features_train = add_names_numpy_array(total_features_train,total_cols1)

    #rbtener readmission
    total_fetura_valid = obtain_readmission_realdata(total_fetura_valid)
    total_features_train = obtain_readmission_realdata(total_features_train)
    # Imprimir la cantidad de ceros en cada fila
    print(total_fetura_valid.shape)
    
    return total_fetura_valid,total_features_train,total_features_synthethic
   