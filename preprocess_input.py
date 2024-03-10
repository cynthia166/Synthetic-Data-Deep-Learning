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
import polars as pl

import pandas as pd
from icdmappings import Mapper
import pandas as pd
import numpy as np
import pandas as pd
import plotly.express as px
import glob
import pandas as pd
from icdmappings import Mapper
import pandas as pd

import plotly.express as px

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

#from utils import getConnection
import polars as pl




#################################################################################################################################################################################################333
#funcion para concatenar archivo en el folde s_data con poalrs
def encoding(res,categorical_cols):
    # Identificar y reemplazar las categorías que representan el 80% inferior
    for col in categorical_cols:
        counts = res[col].value_counts(normalize=True)
        lower_80 = counts[counts.cumsum() > 0.8].index
        res[col] = res[col].replace(lower_80, 'Otra')

    # Aplicar One Hot Encoding
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_cols = encoder.fit_transform(res[categorical_cols])
    encoded_cols_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names(categorical_cols))

    # Concatenar el DataFrame original con el DataFrame codificado
    res_final = pd.concat([res[[i for i in res.columns if i not in categorical_cols]], encoded_cols_df], axis=1)



    for col in categorical_cols:
        print(res[col].unique())
        #res[col] = res[col].replace('Not specified', 'Otra')

    res_final.to_csv("generative_input/"+name_encodeing)      
    return res_final

# Lee todos los archivos con la extensión especificada
def concat_archivo_primeto(procedures,admi,ruta_archivos,save,nom_archivo):
    df = pl.read_csv(admi)
    df_filtered = df
 
    df_filtered = df_filtered.with_columns(pl.col("SUBJECT_ID").cast(pl.Utf8))
    df_filtered = df_filtered.with_columns(pl.col("HADM_ID").cast(pl.Utf8))
    archivos = glob.glob(ruta_archivos)
    archivos
    df_filtered.shape
    unique_subid = []

    for i in archivos:
        aux =  pl.read_csv(i,infer_schema_length=0
                        )
    
        try:
            aux = aux.with_columns(pl.col("SUBJECT_ID").cast(pl.Utf8))
            aux = aux.with_columns(pl.col("HADM_ID").cast(pl.Utf8))
            aux = aux.groupby(['SUBJECT_ID','HADM_ID'], maintain_order=True).all()
            #NOTA  SE ELIMIA ICUSTARY_ID del archivo 'dataset/ICUSTAYS.csv.gz' ya que esta duplicado"ICUSTAY_ID"
            aux = aux.select(pl.exclude("ROW_ID"))
            if i == 'dataset/ICUSTAYS.csv.gz':
                aux = aux.select(pl.exclude("ICUSTAY_ID"))
                
            elif i ==procedures:    
                aux = aux.select(pl.exclude("SEQ_NUM"))
            df_filtered=df_filtered.join(aux, on=['SUBJECT_ID','HADM_ID'], how="left")

            #df_filtered = pd.merge(df_filtered, aux, on=['SUBJECT_ID','HADM_ID'], how='left')
            print("concat, "+i)
        except:
            
                
            aux = aux.with_columns(pl.col("SUBJECT_ID").cast(pl.Utf8))
            aux = aux.filter(pl.col('SUBJECT_ID').is_in( ids["0"]))
            #aux = aux.groupby(['SUBJECT_ID'], maintain_order=True).all()
            
            aux = aux.select(pl.exclude("ROW_ID"))
            df_filtered=df_filtered.join(aux, on=['SUBJECT_ID'], how="left")
            unique_subid.append(i)
            print(i)
    if save == True:


# Ahora intenta escribir el archivo
        df_filtered.write_parquet(nom_archivo)
        print(df_filtered)     

# Assuming df1, df2, df3 are your dataframes
#df_drugs =pd.read_csv('./input_model_pred_drugs_u/ATC3_outs_visit_non_filtered.csv')
#df_diagnosis = pd.read_csv('./input_model_pred_diagnosis_u/CCS_CODES_diagnosis_outs_visit_non_filtered.csv')
#df_procedures = pd.read_csv('./input_model_visit_procedures/CCS CODES_proc_outs_visit_non_filtered.csv')
# Drop the columns categotical

#funcion para poder concatenar los 3 inputs, manteniendo las columasn del mayot



#n = nuevo_df["icd9_category"].unique()



#######parte del preprocesamiento###################


##########nulos##########

# Reemplazar 'Not specified' con 'Otra'



#from utils import getConnection



# Supongamos que ya tienes un DataFrame 'df' con las columnas 'subject_id', 'ham_id' y 'lista_codigos'
def descocatenar_codes(txt,name):
    # Lista para almacenar los datos del nuevo DataFrame
    nuevos_datos = []

    # Recorremos el DataFrame original 'df'
    for index, row in txt.iterrows():
        subject_id = row['SUBJECT_ID']
        ham_id = row['HADM_ID']
        lista_codigos = row['ICD9_CODE']

        # Verificamos si lista_codigos no es None
        if lista_codigos is not None:
            # Creamos un diccionario con los datos para una fila del nuevo DataFrame
            for codigo in lista_codigos:
                nuevo_registro = {
                    'SUBJECT_ID': subject_id,
                    'HADM_ID': ham_id,
                    'ICD9_CODE': codigo
                }

                # Agregamos el registro a la lista de nuevos datos
                nuevos_datos.append(nuevo_registro)

    # Creamos un nuevo DataFrame con los datos recopilados
    nuevo_df = pd.DataFrame(nuevos_datos)

    # Muestra el nuevo DataFrame resultante
    print(nuevo_df)
    nuevo_df = txt.dropna()
    return nuevo_df

def convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        if 'V' in value:
            return 18
        elif 'E' in value:
            return 19
        else:
            return None  # Replace "none" with "None"

def diagnosis(d1,n,name):
   
    df_ = pl.read_csv(d1)
    df_filtered = df_.with_columns(pl.col("SUBJECT_ID").cast(pl.Utf8))
    df_filtered = df_filtered.with_columns(pl.col("HADM_ID").cast(pl.Utf8))
    txt =df_filtered[["HADM_ID","SUBJECT_ID","ICD9_CODE"]].to_pandas()

    nuevos_datos = descocatenar_codes(txt)
    nuevo_df = codes_diag(nuevos_datos)

    nuevo_df = funcion_acum(nuevo_df,n,name)
    return nuevo_df

def procedures(d2,n,name):

    df_ = pl.read_csv(d2)
    df_filtered = df_.with_columns(pl.col("SUBJECT_ID").cast(pl.Utf8))
    df_filtered = df_filtered.with_columns(pl.col("HADM_ID").cast(pl.Utf8))
    nuevos_datos =df_filtered[["HADM_ID","SUBJECT_ID","ICD9_CODE"]].to_pandas()
    nuevo_df = nuevos_datos.dropna()
    nuevo_df = codes_diag(nuevos_datos)

    nuevo_df = funcion_acum(nuevo_df,n,name)
    return nuevo_df

def descocatenar_codes(txt,name):
    nuevos_datos = []
    for index, row in txt.iterrows():
        subject_id = row['SUBJECT_ID']
        ham_id = row['HADM_ID']
        lista_codigos = row[name]
        if lista_codigos is not None:
            for codigo in lista_codigos:
                nuevo_registro = {
                    'SUBJECT_ID': subject_id,
                    'HADM_ID': ham_id,
                   name: codigo
                }
                nuevos_datos.append(nuevo_registro)
    nuevo_df = pd.DataFrame(nuevos_datos)
    print(nuevo_df)
    nuevo_df = txt.dropna()
    return nuevo_df

def convert_to_int(value):
    try:
        return str(value)
    except ValueError:
        if 'V' in value:
            return 18
        elif 'E' in value:
            return 19
        else:
            return None

def codes_diag(nuevo_df):
    nuevo_df["ICD9_CODE"] = nuevo_df["ICD9_CODE"].apply(convert_to_int)
    nuevo_df["ICD9_CODE"] = nuevo_df["ICD9_CODE"].astype(str) 
    icd9codes = list(nuevo_df["ICD9_CODE"])
    mapper = Mapper()
    nuevo_df["CCS CODES"]  =  mapper.map(icd9codes, mapper='icd9toccs')
    nuevo_df["LEVE3 CODES"]  =  mapper.map(icd9codes, mapper='icd9tolevel3')
    return nuevo_df



#n = nuevo_df["icd9_category"].unique()


def cumulative_plot(icd9_codes, num_bins,threshold_value,cat):
    # Create a DataFrame with ICD-9 codes and their frequencies
    icd9_df = pd.DataFrame(icd9_codes, columns=['ICD-9 Code'])
    icd9_df['Frequency'] = icd9_df['ICD-9 Code'].map(icd9_df['ICD-9 Code'].value_counts())
    icd9_df= icd9_df.sort_values(by='Frequency', ascending=False)

    # Drop duplicate rows to get unique ICD-9 codes and their frequencies
    unique_icd9_df = icd9_df.drop_duplicates().sort_values(by='Frequency', ascending=False)

    # Calculate cumulative frequency percentage
    unique_icd9_df['Cumulative Frequency'] = unique_icd9_df['Frequency'].cumsum()
    total_frequency = unique_icd9_df['Cumulative Frequency'].iloc[-1]
    unique_icd9_df['Cumulative F percentage'] = unique_icd9_df['Cumulative Frequency'] / total_frequency

    # Create the plot using Matplotlib
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Adjust the figsize as needed

    # Histogram with fewer bins
      # Adjust the number of bins
    n, bins, patches = ax1.hist(icd9_df['ICD-9 Code'], bins=num_bins, color='blue', alpha=0.7)
    ax1.set_xlabel('ICD-9 Codes')
    ax1.set_ylabel('Frequency', color='blue')

    # Add a vertical dashed line at the threshold value

    threshold_x_value = unique_icd9_df[unique_icd9_df["Cumulative F percentage"] >= threshold_value].sort_values(by="Cumulative F percentage", ascending=False).iloc[-1]['ICD-9 Code']
    
    bins_before_threshold = unique_icd9_df[unique_icd9_df["Cumulative F percentage"] < threshold_value].sort_values(by="Cumulative F percentage", ascending=False)['ICD-9 Code'].nunique()
    bins_before_threshold_i = unique_icd9_df[unique_icd9_df["Cumulative F percentage"] < threshold_value].sort_values(by="Cumulative F percentage", ascending=False)['ICD-9 Code'].unique()

        
    ax1.axvline(x=threshold_x_value, color='red', linestyle='--', linewidth=2, label='Threshold: ' + str(threshold_value) )
    ax1.annotate(f'{bins_before_threshold} Bins Before Threshold', xy=(threshold_x_value, 0), xytext=(10, 20), textcoords='offset points', fontsize=10, color='red')

        
    
    #percentage = count / total_frequency * 100
    #ax1.annotate(f'\n{i}', xy=(bins[i] + (bins[i+1] - bins[i])/2, count), ha='center', va='bottom', fontsize=10)
        #ax1.annotate(f'{int(count)}', xy=(bins[i] + (bins[i+1] - bins[i])/2, count), ha='center', va='bottom', fontsize=10, )

    # Customize the plot
    ax1.set_title('Histogram of:' +str(cat))
    ax1.legend(loc='upper right')



    # Create a secondary y-axis for cumulative frequency
    ax2 = ax1.twinx()
    ax2.plot(unique_icd9_df['ICD-9 Code'], unique_icd9_df['Cumulative F percentage'], color='green', label='Cumulative Frequency')
    ax2.set_ylabel('Cumulative Frequency', color='green')
    ax2.legend(loc='upper right')

    legend_position = (1, .2)  # Adjust the position as needed
    ax1.legend(loc='upper right', bbox_to_anchor=legend_position)
    ax2.legend(loc='lower right', bbox_to_anchor=legend_position)

    # Hide x-axis tick labels
    ax1.set_xticks([])

    # Show the plot
    plt.tight_layout()  # Adjust layout for labels
    plt.show()
    return bins_before_threshold,bins_before_threshold_i


def asignar_valor(series, lista_especifica):
    # Usamos una comprensión de lista para asignar "Otro" a los valores que no están en la lista
    nueva_serie = series.apply(lambda x: x if x in lista_especifica else -1)
    return nueva_serie

def funcion_acum(nuevo_df,n,name):
    
    bins_before_threshold = []
    bins_before_threshold_index = []
    for i in n:
        aux_thresh = nuevo_df.copy()
        #aux_thresh = nuevo_df[nuevo_df["icd9_category"]==i]
        num_bins = len(aux_thresh[name].unique())
        icd9_codes = list(aux_thresh[name])
        a,b = cumulative_plot(icd9_codes, num_bins,i,i)
        bins_before_threshold.append(a)
        bins_before_threshold_index.extend(list(b))
        serie_original = nuevo_df[name]  
        lista_especifica = bins_before_threshold_index
        #lista_especifica = b

        # Llama a la función para asignar valores
        serie_modificada = asignar_valor(serie_original, lista_especifica)
        nuevo_df["threshold_"+str(i)] = serie_modificada
    return nuevo_df


d1 = '..\s_data\PRESCRIPTIONS.csv.gz'
name1 = "DRUG"
def drugs(d1,name1):
    df_ = pl.read_csv(d1, infer_schema_length=10000, ignore_errors=True, )
    df_filtered = df_.with_columns(pl.col("SUBJECT_ID").cast(pl.Utf8))
    df_filtered = df_filtered.with_columns(pl.col("HADM_ID").cast(pl.Utf8))
    nuevo_df =df_filtered[["HADM_ID","SUBJECT_ID", "DRUG"]].to_pandas()

    print(nuevo_df.shape)
    nuevo_df.drop_duplicates( inplace=True)
    nuevo_df = funcion_acum(nuevo_df,n,name1)
    return nuevo_df

