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

from xml.dom.pulldom import ErrorHandler
import pandas as pd
import dill
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import BRICS

def med_process(med_file):
    med_pd = pd.read_csv(med_file, dtype={"NDC": "category"})

    # med_pd.drop(columns=['ROW_ID','DRUG_TYPE','DRUG_NAME_POE','DRUG_NAME_GENERIC',
    #                     'FORMULARY_DRUG_CD','PROD_STRENGTH','DOSE_VAL_RX',
    #                     'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP', 'GSN', 'FORM_UNIT_DISP',
    #                     'ROUTE','ENDDATE','DRUG'], axis=1, inplace=True)
    med_pd.drop(
        columns=[
            "ROW_ID",
            "DRUG_TYPE",
            "DRUG_NAME_POE",
            "DRUG_NAME_GENERIC",
            "FORMULARY_DRUG_CD",
            "PROD_STRENGTH",
            "DOSE_VAL_RX",
            "DOSE_UNIT_RX",
            "FORM_VAL_DISP",
            "FORM_UNIT_DISP",
            "GSN",
            "FORM_UNIT_DISP",
            "ROUTE",
            "ENDDATE",
        ],
        axis=1,
        inplace=True,
    )
    #med_pd.drop(index=med_pd[med_pd["NDC"] == "0"].index, axis=0, inplace=True)
    med_pd.fillna(method="pad", inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd["ICUSTAY_ID"] = med_pd["ICUSTAY_ID"].astype("int64")
    med_pd["STARTDATE"] = pd.to_datetime(
        med_pd["STARTDATE"], format="%Y-%m-%d %H:%M:%S"
    )
    med_pd.sort_values(
        by=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE"], inplace=True
    )
    med_pd = med_pd.reset_index(drop=True)

    med_pd = med_pd.drop(columns=["ICUSTAY_ID"])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    return med_pd

def codeMapping2atc4(med_pd):
    RXCUI2atc4_file = "./data/RXCUI2atc4.csv"
    ndc2RXCUI_file="./data/ndc2RXCUI.txt"
    with open(ndc2RXCUI_file, "r") as f:
        ndc2RXCUI = eval(f.read())
    #med_pd["RXCUI"] = med_pd["NDC"].map(ndc2RXCUI)
    med_pd["RXCUI"] = med_pd["NDC"].apply(lambda x: ndc2RXCUI.get(x, np.nan))
    med_pd.dropna(inplace=True)

    RXCUI2atc4 = pd.read_csv(RXCUI2atc4_file)
    RXCUI2atc4 = RXCUI2atc4.drop(columns=["YEAR", "MONTH", "NDC"])
    RXCUI2atc4.drop_duplicates(subset=["RXCUI"], inplace=True)
    med_pd.drop(index=med_pd[med_pd["RXCUI"].isin([""])].index, axis=0, inplace=True)

    med_pd["RXCUI"] = med_pd["RXCUI"].astype("int64")
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(RXCUI2atc4, on=["RXCUI"])
    #med_pd.drop(columns=["NDC", "RXCUI"], inplace=True)
    med_pd["ATC3"] = med_pd["ATC4"].map(lambda x: x[:4])
    #med_pd = med_pd.rename(columns={"ATC4": "ATC3"})
    #med_pd["ATC4"] = med_pd["ATC4"].map(lambda x: x)
   
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd


def drug2(d1):
    med_pd = med_process(d1)
    RXCUI2atc4_file = "./data/RXCUI2atc4.csv"
    ndc2RXCUI_file="./data/ndc2RXCUI.txt"
    med_pd = codeMapping2atc4(med_pd)
    med_pd.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID', 'ATC3','ATC4','NDC'],inplace=True)
    # Supongamos que df es tu DataFrame
    med_pd["SUBJECT_ID"] = med_pd["SUBJECT_ID"].astype(str)
    med_pd["HADM_ID"] = med_pd["HADM_ID"].astype(str)
    return med_pd[['SUBJECT_ID', 'HADM_ID', 'ATC3','ATC4','NDC']]


def drugs1(d1,n,name):
    df_ = pl.read_csv(d1, infer_schema_length=10000, ignore_errors=True, )
    df_filtered = df_.with_columns(pl.col("SUBJECT_ID").cast(pl.Utf8))
    df_filtered = df_filtered.with_columns(pl.col("HADM_ID").cast(pl.Utf8))
    nuevo_df =df_filtered[["HADM_ID","SUBJECT_ID", "DRUG"]].to_pandas()
   
    print(nuevo_df.shape)
    nuevo_df.drop_duplicates( inplace=True)
    nuevo_df = funcion_acum(nuevo_df,n,name)
    return nuevo_df


#################################################################################################################################################################################################333
#funcion para concatenar archivo en el folde s_data con poalrs
def encoding2(res,categorical_cols):
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

    nuevos_datos = descocatenar_codes(txt,name)
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
    #plt.show()
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

import pandas as pd
import numpy as np
from scipy.stats import mode


def limipiar_Codigos(df,dt):   
    df = df.fillna(-1)
    
    return df
def max_patient_add(x):
    return x.max() - x.min()

def max_visit_add(x):
    return x - x.min()

'''def last_firs(ADMISSIONS):
    
    try: 
        ADMISSIONS[['ADMITTIME','DISCHTIME']] = ADMISSIONS[['ADMITTIME','DISCHTIME']].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
    except:
        ADMISSIONS[['ADMITTIME','DISCHTIME']] = ADMISSIONS[['ADMITTIME','DISCHTIME']].apply(lambda x: pd.to_datetime(x, infer_datetime_format=True))    
        
    ADMISSIONS["L_1s_last_p1"] = ADMISSIONS.groupby('SUBJECT_ID')['ADMITTIME'].transform(lambda x: x - x.min())
    return ADMISSIONS[["SUBJECT_ID","HADM_ID","L_1s_last_p1"]] 
'''

import pandas as pd

def last_firs(ADMISSIONS,level):
    try: 
            ADMISSIONS[['ADMITTIME','DISCHTIME']] = ADMISSIONS[['ADMITTIME','DISCHTIME']].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
    except:
            ADMISSIONS[['ADMITTIME','DISCHTIME']] = ADMISSIONS[['ADMITTIME','DISCHTIME']].apply(lambda x: pd.to_datetime(x, infer_datetime_format=True))    

    if level== "Patient":
                   
        ADMISSIONS["L_1s_last_p1"] = ADMISSIONS.groupby('SUBJECT_ID')['ADMITTIME'].transform(lambda x: x - x.min())
        ADMISSIONS["L_1s_last_p1"] =[int(i.days) for i in ADMISSIONS["L_1s_last_p1"]]

        
    else:    
    # Asegurar que ADMITTIME es tipo datetime
           
        # Ordenar por SUBJECT_ID y ADMITTIME para asegurar el orden cronológico de las visitas
        ADMISSIONS.sort_values(by=['SUBJECT_ID', 'ADMITTIME'], inplace=True)
        
        # Calcular la diferencia en tiempo hasta la visita anterior
        ADMISSIONS['L_1s_last_p1'] = ADMISSIONS.groupby('SUBJECT_ID')['ADMITTIME'].diff().dt.days
        
        # Llenar los valores NaN con cero
        ADMISSIONS['L_1s_last_p1'] = ADMISSIONS['L_1s_last_p1'].fillna(0)
        
    return ADMISSIONS[['SUBJECT_ID', 'HADM_ID', 'L_1s_last_p1']]

# Aplicar la función al DataFrame ADMISSIONS


def calculate_pivot_df(duplicados, real, level,type_g):
    """Calculate the pivot table.
    
    Args:
        duplicados (DataFrame): Pre-cleaned DataFrame used for the pivot table.
        real (str): Name of the columns.
        level (str): Indicates whether it's at patient level or visit level.
        
    Returns:
        pivot_df (DataFrame): The pivoted DataFrame.
    """
    if type_g == "drug2":
         duplicados["SUBJECT_ID"] = duplicados["SUBJECT_ID"].astype(str)
    if type_g == "diagnosis":
       
# Assuming `duplicados` is your DataFrame and `real` is the column you're trying to convert
       duplicados[real] = pd.to_numeric(duplicados[real], errors='coerce')
       duplicados[real] = duplicados[real].fillna(-18).astype(int)
    if type_g != "drug2":   
       duplicados[real] = duplicados[real].astype(int)
       
    duplicados["SUBJECT_ID"] = duplicados["SUBJECT_ID"].astype(str)
    duplicados["HADM_ID"] = duplicados["HADM_ID"].astype(str)

    if level == "Patient":
        pivot_df = duplicados[[real, "SUBJECT_ID"]].pivot_table(
            index='SUBJECT_ID',
            columns=real,
            aggfunc='size',
            fill_value=0
        )

    else:
        pivot_df = duplicados[[real, "SUBJECT_ID", "HADM_ID"]].pivot_table(
            index=['SUBJECT_ID', "HADM_ID"],
            columns=real ,
            aggfunc='size',
            fill_value=0
        )

    pivot_df.reset_index(inplace=True)
    return pivot_df

def calculate_agregacion_cl(adm,pa, categorical_cols, level,cat_considered,prod_ipvot):
    """Calculate the aggregated demographics DataFrame.
    
    Args:
        duplicados (DataFrame): Pre-cleaned DataFrame used for the pivot table.
        categorical_cols (list): List of categorical columns to include in aggregation.
        archivo: File containing additional data.
        level (str): Indicates whether it's at patient level or visit level.
        
    Returns:
        agregacion_cl (DataFrame): DataFrame with aggregated demographics.
    """
    adm = pd.read_csv(adm)
    aux_ad = last_firs(adm,level)
    
    pa = pd.read_csv(pa)
    adm["SUBJECT_ID"] = adm["SUBJECT_ID"].astype(str)
    adm["HADM_ID"] = adm["HADM_ID"].astype(str)
    aux_ad["SUBJECT_ID"] = aux_ad["SUBJECT_ID"].astype(str)
    aux_ad["HADM_ID"] = aux_ad["HADM_ID"].astype(str)

    pa["SUBJECT_ID"] = pa["SUBJECT_ID"].astype(str)




    duplicados = prod_ipvot.merge(adm[cat_considered], on=['SUBJECT_ID', 'HADM_ID'], how='left')
    duplicados=duplicados.merge(pa[['SUBJECT_ID', 'GENDER', 'DOB']], on=['SUBJECT_ID'], how='left')
    duplicados=duplicados.merge(aux_ad, on=['SUBJECT_ID','HADM_ID'], how='left')
    print(duplicados.shape)
    duplicados["DISCHTIME"] = pd.to_datetime(duplicados["DISCHTIME"])
    
    
    

    #duplicados['DOB'] = pd.to_datetime(duplicados['DOB'], format='%Y-%m-%d %H:%M:%S')
    duplicados['DEATHTIME'] = pd.to_datetime(duplicados['DEATHTIME'])
    duplicados['DEATHTIME'] = pd.to_datetime(duplicados['DEATHTIME'])
    duplicados['DOB'] = pd.to_datetime(duplicados['DOB'], format='%Y-%m-%d %H:%M:%S')
    duplicados['DOB'] = pd.to_datetime(duplicados['DOB'])
    duplicados['ADMITTIME'] = pd.to_datetime(duplicados['ADMITTIME'])
    duplicados["LOSRD"] = duplicados["DISCHTIME"] - duplicados["ADMITTIME"]
    duplicados['DOB'] = [timestamp.to_pydatetime() for timestamp in duplicados['DOB']]
    duplicados['ADMITTIME'] = [timestamp.date() for timestamp in duplicados['ADMITTIME']]
    duplicados['DOB'] = [timestamp.date() for timestamp in duplicados['DOB']]
    duplicados["age"] = (duplicados['ADMITTIME'].to_numpy() - duplicados['DOB'].to_numpy())
    duplicados["year_age"] = [i.days/365 for i in duplicados["age"]]

    
    duplicados.loc[duplicados['year_age'] > 100, 'year_age'] = 89  
        
    
    
    
    duplicados['LOSRD']  = [i.days for i in duplicados['LOSRD']]
    duplicados = duplicados[duplicados["LOSRD"]>0]   
    
    if level == "Patient":
        agregacion_cl = duplicados.groupby(['SUBJECT_ID']).agg(
            Age_max=("year_age", 'max'),
            GENDER=("GENDER", lambda x: mode(x)[0][0]),
            ADMISSION_LOCATION=("ADMISSION_LOCATION", lambda x: mode(x)[0][0]),
            ETHNICITY=('ETHNICITY', lambda x: mode(x)[0][0]),
            MARITAL_STATUS=("MARITAL_STATUS", lambda x: mode(x)[0][0]),
            RELIGION=("RELIGION", lambda x: mode(x)[0][0]),
            ADMISSION_TYPE=("ADMISSION_TYPE", lambda x: mode(x)[0][0]),
            INSURANCE=("INSURANCE", lambda x: mode(x)[0][0]),
            DISCHARGE_LOCATION=("DISCHARGE_LOCATION", lambda x: mode(x)[0][0]),
            LOSRD_sum=("LOSRD", 'sum'),
            LOSRD_avg=("LOSRD", np.mean),
            L_1s_last_p1=("L_1s_last_p1",  'max'),
            ADMITTIME_max=("ADMITTIME", 'max'),
            ADMITTIME_min=("ADMITTIME", 'min')
        )
    else:
        agregacion_cl = duplicados.groupby(['SUBJECT_ID', "HADM_ID"]).agg(
            Age_max=("year_age", 'max'),
            GENDER=("GENDER", lambda x: mode(x)[0][0]),
            ADMISSION_LOCATION=("ADMISSION_LOCATION", lambda x: mode(x)[0][0]),
            ETHNICITY=('ETHNICITY', lambda x: mode(x)[0][0]),
            MARITAL_STATUS=("MARITAL_STATUS", lambda x: mode(x)[0][0]),
            RELIGION=("RELIGION", lambda x: mode(x)[0][0]),
            ADMISSION_TYPE=("ADMISSION_TYPE", lambda x: mode(x)[0][0]),
            INSURANCE=("INSURANCE", lambda x: mode(x)[0][0]),
            DISCHARGE_LOCATION=("DISCHARGE_LOCATION", lambda x: mode(x)[0][0]),
            LOSRD_sum=("LOSRD", 'sum'),
            LOSRD_avg=("LOSRD", np.mean),
            L_1s_last_p1=("L_1s_last_p1",  'max'),
            ADMITTIME_max=("ADMITTIME", 'max'),
            ADMITTIME_min=("ADMITTIME", 'min')
        ).reset_index()
        
    #agregacion_cl["L_1s_last_p1"] = (agregacion_cl["ADMITTIME_max"] - agregacion_cl["ADMITTIME_min"]).apply(lambda x: x.days)
    #agregacion_cl["L_1s_last_p1"] =[int(i.days) for i in agregacion_cl["L_1s_last_p1"]]

    agregacion_cl = agregacion_cl[["Age_max", "LOSRD_sum", "L_1s_last_p1", "LOSRD_avg"] +categorical_cols+['SUBJECT_ID', 'HADM_ID']] 
    
    for i in ["Age_max", "LOSRD_sum", "L_1s_last_p1", "LOSRD_avg"]:
        agregacion_cl[i]= agregacion_cl[i].replace(np.nan,0)

    for col in categorical_cols:
        agregacion_cl[col]= agregacion_cl[col].replace(np.nan, "Unknown")
        agregacion_cl[col] = agregacion_cl[col].replace(0, 'Unknown')
        agregacion_cl[col] = agregacion_cl[col].replace('** INFO NOT AVAILABLE **', 'Unknown')
        agregacion_cl[col] = agregacion_cl[col].replace('UNKNOWN (DEFAULT)', 'Unknown')
        agregacion_cl[col] = agregacion_cl[col].replace('UNOBTAINABLE', 'Unknown')
        agregacion_cl[col] = agregacion_cl[col].replace('OTHER', 'Unknown')
        agregacion_cl[col] = agregacion_cl[col].replace('NOT SPECIFIED', 'Unknown')
        agregacion_cl = agregacion_cl.replace('UNKNOWN/NOT SPECIFIED', 'Unknown')

    return agregacion_cl

def merge_df(agregacion_cl, prod_ipvot):
    return agregacion_cl.merge(prod_ipvot, on=["HADM_ID",'SUBJECT_ID'], how='left')
    
import pandas as pd
import numpy as np

import numpy as np
import pandas as pd

def normalize_count_matrix(pivot_df, level):
    """
    Processes the count matrix by normalizing based on zero entries and then concatenates demographic data.
    
    :param pivot_df: DataFrame with count matrix of icd9-codes
    :param level: 'Patient', 'visit', or 'outs_visit'
    :return: DataFrame with normalized counts
    """
    # Drop identifiers based on the level
    if level == "Patient":
        matrix_df = pivot_df.drop('SUBJECT_ID', axis=1)
    else:
        matrix_df = pivot_df.drop(['HADM_ID', 'SUBJECT_ID'], axis=1)
    
    # Calculate the number of zero entries for each column
    num_zeros = (matrix_df == 0).sum()

    # Avoid division by zero by adding a small constant, if needed
    # This step depends on your specific requirements and data characteristics
    num_zeros += (num_zeros == 0)  # This will add 1 to columns with no zeros to avoid division by zero

    # Normalize each column by the number of zero entries
    normalized_matrix_df = matrix_df.div(num_zeros, axis=1)
    
    # Prepare the result DataFrame
    result_df = pd.concat([pivot_df[['SUBJECT_ID', 'HADM_ID']], normalized_matrix_df], axis=1)

    return result_df


def normalize_count_matrix__aux(pivot_df, level):
    """
    Processes the count matrix by normalizing and then concatenates demographic data.
    
    :param pivot_df: DataFrame with count matrix of icd9-codes
    :param stri: 'visit', 'Patient', or 'outs_visit'
    :param agregacion_cl: DataFrame with demographic data
    :param categorical_cols: List of categorical columns to include
    :return: Concatenated DataFrame with normalized counts and demographics
    """
    # Normalize the count matrix
    if level == "Patient":
        matrix = pivot_df.drop('SUBJECT_ID', axis=1).values
        
    else:
        matrix = pivot_df.drop(['HADM_ID','SUBJECT_ID'], axis=1).values
        
    num_non_zeros = np.count_nonzero(matrix)
    normalized_matrix = matrix / num_non_zeros  # Dividing the matrix by the number of non-zero elements
    
    # Create the result DataFrame
    result_df = pd.DataFrame(normalized_matrix, columns=pivot_df.columns.difference(['SUBJECT_ID', 'HADM_ID'], sort=False))
    if 'SUBJECT_ID' in pivot_df.columns:
        result_df['SUBJECT_ID'] = pivot_df['SUBJECT_ID']
    if 'HADM_ID' in pivot_df.columns:
        result_df['HADM_ID'] = pivot_df['HADM_ID']
    
    # Concatenate with demographic data
       
    return result_df

from sklearn.preprocessing import FunctionTransformer


from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

def encoding(res, categorical_cols, encoding_type='onehot', ):
    """
    Aplica codificación a las columnas categóricas de un DataFrame.

    Parámetros:
    - res: DataFrame original.
    - categorical_cols: Lista de columnas categóricas para codificar.
    - encoding_type: Tipo de codificación ('onehot' o 'label').
    - output_file_name: Nombre del archivo CSV para guardar el resultado.

    Retorna:
    - res_final: DataFrame con columnas categóricas codificadas.
    """

        
    for col in ['ADMISSION_TYPE',
    'ADMISSION_LOCATION',
    'DISCHARGE_LOCATION',
    'INSURANCE',
    'RELIGION',
    'ETHNICITY',
    ]:
            counts = res[col].value_counts(normalize=True)
            lower_80 = counts[counts.cumsum() > 0.8].index
            res[col] = res[col].replace(lower_80, 'Otra')
        
        # Aplicar codificación basada en el tipo especificado
    if encoding_type == 'onehot':
        encoder = OneHotEncoder()
        encoded_cols = encoder.fit_transform(res[categorical_cols])
        #encoded_cols_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names(categorical_cols))
        encoded_cols_df = pd.DataFrame(encoded_cols.toarray(), columns=encoder.get_feature_names_out(categorical_cols))

        # Concatenar el DataFrame original con el DataFrame codificado
    elif encoding_type == 'label':
        encoded_cols_df = res[categorical_cols].apply(LabelEncoder().fit_transform)

    # Concatenar el DataFrame original con el DataFrame codificado
    
    res_final = pd.concat([res[[i for i in res.columns if i not in categorical_cols]], encoded_cols_df], axis=1)

    
    return res_final



from sklearn.preprocessing import FunctionTransformer

def apply_log_transformation(merged_df, column_name):
    """
    Applies log transformation to a column in the DataFrame.

    :param merged_df: DataFrame resulting from process_and_concat function
    :param column_name: The name of the column to which log transformation will be applied
    :return: DataFrame with the log-transformed column
    """
    transformer = FunctionTransformer(np.log1p, validate=False)  # np.log1p handles log(0) by returning 0
    merged_df[f"{column_name}"] = transformer.transform(merged_df[[column_name]].values)
    
    # Replace -inf with 0 if there are any -inf values resulting from log(0)
    merged_df[f"{column_name}"] = merged_df[f"{column_name}"].replace(-np.inf, 0)
    
    return merged_df


categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                'MARITAL_STATUS', 'ETHNICITY','GENDER']


from sklearn.preprocessing import StandardScaler, MaxAbsScaler, PowerTransformer
import numpy as np
import pandas as pd

def preprocess(X, prep, columns_to_normalize):
    '''
    Normalizes specified numerical features in a DataFrame.
    
    Parameters:
    - X: DataFrame to preprocess.
    - prep: String indicating the preprocessing method: 'std', 'max', 'power'.
    - columns_to_normalize: List of column names to normalize.
    
    Returns:
    - X: DataFrame with preprocessed numerical features.
    '''
    
    if prep == "std":
        scaler = StandardScaler()
        X[columns_to_normalize] = scaler.fit_transform(X[columns_to_normalize])
        
    elif prep == "max":
        transformer = MaxAbsScaler()
        X[columns_to_normalize] = transformer.fit_transform(X[columns_to_normalize])
        
    elif prep == "power":
        pt = PowerTransformer()
        X[columns_to_normalize] = pt.fit_transform(X[columns_to_normalize])
        
    # No else case needed, if 'prep' is not one of the above, X is returned unchanged
    
    return X

# Ejemplo de uso
# X es tu DataFrame
# 'prep' es el método de preprocesamiento: 'std', 'max', 'power'
# 'columns_to_normalize' es una lista de las columnas numéricas que deseas normalizar
# X_preprocessed = preprocess(X, 'std', ['col1', 'col2', 'col3', 'col4'])