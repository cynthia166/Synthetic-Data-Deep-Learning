
import joblib 
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from model_eval import *
#from yellowbrick.cluster import KElbowVisualizer

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
# Davies Bouldin score for K means
from sklearn.metrics import davies_bouldin_score
from sklearn import metrics
import re
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import PowerTransformer

import ast
import numpy as np
import os


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder



def string_list(x):
    '''funcion que se convierte una lista de string a una lista normal'''
    '''Input '["\'44\'", "\'44\'", "\'50\'", "\'193\'", "\'222\'", "\'222\'", "\'222\'"]'
    x: string as list
    Output
    list: list'''

    try:
        lista = ast.literal_eval(x)
    except:
        lista = np.nan
    return lista

def solo_numeros(x):
        '''funcion que se convierte una lista de string a una lista normal'''
        '''Input ["\'44\'", "\'44\'", "\'50\'", "\'193\'", "\'222\'", "\'222\'", "\'222\'"]
        x: list of strings
        Output
        list: list of string latter formar'''
        
        try: 
            lis = x.split(",")
            r = [lis[i][3:-2] for i in range(len(lis))]
        except:
            r = None
        return r
    
def ultimo_num(x):
    '''funcion que se convierte una lista de string a una lista normal'''
    '''Input ["\'44\'", "\'44\'", "\'50\'", "\'193\'", "\'222\'", "\'222\'", "\'222\'"]
    x: list of strings
    Output
    list: list of string latter formar'''

    try: 

        r = [x[i] if i!= len(x)-1 else x[-1][:-1] for i in range(len(x))]
    except:
        r = None
    return r


#filtrar funciones
def filtrar_dataframe(df,name):
    '''function that filters the subject id isin
    Input
    name: name of the isin to filter
    Output
    list: the dataframe that watns to be filtered'''
    
    lis = pd.read_csv(name)
    condition = df['SUBJECT_ID'].isin(lis)

    # Apply the condition to filter the DataFrame
    nuevo_aux = df[condition]
    return nuevo_aux


#mode
def mode(x):
    '''function to obtain the mode'''
    return x.mode()[0]



#Actual Clustering
def cluster_scores(num,X,method):
    '''function that train  model k_means with num number of clusters it train model 10 times
    and average the metrics
    Input
    X: the input features, that will be clustered
    num: number of clusters
    Output
    silhouette_avg: it tell you the quality of the clustering, the compactnes and how well is the separation with other clusters
    davies_bouldin_avg: minimize this metric
    kmeans_labels: labels of the model
    kmeans:last model that was trained
    nuevo_df2_gen: the dataframe descocatenated
    '''    
    lis_silhouette_avg  =[]
    lis_score = []
    for i in range(2):
        if method == "kmeans":
           kmeans = KMeans(n_clusters=num).fit(X)
        else:
           kmeans = AgglomerativeClustering(n_clusters=num).fit(X)    
        kmeans_labels = kmeans.labels_
        #model = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, kmeans_labels)
        lis_silhouette_avg.append(silhouette_avg)
        score = davies_bouldin_score(X, kmeans_labels)
        lis_score.append(score)    
    silhouette_avg = np.mean(lis_silhouette_avg)
    davies_bouldin_avg = np.mean(lis_score)
    return silhouette_avg,davies_bouldin_avg,kmeans_labels,kmeans

def desconacat_codes(df,real,filtered):
    '''function that  deconcatenate the icd9-codes list and the result is a list  to,subject_id, admission_id and the respective icd9-code
    one per row    
    Input
    df: the input that wants to be desconcatenated
    real: name of the columns of icd9-codes
    Output
    nuevo_df2_gen:el dataframe desconcatenado'''
    if real == 'CCS CODES_proc':
        #'["\'44\'", "\'44\'", "\'50\'", "\'193\'", "\'222\'", "\'222\'", "\'222\'"]'
        name_1 =  real


        df[name_1 + "_preprocess"] = df[real ].apply(  lambda x: ultimo_num(x))    
            
        df[name_1 + "_preprocess"] = df[name_1].apply(  lambda x: solo_numeros(x))
        
        
        name1 = real + "_preprocess"
        txt =df[["HADM_ID","SUBJECT_ID",name1]]
        txt = txt.dropna()
        #txt.fillna(value = "NO DRUG",
        #        inplace = True)
        # Supongamos que ya tienes un DataFrame 'df' con las columnas 'subject_id', 'ham_id' y 'lista_codigos'

        # Lista para almacenar los datos del nuevo DataFrame
        nuevos_datos = []

        # Recorremos el DataFrame original 'df'
        for index, row in txt.iterrows():
            subject_id = row['SUBJECT_ID']
            ham_id = row['HADM_ID']
            lista_codigos = row[name1]
            lista_codigos=[elemento for elemento in lista_codigos if elemento is not None]

            # Creamos un diccionario con los datos para una fila del nuevo DataFrame
            for codigo in lista_codigos:
                nuevo_registro = {
                    'SUBJECT_ID': subject_id,
                    'HADM_ID': ham_id,
                    name1: codigo
                }

                # Agregamos el registro a la lista de nuevos datos
                nuevos_datos.append(nuevo_registro)

        # Creamos un nuevo DataFrame con los datos recopilados
        nuevo_df2 = pd.DataFrame(nuevos_datos)

        # Muestra el nuevo DataFrame resultante
        print(nuevo_df2) 
        nuevo_df2_gen=nuevo_df2.copy()
    elif real == 'cat_threshold .95 most frequent_proc':
        #"['3613', '3615', '3961', '8872', '9904', '9905', '9907']"
        nm2 = real + "_preprocess"
        name2 =  real

        df[name2 + "_preprocess"] = df[name2 ].apply(  lambda x: string_list(x))

        #df1[name2 + "_preprocess"] = df1[real]

        
        txt =df[["HADM_ID","SUBJECT_ID",nm2]]

        #txt[name2] = txt[name2].astype(int)
        txt = txt.dropna()
        #txt.fillna(value = "NO DRUG",
                #inplace = True)
        # Supongamos que ya tienes un DataFrame 'df' con las columnas 'subject_id', 'ham_id' y 'lista_codigos'

        # Lista para almacenar los datos del nuevo DataFrame
        nuevos_datos = []

        # Recorremos el DataFrame original 'df'
        for index, row in txt.iterrows():
            subject_id = row['SUBJECT_ID']
            ham_id = row['HADM_ID']
            lista_codigos = row[nm2]
            lista_codigos=[elemento for elemento in lista_codigos if elemento is not None]

            # Creamos un diccionario con los datos para una fila del nuevo DataFrame
            for codigo in lista_codigos:
                nuevo_registro = {
                    'SUBJECT_ID': subject_id,
                    'HADM_ID': ham_id,
                    nm2: codigo
                }

                # Agregamos el registro a la lista de nuevos datos
                nuevos_datos.append(nuevo_registro)

        # Creamos un nuevo DataFrame con los datos recopilados
        nuevo_df3 = pd.DataFrame(nuevos_datos)
        nuevo_df3[real+"_preprocess"] = nuevo_df3[real+"_preprocess"].replace("Otro",-1)

        # Muestra el nuevo DataFrame resultante
        print(nuevo_df3)
        nuevo_df2_gen=nuevo_df3.copy()
    elif real == "ICD9_CODE_procedures" :
        #'[3613, 3615, 3961, 8872, 9904, 9905, 9907]'
        name2_ =  "ICD9_CODE_procedures" 
        #name2_ =  "ICD9_CODE"
        nm2 = name2_ + "_preprocess"
        df[name2_ + "_preprocess"] = df[name2_ ].apply(  lambda x: string_list(x))
        #df[name2_ + "_preprocess"] = df[name2_ ]

        txt =df[["HADM_ID","SUBJECT_ID",nm2]]

        txt = txt.dropna()
        #txt[name2] = txt[name2].astype(int)
        # se remplaza los nulos ya que 
        #txt.fillna(value = "NO DRUG",
        #        inplace = True)
        # Supongamos que ya tienes un DataFrame 'df' con las columnas 'subject_id', 'ham_id' y 'lista_codigos'

        # Lista para almacenar los datos del nuevo DataFrame
        nuevos_datos = []

        # Recorremos el DataFrame original 'df'
        for index, row in txt.iterrows():
            subject_id = row['SUBJECT_ID']
            ham_id = row['HADM_ID']
            lista_codigos = row[nm2]
            lista_codigos=[elemento for elemento in lista_codigos if elemento is not None]

            # Creamos un diccionario con los datos para una fila del nuevo DataFrame
            for codigo in lista_codigos:
                nuevo_registro = {
                    'SUBJECT_ID': subject_id,
                    'HADM_ID': ham_id,
                    name2_: codigo
                }

                # Agregamos el registro a la lista de nuevos datos
                nuevos_datos.append(nuevo_registro)

        # Creamos un nuevo DataFrame con los datos recopilados
        nuevo_df4 = pd.DataFrame(nuevos_datos)

        # Muestra el nuevo DataFrame resultante
        print(nuevo_df4) 
        nuevo_df2_gen=nuevo_df4.copy()

    else:
        if filtered == True:
    #"['41401', '4111', '4241', 'V4582', '2724', '4019', 'Otro', '3899']"    
            nm2 = real + "_preprocess"
            name2 =  real

            

            df[name2 + "_preprocess"] = df[real]

            
            txt =df[["HADM_ID","SUBJECT_ID",nm2]]

            #txt[name2] = txt[name2].astype(int)
            txt = txt.dropna()
            #txt.fillna(value = "NO DRUG",
            #        inplace = True)
            # Supongamos que ya tienes un DataFrame 'df' con las columnas 'subject_id', 'ham_id' y 'lista_codigos'

            # Lista para almacenar los datos del nuevo DataFrame
            nuevos_datos = []

            # Recorremos el DataFrame original 'df'
            for index, row in txt.iterrows():
                subject_id = row['SUBJECT_ID']
                ham_id = row['HADM_ID']
                lista_codigos = row[nm2]
                lista_codigos=[elemento for elemento in lista_codigos if elemento is not None]

                # Creamos un diccionario con los datos para una fila del nuevo DataFrame
                for codigo in lista_codigos:
                    nuevo_registro = {
                        'SUBJECT_ID': subject_id,
                        'HADM_ID': ham_id,
                        nm2: codigo
                    }

                    # Agregamos el registro a la lista de nuevos datos
                    nuevos_datos.append(nuevo_registro)

            # Creamos un nuevo DataFrame con los datos recopilados
            nuevo_df3 = pd.DataFrame(nuevos_datos)
            nuevo_df3[real+"_preprocess"] = nuevo_df3[real+"_preprocess"].replace("Otro",-1)

            # Muestra el nuevo DataFrame resultante
            print(nuevo_df3)
            nuevo_df2_gen=nuevo_df3.copy()
        else:
            nm2 = real + "_preprocess"
            name2 =  real

            

            df[name2 + "_preprocess"] = df[real]

            
            txt =df[["HADM_ID","SUBJECT_ID",nm2]]

            #txt[name2] = txt[name2].astype(int)
            nuevo_df2_gen = txt.dropna()
            #txt.fillna(value = "NO DRUG",
            #        inplace = True)

                
    return nuevo_df2_gen
            
            
                    
                    
#leans admission, duplication, negative values LORS, only valid icd9codes are obtaines.

def demo_ad(categorical_cols,aux_demo):
    '''Function that encodes categorical features
    Input
    categorcial_cols: list of cols tu encode
    aux_demo: df that will be encoded
    Output:
    aux_demo: df encoded'''
    
            
    aux_demo[["MARITAL_STATUS","RELIGION"]] = aux_demo[["MARITAL_STATUS","RELIGION"]].fillna('Unknown')
    columnTransformer = ColumnTransformer([(OrdinalEncoder(),  LabelEncoder(), [0])], remainder='passthrough')

    enc = OrdinalEncoder()
    enc.fit(aux_demo[categorical_cols])
    aux_demo[categorical_cols] = enc.transform(aux_demo[categorical_cols])
    return aux_demo

def merge_df(df,nuevo_df2_gen,nuevo_df4,real,categorical_cols):
    '''function concatenates the created, desconcatenated datafrane, and the desconcatenated icd9 original codes  
    it also cleans the codes as there are some no numeric values as "NO DRUG" was added to identify does that di no have
    The admision that are kept are the ones that have lenght of stay(discharge time-admitted time) positive, 
    the columns that are duplocated are eliminated, in "CCS CODES_proc there was comma left
    Input
    df: the original input data_preprocess.csv
    nuevo_df2_gen: the dataframe desconcatenated
    nuevo_df4: icd9 codes original desconcatenated
    real:name of the file/classification that is being considered
    Output
    duplicados: without duplications, los postivie, numerical icd9codes.
    '''

    
    nuevo_aux = pd.concat([nuevo_df2_gen, nuevo_df4["ICD9_CODE_procedures"]],axis = 1)

  
    print(nuevo_df2_gen.shape)
    print(nuevo_df4.shape)
    print(nuevo_aux.shape)
    print("Unicos Subjects",nuevo_aux["SUBJECT_ID"].nunique())
    
    df = demo_ad(categorical_cols,df)
                
   
    merge_atc = pd.merge(df[categorical_cols+['ICD9_CODE_procedures',
        'EXPIRE_FLAG', 'DOB', 'SUBJECT_ID',
        'HADM_ID', 'ADMITTIME', 'age', 'year_age', "ADMITTIME","LOSRD"]], nuevo_aux, on=["HADM_ID","SUBJECT_ID"], how='right')

    if real == "cat_threshold .95 most frequent_proc":
        name = name2 = real
        merge_atc[name2 +'_preprocess']= merge_atc[name2 +'_preprocess'].replace("Otro", -1)
        merge_atc[name2 + "_preprocessv1"] = merge_atc[name2 + "_preprocess"].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        merge_atc = merge_atc[merge_atc[name2 + "_preprocessv1"].notnull()]
        
    elif real == "CCS CODES_proc":
        name = real 
        merge_atc[name + "_preprocess"]=[item.replace("'", '') for item in merge_atc["CCS CODES_proc_preprocess"]]
     
        merge_atc[name + "_preprocessv1"] = merge_atc[name + "_preprocess"].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        merge_atc = merge_atc[merge_atc[name + "_preprocessv1"].notnull()]
        
    elif real == "ICD9_CODE_procedures":    
        name = real
        merge_atc=merge_atc.loc[:, ~merge_atc.columns.duplicated(keep='last')]
        merge_atc = merge_atc.rename(columns=lambda x: x.rstrip('_x'))
        merge_atc[name + "_preprocess_aux"] = merge_atc[name+"_y"].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        merge_atc = merge_atc[merge_atc[name + "_preprocess_aux"].notnull()]
        merge_atc
   
    else: 
        name2 = real
            
        merge_atc[name2 + "_preprocessv1"] = merge_atc[name2 + "_preprocess"].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        merge_atc = merge_atc[merge_atc[name2 + "_preprocessv1"].notnull()]

            

    #aux = merge_atc.copy()
    print(merge_atc.shape)
    print(df.shape)
    
    
    #DUPLICADOS
    if real =="ICD9_CODE_procedures":
        def replace_negative_with_zero(x):
             return max(0, x)
        merge_atc = merge_atc[merge_atc["LOSRD"]>0]
        #merge_atc["LOSRD"] = merge_atc["LOSRD"].apply(replace_negative_with_zero)
    else:
        merge_atc = merge_atc[merge_atc["LOSRD"]>0]
            
            

# Aplicar la función a cada elemento en la serie

    

    #duplicados = merge_atc.drop_duplicates(subset=list(merge_atc.columns))
    duplicados = merge_atc.copy()
    print(merge_atc.shape)
    print("Duplicados",duplicados.SUBJECT_ID.nunique())
    duplicados=duplicados.loc[:, ~duplicados.columns.duplicated(keep='first')]
    duplicados['ADMITTIME'] = pd.to_datetime(duplicados["ADMITTIME"])
    return duplicados


def max_patient_add(x):
    return x.max() - x.min()

def max_visit_add(x):
    return x - x.min()

def last_firs(archivo):
    ADMISSIONS = pd.read_csv(archivo)
    try: 
        ADMISSIONS[['ADMITTIME','DISCHTIME']] = ADMISSIONS[['ADMITTIME','DISCHTIME']].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
    except:
        ADMISSIONS[['ADMITTIME','DISCHTIME']] = ADMISSIONS[['ADMITTIME','DISCHTIME']].apply(lambda x: pd.to_datetime(x, infer_datetime_format=True))    
        
    ADMISSIONS["L_1s_last"] = ADMISSIONS.groupby('SUBJECT_ID')['ADMITTIME'].transform(lambda x: x - x.min())
    return ADMISSIONS[["SUBJECT_ID","HADM_ID","L_1s_last"]] 

def pivotm(duplicados,real,stri,categorical_cols,archivo):
    '''It creates the pivot table by patient or visit, it is considered the columns, according the mapping of procedure
    Patient consideres the max of the age, mode gender, mode location, it takes in account the maximum of los and the mean
    it obtains the time bewteen the first visits an the las visit
    Input
    duplicados: dataframe, previosly pre cleaned which will be used for for the pivot table /icd9codes desconcatenated
    real: name of the columns of icd9-codes
    stri: if its patient level or visit level
    Output agregcacion_cl
    pivot_df:el dataframe desconcatenado
    agregacion_cl: dataframe with demographics aggreagated'''

    if stri == "Patient":
        agregacion_cl =duplicados.groupby(['SUBJECT_ID']).agg(

        Age_max=("year_age", 'max'),  GENDER=("GENDER", mode), ADMISSION_LOCATION=("ADMISSION_LOCATION", mode), ETHNICITY=('ETHNICITY', mode)
        ,MARITAL_STATUS=("MARITAL_STATUS", mode),RELIGION=("RELIGION", mode), ADMISSION_TYPE=("ADMISSION_TYPE", mode),INSURANCE=("INSURANCE", mode),DISCHARGE_LOCATION=("DISCHARGE_LOCATION", mode),

        LOSRD_sum=("LOSRD", 'sum'),LOSRD_avg=("LOSRD", np.mean),L_1s_last=("ADMITTIME", max_patient_add),ADMITTIME_max=("ADMITTIME", 'max'),ADMITTIME_min=("ADMITTIME", 'min'))

        agregacion_cl["L_1s_last"] = agregacion_cl["ADMITTIME_max"]-agregacion_cl["ADMITTIME_min"] 
        #aux = last_firs(archivo)
        #agregacion_cl = pd.merge(agregacion_cl, aux, on=["SUBJECT_ID"], how='left')
       
        agregacion_cl["L_1s_last"] =[int(i.days) for i in agregacion_cl["L_1s_last"]]

        agregacion_cl = agregacion_cl[["Age_max","LOSRD_sum","L_1s_last","LOSRD_avg"]+categorical_cols]
        
        name = real
        if real == 'cat_threshold .95 most frequent_proc':
        
            pivot_df = duplicados[[name + "_preprocessv1","SUBJECT_ID"]].pivot_table(index='SUBJECT_ID', columns=name + "_preprocessv1", aggfunc='size', fill_value=0)
        elif real == 'cat_threshold .999 most frequent':
            pivot_df = duplicados[[name + "_preprocessv1" ,"SUBJECT_ID"]].pivot_table(index='SUBJECT_ID', columns=name + "_preprocessv1", aggfunc='size', fill_value=0)

        elif real ==  "ICD9_CODE_procedures":
            pivot_df = duplicados[[name + "_y","SUBJECT_ID"]].pivot_table(index='SUBJECT_ID', columns=name + "_y", aggfunc='size', fill_value=0) 
        else:
            pivot_df = duplicados[[name + "_preprocessv1","SUBJECT_ID"]].pivot_table(index='SUBJECT_ID', columns=name+ "_preprocessv1" , aggfunc='size', fill_value=0)


        print(agregacion_cl.shape)
    #elif stri =="visit":
    else:
            agregacion_cl =duplicados.groupby(['SUBJECT_ID',"HADM_ID"]).agg(

            Age_max=("year_age", 'max'),  GENDER=("GENDER", mode), ADMISSION_LOCATION=("ADMISSION_LOCATION", mode),  ETHNICITY=('ETHNICITY', mode)
            ,MARITAL_STATUS=("MARITAL_STATUS", mode),RELIGION=("RELIGION", mode), ADMISSION_TYPE=("ADMISSION_TYPE", mode),INSURANCE=("INSURANCE", mode),DISCHARGE_LOCATION=("DISCHARGE_LOCATION", mode),


            LOSRD_sum=("LOSRD", 'sum'),LOSRD_avg=("LOSRD", np.mean),ADMITTIME_max=("ADMITTIME", 'max'),).reset_index()
          
            #agregacion_cl["L_1s_last"] = agregacion_cl["ADMITTIME_max"]-agregacion_cl["ADMITTIME_min"] 
            aux = last_firs(archivo)
            agregacion_cl = pd.merge(agregacion_cl, aux, on=["HADM_ID","SUBJECT_ID"], how='left')
       
            agregacion_cl["L_1s_last"] =[int(i.days) for i in agregacion_cl["L_1s_last"]]

            
            agregacion_cl = agregacion_cl[["HADM_ID","SUBJECT_ID","Age_max","LOSRD_sum","LOSRD_avg","L_1s_last"]+categorical_cols]

            print(agregacion_cl.shape)
        
            name = real
            
            if real == 'cat_threshold .95 most frequent_proc':
                
                pivot_df = duplicados[[name + "_preprocessv1","SUBJECT_ID","HADM_ID"]].pivot_table(index=['SUBJECT_ID',"HADM_ID"], columns=name + "_preprocessv1", aggfunc='size', fill_value=0)
            elif real == 'cat_threshold .999 most frequent':
                pivot_df = duplicados[[name + "_preprocessv1" ,"SUBJECT_ID","HADM_ID"]].pivot_table(index=['SUBJECT_ID',"HADM_ID"], columns=name + "_preprocessv1", aggfunc='size', fill_value=0)

            elif real ==  "ICD9_CODE_procedures":
                pivot_df = duplicados[[name + "_y","SUBJECT_ID","HADM_ID"]].pivot_table(index=['SUBJECT_ID',"HADM_ID"], columns=name + "_y", aggfunc='size', fill_value=0) 
            else:
                pivot_df = duplicados[[name + "_preprocessv1","SUBJECT_ID","HADM_ID"]].pivot_table(index=['SUBJECT_ID',"HADM_ID"], columns=name+ "_preprocessv1" , aggfunc='size', fill_value=0)
                

    # Reset the index if needed
    pivot_df.reset_index(inplace=True)
    print(pivot_df.shape)
    return pivot_df, agregacion_cl

def label_fun(days):
    '''Funcion que obtiene las readmissiones que son mayores a un numero de dias '''
    '''input
    days: numero de dias los cuales se consideran como una readmission
    return: regresa un dataframe donde se tiene los intervalos de visita y ultima visitam como el rankeo de las visitas y las etiqueta de 
    readimission.
    '''
    # Admissions 
    ADMISSIONS = pd.read_csv("/Users/cgarciay/Desktop/Laval_Master_Computer/research/data_preprocess.csv")


    #the dataframe is ranked, by admission time, 
    a = ADMISSIONS[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME']]
    a['ADMITTIME_RANK'] = a.groupby('SUBJECT_ID')['ADMITTIME'].rank(method='first')

    # The last admission date is obtained
    b = a.groupby('SUBJECT_ID')['ADMITTIME_RANK'].max().reset_index()
    b.rename(columns={'ADMITTIME_RANK': 'MAX_ADMITTIME_RANK'}, inplace=True)

    #Each rank matches the max rank
    readmit_df = pd.merge(a, b, on='SUBJECT_ID', how='left')

    # date types are changed
    readmit_df[['ADMITTIME','DISCHTIME']] = readmit_df[['ADMITTIME','DISCHTIME']].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
    # readmission is sorted descending considering the admission time

    readmit_df = readmit_df.sort_values(by=['SUBJECT_ID', 'ADMITTIME'])


    # the label is shifted one, to obtain the next  amission addtime, the fist admission time is appended in the end
    next_admittime= readmit_df['ADMITTIME'].values[1:]
    # The fist label is appende in the last to have the  same dimension but it does not matter because the las label is the max of the subject so it will no be taken into account
    next_admittime = np.append(next_admittime, next_admittime[0])
    readmit_df['NEXT_ADMITTIME'] = next_admittime
    readmit_df['NEXT_ADMITTIME'].shape

    # we star with label as -1

    readmit_df[days +'_READMIT'] = -1
    # if the rank is equal to the max the its considered as las admission and there is no readmission
    readmit_df.loc[readmit_df['ADMITTIME_RANK'] == readmit_df['MAX_ADMITTIME_RANK'], days+'_READMIT'] = 0
    # to analyse does that are possible readmission we filter the ones left -1
    readmit_sub_df = readmit_df[readmit_df[days +'_READMIT']==-1]
    print(readmit_sub_df.shape)

    # the interval of day between next admission and the actual discharfe time is obatain
    interval = (readmit_sub_df['NEXT_ADMITTIME'] - readmit_sub_df['DISCHTIME']).dt.days.values

    # if it lower than n days then it would be readmission 0 otherwise
    readmit_df.loc[readmit_df[days+'_READMIT']==-1, 'INTERVAL'] = interval 

    readmit_df.loc[readmit_df['INTERVAL']<=int(days), days+'_READMIT'] = 1
    readmit_df.loc[readmit_df['INTERVAL']>int(days),  days+'_READMIT'] = 0
    readmit_df.to_csv("/Users/cgarciay/Desktop/Laval_Master_Computer/research/y_readmission_label/label_"+days+"j.csv")
    return readmit_df


def firs_preprocesing_aux(pivot_df,stri,agregacion_cl,categorical_cols):
    '''It is used to save the features for the preiction and the matrix (mutual informariton
    anda rand score). No columns is dropped as it need to be considered to merge with the label(readmission for the predictions)
    Input
    pivot_df: dataframe which is the count matrix of icd9-codes
    agregacion_cl:  dataframe demographics
    stri: visit, patient, outs_vist the only one that are considere are visit and out_vitis
    Output
    X: dataframe with the first preprocess features
    '''
    if stri =="visit":
        drop_l = list(pivot_df.reset_index().columns)[1:3]
        matrix = pivot_df.drop(drop_l, axis=1).values
    elif stri == "Patient":
        matrix = pivot_df.drop('SUBJECT_ID', axis=1).values
    elif stri == "outs_visit":
        drop_l = list(pivot_df.reset_index().columns)[1:3]
        matrix = pivot_df.drop(drop_l, axis=1).values

        

            

    num_non_zeros = np.count_nonzero(matrix)

    # Dividir la matriz por el número de elementos no nulos
    result_matrix = matrix / num_non_zeros

    # Crear un nuevo DataFrame con los resultados
    if stri =="visit":
        result_df = pd.DataFrame(result_matrix, columns=pivot_df.columns[2:])

    elif stri == "Patient":
        result_df = pd.DataFrame(result_matrix, columns=pivot_df.columns[1:])
    elif stri =="outs_visit":
        result_ = pd.DataFrame(result_matrix, columns=pivot_df.columns[2:])
        matrix2 = pivot_df.iloc[:,:2]

        result_df = pd.concat([result_,matrix2], axis = 1)



    if stri != "outs_visit":
        result_df['SUBJECT_ID'] = pivot_df['SUBJECT_ID']

    if stri == "outs_visit" or stri =="visit":    
        agregacion_df  = agregacion_cl[["SUBJECT_ID", "HADM_ID","Age_max","LOSRD_sum","L_1s_last","LOSRD_avg"]+categorical_cols]
        merge_input = pd.merge(result_df, agregacion_df, on=["SUBJECT_ID","HADM_ID"], how='inner')

    else: 

        agregacion_df  = agregacion_cl.reset_index()[["SUBJECT_ID", "Age_max","LOSRD_sum","L_1s_last","LOSRD_avg"]+categorical_cols]
        merge_input = pd.merge(result_df, agregacion_df, on=["SUBJECT_ID"], how='inner')

    transformer = FunctionTransformer(np.log)
    merge_input["L_1s_last_p1"] = list(transformer.transform(merge_input["L_1s_last"]))
    merge_input ["L_1s_last_p1"] =np.where(np.isinf(merge_input["L_1s_last_p1"]) , 0, merge_input["L_1s_last_p1"])

    

    print(merge_input.shape)

    return merge_input

def firs_preprocesing(pivot_df,stri,agregacion_cl,categorical_cols):
    '''function that preproces the pivot_df, which is the count matrix of icd9-codes, it divide the matrix by numer of non zeros
    it merges the pivot_df normalized by non zero with the demographic variables, log transformation is applied to variable las visit to 
    first visit, becaus it has large values and all the clustering was done considering this variable, it is done by patien and visit level
    it also drop variable such as ['SUBJECT_ID', "HADM_ID","L_1s_last"], that are not going to be used as in the  as features
    Input
    pivot_df: dataframe which is the count matrix of icd9-codes
    agregacion_cl:  dataframe demographics
    stri: visit, patient, outs_vist the only one that are considere are visit and out_vitis
    Output
    X: dataframe with the first preprocess features
    '''
    if stri =="visit":
        drop_l = list(pivot_df.reset_index().columns)[1:3]
        matrix = pivot_df.drop(drop_l, axis=1).values
    elif stri == "Patient":
        matrix = pivot_df.drop('SUBJECT_ID', axis=1).values
    elif stri == "outs_visit":
        drop_l = list(pivot_df.reset_index().columns)[1:3]
        matrix = pivot_df.drop(drop_l, axis=1).values

        

            

    num_non_zeros = np.count_nonzero(matrix)

    # Dividir la matriz por el número de elementos no nulos
    result_matrix = matrix / num_non_zeros

    # Crear un nuevo DataFrame con los resultados
    if stri =="visit":
        result_df = pd.DataFrame(result_matrix, columns=pivot_df.columns[2:])

    elif stri == "Patient":
        result_df = pd.DataFrame(result_matrix, columns=pivot_df.columns[1:])
    elif stri =="outs_visit":
        result_ = pd.DataFrame(result_matrix, columns=pivot_df.columns[2:])
        matrix2 = pivot_df.iloc[:,:2]

        result_df = pd.concat([result_,matrix2], axis = 1)



    if stri != "outs_visit":
        result_df['SUBJECT_ID'] = pivot_df['SUBJECT_ID']

    if stri == "outs_visit" or stri =="visit":    
        agregacion_df  = agregacion_cl.reset_index()[["SUBJECT_ID", "HADM_ID","Age_max","LOSRD_sum","L_1s_last","LOSRD_avg"] +categorical_cols]
        merge_input = pd.merge(result_df, agregacion_df, on=["SUBJECT_ID","HADM_ID"], how='inner')

    else: 

        agregacion_df  = agregacion_cl.reset_index()[["SUBJECT_ID", "Age_max","LOSRD_sum","L_1s_last","LOSRD_avg"]+categorical_cols]
        merge_input = pd.merge(result_df, agregacion_df, on=["SUBJECT_ID"], how='inner')

    transformer = FunctionTransformer(np.log)
    merge_input["L_1s_last_p1"] = list(transformer.transform(merge_input["L_1s_last"]))
    merge_input["L_1s_last_p1"] =np.where(np.isinf(merge_input["L_1s_last_p1"]) , 0, merge_input["L_1s_last_p1"])
    # REVISAR****PAR CLUSTER PROCEDURES*
    
    if stri != "outs_visit" or stri != "Patient":
        X = merge_input.drop(['SUBJECT_ID', "HADM_ID","L_1s_last"], axis=1)
    elif stri == "Patient":
        X = merge_input.drop(["L_1s_last","SUBJECT_ID"], axis=1)
            
    else :    
        X = merge_input.drop(["L_1s_last"], axis=1)
        print(merge_input.shape)
        X.columns
    
    print(X.shape)

    return X




def preprocess(X, prep):
    ''' The case it has 3 differemt option to normalize the feature, standarization
    Maxabscaler,PowerTransformer
    Input
    X: dataframe which will be preprocessed
    prep: string indicating the preprocessing: std,max,power
    Output
    X: dataframe  preprocessed
    
    '''
    if prep  == "std":
        
        data = X.values
        scaler = StandardScaler()
        scaler.fit(data)
        X = scaler.transform(data)
        print(X)
        print(X.shape)
        
    elif prep ==  "max":
                
        transformer = MaxAbsScaler().fit(X)
        X = transformer.transform(X)
        print(X.shape)
        print(X)
    elif prep == "power":
        numerical = X.select_dtypes(exclude='object')

        for c in numerical.columns:
            pt = PowerTransformer()
            numerical.loc[:, c] = pt.fit_transform(np.array(numerical[c]).reshape(-1, 1))
            X = numerical.copy()
            print(X.shape)
    else: 
        
        X = X        
    return X     

def read_director(ejemplo_dir):
    'Function that give you all the files int and archive'
    with os.scandir(ejemplo_dir) as ficheros:
        ficheros = [fichero.name for fichero in ficheros if fichero.is_file()]
    print(ficheros)
    
    
    
    return [i for i in ficheros if i!= '.DS_Store']       

def fit_kmean_model(ficheros,ejemplo_dir,type_a,num,prepo):
    '''This function reads file input X  and fits a clusterin model 
    it also considers the preprocessing respectively
    Input:
    ficheros:the name of the file
    ejemplo_dir: the name of the path
    type_a = patien or visit 
    num: number of clusters
    Output:
    It  saves the output in research, model cluster as pkl

    '''
    for i in range(len(ficheros)):
        for j in range(3):
        
            X = pd.read_csv(ejemplo_dir+ficheros[i])
            #X = pd.read_csv(ejemplo_dir+X_)
            
            if type_a == "Patient":
                X = X.drop(['SUBJECT_ID',"Unnamed: 0" ,"L_1s_last"], axis=1)    
            else:    
                X = X.drop(['SUBJECT_ID',"HADM_ID","Unnamed: 0" ,"L_1s_last"], axis=1)
            X = preprocess(X, prepo[i])

            name = ficheros[i] + str(j)
            #name = i
            kmeans = KMeans(n_clusters=num,).fit(X)
            print(i)
            if type_a == "outs_visit":
                joblib.dump(kmeans, 'models_cluster/visit/modelo1_'+ name + '_std_4_visit.pkl')
            else:
                joblib.dump(kmeans, 'models_cluster/patient/modelo1_'+ name + '_std_4_visit.pkl')
        



def clustering_icdcodes(df,real,df1,type_a,norm_str,nam_p,categorical_cols,archivo,filtered):
    '''function tha conglomerate all function to obtain the final features, it also apply a general preprocessing to the icdecode matrix and the 
    demographics features
    Input
    df: data_process.csv
    real: name of mapping of icd9-codes
    df1: name of dataframe which includes the mapping with different threshoilds
    type_a: patient,outs_visti
    norm_str: name of preprocessing for all dataframe it can be standarization, maxhab, powertranformer
    name_p: is the name of the mapping of Procedures.
    Output
    X features concatenates preproessed    
    '''
    
    if nam_p == "allicd9Procedures":
        nuevo_df2_gen = desconacat_codes(df,real,filtered)
        nuevo_df4 =  nuevo_df2_gen.rename(columns={"ICD9_CODE_procedures":"col"})

        
        duplicados = merge_df(df,nuevo_df4,nuevo_df2_gen,real,categorical_cols)
        pivot_df,agregacion_cl = pivotm(duplicados,real,type_a,categorical_cols,archivo)

    elif nam_p == "Threshold":
        #nuevo_df2_gen = desconacat_codes(df,real)
        #nuevo_df_x  = nuevo_df2_gen.copy()
        nuevo_df_x = desconacat_codes(df,"ICD9_CODE_procedures",filtered)
        nuevo_df4 = desconacat_codes(df1,real,filtered)
        #print(nuevo_df2_gen.SUBJECT_ID.nunique())
        duplicados = merge_df(df,nuevo_df4,nuevo_df_x,real,categorical_cols)
        pivot_df,agregacion_cl = pivotm(duplicados,real,type_a,categorical_cols,archivo)
    else:
        #nuevo_df2_gen = desconacat_codes(df,real)
        #nuevo_df_x  = nuevo_df2_gen.copy()
        nuevo_df_x = desconacat_codes(df,"ICD9_CODE_procedures",filtered)
        nuevo_df4 = desconacat_codes(df,real,filtered)
        #print(nuevo_df2_gen.SUBJECT_ID.nunique())
        duplicados = merge_df(df,nuevo_df4,nuevo_df_x,real,categorical_cols)
        pivot_df,agregacion_cl = pivotm(duplicados,real,type_a,categorical_cols,archivo)
    X = firs_preprocesing(pivot_df,type_a,agregacion_cl,categorical_cols)
    X = preprocess(X, norm_str)
    return X
    #X = preprocess(X, "std")    
    #obtene X
    #X = firs_preprocesing(pivot_df,"outs_visit")
    #X.to_csv("input_model_pred/cat_"+name+"_procedures.csv") 
    
def clustering_icdcodes_aux(df,real,df1,type_a,norm_str,nam_p,categorical_cols,filtered,archivo):
    '''function tha conglomerate all function to obtain the final features
    the main difference is it does not apply the general preprocessing
    Also eliminares the variable as it does not when its visit level, L_1s_last_p1
    Input
    df: data_process.csv
    real: name of mapping of icd9-codes
    df1: name of datafram which includes the mapping with different threshoilds
    type_a: patient,outs_visti
    norm_str: name of preprocessing for all dataframe it can be standarization, maxhab, powertranformer
    name_p: is the name of the mapping of Procedures.
    Output
    the ouput is X featues with out the las preprocessing this fumction is created to save the features visit-leve
    for the readmission task'''
    
    if nam_p == "allicd9Procedures":
        nuevo_df2_gen = desconacat_codes(df,real,filtered)
        nuevo_df4 =  nuevo_df2_gen.rename(columns={"ICD9_CODE_procedures":"col"})

        
        duplicados = merge_df(df,nuevo_df4,nuevo_df2_gen,real, categorical_cols)
        pivot_df,agregacion_cl = pivotm(duplicados,real,type_a,categorical_cols,archivo)

    elif nam_p == "Threshold":
        #nuevo_df2_gen = desconacat_codes(df,real)
        #nuevo_df_x  = nuevo_df2_gen.copy()
        nuevo_df_x = desconacat_codes(df,"ICD9_CODE_procedures",filtered)
        nuevo_df4 = desconacat_codes(df1,real,filtered)
        #print(nuevo_df2_gen.SUBJECT_ID.nunique())
        duplicados = merge_df(df,nuevo_df4,nuevo_df_x,real,categorical_cols)
        pivot_df,agregacion_cl = pivotm(duplicados,real,type_a,categorical_cols,archivo)
    else:
        #nuevo_df2_gen = desconacat_codes(df,real)
        #nuevo_df_x  = nuevo_df2_gen.copy()
        nuevo_df_x = desconacat_codes(df,"ICD9_CODE_procedures",filtered)
        nuevo_df4 = desconacat_codes(df,real,filtered)
        #print(nuevo_df2_gen.SUBJECT_ID.nunique())
        duplicados = merge_df(df,nuevo_df4,nuevo_df_x,real,categorical_cols)
        pivot_df,agregacion_cl = pivotm(duplicados,real,type_a,categorical_cols,archivo)
    X = firs_preprocesing_aux(pivot_df,type_a,agregacion_cl,categorical_cols)
    
    return X
    #X = preprocess(X, "std")    
    #obtene X
    #X = firs_preprocesing(pivot_df,"outs_visit")
    #X.to_csv("input_model_pred/cat_"+name+"_procedures.csv")                               

def input_for_pred_mutualinfo(df,categorical_cols,real,stri,archivo,type_a,nuevo_df_x):
    
  
     
        
        nuevo_df4 = desconacat_codes_ori(df,real)

        

        duplicados = merge_df_ori(nuevo_df_x,nuevo_df4,df,categorical_cols,real) 
        #duplicados["MARITAL_STATUS"]= duplicados["MARITAL_STATUS"].replace(np.nan, "Unknown")

        
        pivot_df, agregacion_cl = pivotm_ori(duplicados,real,stri,categorical_cols,archivo)
        agregacion_cl = demo_ad(categorical_cols,agregacion_cl)
        X = firs_preprocesing_aux(pivot_df,type_a,agregacion_cl,categorical_cols)

                    
        return X  

def merge_df_ori(nuevo_df_x,nuevo_df4,df,categorical_cols,real):   

    nuevo_aux = pd.concat([nuevo_df4, nuevo_df_x[[nuevo_df_x.columns[-1]]]],axis = 1)

    

    
    print(nuevo_df_x.shape)
    print(nuevo_df4.shape)
    print(nuevo_aux.shape)
    print("Unicos Subjects",nuevo_aux["SUBJECT_ID"].nunique())


    
    merge_atc = pd.merge(df[categorical_cols+['ICD9_CODE_procedures',
            'EXPIRE_FLAG', 'DOB', 'SUBJECT_ID',
            'HADM_ID', 'ADMITTIME', 'age', 'year_age', "ADMITTIME","LOSRD"]], nuevo_aux, on=["HADM_ID","SUBJECT_ID"], how='right')

    print("shape merged",merge_atc.shape)

    print("Unicos Subjects",merge_atc["SUBJECT_ID"].nunique())


    print("HADM_ID unicos",merge_atc["HADM_ID"].nunique())
    if real == "ICD9_CODE_diagnosis":
       merge_atc = merge_atc.iloc[:,:-1]
    merge_atc = merge_atc[merge_atc[real + "_preprocess"].notnull()]
    print("shape merged, despues de eliminar nulos",merge_atc.shape)

    #verificas que solo se tienen lenght of stay

    merge_atc = merge_atc[merge_atc["LOSRD"]>0]   
    
    merge_atc=merge_atc.loc[:, ~merge_atc.columns.duplicated(keep='first')]
    merge_atc['ADMITTIME'] = pd.to_datetime(merge_atc["ADMITTIME"])
    #eliminar nulos de variables demograficas
    for i in categorical_cols:
        merge_atc[i]= merge_atc[i].replace(np.nan, "Unknown")

    return merge_atc

def clustering_prepo2(X,name,num_clusters,method):
    '''function that obtain the the performance metrics of the clustering davis_bouldin and silhouette_avg
    it reruns the algorith 10 times and average the metric to obtain a more robust metric
    Input: 
    X: dataframe preprocessed
    name: type of mapping
    num_clusters: int number of clusters
    
    Output:
    silhouette_avg,davies_bouldin_avg: evaluation metrics
    '''
    kmeans_labels_l=[]
    ccscodes_thhreshold_l = []
    ccscodes_rand_l = []
    silhouette_avg,davies_bouldin_avg,kmeans_labels,kmeans = cluster_scores(num_clusters,X,method)
    kmeans_labels_l.append(kmeans_labels)
    return silhouette_avg,davies_bouldin_avg



def get_kmeans_score(data, center):
    '''
    returns the kmeans score regarding Davies Bouldin for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the Davies Bouldin score for the kmeans model fit to the data
    '''
    #instantiate kmeans
    kmeans = KMeans(n_clusters=center)
# Then fit the model to your data using the fit method
    model = kmeans.fit_predict(X)
    
    # Calculate Davies Bouldin score
    score = davies_bouldin_score(X, model)
    
    return score

def clustering_(X,name,num_clusters,meethod,type_a):
    ''' Function that evaluates the performance of the cluster
    with m ,silhouette_avg,davies_bouldin_avg scores it re runs the k mean algorith
    10 times and average the score, 
    
    it also evaluates the coherence of the clustering it luste
    Input
    X: dataframe which will be preprocessed
    prep: string indicating the preprocessing: std,max,power
    Output
    X: dataframe  preprocessed
    
    ''' 
    
    kmeans_labels_l=[]
    ccscodes_thhreshold_l = []
    ccscodes_rand_l = []
    silhouette_avg,davies_bouldin_avg,kmeans_labels,kmeans = cluster_scores(num_clusters,X,meethod) 
    kmeans_labels_l.append(kmeans_labels)
    if type_a == "outs_visit":
       joblib.dump(kmeans, 'models_cluster/visit/modelo1_'+ name + '_std_4_visit.pkl')
    else:
       joblib.dump(kmeans, 'models_cluster/patient/modelo1_'+ name + '_std_4_visit.pkl')
    return silhouette_avg,davies_bouldin_avg
 
 

def calculare_changes(i,df,df1,nam_p,v,real,filtered):
    real = real
    if nam_p == "Threshold":
        #nuevo_df2_gen = desconacat_codes(df,real)
        #nuevo_df_x  = nuevo_df2_gen.copy()
        nuevo_df_x = desconacat_codes(df,"ICD9_CODE_procedures",filtered)
        nuevo_df4 = desconacat_codes(df1,real,filtered)
        #print(nuevo_df2_gen.SUBJECT_ID.nunique())
        
        
    else:
        #nuevo_df2_gen = desconacat_codes(df,real)
        #nuevo_df_x  = nuevo_df2_gen.copy()
        nuevo_df_x = desconacat_codes(df,"ICD9_CODE_procedures",filtered)
        nuevo_df4 = desconacat_codes(df,real,filtered)
      

        #print(nuevo_df2_gen.SUBJECT_ID.nunique())
    nuevo_df_x = nuevo_df_x.sort_values(by=["SUBJECT_ID","HADM_ID"])
    nuevo_df4 = nuevo_df4.sort_values(by=["SUBJECT_ID","HADM_ID"])
    
    duplicados = pd.concat([nuevo_df4.reset_index(), nuevo_df_x["ICD9_CODE_procedures"]],axis = 1)
    #merged_df = pd.merge(nuevo_df4, nuevo_df_x, on=["SUBJECT_ID","HADM_ID"], how='left')

    #duplicados = pd.merge(nuevo_df_x.reset_index(),nuevo_df4.reset_index(), on=["SUBJECT_ID","HADM_ID",], how='left')
    #realizar una fFUNCION QUE ME AYUDE A LIMPIAR PREPROCESINGG DE real preprocess porque corata 30% de datos
    nuevo_df4[real +'_preprocess']= nuevo_df4[real +'_preprocess'].replace("Otro", -1)
    duplicados[real +'_preprocess']= duplicados[real +'_preprocess'].replace("Otro", -1)
    duplicados['ICD9_CODE_procedures']= duplicados['ICD9_CODE_procedures'].replace("Otro", -1)
    
    if real == "cat_threshold .95 most frequent_proc":
        name = name2 = real
        
        duplicados[name2 + "_preprocess"] = duplicados[name2 + "_preprocess"].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        duplicados = duplicados[duplicados[name2 + "_preprocess"].notnull()]

    elif real == "CCS CODES_proc":
        name = real 
        duplicados[name + "_preprocess"]=[item.replace("'", '') for item in duplicados["CCS CODES_proc_preprocess"]]

        duplicados[name + "_preprocess"] = duplicados[name + "_preprocess"].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        duplicados = duplicados[duplicados[name + "_preprocess"].notnull()]

   
    elif nam_p == 'Threshold':
        name2 = real
            
        duplicados[name2 + "_preprocess"] = duplicados[name2 + "_preprocess"].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        duplicados = duplicados[duplicados[name2 + "_preprocess"].notnull()]

  

    
      

    aux = duplicados[["SUBJECT_ID","HADM_ID",real+"_preprocess","ICD9_CODE_procedures"]]
             
# Create 'changes' column based on the condition
    aux['changes'] = (aux['ICD9_CODE_procedures'] != aux[real+"_preprocess"]).astype(int)


    # Calculate the total number of changes per patient
    if v == "visit":
        #changes_per_patient = aux.groupby('SUBJECT_ID','HAMD_ID')['changes'].sum().reset_index()
        unique_icd9 = duplicados['ICD9_CODE_procedures'].nunique()
        unique_r= duplicados[real+"_preprocess"].nunique()    
        if real!= 'CCS CODES_proc':
            changes_per_patient = nuevo_df4[nuevo_df4[real +'_preprocess'] == -1].groupby(['SUBJECT_ID','HADM_ID']).size().reset_index(name='count_minus_one')
        else:
            changes_per_patient = aux.groupby(['SUBJECT_ID','HADM_ID'])['changes'].sum().reset_index()
            

    else:
        #changes_per_patient = aux.groupby('SUBJECT_ID')['changes'].sum().reset_index()
        unique_icd9 = duplicados['ICD9_CODE_procedures'].nunique()
        unique_r= duplicados[real+"_preprocess"].nunique()
        if real!= 'CCS CODES_proc':
            changes_per_patient = nuevo_df4[nuevo_df4[real +'_preprocess'] == -1].groupby('SUBJECT_ID').size().reset_index(name='count_minus_one')
        else:
            changes_per_patient = aux.groupby('SUBJECT_ID')['changes'].sum().reset_index()
            
    return changes_per_patient,real,unique_icd9 ,unique_r


def create_results(result_stat,changes_per_patient,real,unique_r,unique_icd9):
    if real == 'CCS CODES_proc':
       num_w = changes_per_patient["changes"].describe().loc["min"]
       changes=(changes_per_patient[changes_per_patient["changes"]>=num_w].shape[0]/changes_per_patient.shape[0])
       res = changes_per_patient["changes"].describe().to_dict()
    else:    
       num_w = changes_per_patient["count_minus_one"].describe().loc["min"]
       changes=(changes_per_patient[changes_per_patient["count_minus_one"]>=num_w].shape[0]/changes_per_patient.shape[0])
       res = changes_per_patient["count_minus_one"].describe().to_dict()
    
    auc_d = {**{"Name":real,"Min >":changes,"Unique_codes":unique_r,"Unique_codes_icd9":unique_icd9},**res}



    return    auc_d
    
if __name__ == "__main__":
    print("hola")
