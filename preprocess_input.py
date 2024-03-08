import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Assuming df1, df2, df3 are your dataframes
#df_drugs =pd.read_csv('./input_model_pred_drugs_u/ATC3_outs_visit_non_filtered.csv')
#df_diagnosis = pd.read_csv('./input_model_pred_diagnosis_u/CCS_CODES_diagnosis_outs_visit_non_filtered.csv')
#df_procedures = pd.read_csv('./input_model_visit_procedures/CCS CODES_proc_outs_visit_non_filtered.csv')
# Drop the columns categotical
numerical_cols =  ['Age_max', 'LOSRD_sum',
       'L_1s_last', 'LOSRD_avg','L_1s_last_p1']

categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                'MARITAL_STATUS', 'ETHNICITY','GENDER']

name_df = "raw_input.csv"
########onehot encoding y agrupacion de categoria de 80##
name_encodeing = "input_onehot_encoding.csv"
'''La primeras do gunciones se corren si debo volver a correl el input 
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
    


    #adm = pd.read_csv('./data/data_preprocess_nonfilteres.csv')
    ad_f = "ADMISSIONS.csv.gz"
    adm = pd.read_csv(ad_f)
    res = pd.merge(adm[categorical_cols+["ADMITTIME","SUBJECT_ID","HADM_ID"]],result_final, on=["SUBJECT_ID","HADM_ID"], how='right')

    # Assuming df is your DataFrame

    # Find columns that contain 'unnamed' in their name
    cols_to_drop = res.filter(like='Unnamed', axis=1).columns

    # Drop these columns
    res.drop(cols_to_drop, axis=1, inplace=True)
    res = res.fillna(0)
    res.to_csv("generative_input/"+ name_df)
    return res

#######parte del preprocesamiento###################


##########nulos##########

# Reemplazar 'Not specified' con 'Otra'



def encoding(res):
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
    
