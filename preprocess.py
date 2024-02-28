
import pandas as  pd
from function_mapping import *
import ast
import numpy as np
#'threshold_0.88', 'threshold_0.95',
 #      'threshold_0.98', 'threshold_0.999'

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

def desconacat_codes_ori(drugs,real):
    ex_drug = drugs[drugs[real].notnull()][real].iloc[0]

    if type(string_list(ex_drug)) == list and type(ex_drug) == str :
            nm2 = real + "_preprocess"
            name2 =  real

            drugs[name2 + "_preprocess"] = drugs[name2 ].apply(  lambda x: string_list(x))

            #df1[name2 + "_preprocess"] = df1[real]

            
            txt =drugs[["HADM_ID","SUBJECT_ID",nm2]]

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
            print(nuevo_df3.shape)

            nuevo_df2_gen=nuevo_df3.copy()
            
          

    return nuevo_df2_gen

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


