
import pandas as  pd
from function_mapping import *
from preprocess import *

def pivotm_ori(duplicados,real,stri,categorical_cols,archivo):

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
            print(agregacion_cl.shape)
            
            #creacion de pivot, matriz que cuenta la ocurrencia por subject_id
            pivot_df = duplicados[[real + "_preprocess","SUBJECT_ID"]].pivot_table(index='SUBJECT_ID', columns=real + "_preprocess", aggfunc='size', fill_value=0)
    else:
        agregacion_cl =duplicados.groupby(['SUBJECT_ID',"HADM_ID"]).agg(

        Age_max=("year_age", 'max'),  GENDER=("GENDER", mode), ADMISSION_LOCATION=("ADMISSION_LOCATION", mode),  ETHNICITY=('ETHNICITY', mode)
        ,MARITAL_STATUS=("MARITAL_STATUS", mode),RELIGION=("RELIGION", mode), ADMISSION_TYPE=("ADMISSION_TYPE", mode),INSURANCE=("INSURANCE", mode),DISCHARGE_LOCATION=("DISCHARGE_LOCATION", mode),


        LOSRD_sum=("LOSRD", 'sum'),LOSRD_avg=("LOSRD", np.mean),ADMITTIME_max=("ADMITTIME", 'max'),).reset_index()
        aux = last_firs(archivo)
        agregacion_cl = pd.merge(agregacion_cl, aux, on=["HADM_ID","SUBJECT_ID"], how='left')
       
        agregacion_cl["L_1s_last"] =[int(i.days) for i in agregacion_cl["L_1s_last"]]

            
        agregacion_cl = agregacion_cl[["HADM_ID","SUBJECT_ID","Age_max","LOSRD_sum","LOSRD_avg","L_1s_last"]+categorical_cols]

        print(agregacion_cl.shape)
        
        #creacion de pivot table
        pivot_df = duplicados[[real + "_preprocess" ,"SUBJECT_ID","HADM_ID"]].pivot_table(index=['SUBJECT_ID',"HADM_ID"], columns=real + "_preprocess", aggfunc='size', fill_value=0)

        

                
        
    pivot_df.reset_index(inplace=True)
    print(pivot_df.shape)    
    return pivot_df, agregacion_cl

def demo_ad(categorical_cols,aux_demo):
    '''Function that encodes categorical features
    Input
    categorcial_cols: list of cols tu encode
    aux_demo: df that will be encoded
    Output:
    aux_demo: df encoded'''
              
    columnTransformer = ColumnTransformer([(OrdinalEncoder(),  LabelEncoder(), [0])], remainder='passthrough')

    enc = OrdinalEncoder()
    enc.fit(aux_demo[categorical_cols])
    aux_demo[categorical_cols] = enc.transform(aux_demo[categorical_cols])
    return aux_demo



def firs_preprocesing_ori(pivot_df,stri,agregacion_cl,categorical_cols):

    if stri =="visit" or stri == "outs_visit":
        drop_l = list(pivot_df.reset_index().columns)[1:3]
        matrix = pivot_df.drop(drop_l, axis=1).values
    elif stri == "Patient":
        matrix = pivot_df.drop('SUBJECT_ID', axis=1).values
    
    num_non_zeros = np.count_nonzero(matrix)

    # Dividir la matriz por el número de elementos no nulos
    result_matrix = matrix / num_non_zeros

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
    merge_input.columns

    # CORRECCION DEL ORIGINAL
    if stri =="visit" or stri =="outs_visit":
        X = merge_input.drop(['SUBJECT_ID', "HADM_ID","L_1s_last"], axis=1)
    elif stri == "Patient":
        X = merge_input.drop(["L_1s_last","SUBJECT_ID"], axis=1)
            


    print(X.shape)
    return X




