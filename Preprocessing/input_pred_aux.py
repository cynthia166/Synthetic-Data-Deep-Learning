from Preprocessing.function_mapping import *
import pandas as pd
import argparse
def main():
    nom_t = "Drugs"
    ori = "DRUG_x"
    categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                    'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                    'MARITAL_STATUS', 'ETHNICITY','GENDER']
    

    #ren el caso que solo se incluya una visita
    #archivo = "data_preprocess.csv"
    parser = argparse.ArgumentParser()
    #Patient/outs_visit
    parser.add_argument('--type_a', type=str, default='Patient')

    global df_res
    #todas las visitass
    archivo = "data/data_preprocess_nonfilteres.csv"
    file_save = "input_model_patient_drugs/"

    df = pd.read_csv(archivo)

    pd.set_option('display.max_columns', None)
        
    #Lectures of dataframe that have the procedures icd-9 codes with different threshold
    #mas de dos visitas
    #archivo_procedures = "procedures_preprocess_threshold.csv"



    #proc = pd.read_csv("/Users/cgarciay/Desktop/Laval_Master_Computer/research/procedures_preprocess_threshold.csv")
    #grouped = proc.groupby(['SUBJECT_ID', 'HADM_ID']).agg(lambda x: x.tolist())

    # Reset
    # the index to make 'SUBJECffT_ID' and 'HADM_ID' regular columns
    #grouped_proc = grouped.reset_index()


    #todas las visitas

    #df1=grouped_proc.copy()
    archivo_procedures = "procedures_preprocess_threshold_nonfiltered.csv"
    df1 = pd.read_csv("data/"+archivo_procedures)
    filtered = False
    #procedures threshold
    archivo = "data/data_preprocess_non_filtered.csv"

    #todas las visitas
    #df1 = df.copy()

    # the results are saved
    #df_res.to_csv("/prepro_experiment_"+type_a+method+"_nonfiltered.csv")

    #procedures
    '''list_cat = ["ICD9_CODE_procedures",'CCS CODES_proc', 'cat_threshold .95 most frequent_proc','cat_threshold .88 most frequent', 'cat_threshold .98 most frequent',
        'cat_threshold .999 most frequent']
    #drugs
    list_cat = [
       'ATC4', 'ATC3',  'threshold_0.88', 'threshold_0.95',
       'threshold_0.98', 'threshold_0.999','DRUG_y']  '''
    #
      #patient procedures
    #prepo_li= ["std","max","power","std","std","power","std"]
    #visit procedures
    #prepo_li = ["std","max","max","std","std","std","std"]
    
    #patient drug
    #prepo_li= ["max","max","max","max","std","std","std","std"]
    #visit drug 
    #prepo_li = ["max","max","max","max","power","power","power"]
    
    #patient diagnosis 
    #prepo_li = ["max","max","max","max","std","power","power"]

    #visit diagnosis  
    #prepo_li= ["max","max","std","max","max","std","std"]
    list_cat = [
       'ATC4', 'ATC3',  'threshold_0.88', 'threshold_0.95',
       'threshold_0.98', 'threshold_0.999','DRUG_y']  
 
  
   
    result = {'Name':[],
                'silhouette_avg':[],
            'davies_bouldin_avg':[],
                        } 
    nam_p_list = ["allicd9Procedures",'CCS CODES_proc', 'cat_threshold .95 most frequent_proc','Threshold', 'Threshold',   'Threshold',   'Threshold']
    
    args = parser.parse_args()
    #type_a=stri =args.type_a
    #Patient,outs_visit
    type_a = stri = "Patient"
    if type_a == "Patient":
       file_save =file_save
    else:
       file_save = file_save   
    #prep_type = ["std","std","std","power","power","std"]
    #patient
    #prep_type = ["std","max","power","std","std","power"]
    
    #visit
    #prep_type = ["std","max","max","std","std","std"]

    
    if nom_t == "Drugs" or nom_t == "Diagnosis":
        archivo = "data/data_preprocess_nonfilteres.csv"
        df = pd.read_csv(archivo)
        nuevo_df_x = desconacat_codes_ori(df,ori)

    
    for i in range(len(list_cat)):
        real = list_cat[i]
        #nam_p = nam_p_list[i]
        name = list_cat[i]
        if nom_t == "Drugs" or nom_t == "Diagnosis":
            #std max power
            #norm_str =[ "power"]
            archivo = "data/data_preprocess_nonfilteres.csv"
            df = pd.read_csv(archivo)

       
      
            X = input_for_pred_mutualinfo(df,categorical_cols,real,stri,archivo,type_a,nuevo_df_x)
        else:    
            X = clustering_icdcodes_aux(df,real,df1,type_a,prep_type[i],nam_p,categorical_cols,filtered,archivo)
        X.to_csv(file_save+real+"_"+type_a+"_non_filtered.csv")


if __name__ == "__main__":
    main()