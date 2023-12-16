from function_mapping import *
import pandas as pd
import argparse
def main():
    #ren el caso que solo se incluya una visita
    #archivo = "data_preprocess.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_a', type=str, default='outs_visit')

    global df_res
    #todas las visitass
    archivo = "data/data_preprocess_non_filtered.csv"


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
    #todas las visitas
    #df1 = df.copy()

    # the results are saved
    df_res.to_csv("/prepro_experiment_"+type_a+method+"_nonfiltered.csv")
    ategorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                'MARITAL_STATUS', 'ETHNICITY','GENDER']

    list_cat = ["ICD9_CODE_procedures",'CCS CODES_proc', 'cat_threshold .95 most frequent_proc','cat_threshold .88 most frequent', 'cat_threshold .98 most frequent',
        'cat_threshold .999 most frequent']

    result = {'Name':[],
                'silhouette_avg':[],
            'davies_bouldin_avg':[],
                        } 
    nam_p_list = ["allicd9Procedures",'CCS CODES_proc', 'cat_threshold .95 most frequent_proc','Threshold', 'Threshold',
        'Threshold']
    type_a=stri =arg.type_a
    #prep_type = ["std","std","std","power","power","std"]
    prep_type = ["std","std","std","std","std","power"]
    mean_mutual_information_l=[]
    mean_ccscodes_randindex_l=[]
    silhouette_avg_l=[]
    davies_bouldin_avg_l=[]
    real_l=[]
    num_clusters = 4
    for i in range(len(list_cat)):
        real = list_cat[i]
        nam_p = nam_p_list[i]
        name = list_cat[i]
        X = clustering_icdcodes_aux(df,real,df1,type_a,prep_type[i],nam_p,categorical_cols)
        X.to_csv("input_model_pred/"+real+"_"+type_a+"_non_filtered.csv")


if __name__ == "__main__":
    main()