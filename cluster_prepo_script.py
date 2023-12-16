

from function_mapping import *
import pandas as pd

def main():
    # Preprocessing of the data
    #data_preprocess()
    #data_preprocess_nonfiltered()

    #data_preproces
    #ren el caso que solo se incluya una visita
    #archivo = "data_preprocess.csv"
    global df_res,result
    #todas las visitass
    filtered = False
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

    aux_demo = df[['ADMISSION_TYPE', 'ADMISSION_LOCATION',
            'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
            'MARITAL_STATUS', 'ETHNICITY','GENDER']]

    #PREPROCESSING
    list_cat = ["ICD9_CODE_procedures",'CCS CODES_proc', 'cat_threshold .95 most frequent_proc','cat_threshold .88 most frequent', 'cat_threshold .98 most frequent',
        'cat_threshold .999 most frequent']
    categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                    'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                    'MARITAL_STATUS', 'ETHNICITY','GENDER']
    # the dictionary where the results will be stores
    result = {'Name':[],
            'Prepro':[],
            'Num Cluster':[],
            'silhouette_avg':[],
            'davies_bouldin_avg':[],
                        } 
    #type of mapping, 
    nam_p_list = ["allicd9Procedures",'CCS CODES_proc', 'cat_threshold .95 most frequent_proc','Threshold', 'Threshold',
        'Threshold']
    #type of leve.
    type_a=stri ="outs_visit"
    #preprocessing
    norm_str =["std","max", "power"]
    #number of clusters
    num_cluste = [4,8,12]
    #information lists
    # "agglomerative" / kmeans
    method = "kmeans"

    mean_mutual_information_l=[]
    mean_ccscodes_randindex_l=[]
    silhouette_avg_l=[]
    davies_bouldin_avg_l=[]
    real_l=[]

    #for i in range(len(list_cat)-1, len(list_cat)):
    for i in range(1,3):
        print(i)
        # Your code here
        for j in norm_str:
            for n in num_cluste:
                real = list_cat[i]
                nam_p = nam_p_list[i]
                name = list_cat[i]
                X = clustering_icdcodes(df,real,df1,type_a,j,nam_p,categorical_cols,archivo,filtered)
                silhouette_avg,davies_bouldin_avg = clustering_prepo2(X,name,n,method)
                result["Prepro"].append(j)
                result["Num Cluster"].append(n)
                result["silhouette_avg"].append(silhouette_avg)
                result["davies_bouldin_avg"].append(davies_bouldin_avg)
                result["Name"].append(name)

                df_res = pd.DataFrame(result)
                df_res   
                # the results are saved
                df_res.to_csv("experiment_prepo/prepro_experiment_"+type_a+method+"_"+name+"_nonfiltered.csv")

if __name__ == "__main__":
    main()