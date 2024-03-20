



from Preprocessing.function_mapping import *
import pandas as pd

def main():
    #ren el ca
    archivo = "data/data_preprocess_non_filtered.csv"
    filtered = False
    output_path = "models_cluster/metricas_cluster/"
    df = pd.read_csv(archivo)

    pd.set_option('display.max_columns', None)
    method = "kmeans"    
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
    

    categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
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
    type_a=stri ="Patient"
    #prep_type = ["std","std","std","power","power","std"]
    #este corresponde a las visitas filtradas
    #prep_type = ["std","std","std","std","std","power"]
    prep_type = ["power","max","max","max","power","power"]
    mean_mutual_information_l=[]
    mean_ccscodes_randindex_l=[]
    silhouette_avg_l=[]
    davies_bouldin_avg_l=[]
    real_l=[]
    num_clusters = 4
    for i in range(len(list_cat)):
        print(i)
        real = list_cat[i]
        nam_p = nam_p_list[i]
        name = list_cat[i]
        X = clustering_icdcodes(df,real,df1,type_a,prep_type[i],nam_p,categorical_cols,archivo,filtered)
        #pd.DataFrame(X).to_csv("/Users/cgarciay/Desktop/Laval_Master_Computer/research/input_model_pred/"+real+".csv")
        silhouette_avg,davies_bouldin_avg =clustering_(X,name,num_clusters,method,type_a)
        result["silhouette_avg"].append(silhouette_avg)
        result["davies_bouldin_avg"].append(davies_bouldin_avg)
        result["Name"].append(name)
        df_res = pd.DataFrame(result)
        df_res.to_csv(output_path+type_a+"_metricas_clustering_non_filtered"+name+".csv")


      
    
    #after its done without the clustering
    X = X.iloc[:,-7:]


    silhouette_avg,davies_bouldin_avg =clustering_(X,"sin_codigo",num_clusters,method,type_a)
    result["silhouette_avg"].append(silhouette_avg)
    result["davies_bouldin_avg"].append(davies_bouldin_avg)
    result["Name"].append("sincodigo")

    df_res = pd.DataFrame(result)

    
    df_res.to_csv(output_path+type_a+"_metricas_clustering_non_filtered.csv")

if __name__ == "__main__":
    main()    