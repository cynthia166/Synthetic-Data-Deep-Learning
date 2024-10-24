import sys
sys.path.append("/Users/cynthiagarcia/Desktop/Synthetic-Data-Deep-Learning/preprocessing")

from function_mapping import *
import pandas as pd

def main():

    filtered = False
    output_path = "/Users/cynthiagarcia/Desktop/Synthetic-Data-Deep-Learning/results/models_cluster/drugs/final_clustering/"
    archivo ="/Users/cynthiagarcia/Desktop/Synthetic-Data-Deep-Learning/data/intermedi/SD/inpput/second/"
 
    pd.set_option('display.max_columns', None)
    method = "kmeans"    
    output_path_prepoexperimet = "results/models_cluster/drugs/"
    type_a="Patient"
    if type_a == "Patient":
        archivo_completa =archivo + "drugs_Patient"
        prep_file = pd.read_csv(output_path_prepoexperimet +"best_silhoutte_scores_drugs_patient.csv")
        list_cat_aux= listar_archivos(archivo_completa)
    else:
        archivo_completa =archivo + "drugs_visit"
        list_cat_aux= listar_archivos(archivo_completa)
        prep_file = pd.read_csv(output_path_prepoexperimet +"best_silhoutte_scores_drugs.csv")

    list_cat_aux= listar_archivos(archivo_completa)
    result = {'Name':[],
                'silhouette_avg':[],
            'davies_bouldin_avg':[],
                        } 
    
   

    for i in range(len(list_cat_aux)):
        print(i)
        real = list_cat_aux[i]
        
        #lee el respectivo file
        aux = prep_file[prep_file["Name"]==real]
        prepro =aux['Prepro'].values[0]
        num_clusters =aux["Num Cluster"].values[0]
        
        name = list_cat_aux[i]
        ruta = archivo_completa+ "/"+real
        df = pd.read_csv(ruta)
        df = df.iloc[:, 1:]
        #df= df[:2000]
        # Assuming df and demographic are your two DataFrames
        print("null df vales", df.isnull().sum())
        df.drop(columns='DISCHTIME',inplace=True)
        X = preprocess(df,prepro)
        silhouette_avg,davies_bouldin_avg =clustering_(X,name,num_clusters,method,type_a,output_path)
        result["silhouette_avg"].append(silhouette_avg)
        result["davies_bouldin_avg"].append(davies_bouldin_avg)
        result["Name"].append(name)
        df_res = pd.DataFrame(result)
        df_res.to_csv(output_path+type_a+"/_metricas_clustering_non_filtered"+name+".csv")
        
     
        
    #after its done without the clustering
    X = X[:,:20]


    silhouette_avg,davies_bouldin_avg =clustering_(X,"sin_codigo",num_clusters,method,type_a,output_path)
    result["silhouette_avg"].append(silhouette_avg)
    result["davies_bouldin_avg"].append(davies_bouldin_avg)
    result["Name"].append("sincodigo")

    df_res = pd.DataFrame(result)

    
    df_res.to_csv(output_path+type_a+"_metricas_clustering_non_filtered"+name+".csv")
    
    
if __name__ == "__main__":
    main()    