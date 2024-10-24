import sys
sys.path.append("/Users/cynthiagarcia/Desktop/Synthetic-Data-Deep-Learning/preprocessing")
sys.path.append("/Users/cynthiagarcia/Desktop/Synthetic-Data-Deep-Learning/preprocessing")

from function_mapping import *
import pandas as pd
import pandas as pd



def main():
    global df_res,result
    #todas las visitass
    category="diagnosis"  #procedures #drugs
    filtered = False
    archivo ="/Users/cynthiagarcia/Desktop/Synthetic-Data-Deep-Learning/data/intermedi/SD/inpput/second/"+category+"/"
    #procedures threshold
 
    output_path = "results/models_cluster/"+category+"/"
    
    #list_cat_aux = ["threshold_presence_0.9_drugst_non_prepo.csv"]

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
    
    #type of leve.
    type_a="Patient" #Patien #visit
    if type_a == "Patient":
        archivo_completa =archivo + category+"_Patient"
    
        list_cat_aux= listar_archivos(archivo_completa)
    else:
        archivo_completa =archivo + category+"_visit"
        list_cat_aux= listar_archivos(archivo_completa)
    
    for i in range(0,len(list_cat_aux)):
        #for i in range(1,3):
        print(i)
        real = list_cat_aux[i]
        name = list_cat_aux[i]
        ruta = archivo_completa +'/'+real
        
        df = pd.read_csv(ruta)
        print(real,df.shape)
    
       #preprocessing
    norm_str =["std","max", "power"]
    #norm_str =[ "power"]
    #number of clusters
    num_cluste = [4,8,12]
    #information lists
    # "agglomerative" / kmeans
    method = "kmeans"
     

    #for i in range(5,len(list_cat)):
    for i in range(0,len(list_cat_aux)):
        #for i in range(1,3):
        print(i)
        # Your code here
        for j in norm_str:
            for n in num_cluste:
                real = list_cat_aux[i]
                name = list_cat_aux[i]
                ruta = archivo_completa +'/'+real
                
                df = pd.read_csv(ruta)
                df = df.iloc[:, 1:]
                #df= df[:2000]
                df.drop(columns='DISCHTIME',inplace=True)
                # Assuming df and demographic are your two DataFrames
                print("null df vales", df.isnull().sum())
                
                X = preprocess(df,j)                
                silhouette_avg,davies_bouldin_avg = clustering_prepo2(X,name,n,method)
                
                result["Prepro"].append(j)
                result["Num Cluster"].append(n)
                result["silhouette_avg"].append(silhouette_avg)
                result["davies_bouldin_avg"].append(davies_bouldin_avg)
                result["Name"].append(name)
                 
                df_res = pd.DataFrame(result)
                
                # the results are saved
                df_res.to_csv(output_path + "prepro_experiment_"+type_a+method+"_"+name+".csv")
    get_best_prepro_by_silhouette(df_res,output_path +"best_silhoutte_scores_drugs.csv")
if __name__ == "__main__":
    main()