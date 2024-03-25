
from sklearn.metrics import rand_score
from preprocessing.function_mapping import *
import pandas as pd
import csv

def main():
    
    u = "patient/"    
    type_a=stri ="Patient"
    ejemplo_dir = 'input_model_pred/'
    ficheros = read_director(ejemplo_dir)
    for i in ficheros:
        print(i)
    #prepo_li = ["std", "std","power","std","power","std","std"]
    prepo_li=["max","max","power","power","std","max","power"]
    
    #fit_kmean_model(ficheros,ejemplo_dir,type_a,4,prepo_li)

    
    ejemplo_dir2= 'models_cluster/'+u
    ficheros2 = read_director(ejemplo_dir2)

    
    res = {
        "Name2":[],
        "Name":[],
    
        "mutual_information":[],
        "rand index":[],
        
    }
    #Bert is not taken in accounts 
    ficheros = [i for i in ficheros if i != 'clinicalBertembedding.csv']
    for i in ficheros:
        
        na = i[:-4]
        #new_l  =  [j for j in ficheros if na not in j]
        for j in ficheros: 
            # the first 3 files are  filter mappin i the other are takenin account in a cicler for, of all the others
            archivos_filtrados = [archivo for archivo in ficheros2 if na in archivo]
            archivos_filtrados2 = [archivo for archivo in ficheros2 if j[:-4]  in archivo]
            #archivods del indice i
            f1 = joblib.load('models_cluster/'+u+archivos_filtrados[0]).labels_
            print(archivos_filtrados[0])
            f2 = joblib.load('models_cluster/'+u+archivos_filtrados[1]).labels_
            print(archivos_filtrados[1])
            f3 = joblib.load('models_cluster/'+u+archivos_filtrados[2]).labels_
            print(archivos_filtrados[2])
            #archivo de las demas combinaciones
            f4 = joblib.load('models_cluster/'+u+archivos_filtrados2[0]).labels_
            print(archivos_filtrados2[0])
            f5 = joblib.load('models_cluster/'+u+archivos_filtrados2[1]).labels_
            print(archivos_filtrados2[1])
            f6 = joblib.load('models_cluster/'+u+archivos_filtrados2[2]).labels_
            print(archivos_filtrados2[2])
            list_arc = [f1,f2,f3,f4,f5,f6]
            #limpiar listas
            ccscodes_thhreshold_l=[]
            ccscodes_rand_l = []
            #for each file, the file i is obtain mutual inforamtion, it is done 3 times
            #and the results are averaged
            for m in range(3):

                for k in range (3):
                    ccscodes_thhreshold = metrics.mutual_info_score(  list(list_arc[m]), list(list_arc[k+3])) 
                    ccscodes_rand  = rand_score(list(list_arc[m]), list(list_arc[k+3]))
                    ccscodes_rand_l.append(ccscodes_rand)
                    ccscodes_thhreshold_l.append(ccscodes_thhreshold)
                    
            mean_mmutual_information = np.mean(ccscodes_thhreshold_l)
            mean_ccscodes_randindex = np.mean(ccscodes_rand_l)
            res["Name2"].append(i)
            res["Name"].append(j)
            res["mutual_information"].append(mean_mmutual_information)
            res["rand index"].append(mean_ccscodes_randindex)


                
                
                
    df_res = pd.DataFrame(res)
    df_res   
    #df_res.to_csv("./models_cluster/mi_"+u+"patient_m.csv")    
    
# Assuming 'df' is your dataframe

    with open('output_mi_patient.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(df_res.values)
if __name__ == "__main__":
    main()  