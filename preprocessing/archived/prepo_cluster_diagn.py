from model_eval import *
#stri = "Patient"
#option-->outs_visit /Patient

filtered = False
stri = "Patient"
num_run = 11
#norm_str =["std","max", "power"]
norm_str =["power"]
archivo = "data/data_preprocess_nonfilteres.csv"
df = pd.read_csv(archivo)
#real = 'threshold_0.95'
ori = 'ICD9_CODE_diagnosis'
# kmeans or other
clustering_method = "kmeans"
categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                    'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                    'MARITAL_STATUS', 'ETHNICITY','GENDER']

mean_mutual_information_l=[]
mean_ccscodes_randindex_l=[]
silhouette_avg_l=[]
davies_bouldin_avg_l=[]
real_l=[]
save = False
num_clusters = [4,8,12]
#num_clusters = [12]
#num_clusters = [4,8]

'''list_cat = [
        
       'CCS_CODES_diagnosis', 'LEVE3 CODES',
       'threshold_0.95_diagnosis', 'threshold_0.88_diagnosis',
       'threshold_0.98_diagnosis', 'threshold_0.999_diagnosis', 'ICD9_CODE_diagnosis']
list_cat = [
        
        'ICD9_CODE_diagnosis']'''


# the dictionary where the results will be stores
list_cat = [
      'ICD9_CODE_diagnosis']

result = {'Name':[],
        'Prepro':[],
        'Num Cluster':[],
        'silhouette_avg':[],
        'davies_bouldin_avg':[],
                    } 

nuevo_df_x = desconacat_codes_ori(df,ori)

for i in range(len(list_cat)):
    print(i)
    for j in norm_str:
        for n in num_clusters:
                real = list_cat[i]
                num = n
                nuevo_df4 = desconacat_codes_ori(df,real)



                duplicados = merge_df_ori(nuevo_df_x,nuevo_df4,df,categorical_cols,real) 
                #duplicados["MARITAL_STATUS"]= duplicados["MARITAL_STATUS"].replace(np.nan, "Unknown")

               
                pivot_df, agregacion_cl = pivotm_ori(duplicados,real,stri,categorical_cols,archivo)
                agregacion_cl = demo_ad(categorical_cols,agregacion_cl)
                X = firs_preprocesing_ori(pivot_df,stri,agregacion_cl,categorical_cols)    
                    
                #se normaliza
                X = preprocess(X, j)

                silhouette_avg,davies_bouldin_avg=clustering_ori(X,num,clustering_method,stri,save,ori,j,real)
                result["Prepro"].append(j)
                result["Num Cluster"].append(n)
                result["silhouette_avg"].append(silhouette_avg)
                result["davies_bouldin_avg"].append(davies_bouldin_avg)
                result["Name"].append(real)

                df_res = pd.DataFrame(result)
                   
                # the results are saved
                df_res.to_csv("experiment_prepo/prepro_experiment_"+stri+'_'+clustering_method+"_"+ori+"_nonfiltered_"+str(num_run)+".csv")
