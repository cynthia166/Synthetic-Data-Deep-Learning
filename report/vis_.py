import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
sys.path.append('')
#os.chdir('../')
from preprocessing.config import *
from preprocessing.function_pred import *
from preprocessing.config import *
import importlib
importlib.reload(sys.modules['preprocessing.config'])
from function_vis import *
importlib.reload(sys.modules['function_vis'])
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
import matplotlib.pyplot as plt
import seaborn as sns
#import pacmap
import pandas as pd
directorio_actual = os.getcwd()
print("Directorio actual:", directorio_actual)
import pandas as pd

def create_graphs(graph,type_procedur= None):
    import numpy as np  
    
    import matplotlib.pyplot as plt
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    import pandas as pd
    
    if graph == "hyperparameter" or graph == "Readmission_Proceduers":
        df = pd.read_pickle("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/procedures/prediction/final_results_visit_30.pkl")
        df_d = pd.concat(df)
        df =df_d.groupby(["Name","Classifier"]).mean().reset_index()
        print(df.shape)
      
        import regex as re
        def process_dataframe(df):
            def process_column_name(col):
                if 'sin_codigo' in col:
                    
                    return 'without code'
                elif 'CCS CODES' in col:
                    return 'ccs codes' 
                else:
                    threshold = re.search(r'threshold_([0-9.]+)', col)
                    if threshold:
                        return 'threshold ' + threshold.group(1)
                    else:
                        return col.replace('_drugst_non_prepo.csv', '')
            
            df = df.rename(columns={df.columns[0]: 'Method'})
            
            df['Method'] = [process_column_name(col) for col in list(df.Method)]
            return df
        df = process_dataframe(df)    
        #df["Mapping"] = df["Mapping"].replace(reemplazos1)

        cols = ['f1_test', 'f1_train', 'sensitivity_test',
            'specificity_test', 'precision_test', 'accuracy_test',
            'sensitivity_train', 'specificity_train', 'precision_train',
            'accuracy_train',
            'Classifier', 	 ]

        print("real",df.shape)
        
        
        print(list(df.Classifier.unique()))
       
        
        
        aux_logistic= df[df["Classifier"]=="logistic"]
        name = "Logistic Regression"
        create_heatmap(aux_logistic, name,"Logistic_drud",type_procedur)
        
        aux_xgboost= df[df["Classifier"]=="xgboost"]
        name = "XGB Classfier"
        create_heatmap(aux_xgboost, name,"XGboost_drud",type_procedur)
            
    if graph == "hyperparameter_drugs" or graph == "Readmission_drugs":
        df = pd.read_pickle("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/drugs/prediction/final_results_visit_30.pkl")
        df_d = pd.concat(df)
        df =df_d.groupby(["Name","Classifier"]).mean().reset_index()
        print(df.shape)
      
        import regex as re
        def process_column_name(col):
            if 'sin_codigo' in col:
                return 'without code'
            else:
                threshold = re.search(r'threshold_presence_([0-9.]+)', col)
                if threshold:
                    return 'threshold ' + threshold.group(1)
                else:
                    return col.replace('_drugst_non_prepo.csv', '')
        df = df.rename(columns={df.columns[0]: 'Method'})
        df['Method'] = [process_column_name(col) for col in list(df.Method)]
        #df["Mapping"] = df["Mapping"].replace(reemplazos1)

        cols = ['f1_test', 'f1_train', 'sensitivity_test',
            'specificity_test', 'precision_test', 'accuracy_test',
            'sensitivity_train', 'specificity_train', 'precision_train',
            'accuracy_train',
            'Classifier', 	 ]

        print("real",df.shape)
        
        
        print(list(df.Classifier.unique()))
       
        
        
        aux_logistic= df[df["Classifier"]=="logistic"]
        name = "Logistic Regression"
        create_heatmap(aux_logistic, name,"Logistic_drud")
        
        aux_xgboost= df[df["Classifier"]=="xgboost"]
        name = "XGB Classfier"
        create_heatmap(aux_xgboost, name,"XGboost_drud")
        
        if  graph == "hyperparameter_drugs":
            #######################################################################################
            #######################################################################################
            ########Best hyperparameters Logistic Regression######################################
        
            df1= pd.read_csv(DATA_DIRECTORY_results/'drugs/xgbost_rp_druga.csv')
            df1["Mapping"] = df1["Mapping"].replace(reemplazos1)
            lista2 = ['f1_test_a', 'f1_train_a', 'sensitivity_test_a', 'specificity_test_a',
                'precision_test_a', 'accuracy_test_a', 'sensitivity_train_a',
                'specificity_train_a', 'precision_train_a', 'accuracy_train_a']
            lista = ['f1_test', 'f1_train', 'sensitivity_test', 'specificity_test', 'precision_test', 'accuracy_test', 'sensitivity_train', 'specificity_train', 'precision_train', 'accuracy_train']
            metric_list = ['sensitivity', 'f1', 'accuracy', 'precision','specificity']
            title_ = "Model Validation Xgboost model (Drugs)"
            df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression")&(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression")&(df_sin_duplicados_columnas_especificas["Sampling"]=="non")&(df_sin_duplicados_columnas_especificas["Feature selection"]==True)]
            df_sin_duplicados_columnas_especificas_ = agregar_anterior_hp_S(df_sin_duplicados_columnas_especificas_,lista)
            crear_tabla_results(df_sin_duplicados_columnas_especificas_,df1,title_,metric_list,lista,"lr_graph_Hyperparameterruning")
            #df_sin_duplicados_columnas_especificas_.groupby("Mapping")["sensitivity_test"].idxmax()
            #df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="XGBClassifier") ]
            #df_sin_duplicados_columnas_especificas_.loc[df_sin_duplicados_columnas_especificas_.groupby("Mapping")["f1_test"].idxmax()].drop_duplicates(subset='Mapping', keep='first')
            crear_datafram_hyperparametros(df1)
            df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression") ]
            df_sin_duplicados_columnas_especificas_ = df_sin_duplicados_columnas_especificas_.loc[df_sin_duplicados_columnas_especificas_.groupby("Mapping")["f1_test"].idxmax()].drop_duplicates(subset='Mapping', keep='first')
            aux3 = df_sin_duplicados_columnas_especificas_[df_sin_duplicados_columnas_especificas_["Mapping"]!= "No ICD9-Codes"]
            #hyperparameter from wandb
            df = pd.read_csv(DATA_DIRECTORY_results/'drugs/drugs_lr_hp.csv')
            df["Mapping"] = df["Mapping"].replace(reemplazos1)
            lista2 = ['f1_test_a', 'f1_train_a', 'sensitivity_test_a', 'specificity_test_a',
                'precision_test_a', 'accuracy_test_a', 'sensitivity_train_a',
                'specificity_train_a', 'precision_train_a', 'accuracy_train_a']
            lista = ['f1_test', 'f1_train', 'sensitivity_test', 'specificity_test', 'precision_test', 'accuracy_test', 'sensitivity_train', 'specificity_train', 'precision_train', 'accuracy_train']
            metric_list = ['sensitivity', 'f1', 'accuracy', 'precision','specificity']
            title_ = "Model Validation Logistic Regression model (Drugs)"
            df_sin_duplicados_columnas_especificas_ = agregar_anterior_hp_S(df_sin_duplicados_columnas_especificas_,lista)
            crear_tabla_results(df_sin_duplicados_columnas_especificas_,df,title_, metric_list,lista,"drugs_hyperparametes")
            create_best_lg(df,reemplazos1)
            #### CREATE PREDICTIONS" Logistic regression###
            df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression") ]
            df_sin_duplicados_columnas_especificas_ = df_sin_duplicados_columnas_especificas_.loc[df_sin_duplicados_columnas_especificas_.groupby("Mapping")["f1_test"].idxmax()].drop_duplicates(subset='Mapping', keep='first')
            df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="XGBClassifier") ]
            df_sin_duplicados_columnas_especificas_ = df_sin_duplicados_columnas_especificas_.loc[df_sin_duplicados_columnas_especificas_.groupby("Mapping")["f1_test"].idxmax()].drop_duplicates(subset='Mapping', keep='first')
            create_heatmap(df_sin_duplicados_columnas_especificas_,"XGB Classifier", "_xgbost_drugs_")
            df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression")&(df_sin_duplicados_columnas_especificas["Sampling"]=="non")&(df_sin_duplicados_columnas_especificas["Feature selection"]==True)]
            create_graph(df_sin_duplicados_columnas_especificas_.reset_index(),"Logistic Regression", "LR")
            df_sin_duplicados_columnas_especificas_
            df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="XGBClassifier")&(df_sin_duplicados_columnas_especificas["Sampling"]=="non")&(df_sin_duplicados_columnas_especificas["Feature selection"]==True)]
            create_graph(df_sin_duplicados_columnas_especificas_,"XGB Classifier","Lr_")
            print(len(list(df_sin_duplicados_columnas_especificas.Sampling.unique())))
            print("real",df_sin_duplicados_columnas_especificas.shape)
            print("sin_duplciados",df_sin_duplicados_columnas_especificas.shape)
            print(list(df_sin_duplicados_columnas_especificas.Mapping.unique()))
            print(list(df_sin_duplicados_columnas_especificas.Classifiers.unique()))
            print(list(df_sin_duplicados_columnas_especificas.Mapping.unique()))
            print(list(df_sin_duplicados_columnas_especificas.Mapping.unique()))
            print(list(df_sin_duplicados_columnas_especificas['Feature selection'].unique()))
            print(len(list(df_sin_duplicados_columnas_especificas.Mapping.unique())))
            print(len(list(df_sin_duplicados_columnas_especificas.Sampling.unique())))
            df_sin_duplicados_columnas_especificas.Sampling.unique()
    if graph == "hyperparameter_diagnosis" or graph == "Readmission_diagnosis":
        df = pd.read_pickle("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/diagnosis/prediction/final_results_visit_30.pkl")
        df_d = pd.concat(df)
        df =df_d.groupby(["Name","Classifier"]).mean().reset_index()
        print(df.shape)
        
        import regex as re
        def process_dataframe(df):
            def process_column_name(col):
                if 'sin_codigo' in col:
                    
                    return 'without code'
                elif 'CCS CODES' in col:
                    return 'ccs codes' 
                elif 'LEVE3' in col:
                    return 'level 3 codes' 
                else:
                    threshold = re.search(r'threshold_([0-9.]+)', col)
                    if threshold:
                        return 'threshold ' + threshold.group(1)
                    else:
                        return col.replace('_drugst_non_prepo.csv', '')
            
            df = df.rename(columns={df.columns[0]: 'Method'})
            
            df['Method'] = [process_column_name(col) for col in list(df.Method)]
            return df
        df = process_dataframe(df)    
        #df["Mapping"] = df["Mapping"].replace(reemplazos1)

  

        print("real",df.shape)
        
        
        print(list(df.Classifier.unique()))
       
        
        
        aux_logistic= df[df["Classifier"]=="logistic"]
        name = "Logistic Regression"
        create_heatmap(aux_logistic, name,"Logistic_drud",type_procedur)
        
        aux_xgboost= df[df["Classifier"]=="xgboost"]
        name = "XGB Classfier"
        create_heatmap(aux_xgboost, name,"XGboost_drud",type_procedur)
       
    if  graph =="get_input_preprocess_results":
        arch_name = 'prepro_experiment_Patient_kmeans_ICD9_CODE_diagnosis'
        def get_concatenated_archives(arch_name):
            #directory_path = Path("results/experimernt_prepro/diagnosis/experiment_prepo")
            directory_path = DIAGNOSIS_DIRECTORY_e
            # List all files in the directory
            files = [str(f) for f in directory_path.glob('*') if arch_name in str(f)]
            list_data =[]
            for i in files:
                list_data.append(pd.read_csv(i))    
            concatenated_df = pd.concat(list_data, axis=0)
            print(concatenated_df[concatenated_df.notnull()].head()  )
            concatenated_df = concatenated_df.drop_duplicates(subset=set(concatenated_df.columns).difference(
                    {'Unnamed: 0.1','index','silhouette_avg','davies_bouldin_avg','Unnamed: 0','level_0'}),
                ignore_index=True)
            df_p1 = concatenated_df.reset_index() 
            return df_p1
        arch_name = 'prepro_experiment_Patient_kmeans_ICD9_CODE_diagnosis'
        df_p = get_concatenated_archives(arch_name)
        print(df_p.shape)
        arch_name_v = 'prepro_experiment_outs_visit_kmeans_ICD9_CODE_diagnosis'
        df_p1 = get_concatenated_archives(arch_name_v)
        print(df_p1.shape)
        # List all files that contain 'prepro_experiment_outs_visitkmeans' in their names
        #df_p1 = concatenated_df2.reset_index() #visit
        #df_p = concatenated_df.reset_index()
        df_p1.to_csv(DIAGNOSIS_DIRECTORY/"visit_results_prepro_nonfilteres.csv")
        df_p1 = df_p1.rename(columns={'silhouette_avg': 'silhouette_avg_v',
                                    'davies_bouldin_avg': 'davies_bouldin_avg_v',
                                    })
        merged_df = pd.merge(df_p, df_p1, on=["Name", "Prepro", "Num Cluster"] , how = 'inner')
        print(merged_df.shape)
        merged_df['Name'] = merged_df['Name'].replace('CCS_CODES_diagnosis', 'CCS CODES')    
        merged_df['Name'] = merged_df['Name'].replace('LEVE3 CODES', 'LEVEL 3 CODES')   
        merged_df['Name'] = merged_df['Name'].replace('threshold_0.88_diagnosis', 'threshold 0.88')   
        merged_df['Name'] = merged_df['Name'].replace('threshold_0.95_diagnosis', 'threshold 0.95') 
        merged_df['Name'] = merged_df['Name'].replace('threshold_0.98_diagnosis', 'threshold 0.98')   
        merged_df['Name'] = merged_df['Name'].replace('threshold_0.999_diagnosis', 'threshold 0.999')   
        merged_df['Name'] = merged_df['Name'].replace('DRUG_y', 'Medicament')            
        merged_df['Name'] = merged_df['Name'].replace('threshold_0.95', 'threshold 0.95')   
        merged_df['Name'] = merged_df['Name'].replace('threshold_0.88', 'threshold 0.88')  
        merged_df['Name'] = merged_df['Name'].replace('threshold_0.98', 'threshold 0.98')   
        merged_df['Name'] = merged_df['Name'].replace('threshold_0.999', 'threshold 0.999')   
        merged_df.to_csv(DIAGNOSIS_DIRECTORY/ "results_prepro_nonfilteres_diagnosis_final.csv")
    replace_dict_prod = {
        'CCS CODES_proc': 'CCS CODES',
        'ICD9_CODE_procedures': 'ICD9_CODE',
        'cat_threshold .88 most frequent': 'threshold .88',
        'cat_threshold .999 most frequent': 'threshold .999',
        'cat_threshold .98 most frequent': 'threshold .98',
        'cat_threshold .95 most frequent_proc': 'threshold .95',
        'sin_codigo': 'No ICD9-Code'
    }
    if graph == "preprocessing_clustering_exp" :
        #drugs = DRUGS_DIRECTORY/"experiment_preporesults_prepro_nonfilteres_DRUGS_final.csv"\
        #diagnosis = DIAGNOSIS_DIRECTORY/ "results_prepro_nonfilteres_diagnosis_final.csv"
        #procedures = PROCEDURES_DIRECTORY/"results_final_merged_procedures_prepo.csv"
        #drugs_patient = pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/drugs/Patient/prepro_experiment_Patientkmeans_threshold_presence_0.999_drugst_non_prepo.csv.csv")
       # drugs_patient = pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/procedures/Patient/prepro_experiment_Patientkmeans_threshold_0.8_procedures_non_prepo.csv.csv")
       
        drugs_patient = pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/diagnosis/Patient/conatenated_res_patients.csv")
        #drugs_visit =  pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/drugs/final_clustering/visit/prepro_experiment_visitkmeans_threshold_presence_0.999_drugst_non_prepo.csv.csv")
        #drugs_visit= pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/procedures/visit/prepro_experiment_visitkmeans_threshold_0.9_procedures_non_prepo.csv.csv")
        drugs_visit= pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/diagnosis/visit/prepro_experiment_visitkmeans_threshold_0.999_diagnosis_non_prepo.csv.csv")
        drugs_visit = drugs_visit.rename(columns={'silhouette_avg': 'silhouette_avg_v',
                                    'davies_bouldin_avg': 'davies_bouldin_avg_v',
                                    })
        drugs_merged = pd.merge(drugs_patient, drugs_visit,  on=['Name','Prepro','Num Cluster'], how='inner')
        import regex as re
  
        def process_column_name(col):
                if 'sin_codigo' in col:
                    return 'without code'
                elif 'CCS CODES' in col:
                    return 'ccs codes'
                elif 'LEVE3 CODES' in col:
                    return 'level 3 codes'
                else:
                    threshold = re.search(r'threshold_([0-9.]+)', col)
                    if threshold:
                        return 'threshold ' + threshold.group(1)
                    else:
                        return col.replace('_drugst_non_prepo.csv', '')
            
        
        drugs_merged['Name'] = [process_column_name(col) for col in list(drugs_merged.Name)]
       
          #merged_df = pd.read_csv("experiment_prepo"+"results_prepro_nonfilteres_diagnosis_final.csv"
        #,diagnosis,procedures
        for j,i in enumerate([drugs_merged]):
            if j ==2:     
                merged_df = pd.read_csv(i)      
                #merged_df["Name"] = merged_df["Name"].replace(replace_dict_prod)       
                merged_df = merged_df[merged_df['Name']!='ICD9_CODE']                                                                                         
            else:                                                                                                                                  
               # merged_df = pd.read_csv(i)
                merged_df=i
               
            idx = merged_df.groupby(['Name'])['silhouette_avg'].idxmax()
            # Usar los índices para obtener las filas correspondientes
            top_silhouette_avg_per_name = merged_df.loc[idx]
            top_silhouette_avg_per_name
            idx = merged_df.groupby('Name')['silhouette_avg'].idxmax()
            # Usar los índices para obtener las filas correspondientes
            top_silhouette_avg_per_name = merged_df.loc[idx]
            top_silhouette_avg_per_name = top_silhouette_avg_per_name[top_silhouette_avg_per_name['Name']!="Medicament"]
            top_silhouette_avg_per_name
            idx = merged_df.groupby('Name')['silhouette_avg_v'].idxmax()
            # Usar los índices para obtener las filas corre}
            # spondientes
            top_silhouette_avg_per_name_v = merged_df.loc[idx]
            top_silhouette_avg_per_name_v = top_silhouette_avg_per_name_v[top_silhouette_avg_per_name_v['Name']!="Medicament"]
            top_silhouette_avg_per_name_v
         
            # Definir patrones para el número de clusters (no se usará para el color en este ejemplo)
            patterns = {4: "//", 8: "+++", 12: "xx"}
            # Definir paleta de colores de tonos de azul basada en el tipo de preprocessing
            palette = {"power": "lightblue", "max": "blue", "std": "steelblue"}
            # Crear DataFrames de muestra con una columna 'Preprocessing'
            # Crear una grilla de subtramas 1x2
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            titles = ["Preprocessing patient level", "Preprocessing visit level"]
            scores = ["Silhouette score"]*2
            for i, (ax, title) in enumerate(zip(axs, titles)):
                df = top_silhouette_avg_per_name if i == 0 else top_silhouette_avg_per_name_v
                metric = 'silhouette_avg' if i == 0 else 'silhouette_avg_v'
                # Asignar colores basados en el tipo de preprocessing
                colors = [palette[pre] for pre in df["Prepro"]]
                ax.bar(df["Name"], df[metric], color=colors, edgecolor='black')     
                ax.set_title(title, fontsize=16)
                ax.set_ylabel(scores[i], fontsize=14)
                ax.set_xlabel("ICD-9 Codes", fontsize=14)
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.set_xticklabels(df["Name"], rotation=90)
            # Crear leyendas personalizadas para la paleta de colores
            prepro_patches = [Patch(color=palette[key], label=key.capitalize()) for key in palette]
            plt.legend(handles=prepro_patches, loc='upper left', bbox_to_anchor=(1, 1), title="Preprocessing Types")
            plt.tight_layout()
            #plt.savefig(IMAGES_Cluster_DICT+'silhouate_scores_'+str(i)+'_.svg')
            plt.savefig('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/images/'+type_procedur+'/silhouate_scores_'+str(i)+'_.svg')
            plt.show()
            for i in [0,1]:
                if i ==0:
                    level = "patient"
                else:
                    level = "visit"    
                df = top_silhouette_avg_per_name if i == 0 else top_silhouette_avg_per_name_v
                # Definir paleta de colores para preprocessing
                palette = {"power": "lightblue", "max": "cyan", "std": "steelblue"}
                # Definir patrones para el número de clusters
                patterns = {4: "//", 8: "xx", 12: "++"}
                # Crear la figura y el eje
                fig, ax = plt.subplots(figsize=(10, 7))
                # Bucle para dibujar las barras
                for i, row in df.iterrows():
                    color = palette[row['Prepro']]
                    hatch = patterns.get(row['Num Cluster'], '')  # Obtener el patrón o '' si no existe
                    ax.bar(row['Name'], row['silhouette_avg'], color=color, hatch=hatch, edgecolor='black', label=row['Preprocessing'] if i in palette else "_nolegend_")
                # Ajustar detalles del gráfico
                #ax.set_xlabel('Simplification', fontsize=12)
                ax.set_ylabel('Silhouette Score', fontsize=12)
                ax.set_title('Silhouette Scores by preprocessing type and number cluster ('+level+' level)', fontsize=16)
                # Crear leyendas personalizadas
                prepro_patches = [Patch(color=palette[key], label=key) for key in palette]
                pattern_patches = [Patch(facecolor='white', edgecolor='black', hatch=patterns[key], label=f'Cluster {key}') for key in patterns]
                # Añadir leyendas al gráfico
                legend1 = ax.legend(handles=prepro_patches, title="Preprocessing Types", loc='upper left', bbox_to_anchor=(1, 0.5))
                legend2 = ax.legend(handles=pattern_patches, title="Num Cluster", loc='upper left', bbox_to_anchor=(1, 0.3))
                ax.add_artist(legend1)  # Añadir de nuevo la primera leyenda después de la segunda
                # Set the fontsize for x-axis tick labels
                ax.tick_params(axis='x', labelsize=14,rotation = 45)
                plt.savefig('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/images/'+type_procedur+'/real_Preprocessing'+level+'_'+str(i)+'.svg')
                plt.tight_layout()
                plt.show()
    if graph =="get_rand_index":
        type_in = type_procedur
        
        if type_in == "drugs":
            directory_path = DRUGS_DIRECTORY
            mi_pa = pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/"+type_procedur+"/mutual_information/randindex_drugs_Patient.csv") #Patient
            mi_pa1 = pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/"+type_procedur+"/mutual_information/randindex_drugs_visit.csv") #visit
        elif type_in == "diagnosis":
        # Drop the extra column from mi_pa1
            directory_path =DIAGNOSIS_DIRECTORY 
            mi_pa = pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/diagnosis/mutual_information/randindex_drugs_patient.csv")
            mi_pa1 = pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/diagnosis/mutual_information/randindex_drugs_visit.csv")
        elif type_in == "procedures":
            directory_path =PROCEDURES_DIRECTORY 
            mi_pa = pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/"+type_procedur+"/mutual_information/randindex_drugs_Patient.csv") #Patient
            mi_pa1 = pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/"+type_procedur+"/mutual_information/randindex_drugs_visit.csv") #visit
       #mi_pa1 = mi_pa1.drop(columns=['Unnamed: 0'])
        mi_pa.rename(columns={"mutual_information":"mutual_information_p","rand index":"rand index_p","Name":"Mapping","Name2":"Column2"}, inplace = True)
        mi_pa1.rename(columns={"Name":"Column1","Name2":"Column2"}, inplace = True)
        mi_pa.rename(columns={"Mapping":"Column1","Name2":"Column2"}, inplace = True)
        

        import regex as re
        def process_dataframe(df):
            def process_column_name(col):
                if 'sin_codigo' in col:
                    
                    return 'without code'
                elif 'CCS CODES' in col:
                    return 'ccs codes' 
                elif 'LEVE3 CODES' in col:
                    return 'level 3 codes'
                else:
                    threshold = re.search(r'threshold_([0-9.]+)', col)
                    if threshold:
                        return 'threshold ' + threshold.group(1)
                    else:
                        return col.replace('_drugst_non_prepo.csv', '')
            
            df = df.rename(columns={df.columns[0]: 'Method'})
           
            df['Method'] = [process_column_name(col) for col in list(df.Method)]
            # Rename the first column to 'Method'
                
            # Add "without code" for methods containing 'sin_codigo'
            
            # Process column names
            new_columns = ['Method']
            for col in df.columns[1:]:
                if 'sin_codigo' in col:
                    new_col=' without code'
                elif 'CCS CODES' in col:
                     new_col='ccs codes'    
                elif 'LEVE3 CODES' in col:
                    new_col= 'level 3 codes'     
                else:    
                    threshold = re.search(r'threshold_([0-9.]+)', col)
                    if threshold:
                        new_col = 'threshold ' + threshold.group(1)
                    else:
                        new_col = col
                new_columns.append(new_col)
                
            df.columns = new_columns
            
            # Set 'Method' as index
            df = df.set_index('Method')
            
            # Convert to float and round
            df = df.astype(float).round(3)
            
            return df
        pivot_df1 = process_dataframe(mi_pa)
        pivot_df2 = process_dataframe(mi_pa1)

        if type_procedur =="drugs":
           column_order = ['threshold 0.8', 'threshold 0.85', 'threshold 0.9', 'threshold 0.95', 'threshold 0.98', 'threshold 0.999', 'without code']
        elif type_procedur =="procedures":
            column_order = ['ccs codes','threshold 0.8', 'threshold 0.85', 'threshold 0.9', 'threshold 0.95', 'threshold 0.98', 'threshold 0.999', 'without code']
        else:
                column_order = ['ccs codes','level 3 codes', 'threshold 0.85', 'threshold 0.9', 'threshold 0.95', 'threshold 0.98', 'threshold 0.999', 'without code']
        # Define the desired index order (assuming these are the ICD-9 codes)
        index_order = pivot_df1.index.tolist()
        index_order.sort()  # Sort the ICD-9 codes alphanumerically

        # Reorder the columns and index for both DataFrames
        pivot_df1 = pivot_df1.reindex(columns=column_order, index=index_order)
        pivot_df2 = pivot_df2.reindex(columns=column_order, index=index_order)
        if type_procedur =="drugs":
            pivot_df1['without code'] = [i[0] for i in pivot_df1[pivot_df1.index == 'without code'].values[:,:-1].T] +[1]
            pivot_df2['without code'] = [i[0] for i in pivot_df2[pivot_df2.index == 'without code'].values[:,:-1].T] +[1]
            
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.heatmap(pivot_df1, cmap='Blues', annot=True, fmt=".3f", cbar=False)
        plt.xlabel('ICD-9 Codes')
        plt.ylabel('')
        plt.title('Heatmap of rand index patient level')
        # Create the second subplot for the second heatmap
        plt.subplot(1, 2, 2)
        sns.heatmap(pivot_df2, cmap='Blues', annot=True, fmt=".3f")
        plt.xlabel('ICD-9 Codes')
        plt.ylabel('')
        plt.title('Heatmap of rand index visit level')
        # Adjust the spacing between subplots
        plt.tight_layout()
        plt.savefig("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/images/"+type_in+'/rand_index.svg')
        # Show the plot
        plt.show()
    
              
    if graph =="get_mutula_information":
        type_in = type_procedur
        
        if type_in == "drugs":
            directory_path = DRUGS_DIRECTORY
            mi_pa = pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/drugs/mutual_information/mutualinf_drugs_Patient.csv") #Patient
            mi_pa1 = pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/drugs/mutual_information/mutualinf_drugs_visit.csv") #visit
        elif type_in == "diagnosis":
        # Drop the extra column from mi_pa1
            directory_path =DIAGNOSIS_DIRECTORY 
            mi_pa = pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/diagnosis/mutual_information/mutualinf_drugs_patient.csv")
            mi_pa1 = pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/diagnosis/mutual_information/mutualinf_drugs_visit.csv")
        elif type_in == "procedures":
            directory_path =PROCEDURES_DIRECTORY 
            mi_pa = pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/"+type_procedur+"/mutual_information/randindex_drugs_Patient.csv") #Patient
            mi_pa1 = pd.read_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/models_cluster/"+type_procedur+"/mutual_information/randindex_drugs_visit.csv") #visit
   #mi_pa1 = mi_pa1.drop(columns=['Unnamed: 0'])
        mi_pa.rename(columns={"mutual_information":"mutual_information_p","rand index":"rand index_p","Name":"Mapping","Name2":"Column2"}, inplace = True)
        mi_pa1.rename(columns={"Name":"Column1","Name2":"Column2"}, inplace = True)
        mi_pa.rename(columns={"Mapping":"Column1","Name2":"Column2"}, inplace = True)
        

        import regex as re
        def process_dataframe(df):
            def process_column_name(col):
                if 'sin_codigo' in col:
                    return 'without code'
                elif 'CCS CODES' in col:
                    return 'ccs codes' 
                elif 'LEVE3 CODES' in col:
                    return 'level 3 codes'
            
              
                else:
                    threshold = re.search(r'threshold_([0-9.]+)', col)
                    if threshold:
                        return 'threshold ' + threshold.group(1)
                    else:
                        return col.replace('_drugst_non_prepo.csv', '')
            
            df = df.rename(columns={df.columns[0]: 'Method'})
           
            df['Method'] = [process_column_name(col) for col in list(df.Method)]
            # Rename the first column to 'Method'
                
            # Add "without code" for methods containing 'sin_codigo'
            
            # Process column names
            new_columns = ['Method']
            for col in df.columns[1:]:
                if 'sin_codigo' in col:
                    new_col=' without code'
                elif 'CCS CODES' in col:
                    new_col= 'ccs codes' 
                elif 'LEVE3 CODES' in col:
                    new_col='level 3 codes'
                
                
                else:    
                    threshold = re.search(r'threshold_([0-9.]+)', col)
                    if threshold:
                        new_col = 'threshold ' + threshold.group(1)
                    else:
                        new_col = col
                new_columns.append(new_col)
                
            df.columns = new_columns
            
            # Set 'Method' as index
            df = df.set_index('Method')
            
            # Convert to float and round
            df = df.astype(float).round(3)
            
            return df
        pivot_df1 = process_dataframe(mi_pa)
        pivot_df2 = process_dataframe(mi_pa1)
        
        if type_procedur == "diagnosis":
            column_order = ['ccs codes','level 3 codes', 'threshold 0.85', 'threshold 0.9', 'threshold 0.95', 'threshold 0.98', 'threshold 0.999', 'without code']
        else:
            column_order = ['ccs codes','threshold 0.8', 'threshold 0.85', 'threshold 0.9', 'threshold 0.95', 'threshold 0.98', 'threshold 0.999', 'without code']
        # Define the desired index order (assuming these are the ICD-9 codes)
        index_order = pivot_df1.index.tolist()
        index_order.sort()  # Sort the ICD-9 codes alphanumerically

        # Reorder the columns and index for both DataFrames
        pivot_df1 = pivot_df1.reindex(columns=column_order, index=index_order)
        pivot_df2 = pivot_df2.reindex(columns=column_order, index=index_order)


        # Assuming you have a DataFrame named 'df' with the columns 'Column1', 'Column2', 'MI', and 'RI'
        # Reshape the DataFrame using pivot
        #pivot_df1 = mi_pa.pivot(index='Column2', columns='Column1', values='mutual_information_p')
        #pivot_df2 = mi_pa1.pivot(index='Column2', columns='Column1', values='mutual_information')

        pivot_df1 = pivot_df1.astype(float)
        pivot_df1 = pivot_df1.round(3)
        pivot_df2 = pivot_df2.astype(float)
        pivot_df2 = pivot_df2.round(3)
        # Create the first subplot for the first heatmap
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.heatmap(pivot_df1, cmap='Blues', annot=True, fmt=".3f", cbar=False)
        plt.xlabel('ICD-9 Codes')
        plt.ylabel('')
        plt.title('Heatmap of mutual information patient level')
        # Create the second subplot for the second heatmap
        plt.subplot(1, 2, 2)
        sns.heatmap(pivot_df2, cmap='Blues', annot=True, fmt=".3f")
        plt.xlabel('ICD-9 Codes')
        plt.ylabel('')
        plt.title('Heatmap of mutual information visit level')
        # Adjust the spacing between subplots
        
        
        # Adjust the spacing between subplots
        plt.tight_layout()
        
        plt.tight_layout()
        plt.savefig("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/images/"+type_in+'/mutual_information.svg')
        # Show the plot
        plt.show()
    if graph == "demo_pie" :
        import matplotlib.pyplot as plt
        #demos  juntos
        df, grouped_df = demos_pies()
        grouped_df = df
        variables = ['ADMISSION_TYPE', 'INSURANCE', 'GENDER', 'EXPIRE_FLAG', 'ADMISSION_LOCATION','DISCHARGE_LOCATION','MARITAL_STATUS','LANGUAGE','RELIGION','ETHNICITY',]
        title = ['ADMISSION TYPE', 'INSURANCE', 'GENDER',  'DEATH','ADMISSION LOCATION','DISCHARGE LOCATION','MARITAL STATUS','LANGUAGE','RELIGION','ETHNICITY',]
        fig, axes = plt.subplots(5, 2, figsize=(60, 60))
        fig.patch.set_facecolor('white')
        axes = axes.flatten()
        for i, variable in enumerate(variables):          
            counts = grouped_df[variable].value_counts()
            if variable != 'EXPIRE_FLAG' and variable !="GENDER":  
                top_2 = counts.head(2)  
                other_count = counts.shape[0] - 2
                other_label = f'Other ({other_count:,})'
                top_2_counts = top_2.append(pd.Series([counts.sum() - top_2.sum()], index=[other_label]))
                colors = sns.color_palette('Blues', len(top_2_counts))
            else:
                top_2_counts = counts.head(2)  # Select top 2 categories for other variables
                colors = sns.color_palette('Blues', len(top_2_counts))       
            pie = axes[i].pie(top_2_counts, labels=None, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 30}) 
            axes[i].legend(pie[0], top_2_counts.index, loc='center left', fontsize=30, bbox_to_anchor=(1, 0.5))
            axes[i].set_title(title[i], fontsize=30)  
            axes[i].set_aspect('equal')  # Ensure pie is circular
            axes[i].set_xlabel('')  # Remove x-axis label
            axes[i].set_ylabel('')  # Remove y-axis label
        plt.savefig(IMAGES_Demo+'demo_pie.svg')
        plt.tight_layout()
        #plt.show()
        #demos separated
    if graph =="Kernel_density_estimation":    
        grouped_df1,df2 = get_dft()
        plot_age(df2)
        plot_los(df2)
    if graph== "pacmap_representation":
        dataframes =get_dataframes()
        # Assuming you have a list of 7 DataFrames
        #dataframes = [df1, df2, df3, df4, df5, df6, df7]
        title = ['threshold .88',
        'threshold .95',
        'threshold .999',
        'threshold .98',
        'CCS CODES',
        'ICD-9 CODE']
        # Set up the matplotlib figure
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))  # Adjust the size as needed
        axes = axes.flatten()  # Flatten the axes array for easy looping
        # Loop through each DataFrame, perform PaCMAP embedding, and create a subplot
        for i, df in enumerate(dataframes):
            if dataframes[0].shape[1]!=1:
                # Perform PaCMAP embedding for each DataFrame
                embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0) 
                X_transformed = embedding.fit_transform(df, init="pca")
            # Plotting the transformed data
                sns.scatterplot(
                    x=X_transformed[:, 0],
                    y=X_transformed[:, 1],
                    ax=axes[i]
                )
                axes[i].set_title(title[i]+' Embedding')
                axes[i].set_xlabel('Component 1')
                axes[i].set_ylabel('Component 2')
                # Hide any unused subplots
                for j in range(i + 1, len(axes)):
                    axes[j].set_visible(False)
                plt.tight_layout()
                plt.savefig(IMAGES_Cluster_DICT+'pacmap_prod.svg')
                plt.show()
    if graph== "10_most_frequent":            
               
        if type_procedur=="procedures":
            group = "'ICD-9-CM CODE DESCRIPTION'"
            df = pd.read_csv( DARTA_INTERM/"procedures_preprocess_threshold_nonfiltered.csv")
        elif type_procedur=="diagnosis":
            group = "'CCS CATEGORY DESCRIPTION'"
            df = pd.read_csv(DARTA_INTERM/"diagnosis_preprocess_nonfiltered.csv")
        elif type_procedur=="medicament":    
            df = pd.read_csv(DARTA_INTERM/"drugs2_preprosss_non_preprocess.csv")
            group  ="DRUG"
        result = df.groupby([group]).size().reset_index(name='Count')
        result = result.sort_values(by="Count", ascending=False)[:10]
        plt.figure(figsize=(10, 6))

        # La anchura de las barras puede ser ajustada con el parámetro 'height'
        sns.barplot(x='Count', y=group, data=result, color='skyblue') 

        # Eliminar el borde de cada barra para un aspecto más limpio
        sns.despine()

        # Ajusta la estética del gráfico
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Count', fontsize=14)
        plt.ylabel(type_procedur, fontsize=14)
        plt.title('Top 10 Most Frequent ' + type_procedur, fontsize=14)

        # Ajusta el layout para asegurarse de que todo encaje bien
        plt.tight_layout()

        # Guardar la figura con calidad alta
        #plt.savefig(IMAGES_Demo + type_procedur + 'top_tencounts_.png', format='png', dpi=300)
        plt.savefig(IMAGES_Demo+type_procedur+'top_tencounts_.svg') 
        plt.show()
        '''plt.figure(figsize=(8, 8))  # Cambia el tamaño de la figura para acomodar las barras verticales
        sns.barplot(x='Count', y=group, data=result, color='lightblue')  # Intercambia x e y
        plt.yticks(rotation=0, fontsize=12)  # Cambia xticks a yticks
        plt.ylabel(type_procedur, fontsize=14)  # Intercambia xlabel y ylabel
        plt.xlabel('Count', fontsize=14)  # Intercambia xlabel y ylabel
        plt.title('Top 10 Most Frequent '+type_procedur, fontsize=16)
        plt.savefig(IMAGES_Demo+type_procedur+'top_tencounts_.svg')
        plt.show()'''
        sns.set(style="darkgrid")
    # Load the iris dataset
    if graph== "countpr_admi_patient": 
        if type_procedur=="procedures":
            group = "'ICD-9-CM CODE DESCRIPTION'"
            df = pd.read_csv( DARTA_INTERM/"procedures_preprocess_threshold_nonfiltered.csv")
        elif type_procedur=="diagnosis":
            group = "'CCS CATEGORY DESCRIPTION'"
            df = pd.read_csv(DARTA_INTERM/"diagnosis_preprocess_nonfiltered.csv")
        elif type_procedur=="drugs":    
            df = pd.read_pickle("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/images/drugs/drugs_raw.pkl")
            group  ="DRUG"
 
        if type_procedur =="procedures":
            result_subject = df.groupby("SUBJECT_ID").size().reset_index(name='Count')
            result_admission = df.groupby("HADM_ID").size().reset_index(name='Count')
            result_subject_1 = result_subject[result_subject["Count"]<100]
            result_admission_1 = result_admission[result_admission["Count"]<1000] 
            label_xl = 'Count of ICD-9 codes'
            type_codes="ICD-9 codes"
            type_procedur_l =type_procedur
        elif type_procedur=="diagnosis":
            result_subject = df.groupby("SUBJECT_ID").size().reset_index(name='Count')
            result_admission = df.groupby("HADM_ID").size().reset_index(name='Count')
            result_subject_1 = result_subject[result_subject["Count"]<100]
            result_admission_1 = result_admission[result_admission["Count"]<100]
            label_xl = 'Count of ICD-9 codes'
            type_codes="ICD-9 codes"
            type_procedur_l =type_procedur
        elif type_procedur=="drugs":          
            result_subject = df.groupby("SUBJECT_ID").size().reset_index(name='Count')
            result_admission = df.groupby("HADM_ID").size().reset_index(name='Count')
            result_subject_1 = result_subject[result_subject["Count"]<300]
            result_admission_1 = result_admission[result_admission["Count"]<150]
            label_xl = 'Count of drugs'
            type_codes="drugs"
            type_procedur_l=""
        else:
            print( "Type of procedure not found"   ) 
        # Create a figure with two matplotlib.Axes objects
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        # Assigning a graph to each ax
        sns.histplot(data=result_subject_1, x="Count", ax=ax1, color='darkblue',bins = 50)
        sns.histplot(data=result_admission_1, x="Count", ax=ax2, color='lightblue',bins = 30)
        # Set x-axis and y-axis labels for each subplot
        ax1.set(xlabel='Count of '+type_codes+' per patient', ylabel='Frequency')
        ax2.set(xlabel=label_xl+' per admission', ylabel='Frequency')
        ax1.text(-0.1, -0.2, '(a)', transform=ax1.transAxes, size=10, )
        ax2.text(-0.1, -0.2, '(b)', transform=ax2.transAxes, size=10, )
        fig.suptitle('Count of  '+type_codes +type_procedur_l, fontsize=14)  # Increase the font size of the title
        # Show the plot
        plt.show()
        plt.savefig("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/results/images/drugs/drugs_counts.svg")
def main(graph,type_procedur=None):

    create_graphs(graph,type_procedur)            
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    import pandas as pd
    import argparse
    #import config
    import os
    name_entire = "entire_ceros"
    ruta = DARTA_INTERM_intput+ name_entire+"_.parquet"
    if os.path.exists(ruta):
        print("La ruta existe.")
    else:
        print("La ruta no existe.")
    parser = argparse.ArgumentParser(description="Scripti input special SD model")
    # Este argumento es obligatorio, así que no tiene un valor por defecto.
    parser.add_argument("graph", type=str, choices=["hyperparameter","Readmission_Proceduers","hyperparameter_drugs",
                                                    "Readmission_drugs",
                                                    "hyperparameter_diagnosis","Readmission_diagnosis",
                                                    "get_input_preprocess_results","preprocessing_clustering_exp",
                                                    "get_mutula_information","demo_pie","pacmap_representation",
                                                    "10_most_frequent","countpr_admi_patient","Kernel_density_estimation"],
                        default="temporal_state",
                        help="Tipo de procesamiento a realizar.")
    parser.add_argument("--type_procedur", default=None, choices=['procedures', 'diagnosis', 'medicament'], 
                    help="Tipo de procedimiento a realizar. Las opciones disponibles son: 'procedures', 'diagnosis', 'medicament'.")
    #python report/vis_.py "10_most_frequent" '--type_procedur diagnosis
    try:
        args = parser.parse_args()
        graph = args.graph
        type_procedur = args.type_procedur
    except:    
        graph = "get_rand_index"      #preprocessing_clustering_exp #get_mutula_information #  #get_rand_index   Readmission_drugs #Readmission_Proceduers  #Readmission_diagnosis
        type_procedur = "diagnosis"
    main(graph,type_procedur)
            