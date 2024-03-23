# %%
import sys
sys.path.append('')

from preprocessing.function_pred import *
from preprocessing.config import *
import importlib
importlib.reload(sys.modules['preprocessing.config'])
from function_vis import *
importlib.reload(sys.modules['function_vis'])

# %%
import os

directorio_actual = os.getcwd()
print("Directorio actual:", directorio_actual)



DATA_DIRECTORY_results


# %%
import pandas as pd

graph = "other"
if graph == "hyperparameter" or graph == "Readmission_Proceduers":
    DATA_DIRECTORY_results= DATA_DIRECTORY_results
    df1= pd.read_csv(RESULTS_PREDICTION_FILE_1 )
    df3= pd.read_csv(RESULTS_PREDICTION_FILE_2 )
    df2= pd.read_csv(RESULTS_PREDICTION_FILE_3)
    df = pd.concat([df1,df2,df3],axis=0)
    df_sin_duplicados = df.drop_duplicates(keep='first')
    cols = ['f1_test', 'f1_train', 'sensitivity_test',
        'specificity_test', 'precision_test', 'accuracy_test',
        'sensitivity_train', 'specificity_train', 'precision_train',
        'accuracy_train', 'confusion matrix','Feature selection',
        'Classifiers', 	 ]
    
    df_sin_duplicados_columnas_especificas = df.iloc[:,1:].drop_duplicates(subset=cols, keep='first')
    print("real",df.shape)
    print("sin_duplciados",df_sin_duplicados_columnas_especificas.shape)
    print(list(df.Mapping.unique()))
    print(list(df.Classifiers.unique()))
    print(list(df.Mapping.unique()))
    print(list(df.Mapping.unique()))
    print(list(df['Feature selection'].unique()))

    print(len(list(df.Mapping.unique())))
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('CCS CODES_proc_outs_visit_non_filtered.csv', 'CCS CODES')
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('ICD9_CODE_procedures_outs_visit_non_filtered.csv', 'ICD-9 CODES')
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('CCS CODES_proc_outs_visit_non_filtered.csv', 'CCS CODES')
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('ICD9_CODE_procedures_outs_visit_non_filtered.csv', 'ICD-9 CODES')

    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('cat_threshold .95 most frequent_proc_outs_visit_non_filtered.csv', 'threshold .95')
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('cat_threshold .88 most frequent_outs_visit_non_filtered.csv', 'threshold .88')
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('cat_threshold .999 most frequent_outs_visit_non_filtered.csv', 'threshold .999')
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('cat_threshold .98 most frequent_outs_visit_non_filtered.csv', 'threshold .98')
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('sin_codigo.csv', 'No ICD9-Codes')
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('sin_codigo.csv', 'No ICD9-Codes')


    # %%


    # %%
    #df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression")&(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression")&(df_sin_duplicados_columnas_especificas["Sampling"]=="non")&(df_sin_duplicados_columnas_especificas["Feature selection"]==True)]
    #days = 30
    #li = ["sensitivity","specificity","accuracy","f1","precision"]
    #for i in li:
        
    #    plot_readmission(i,df_sin_duplicados_columnas_especificas_)

    # %% [markdown]
    # Grafica de hiperparametros

    # %%
    #DATA_DIRECTORY_results = "/Users/cgarciay/Desktop/results_SD/prepro/experimernt_prepro/"
    #df1= pd.read_csv(PROCEDURES_DIRECTORY +'hp_prod_xgbos.csv')
    #df_sin_duplicados_columnas_especificas = df1.copy()
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('CCS CODES_proc_outs_visit_non_filtered.csv', 'CCS CODES')
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('ICD9_CODE_procedures_outs_visit_non_filtered.csv', 'ICD-9 CODES')
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('CCS CODES_proc_outs_visit_non_filtered.csv', 'CCS CODES')
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('ICD9_CODE_procedures_outs_visit_non_filtered.csv', 'ICD-9 CODES')

    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('cat_threshold .95 most frequent_proc_outs_visit_non_filtered.csv', 'threshold .95')
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('cat_threshold .88 most frequent_outs_visit_non_filtered.csv', 'threshold .88')
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('cat_threshold .999 most frequent_outs_visit_non_filtered.csv', 'threshold .999')
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('cat_threshold .98 most frequent_outs_visit_non_filtered.csv', 'threshold .98')
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('sin_codigo.csv', 'No ICD9-Codes')
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace('sin_codigo.csv', 'No ICD9-Codes')


    # Crear un diccionario de reemplazos
    reemplazos = {
        'CCS CODES_proc_outs_visit_non_filtered.csv': 'CCS CODES',
        'ICD9_CODE_procedures_outs_visit_non_filtered.csv': 'ICD-9 CODES',
        'cat_threshold .95 most frequent_proc_outs_visit_non_filtered.csv': 'threshold .95',
        'cat_threshold .88 most frequent_outs_visit_non_filtered.csv': 'threshold .88',
        'cat_threshold .999 most frequent_outs_visit_non_filtered.csv': 'threshold .999',
        'cat_threshold .98 most frequent_outs_visit_non_filtered.csv': 'threshold .98',
        'sin_codigo.csv': 'No ICD9-Codes'
    }

    #file from wandb
    aux_2 = pd.read_csv(LG_PROD_FILE)
    cols_hp = ['params.C', 'params.l1_ratio', 'params.max_iter', 'params.penalty', 'params.solver']
    cols_p = [x.replace('params.', '') for x in cols_hp]
    cols_p
    create_best_lg(aux_2,reemplazos)

    # %% [markdown]
    # 
    # Best Hyperparameter Search Xgboost (Procedures)

    # %%
    #df_sin_duplicados_columnas_especificas_ datafram sin hyperparameter SEARCH
    #df_sin_duplicados_columnas_especificas Hhyperparameter search
    # archivo de wandb
    df1= pd.read_csv(HP_PROD_XGBOS_FILE)

    lista2 = ['f1_test_a', 'f1_train_a', 'sensitivity_test_a', 'specificity_test_a',
        'precision_test_a', 'accuracy_test_a', 'sensitivity_train_a',
        'specificity_train_a', 'precision_train_a', 'accuracy_train_a']
    lista = ['f1_test', 'f1_train', 'sensitivity_test', 'specificity_test', 'precision_test', 'accuracy_test', 'sensitivity_train', 'specificity_train', 'precision_train', 'accuracy_train']

    metric_list = ['sensitivity', 'f1', 'accuracy', 'precision','specificity']
    title_ = "Model Validation Xgboost model (Procedures)"
    #df_sin_duplicados_columnas_especificas_ = agregar_anterior_hp_S(df_sin_duplicados_columnas_especificas_,lista)
    #crear_tabla_results(df_sin_duplicados_columnas_especificas_,df_sin_duplicados_columnas_especificas,title_)

    # %% [markdown]
    # Best Hyperparameter Logistic Regression (Procedures)

    # %%
    #df_sin_duplicados_columnas_especificas_ datafram sin hyperparameter SEARCH
    #df_sin_duplicados_columnas_especificas Hhyperparameter search
    #DATA_DIRECTORY_results = "/Users/cgarciay/Desktop/results_SD/prepro/experimernt_prepro/"
    #df1= pd.read_csv(PROCEDURES_DIRECTORY +'hp_prod_xgbos.csv')
    df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression")&(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression")&(df_sin_duplicados_columnas_especificas["Sampling"]=="non")&(df_sin_duplicados_columnas_especificas["Feature selection"]==True)]

    lista2 = ['f1_test_a', 'f1_train_a', 'sensitivity_test_a', 'specificity_test_a',
        'precision_test_a', 'accuracy_test_a', 'sensitivity_train_a',
        'specificity_train_a', 'precision_train_a', 'accuracy_train_a']
    lista = ['f1_test', 'f1_train', 'sensitivity_test', 'specificity_test', 'precision_test', 'accuracy_test', 'sensitivity_train', 'specificity_train', 'precision_train', 'accuracy_train']

    metric_list = ['sensitivity', 'f1', 'accuracy', 'precision','specificity']
    title_ = "Model Validation Logistic Regression model (Procedures)"
    df_sin_duplicados_columnas_especificas_ = agregar_anterior_hp_S(df_sin_duplicados_columnas_especificas_,lista)
    #crear_tabla_results(df_sin_duplicados_columnas_especificas_,aux_2,title_,metric_list,lista)

    # %%
    ##################################################################################
    ##################################################################################
    ###################################################################################
    #Readmission Procedures##########################################################
    df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression")&(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression")&(df_sin_duplicados_columnas_especificas["Sampling"]=="non")&(df_sin_duplicados_columnas_especificas["Feature selection"]==True)]

    create_heatmap(df_sin_duplicados_columnas_especificas_, 'Logistic Regression',"_pred_lg_prod_")

    df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="XGBClassifier")&(df_sin_duplicados_columnas_especificas["Sampling"]=="non")&(df_sin_duplicados_columnas_especificas["Feature selection"]==True)]

    create_heatmap(df_sin_duplicados_columnas_especificas_,"XGB Classifier","_pred_xgb_prd_")

if graph == "hyperparameter drugs" or graph == "Readmission_Proceduers drugs":
    #archivos que se obtiene con script readmission_ped.py
    df1= pd.read_csv(DRUGS_DIRECTORY/'results_prediction_30+non_filteredDrugs_3.csv')
    df2 = pd.read_csv(DRUGS_DIRECTORY/'results_prediction_30+non_filteredDrugs_r.csv')
    df3 = pd.read_csv(DRUGS_DIRECTORY/'results_prediction_30+non_filteredDrugs.csv')
    df4= pd.read_csv(DRUGS_DIRECTORY/'results_prediction_30+non_filteredDrugs_2_r.csv')
    df5= pd.read_csv(DRUGS_DIRECTORY/'results_prediction_30+non_filteredDrugs_threshold.csv')

    print(df1.shape)

    print(df2.shape)
    df = pd.concat([df1,df2,df3,df4,df5],axis = 0)

    reemplazos2 = {
        
        'CCS_CODES_diagnosis_outs_visit_non_filtered.csv': 'CCS CODES',
        'ICD9_CODE_diagnosis_outs_visit_non_filtered.csv': 'ICD-9 CODES',
        'threshold_0.95_diagnosis_outs_visit_non_filtered.csv': 'threshold .95',
        'threshold_0.88_diagnosis_outs_visit_non_filtered.csv': 'threshold .88',
        'threshold_0.999_diagnosis_outs_visit_non_filtered.csv': 'threshold .999',
        'threshold_0.98_diagnosis_outs_visit_non_filtered.csv': 'threshold .98',
        'sin_codigo.csv': 'No ICD9-Codes',
        'LEVE3 CODES_outs_visit_non_filtered.csv':'LEVE3 CODES'
    }



    reemplazos1 = {
        
        'ATC4_outs_visit_non_filtered.csv': 'ATC4',
        'DRUG_y_outs_visit_non_filtered.csv': 'Drugs',
        'threshold_0.95_outs_visit_non_filtered.csv': 'threshold .95',
        'threshold_0.88_outs_visit_non_filtered.csv': 'threshold .88',
        'threshold_0.999_outs_visit_non_filtered.csv': 'threshold .999',
        'threshold_0.98_outs_visit_non_filtered.csv': 'threshold .98',
        'sin_codigo.csv': 'No ICD9-Codes',
        'ATC3_outs_visit_non_filtered.csv':'ATC3'
    }

    # Reemplazos para la columna 1
    df["Mapping"] = df["Mapping"].replace(reemplazos1)



    # %%
    df.Mapping.unique()

    # %%

    cols = ['f1_test', 'f1_train', 'sensitivity_test',
        'specificity_test', 'precision_test', 'accuracy_test',
        'sensitivity_train', 'specificity_train', 'precision_train',
        'accuracy_train',
        'Classifiers', 	 ]
    
    df_sin_duplicados_columnas_especificas = df.iloc[:,1:].drop_duplicates(subset=cols, keep='first')
    print("real",df.shape)
    print("sin_duplciados",df_sin_duplicados_columnas_especificas.shape)
    print(list(df.Mapping.unique()))
    print(list(df.Classifiers.unique()))
    print(list(df.Mapping.unique()))
    print(list(df.Mapping.unique()))
    print(list(df['Feature selection'].unique()))

    print(len(list(df.Mapping.unique())))
    print(len(list(df.Sampling.unique())))
    df.Sampling.unique()

    # %%
    df_sin_duplicados_columnas_especificas["Feature selection"].unique()

    # %%
    df.Classifiers.unique()
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
    # %%
    #df_sin_duplicados_columnas_especificas_.groupby("Mapping")["sensitivity_test"].idxmax()
    #df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="XGBClassifier") ]
    #df_sin_duplicados_columnas_especificas_.loc[df_sin_duplicados_columnas_especificas_.groupby("Mapping")["f1_test"].idxmax()].drop_duplicates(subset='Mapping', keep='first')

    # %%
    crear_datafram_hyperparametros(df1)

    # %%


    # %%

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

    # %%
    create_best_lg(df,reemplazos1)

    # %%
    #### CREATE PREDICTIONS" Logistic regression###

    df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression") ]
    df_sin_duplicados_columnas_especificas_ = df_sin_duplicados_columnas_especificas_.loc[df_sin_duplicados_columnas_especificas_.groupby("Mapping")["f1_test"].idxmax()].drop_duplicates(subset='Mapping', keep='first')

    create_heatmap(df_sin_duplicados_columnas_especificas_, 'Logistic Regression',"_LR_drugs_")

    df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="XGBClassifier") ]
    df_sin_duplicados_columnas_especificas_ = df_sin_duplicados_columnas_especificas_.loc[df_sin_duplicados_columnas_especificas_.groupby("Mapping")["f1_test"].idxmax()].drop_duplicates(subset='Mapping', keep='first')
    create_heatmap(df_sin_duplicados_columnas_especificas_,"XGB Classifier", "_xgbost_drugs_")

    # %%
    df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression")&(df_sin_duplicados_columnas_especificas["Sampling"]=="non")&(df_sin_duplicados_columnas_especificas["Feature selection"]==True)]

    create_graph(df_sin_duplicados_columnas_especificas_.reset_index(),"Logistic Regression", "LR")
    df_sin_duplicados_columnas_especificas_
    df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="XGBClassifier")&(df_sin_duplicados_columnas_especificas["Sampling"]=="non")&(df_sin_duplicados_columnas_especificas["Feature selection"]==True)]

    create_graph(df_sin_duplicados_columnas_especificas_,"XGB Classifier","Lr_")



    # %%
    # %%

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




if graph == "hyperparameter diagnosis" or graph == "Readmission diagnosis":
    
    df1= pd.read_csv(DIAGNOSIS_DIRECTORY/'results_prediction_30+non_filteredDiagnosis_.csv')
    df2 = pd.read_csv(DIAGNOSIS_DIRECTORY/'results_prediction_30+non_filteredDiagnosis_1.csv')
    df3 = pd.read_csv(DIAGNOSIS_DIRECTORY/'results_prediction_30+non_filteredDiagnosis_2.csv')
    df4= pd.read_csv(DIAGNOSIS_DIRECTORY/'results_prediction_30+non_filteredDiagnosis.csv')
    df5= pd.read_csv(DIAGNOSIS_DIRECTORY/'results_prediction_30+non_filteredDiagnosis_threshopl_.csv')
    aux_name = 'results/experimernt_prepro/diagnosis/results_prediction_30+non_filteredDiagnosis_threshopl2'
    df6 = pd.read_csv(aux_name+'.csv'  )
    df = pd.concat([df1,df2,df3,df4,df5,df6],axis = 0)
    df_sin_duplicados = df.drop_duplicates(keep='first')
    cols = ['f1_test', 'f1_train', 'sensitivity_test',
        'specificity_test', 'precision_test', 'accuracy_test',
        'sensitivity_train', 'specificity_train', 'precision_train',
        'accuracy_train', 'confusion matrix','Feature selection',
        'Classifiers', 	 ]
    
    df_sin_duplicados_columnas_especificas = df.iloc[:,1:].drop_duplicates(subset=cols, keep='first')
    print("real",df.shape)
    print("sin_duplciados",df_sin_duplicados_columnas_especificas.shape)
    print(list(df.Mapping.unique()))
    print(list(df.Classifiers.unique()))
    print(list(df.Mapping.unique()))
    print(list(df.Mapping.unique()))
    print(list(df['Feature selection'].unique()))

    print(len(list(df.Mapping.unique())))


    reemplazos2 = {
        
        'CCS_CODES_diagnosis_outs_visit_non_filtered.csv': 'CCS CODES',
        'ICD9_CODE_diagnosis_outs_visit_non_filtered.csv': 'ICD-9 CODES',
        'threshold_0.95_diagnosis_outs_visit_non_filtered.csv': 'threshold .95',
        'threshold_0.88_diagnosis_outs_visit_non_filtered.csv': 'threshold .88',
        'threshold_0.999_diagnosis_outs_visit_non_filtered.csv': 'threshold .999',
        'threshold_0.98_diagnosis_outs_visit_non_filtered.csv': 'threshold .98',
        'sin_codigo.csv': 'No ICD9-Codes',
        'LEVE3 CODES_outs_visit_non_filtered.csv':'LEVE3 CODES',
        'input_model_pred_diagnosisthreshold_0.999_diagnosis_outs_visit_non_filtered.csv': 'threshold .999',
    }




    # Reemplazos para la columna 1
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace(reemplazos2)


    df1= pd.read_csv(DIAGNOSIS_DIRECTORY/"diagnosis_xgbost_hp.csv")
    df1["Mapping"] = df1["Mapping"].replace(reemplazos2)
    lista2 = ['f1_test_a', 'f1_train_a', 'sensitivity_test_a', 'specificity_test_a',
        'precision_test_a', 'accuracy_test_a', 'sensitivity_train_a',
        'specificity_train_a', 'precision_train_a', 'accuracy_train_a']
    lista = ['f1_test', 'f1_train', 'sensitivity_test', 'specificity_test', 'precision_test', 'accuracy_test', 'sensitivity_train', 'specificity_train', 'precision_train', 'accuracy_train']

    metric_list = ['sensitivity', 'f1', 'accuracy', 'precision','specificity']
    title_ = "Model Validation Xgboost model (Diagnosis)"
    df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="XGBClassifier")&(df_sin_duplicados_columnas_especificas["Feature selection"]==True)]

    df_sin_duplicados_columnas_especificas_ = agregar_anterior_hp_S(df_sin_duplicados_columnas_especificas_,lista)
    crear_tabla_results(df_sin_duplicados_columnas_especificas_,df1,title_,metric_list,lista,"xgbost_graph_Hyperparameterruning_diagnosis")

    crear_datafram_hyperparametros(df1)

    # %%
    ar = "diagnosis_lr_hp.csv"

    df1= pd.read_csv(DIAGNOSIS_DIRECTORY/"diagnosis_lr_hp.csv")
    df1["Mapping"] = df1["Mapping"].replace(reemplazos2)

    df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression")&(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression")]
    df_sin_duplicados_columnas_especificas_ = df_sin_duplicados_columnas_especificas_.loc[df_sin_duplicados_columnas_especificas_.groupby("Mapping")["f1_test"].idxmax()].drop_duplicates(subset='Mapping', keep='first')


    ista2 = ['f1_test_a', 'f1_train_a', 'sensitivity_test_a', 'specificity_test_a',
        'precision_test_a', 'accuracy_test_a', 'sensitivity_train_a',
        'specificity_train_a', 'precision_train_a', 'accuracy_train_a']
    lista = ['f1_test', 'f1_train', 'sensitivity_test', 'specificity_test', 'precision_test', 'accuracy_test', 'sensitivity_train', 'specificity_train', 'precision_train', 'accuracy_train']

    metric_list = ['sensitivity', 'f1', 'accuracy', 'precision','specificity']
    title_ = "Model Validation Logistic Regression model (Diagnosis)"
    df_sin_duplicados_columnas_especificas_ = agregar_anterior_hp_S(df_sin_duplicados_columnas_especificas_,lista)
    crear_tabla_results(df_sin_duplicados_columnas_especificas_,df1,title_,metric_list,lista,"lr_graph_Hyperparameterruning_diagnosis")






    create_best_lg(df1,reemplazos2)

    # %%
    df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression")&(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression")]
    df_sin_duplicados_columnas_especificas_ = df_sin_duplicados_columnas_especificas_.loc[df_sin_duplicados_columnas_especificas_.groupby("Mapping")["f1_test"].idxmax()].drop_duplicates(subset='Mapping', keep='first')

    create_heatmap(df_sin_duplicados_columnas_especificas_, 'Logistic Regression', "_lr_diagnosis_")

    df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="XGBClassifier")&(df_sin_duplicados_columnas_especificas["Feature selection"]==True)]

    create_heatmap(df_sin_duplicados_columnas_especificas_,"XGB Classifier", "_xgbost_diagnosis_")

    # %%
    df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression")&(df_sin_duplicados_columnas_especificas["Sampling"]=="non")&(df_sin_duplicados_columnas_especificas["Feature selection"]==True)]

    create_graph(df_sin_duplicados_columnas_especificas_.reset_index(),"Logistic Regression","LR")
    df_sin_duplicados_columnas_especificas_
    df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="XGBClassifier")&(df_sin_duplicados_columnas_especificas["Sampling"]=="non")&(df_sin_duplicados_columnas_especificas["Feature selection"]==True)]

    create_graph(df_sin_duplicados_columnas_especificas_,"XGB Classifier","LR")

    # %%
    # Filter the DataFrame with the rows having the highest sensitivity_train
    import pandas as pd
    import matplotlib.pyplot as plt
    df_sin_duplicados_columnas_especificas_ =df_sin_duplicados_columnas_especificas[(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression")&(df_sin_duplicados_columnas_especificas["Classifiers"]=="LogisticRegression")&(df_sin_duplicados_columnas_especificas["Feature selection"]==True)]
    # Given DataFrame
    days = 30


    o = "f1"
    # Generate positions for the bars


    highest_sensitivity_idx =df_sin_duplicados_columnas_especificas_.groupby('Mapping')[o+'_train'].idxmax()
    highest_sensitivity_rows = df_sin_duplicados_columnas_especificas_.loc[highest_sensitivity_idx]
    highest_sensitivity_rows
    # Bar plot
    positions = range(len(highest_sensitivity_rows))
    plt.figure(figsize=(12, 6))
    plt.bar(positions, highest_sensitivity_rows[o+'_train'], width=0.4, label=o+"Train")
    plt.bar([pos + 0.4 for pos in positions], highest_sensitivity_rows[o+'_train'], width=0.4, label=o+"Test", alpha=0.5)
    plt.xticks([pos + 0.2 for pos in positions], highest_sensitivity_rows["Mapping"], rotation=90)
    plt.xlabel('Simplification')
    plt.ylabel(o + "score")
    plt.title(o+'Train vs'+o+'Test for aach Mapping, model Logistic Regression ' +str(days))
    plt.legend()
    plt.tight_layout()

    plt.show()

    # %%
    # For non filtere aproach

    # %%


# %%


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
    # %%
        
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

    # %%
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

#if graph == "silhouate_scores" :
drugs = DRUGS_DIRECTORY/"experiment_preporesults_prepro_nonfilteres_DRUGS_final.csv"
diagnosis = DIAGNOSIS_DIRECTORY/ "results_prepro_nonfilteres_diagnosis_final.csv"
procedures = PROCEDURES_DIRECTORY/"results_final_merged_procedures_prepo.csv"
#merged_df = pd.read_csv("experiment_prepo"+"results_prepro_nonfilteres_diagnosis_final.csv"
for j,i in enumerate([drugs,diagnosis,procedures]):
    if j ==0:                                                                                                           
        procedures=nomb_(procedures,replace_dict_prod)                                                                                                                                   
    merged_df = pd.read_csv(i)
    
    idx = merged_df.groupby(['Name'])['silhouette_avg'].idxmax()

    # Usar los índices para obtener las filas correspondientes
    top_silhouette_avg_per_name = merged_df.loc[idx]
    top_silhouette_avg_per_name

    # %%
    idx = merged_df.groupby('Name')['silhouette_avg'].idxmax()

    # Usar los índices para obtener las filas correspondientes
    top_silhouette_avg_per_name = merged_df.loc[idx]
    top_silhouette_avg_per_name = top_silhouette_avg_per_name[top_silhouette_avg_per_name['Name']!="Medicament"]
    top_silhouette_avg_per_name

    # %%
    idx = merged_df.groupby('Name')['silhouette_avg_v'].idxmax()

    # Usar los índices para obtener las filas corre}
    # spondientes
    top_silhouette_avg_per_name_v = merged_df.loc[idx]
    top_silhouette_avg_per_name_v = top_silhouette_avg_per_name_v[top_silhouette_avg_per_name_v['Name']!="Medicament"]
    top_silhouette_avg_per_name_v

    # %%
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

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
    plt.savefig(IMAGES_Cluster_DICT+'silhouate_scores_'+str(i)+'_.svg')
    plt.show()


    # %%


    # %%
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
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
        plt.savefig(IMAGES_Cluster_DICT+'real_Preprocessing'+level+'_'+str(i)+'.svg')
        plt.tight_layout()
        plt.show()






# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Assuming merged_df is already defined and contains the relevant data, including a 'num_cluster' column

# Define a shades of blue color palette
palette = {"power": "lightblue", "max": "skyblue", "std": "steelblue"}

# Hatch patterns mapping, assuming 'num_cluster' is categorical or discrete numeric
hatch_patterns = {1: "/", 2: "\\", 3: "|", 4: "-", 5: "+"}  # Example mapping

# Create a 1x2 grid of subplots for the two Silhouette score plots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Flatten the array of axes for easy iterating
axs = axs.flatten()
scores = ["Silhouette score"]*2
titles = ["Preprocessing patient level", "Preprocessing visit level"]

# Loop over the two Silhouette score plots
for i, metric in enumerate(["silhouette_avg", "silhouette_avg_v"]):
    ax = axs[i]
    # Assuming 'num_cluster' is available in merged_df and used for differentiation
    sns.barplot(x="Name", y=metric, hue="Num Cluster", data=merged_df, ax=ax, palette=palette, ci=None)
    ax.set_title(titles[i])
    ax.set_ylabel(scores[i])
    ax.set_xlabel("ICD-9 Codes")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    if i == 0:
        ax.legend_.remove()
    
    # Add hatch patterns to bars
    for bar, pattern in zip(ax.patches, merged_df['num_cluster'].map(hatch_patterns).values):
        bar.set_hatch(pattern)

# Adjust legend for preprocessing and num_cluster
# Note: You'll need to adjust this part to correctly reflect your data and the hatch patterns used
axs[1].legend(handles=prepro_patches + [Patch(hatch=hatch_patterns[nc], label=f'Cluster {nc}') for nc in sorted(hatch_patterns.keys())], loc='center left', bbox_to_anchor=(1, 0.5), title="Legend")

plt.tight_layout()
plt.show()


# %%
print(pattern_patches)

# %%
# Assuming df_p1 is already defined and contains the relevant data
# Replace df_p with df_p1 in your code

merged_df["Name"] = merged_df["Name"].replace('CCS CODES_proc', 'CCS CODES')
merged_df["Name"] = merged_df["Name"].replace('ICD9_CODE_procedures', 'ICD9_CODE')
merged_df["Name"] = merged_df["Name"].replace('cat_threshold .88 most frequent', 'threshold .88')
merged_df["Name"] = merged_df["Name"].replace('cat_threshold .999 most frequent', 'threshold .999')
merged_df["Name"] = merged_df["Name"].replace('cat_threshold .98 most frequent', 'threshold .98')
merged_df["Name"] = merged_df["Name"].replace('cat_threshold .95 most frequent_proc', 'threshold .95')
merged_df["Name"] = merged_df["Name"].replace('sin_codigo', 'No ICD9-Code')

# This will update the "Name" column in df_p1 with the specified replacements.


# %%
type_a=stri ="visit.csv"
df_p =pd.read_csv("/Users/cgarciay/Desktop/Laval_Master_Computer/research/experiment_prepo/prepro_experiment_"+type_a+"")

# %%
df_p["Name"] = df_p["Name"].replace('CCS CODES_proc', 'CCS CODES')
df_p["Name"] = df_p["Name"].replace('ICD9_CODE_procedures', 'ICD9_CODE')
df_p["Name"] = df_p["Name"].replace('cat_threshold .88 most frequent', 'threshold .88')
df_p["Name"] = df_p["Name"].replace('cat_threshold .999 most frequent', 'threshold .999')
df_p["Name"] = df_p["Name"].replace('cat_threshold .98 most frequent', 'threshold .98')
df_p["Name"] = df_p["Name"].replace('cat_threshold .95 most frequent_proc', 'threshold .95')
df_p["Name"] = df_p["Name"].replace('sin_codigo', 'No ICD9-Code')


# %%


# %%
df_p.iloc[:, 1:6].head()

# %%
pivot_table

# %%

#'davies_bouldin_avg'
# Suponiendo que 'data' es tu DataFrame
pivot_table = pd.pivot_table(df_p, 
                             index=['Name', 'Prepro', 'Num Cluster'], 
                             values=['silhouette_avg', ])


# %%


# %%
df.columns

# %%
pivot_table.head()

# %% [markdown]
# # prepo

# %%
ruta = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/experiment_prepo/"
archivo = "prepro_experiment_Patientagglomerative_v2.csv"
archivo2 = "prepro_experiment_outs_visitagglomerative_v2.csv"
df = pd.read_csv(ruta + archivo2)


# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Pivot the DataFrame for the heatmap
pivot_df = df.pivot_table(index=['Name', 'Prepro', 'Num Cluster'], values='silhouette_avg')

plt.figure(figsize=(4, 14))
sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt=".6f")
plt.title('Silhouette_avg')
plt.show()

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = df_p.to_dict(orient='list')  
print(data)

df = pd.DataFrame(data)

# Pivot the DataFrame for the heatmap
pivot_df = df.pivot_table(index=['Name', 'Prepro', 'Num Cluster'], values='silhouette_avg')

plt.figure(figsize=(4, 14))
sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt=".6f")
plt.title('Silhouette Average Heatmap')
plt.show()


# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = df_p.to_dict(orient='list')
print(data)

df = pd.DataFrame(data)

# Pivot the DataFrame for the heatmap
pivot_df = df.pivot_table(index=['Name', 'Prepro', 'Num Cluster'], values='silhouette_avg')

# Find the highest score
max_score = pivot_df.max().max()

# Create a mask to highlight the highest score
mask = pivot_df == max_score

plt.figure(figsize=(4, 14))
sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt=".6f", mask=mask, cbar=False)
plt.title('Silhouette Average Heatmap')
plt.show()

# %%
#df_metric = pd.read_csv("/Users/cgarciay/Desktop/Laval_Master_Computer/research/metricas_clustering_mapeo_visit.csv")
df_metric = pd.read_csv("/Users/cgarciay/Desktop/Laval_Master_Computer/research/metricas_clustering_mapeo_patient.csv")


# %%
df_metric.iloc[:,1:]

# %%
data = df_metric.to_dict(orient='list')  
print(data)

df = pd.DataFrame(data)

# Pivot the DataFrame for the heatmap
pivot_df = df.pivot_table(index=['Name', ], values=['silhouette_avg','davies_bouldin_avg'])

plt.figure(figsize=(4, 4))
sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt=".6f")
plt.title('Silhouette Average Heatmap')
plt.show()


# %% [markdown]
# # PREDICTIONS OF READMISSION

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data (the provided DataFrame)
data = {
    'Variable': ['Age_max', 'LOSRD_sum', 'LOSRD_avg'],
      'sensitivity_test': [0.0, 0.0, 0.0],
    'specificity_test': [1.0, 0.999643, 0.999643],
    'precision_test': [None, 0.0, 0.0],
    'accuracy_test': [0.830071, 0.829775, 0.829775],
    'sensitivity_train': [0.0, 0.0, 0.0],
    'specificity_train': [1.0, 0.999643, 0.999643],
  
    'accuracy_train': [0.830071, 0.829775, 0.829775]
}

df = pd.DataFrame(data)

# Set 'Variable' as the index
df.set_index('Variable', inplace=True)
pivot_df = df.pivot_table(index=['Variable', ], values=[i for i in df if i!= 'Variable'])


plt.figure(figsize=(4, 4))
sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt=".6f")
plt.title('Silhouette Average Heatmap')
plt.show()
plt.savefig("fig.png")

# %%
df

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data setup (assuming the data is stored in a DataFrame named 'df')
data = {
    'Variable': ['X1', 'X2'],
       'sensitivity_test': [0.0, 0.0],
    'specificity_test': [1.0, 1.0],
    'precision_test': [None, None],
    'accuracy_test': [0.830071, 0.830071],
    'sensitivity_train': [0.0, 0.0],
    'specificity_train': [1.0, 1.0],
    'precision_train': [None, None],
    'accuracy_train': [0.830071, 0.830071]
}

df = pd.DataFrame(data)

df

# %%
pivot_df

# %%
df_metric.columns

# %%
df_metric["Classifiers"].unique()


# %% [markdown]
# # Vis for results readmission

# %%
days = "90"

ruta2 ="/Users/cgarciay/Desktop/Laval_Master_Computer/research/results_pred/"
ar = "results_prediction_30+_realv3.csv"
archivo = "results_prediction_"+days+"+_realv3.csv"
archivo2 = "results_prediction_"+days+"+_realv2.csv"
df_metric = pd.read_csv(ruta2+ar)
df_metric2 = pd.read_csv(ruta2+archivo2)
df_metric.head()
are_identical = df_metric.equals(df_metric2)
are_identical

# %%
archivo

# %%
df_metric = df_metric[df_metric["Sampling"]=="non"]

# %%

df_metric["Mapping"] = df_metric["Mapping"].replace('CCS CODES_proc.csv', 'CCS CODES')
df_metric["Mapping"] = df_metric["Mapping"].replace('ICD9_CODE_procedures.csv', 'ICD9_CODE')
df_metric["Mapping"] = df_metric["Mapping"].replace('cat_threshold .88 most frequent.csv', 'threshold .88')
df_metric["Mapping"] = df_metric["Mapping"].replace('cat_threshold .999 most frequent.csv', 'threshold .999')
df_metric["Mapping"] = df_metric["Mapping"].replace('cat_threshold .98 most frequent.csv', 'threshold .98')
df_metric["Mapping"] = df_metric["Mapping"].replace('cat_threshold .95 most frequent_proc.csv', 'threshold .95')
df_metric["Mapping"] = df_metric["Mapping"].replace('sin_codigo.csv', 'No ICD9-Code')
df_metric["Classifiers"] = df_metric["Classifiers"].replace('LogisticRegression', 'LG')
df_metric["Classifiers"] = df_metric["Classifiers"].replace('XGBClassifier', 'XGB')
df_metric["Classifiers"] = df_metric["Classifiers"].replace('XGBClassifier', 'XGB')
df_metric["Sampling"] = df_metric["Sampling"].replace('non', 'n')
df_metric["Sampling"] = df_metric["Sampling"].replace('over', 'o')
df_metric["Feature selection"] = df_metric["Feature selection"].replace(True, 'T')
df_metric["Feature selection"] = df_metric["Feature selection"].replace(False, 'F')

                                                            



# %%
df_metric["Mapping"].unique()

# %%
df_metric["Mapping"] = df_metric["Mapping"].replace('CCS CODES_proc_v2.csv', 'CCS CODES')
df_metric["Mapping"] = df_metric["Mapping"].replace('ICD9_CODE_procedures_v2.csv', 'ICD9_CODE')
df_metric["Mapping"] = df_metric["Mapping"].replace('cat_threshold .88 most frequent_v2.csv', 'threshold .88')
df_metric["Mapping"] = df_metric["Mapping"].replace('cat_threshold .999 most frequent_v2.csv', 'threshold .999')
df_metric["Mapping"] = df_metric["Mapping"].replace('cat_threshold .98 most frequent_v2.csv', 'threshold .98')
df_metric["Mapping"] = df_metric["Mapping"].replace('cat_threshold .95 most frequent_proc_v2.csv', 'threshold .95')
df_metric["Mapping"] = df_metric["Mapping"].replace('sin_codigo_v2.csv', 'No ICD9-Code')
df_metric["Classifiers"] = df_metric["Classifiers"].replace('LogisticRegression', 'LG')
df_metric["Classifiers"] = df_metric["Classifiers"].replace('XGBClassifier', 'XGB')
df_metric["Classifiers"] = df_metric["Classifiers"].replace('XGBClassifier', 'XGB')
df_metric["Sampling"] = df_metric["Sampling"].replace('non', 'n')
df_metric["Sampling"] = df_metric["Sampling"].replace('over', 'o')
df_metric["Feature selection"] = df_metric["Feature selection"].replace(True, 'T')
df_metric["Feature selection"] = df_metric["Feature selection"].replace(False, 'F')

        

# %%
df_metric.head()

# %%


# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



df = pd.DataFrame(data)

# Pivot the DataFrame for the heatmap
pivot_df = df.pivot_table(index=['Mapping', 'Classifiers','Feature selection'], values=[ 'sensitivity_test', 'specificity_test','precision_test', 'accuracy_test',
        'f1_test', 'f1_train', 'sensitivity_train',
       'specificity_train', 'precision_train', 'accuracy_train',])


plt.figure(figsize=(16, 14))
sns.heatmap(pivot_df, cmap='YlGnBu', annot=True, fmt=".5f")
plt.title('Results for '+days+' days readmission')
plt.savefig("./images/metrics"+days+".png")
plt.show()

# %%
df_metric["Feature selection"].unique()

# %%


# %%


df = df_metric[df_metric["Feature selection"]=='T']

# Pivot the DataFrame for the heatmap
pivot_df = df.pivot_table(index=['Mapping', 'Classifiers','Feature selection'], values=[ 'var_change','var_ini'])


plt.figure(figsize=(6, 4))
sns.heatmap(pivot_df, cmap='YlGnBu', annot=True, fmt=".5f")
plt.title('Feature Reduction')
plt.savefig("./images/metrics"+days+".png")
plt.show()

# %%
df.head()

# %%
df["Concat_name"] = df["Mapping"]+ df["Classifiers"]

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Given DataFrame

o = "sensitivity"
#o = "specificity"
#o = "accuracy"
#o = "precision"
# Generate positions for the bars


highest_sensitivity_idx = df.groupby('Concat_name')[o+'_train'].idxmax()

# Filter the DataFrame with the rows having the highest sensitivity_train
highest_sensitivity_rows = df.loc[highest_sensitivity_idx]
highest_sensitivity_rows
# Bar plot
positions = range(len(highest_sensitivity_rows))
plt.figure(figsize=(12, 6))
plt.bar(positions, highest_sensitivity_rows[o+'_train'], width=0.4, label=o+"Train")
plt.bar([pos + 0.4 for pos in positions], highest_sensitivity_rows[o+'_train'], width=0.4, label=o+"Test", alpha=0.5)
plt.xticks([pos + 0.2 for pos in positions], highest_sensitivity_rows["Concat_name"], rotation=90)
plt.xlabel('Concat Name')
plt.ylabel(o)
plt.title(o+'Train vs'+o+'Test for Each Mapping/Classifier ' +days)
plt.legend()
plt.tight_layout()
plt.savefig('./images/Sensitivity_Bar_Plot'+o+'.png')
plt.show()


# %%


# %%
highest_sensitivity_idx

# %% [markdown]
# # changes

# %%
import ast
def string_list(x):
    '''funcion que se convierte una lista de string a una lista normal'''
    '''Input '["\'44\'", "\'44\'", "\'50\'", "\'193\'", "\'222\'", "\'222\'", "\'222\'"]'
    x: string as list
    Output
    list: list'''

    try:
        lista = ast.literal_eval(x)
    except:
        lista = np.nan
    return lista

# %%
import pandas as pd
df = pd.read_csv("./data/data_preprocess_non_filtered.csv")

df.columns

# %%
df.SUBJECT_ID.nunique()

# %%
df.HADM_ID.nunique()

# %%
admissions_count = df.groupby('SUBJECT_ID')['HADM_ID'].nunique()

# Filter subjects with more than one admission
subjects_more_than_one_admission = admissions_count[admissions_count > 1]

print("Subjects with more than one admission:")
print(subjects_more_than_one_admission)

# %%
def obtener_ghist(i,df,df1,nam_p,v,real,filtered):
    real = real
    if nam_p == "Threshold":
        #nuevo_df2_gen = desconacat_codes(df,real)
        #nuevo_df_x  = nuevo_df2_gen.copy()
        nuevo_df_x = desconacat_codes(df,"ICD9_CODE_procedures",filtered)
        nuevo_df4 = desconacat_codes(df1,real,filtered)
        #print(nuevo_df2_gen.SUBJECT_ID.nunique())
        
        
    else:
        #nuevo_df2_gen = desconacat_codes(df,real)
        #nuevo_df_x  = nuevo_df2_gen.copy()
        nuevo_df_x = desconacat_codes(df,"ICD9_CODE_procedures",filtered)
        nuevo_df4 = desconacat_codes(df,real,filtered)
      

        #print(nuevo_df2_gen.SUBJECT_ID.nunique())
    nuevo_df_x = nuevo_df_x.sort_values(by=["SUBJECT_ID","HADM_ID"])
    nuevo_df4 = nuevo_df4.sort_values(by=["SUBJECT_ID","HADM_ID"])
    
    duplicados = pd.concat([nuevo_df4.reset_index(), nuevo_df_x["ICD9_CODE_procedures"]],axis = 1)
    #merged_df = pd.merge(nuevo_df4, nuevo_df_x, on=["SUBJECT_ID","HADM_ID"], how='left')

    #duplicados = pd.merge(nuevo_df_x.reset_index(),nuevo_df4.reset_index(), on=["SUBJECT_ID","HADM_ID",], how='left')
    #realizar una fFUNCION QUE ME AYUDE A LIMPIAR PREPROCESINGG DE real preprocess porque corata 30% de datos
    nuevo_df4[real +'_preprocess']= nuevo_df4[real +'_preprocess'].replace("Otro", -1)
    duplicados[real +'_preprocess']= duplicados[real +'_preprocess'].replace("Otro", -1)
    duplicados['ICD9_CODE_procedures']= duplicados['ICD9_CODE_procedures'].replace("Otro", -1)
    
    if real == "cat_threshold .95 most frequent_proc":
        name = name2 = real
        
        duplicados[name2 + "_preprocess"] = duplicados[name2 + "_preprocess"].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        duplicados = duplicados[duplicados[name2 + "_preprocess"].notnull()]

    elif real == "CCS CODES_proc":
        name = real 
        duplicados[name + "_preprocess"]=[item.replace("'", '') for item in duplicados["CCS CODES_proc_preprocess"]]

        duplicados[name + "_preprocess"] = duplicados[name + "_preprocess"].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        duplicados = duplicados[duplicados[name + "_preprocess"].notnull()]

   
    elif nam_p == 'Threshold':
        name2 = real
            
        duplicados[name2 + "_preprocess"] = duplicados[name2 + "_preprocess"].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        duplicados = duplicados[duplicados[name2 + "_preprocess"].notnull()]

  

    
      

  
    return    duplicados

# %%
obtener_ghist(i,df,df1,nam_p,v,real,filtered)

from function_mapping import *

df = pd.read_csv("./data/data_preprocess_non_filtered.csv")
pd.set_option('display.max_columns', None)

#Lectures of dataframe that have the procedures icd-9 codes with different threshold
proc = pd.read_csv("./data/procedures_preprocess_threshold_nonfiltered.csv")
grouped = proc.groupby(['SUBJECT_ID', 'HADM_ID']).agg(lambda x: x.tolist())

# Reset
# the index to make 'SUBJECT_ID' and 'HADM_ID' regular columns
grouped_proc = grouped.reset_index()
df1=grouped_proc.copy()

list_cat = ['CCS CODES_proc', 'cat_threshold .95 most frequent_proc','cat_threshold .88 most frequent', 'cat_threshold .98 most frequent',
              'cat_threshold .999 most frequent']

categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
              'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
              'MARITAL_STATUS', 'ETHNICITY','GENDER']


nam_p_list = ['CCS CODES_proc', 'cat_threshold .95 most frequent_proc','Threshold', 'Threshold',
       'Threshold']

filtered = True
v = "patient" 


result_stat  = {'Name' : [],
       'count': [],
'mean': [],
'std': [],
'min': [],
'25%': [],
'50%': [],
'75%': [],
'max': [] ,
"Min >":[],
"Unique_codes":[],
"Unique_codes_icd9":[]}

for i in range(len(list_cat)):
       nam_p = nam_p_list[i]
       real = list_cat[i]
       changes_per_patient,real,unique_icd9 ,unique_r = calculare_changes(i,df,df1,nam_p,v,real,filtered)
       auc_d = create_results(result_stat,changes_per_patient,real,unique_r,unique_icd9)

       for key in auc_d:
           result_stat[key].append(auc_d[key])
       

df_res = pd.DataFrame(result_stat)
df_res.to_csv("./results_changes/"+v+"_nonfiltered.csv")

# %%



# %%
df.HADM_ID.nunique()

# %%


df_res.to_csv("./results_changes/"+v+"_nonfiltered.csv")

# %%
nuevo_df2_gen.head()

# %%
# visit level*

# %%
archivo = "visit_nonfiltered.csv"
ruta = "./results_changes/"
cpatient = pd.read_csv(ruta+archivo)

cpatient["% Codes"] = cpatient["Unique_codes"]/cpatient["Unique_codes_icd9"]
cpatient = cpatient.rename(columns ={"Unnamed: 3":'% Change'} )
cpatient["% Change"] = cpatient["count"]/52243.0

cpatient["Name"] = cpatient["Name"].replace('CODES_proc', 'CCS CODES')
cpatient["Name"] = cpatient["Name"].replace('cat_threshold .95 most frequent_proc', 'threshold .95')
cpatient["Name"] = cpatient["Name"].replace('cat_threshold .88 most frequent', 'threshold .88')
cpatient["Name"] = cpatient["Name"].replace('cat_threshold .999 most frequent', 'threshold .999')
cpatient["Name"] = cpatient["Name"].replace('cat_threshold .98 most frequent', 'threshold .98')
cpatient.columns


# %%
cpatient

# %%
import matplotlib.pyplot as plt
import seaborn as sns

df = cpatient[['Name', 'count', '% Change', 'max', 'Unique_codes', 'Unique_codes_icd9', '% Codes']].sort_values(by="Unique_codes")

df = df.rename(columns={"Unique_codes": 'Unique codes', "Unique_codes_icd9": 'Unique ICD-9 codes', "max": 'Max', "count": 'Count'})
# Set 'Name' column as index
df.set_index('Name', inplace=True)

# Create a custom color palette with different shades of blue
cmap = sns.color_palette("Blues", as_cmap=True)

# Create a heatmap
plt.figure(figsize=(13, 2))
sns.heatmap(df, annot=True, cmap=cmap, fmt=".2f", linewidths=.5)
plt.title('Heatmap of changes in simplification (visit level)')

# Modify the y-axis label name
plt.ylabel('Simplification')

plt.show()

# %%


# %% [markdown]
# # Visit

# %%
archivo = "patient_nonfiltered.csv"
ruta = "./results_changes/"
cpatient = pd.read_csv(ruta+archivo)

cpatient["% Codes"] = cpatient["Unique_codes"]/cpatient["Unique_codes_icd9"]
cpatient = cpatient.rename(columns ={"change":'% Change'} )
cpatient["% Change"] = cpatient["count"]/42214.0

cpatient["Name"] = cpatient["Name"].replace('CODES_proc', 'CCS CODES')
cpatient["Name"] = cpatient["Name"].replace('cat_threshold .95 most frequent_proc', 'threshold .95')
cpatient["Name"] = cpatient["Name"].replace('cat_threshold .88 most frequent', 'threshold .88')
cpatient["Name"] = cpatient["Name"].replace('cat_threshold .999 most frequent', 'threshold .999')
cpatient["Name"] = cpatient["Name"].replace('cat_threshold .98 most frequent', 'threshold .98')
cpatient.columns


# %%
cpatient

# %%
import matplotlib.pyplot as plt
import seaborn as sns

df1 = cpatient[['Name', 'count', '% Change', 'max', 'Unique_codes', 'Unique_codes_icd9', '% Codes']].sort_values(by="Unique_codes")

df1 = df1.rename(columns={"Unique_codes": 'Unique codes', "Unique_codes_icd9": 'Unique ICD-9 codes', "max": 'Max', "count": 'Count'})
# Set 'Name' column as index
df1.set_index('Name', inplace=True)

# Create a custom color palette with shades of blue
cmap = sns.color_palette("Blues", as_cmap=True)

# Create a heatmap
plt.figure(figsize=(12, 2))
sns.heatmap(df1, annot=True, cmap=cmap, fmt=".2f", linewidths=.5)
plt.title('Heatmap of changes in simplification (patient level)')


# Modify the y-axis label
plt.ylabel('Custom Y-Axis Label')

plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Create a custom color palette with shades of blue
cmap = sns.color_palette("Blues", as_cmap=True)

# Create a figure with a grid of subplots
fig, axes = plt.subplots(nrows=2, figsize=(12, 4))

# First Heatmap
sns.heatmap(df, annot=True, cmap=cmap, fmt=".2f", linewidths=.5, ax=axes[0])
axes[0].set_title('Heatmap of changes in simplification (patient level)')
axes[0].set_ylabel('Custom Y-Axis Label')

# Second Heatmap
sns.heatmap(df1, annot=True, cmap=cmap, fmt=".2f", linewidths=.5, ax=axes[1])
axes[1].set_title('Heatmap of changes in simplification (visit level)')
axes[1].set_ylabel('Simplification')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()

# %% [markdown]
# # Mutual information

# %%
DATA_DIRECTORY_results = "/Users/cgarciay/Desktop/results_SD/prepro/experimernt_prepro/"
directory_path = './models_cluster/'
directory_path = DATA_DIRECTORY_results +'drugs/'
mi_pa = pd.read_csv(directory_path+'mutual_info_patient_Patient_Drugs.csv')
mi_pa1 = pd.read_csv(directory_path+'mutual_info_visit_outs_visit_Drugs.csv')

# Drop the extra column from mi_pa1
mi_pa1 = mi_pa1.drop(columns=['Unnamed: 0'])

# Uncomment the following line if you want to rename the columns in mi_pa
# mi_pa = mi_pa.rename(columns={'MI': 'MI_vi', 'RI': 'RI_vi'})


# %%
directory_path = './models_cluster/'
directory_path = DATA_DIRECTORY_results +'diagnosis/'
mi_pa = pd.read_csv(directory_path+'mutual_info_patient_Patient_Diagnosis.csv')
mi_pa1 = pd.read_csv(directory_path+'mutual_info_visit_outs_visit_Diagnosis.csv')

# Drop the extra column from mi_pa1
mi_pa1 = mi_pa1.drop(columns=['Unnamed: 0'])

# Uncomment the following line if you want to rename the columns in mi_pa
# mi_pa = mi_pa.rename(columns={'MI': 'MI_vi', 'RI': 'RI_vi'})

# %%
mi_pa.rename(columns={"mutual_information":"mutual_information_p","rand index":"rand index_p","Name":"Mapping","Name2":"Column2"}, inplace = True)

# %%
mi_pa1.rename(columns={"Name":"Column1","Name2":"Column2"}, inplace = True)
mi_pa.rename(columns={"Mapping":"Column1","Name2":"Column2"}, inplace = True)

# %%
u =mi_pa1.groupby("Column1")["rand index"].idxmin()
mi_pa1.loc[u]


# %%
mi_pa["Column1"] = mi_pa["Column1"].replace('CCS CODES_proc_Patient_non_filtered.csv', 'CCS CODES')
mi_pa["Column1"] = mi_pa["Column1"].replace('ICD9_CODE_procedures_Patient_non_filtered.csv', 'ICD-9 CODES')
mi_pa["Column2"] = mi_pa["Column2"].replace('CCS CODES_proc_Patient_non_filtered.csv', 'CCS CODES')
mi_pa["Column2"] = mi_pa["Column2"].replace('ICD9_CODE_procedures_Patient_non_filtered.csv', 'ICD-9 CODES')

mi_pa["Column1"] = mi_pa["Column1"].replace('cat_threshold .95 most frequent_proc_Patient_non_filtered.csv', 'threshold .95')
mi_pa["Column1"] = mi_pa["Column1"].replace('cat_threshold .88 most frequent_Patient_non_filtered.csv', 'threshold .88')
mi_pa["Column1"] = mi_pa["Column1"].replace('cat_threshold .999 most frequent_Patient_non_filtered.csv', 'threshold .999')
mi_pa["Column1"] = mi_pa["Column1"].replace('cat_threshold .98 most frequent_Patient_non_filtered.csv', 'threshold .98')
mi_pa["Column1"] = mi_pa["Column1"].replace('sin_codigo.csv', 'No ICD9-Codes')
mi_pa["Column2"] = mi_pa["Column2"].replace('sin_codigo.csv', 'No ICD9-Codes')
mi_pa["Column2"] = mi_pa["Column2"].replace('cat_threshold .95 most frequent_proc_Patient_non_filtered.csv', 'threshold .95')
mi_pa["Column2"] = mi_pa["Column2"].replace('cat_threshold .88 most frequent_Patient_non_filtered.csv', 'threshold .88')
mi_pa["Column2"] = mi_pa["Column2"].replace('cat_threshold .999 most frequent_Patient_non_filtered.csv', 'threshold .999')
mi_pa["Column2"] = mi_pa["Column2"].replace('cat_threshold .98 most frequent_Patient_non_filtered.csv', 'threshold .98')


# %%
mi_pa1["Column1"] = mi_pa1["Column1"].replace('CCS CODES_proc_outs_visit_non_filtered.csv', 'CCS CODES')
mi_pa1["Column1"] = mi_pa1["Column1"].replace('ICD9_CODE_procedures_outs_visit_non_filtered.csv', 'ICD-9 CODES')
mi_pa1["Column2"] = mi_pa1["Column2"].replace('CCS CODES_proc_outs_visit_non_filtered.csv', 'CCS CODES')
mi_pa1["Column2"] = mi_pa1["Column2"].replace('ICD9_CODE_procedures_outs_visit_non_filtered.csv', 'ICD-9 CODES')

mi_pa1["Column1"] = mi_pa1["Column1"].replace('cat_threshold .95 most frequent_proc_outs_visit_non_filtered.csv', 'threshold .95')
mi_pa1["Column1"] = mi_pa1["Column1"].replace('cat_threshold .88 most frequent_outs_visit_non_filtered.csv', 'threshold .88')
mi_pa1["Column1"] = mi_pa1["Column1"].replace('cat_threshold .999 most frequent_outs_visit_non_filtered.csv', 'threshold .999')
mi_pa1["Column1"] = mi_pa1["Column1"].replace('cat_threshold .98 most frequent_outs_visit_non_filtered.csv', 'threshold .98')
mi_pa1["Column1"] = mi_pa1["Column1"].replace('sin_codigo.csv', 'No ICD9-Codes')
mi_pa1["Column2"] = mi_pa1["Column2"].replace('sin_codigo.csv', 'No ICD9-Codes')
mi_pa1["Column2"] = mi_pa1["Column2"].replace('cat_threshold .95 most frequent_proc_outs_visit_non_filtered.csv', 'threshold .95')
mi_pa1["Column2"] = mi_pa1["Column2"].replace('cat_threshold .88 most frequent_outs_visit_non_filtered.csv', 'threshold .88')
mi_pa1["Column2"] = mi_pa1["Column2"].replace('cat_threshold .999 most frequent_outs_visit_non_filtered.csv', 'threshold .999')
mi_pa1["Column2"] = mi_pa1["Column2"].replace('cat_threshold .98 most frequent_outs_visit_non_filtered.csv', 'threshold .98')


# %%
# Diccionario de reemplazos
reemplazos = {
    
    'CCS_CODES_diagnosis_Patient_non_filtered.csv': 'CCS CODES',
    'ICD9_CODE_diagnosis_Patient_non_filtered.csv': 'ICD-9 CODES',
       'threshold_0.95_diagnosis_Patient_non_filtered.csv': 'threshold .95',
    'threshold_0.88_diagnosis_Patient_non_filtered.csv': 'threshold .88',
    'threshold_0.999_diagnosis_Patient_non_filtered.csv': 'threshold .999',
     'threshold_0.98_diagnosis_Patient_non_filtered.csv': 'threshold .98',
    'sin_codigo.csv': 'No ICD9-Codes',
    'LEVE3 CODES_Patient_non_filtered.csv':'LEVE3 CODES'
}

reemplazos2 = {
    
    'CCS_CODES_diagnosis_outs_visit_non_filtered.csv': 'CCS CODES',
    'ICD9_CODE_diagnosis_outs_visit_non_filtered.csv': 'ICD-9 CODES',
       'threshold_0.95_diagnosis_outs_visit_non_filtered.csv': 'threshold .95',
    'threshold_0.88_diagnosis_outs_visit_non_filtered.csv': 'threshold .88',
    'threshold_0.999_diagnosis_outs_visit_non_filtered.csv': 'threshold .999',
     'threshold_0.98_diagnosis_outs_visit_non_filtered.csv': 'threshold .98',
    'sin_codigo.csv': 'No ICD9-Codes',
    'LEVE3 CODES_outs_visit_non_filtered.csv':'LEVE3 CODES'
}


reemplazos3 = {
    
    'ATC4_Patient_non_filtered.csv': 'ATC4',
    'DRUG_y_Patient_non_filtered.csv': 'Drugs',
       'threshold_0.95_Patient_non_filtered.csv': 'threshold .95',
    'threshold_0.88_Patient_non_filtered.csv': 'threshold .88',
    'threshold_0.999_Patient_non_filtered.csv': 'threshold .999',
     'threshold_0.98_Patient_non_filtered.csv': 'threshold .98',
    'sin_codigo.csv': 'No Drugs',
    'ATC3_Patient_non_filtered.csv':'ATC3'
}

reemplazos4 = {
    
    'ATC4_outs_visit_non_filtered.csv': 'ATC4',
    'DRUG_y_outs_visit_non_filtered.csv': 'Drugs',
       'threshold_0.95_outs_visit_non_filtered.csv': 'threshold .95',
    'threshold_0.88_outs_visit_non_filtered.csv': 'threshold .88',
    'threshold_0.999_outs_visit_non_filtered.csv': 'threshold .999',
     'threshold_0.98_outs_visit_non_filtered.csv': 'threshold .98',
    'sin_codigo.csv': 'No Drugs',
    'ATC3_outs_visit_non_filtered.csv':'ATC3'
}

# Reemplazos para la columna 1
mi_pa1["Column1"] = mi_pa1["Column1"].replace(reemplazos4)

# Reemplazos para la columna 2
mi_pa1["Column2"] = mi_pa1["Column2"].replace(reemplazos4)

mi_pa["Column1"] = mi_pa["Column1"].replace(reemplazos3)

# Reemplazos para la columna 2
mi_pa["Column2"] = mi_pa["Column2"].replace(reemplazos3)


##DIAGNOSI###
# Reemplazos para la columna 1
mi_pa1["Column1"] = mi_pa1["Column1"].replace(reemplazos2)

# Reemplazos para la columna 2
mi_pa1["Column2"] = mi_pa1["Column2"].replace(reemplazos2)

mi_pa["Column1"] = mi_pa["Column1"].replace(reemplazos)

# Reemplazos para la columna 2
mi_pa["Column2"] = mi_pa["Column2"].replace(reemplazos)


# %%
mi_pa["Column2"].unique()

# %%
mi_pa1["Column2"].unique()

# %%
mi_pa = mi_pa.fillna(.999)
mi_pa1 = mi_pa1.fillna(.98)

# %%
mi_pa["Column2"].unique()

# %%
#diagnosis
mi_pa = mi_pa[mi_pa["Column1"] != "threshold .88"]
mi_pa = mi_pa[mi_pa["Column2"] != "threshold .88"]
mi_pa1 = mi_pa1[mi_pa1["Column1"] != "threshold .88"]
mi_pa1 = mi_pa1[mi_pa1["Column2"] != "threshold .88"]


# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'df' with the columns 'Column1', 'Column2', 'MI', and 'RI'

# Reshape the DataFrame using pivot
pivot_df1 = mi_pa.pivot(index='Column2', columns='Column1', values='rand index_p')
pivot_df2 = mi_pa1.pivot(index='Column2', columns='Column1', values='rand index')
#pivot_df1['threshold .88'] = pivot_df1['threshold .88'].fillna(.984)
#pivot_df2['threshold .88'] = pivot_df2['threshold .88'].fillna(.9774)

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
plt.title('Heatmap of rand index patient level')

# Create the second subplot for the second heatmap
plt.subplot(1, 2, 2)
sns.heatmap(pivot_df2, cmap='Blues', annot=True, fmt=".3f")
plt.xlabel('ICD-9 Codes')
plt.ylabel('')
plt.title('Heatmap of rand index visit level')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# %%



# Assuming you have a DataFrame named 'df' with the columns 'Column1', 'Column2', 'MI', and 'RI'

# Reshape the DataFrame using pivot
pivot_df1 = mi_pa.pivot(index='Column2', columns='Column1', values='mutual_information_p')
pivot_df2 = mi_pa1.pivot(index='Column2', columns='Column1', values='mutual_information')
#pivot_df1['threshold .88'] = pivot_df1['threshold .88'].fillna(.984)
#pivot_df2['threshold .88'] = pivot_df2['threshold .88'].fillna(.9774)

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
plt.tight_layout()

# Show the plot
plt.show()

# %%
mi_pa.columns

# %%
mi_pa = mi_pa[(mi_pa['Column1']!='threshold .98')&(mi_pa['Column2']!='threshold .98')]
mi_pa1 = mi_pa1[(mi_pa1['Column1']!='threshold .98')&(mi_pa1['Column2']!='threshold .98')]

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'df' with the columns 'Column1', 'Column2', 'MI', and 'RI'

# Reshape the DataFrame using pivot
pivot_df1 = mi_pa.pivot(index='Column2', columns='Column1', values='rand index_p')
pivot_df2 = mi_pa1.pivot(index='Column2', columns='Column1', values='rand index')
#pivot_df1['threshold .88'] = pivot_df1['threshold .88'].fillna(.984)
#pivot_df2['threshold .88'] = pivot_df2['threshold .88'].fillna(.9774)

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
plt.title('Heatmap of rand index patient level')

# Create the second subplot for the second heatmap
plt.subplot(1, 2, 2)
sns.heatmap(pivot_df2, cmap='Blues', annot=True, fmt=".3f")
plt.xlabel('ICD-9 Codes')
plt.ylabel('')
plt.title('Heatmap of rand index visit level')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# %%


# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'df' with the columns 'Column1', 'Column2', 'MI', and 'RI'

# Reshape the DataFrame using pivot
pivot_df1 = mi_pa.pivot(index='Column2', columns='Column1', values='RI')
pivot_df2 = mi_pa1.pivot(index='Column2', columns='Column1', values='RI_vi')
pivot_df1['threshold .88'] = pivot_df1['threshold .88'].fillna(.984)
pivot_df2['threshold .88'] = pivot_df2['threshold .88'].fillna(.9774)

pivot_df1 = pivot_df1.astype(float)
pivot_df1 = pivot_df1.round(3)
pivot_df2 = pivot_df2.astype(float)
pivot_df2 = pivot_df2.round(3)

# Create the first subplot for the first heatmap
plt.figure(figsize=(13, 8))
plt.subplot(1, 2, 1)

sns.heatmap(pivot_df1, cmap='Blues', annot=True, fmt=".3f", cbar=False)
plt.xlabel('ICD-9 Codes')
plt.ylabel('')
plt.title('Heatmap of rand index\n(patient level)')  # Add a line break for spacing in the title

# Create the second subplot for the second heatmap
plt.subplot(1, 2, 2)
sns.heatmap(pivot_df2, cmap='Blues', annot=True, fmt=".3f")
plt.xlabel('ICD-9 Codes')
plt.ylabel('')
plt.title('Heatmap of rand index\n(visit level)')  # Add a line break for spacing in the title

# Adjust the spacing between subplots and increase spacing between graphs
plt.tight_layout(pad=5.0)  # Increase the pad value for more spacing

# Show the plot
plt.show()

# %% [markdown]
# # Demographics

# %%
import pandas as pd

# %%
#run  2_input_2

archivo_input_label = 'data_preprocess_non_filtered.csv'
df  = pd.read_csv('./data/'+archivo_input_label)
df.shape

# %%
df

# %%
df  = pd.read_csv('./data/'+archivo_input_label)

# %%
adm = pd.read_csv('/Users/cgarciay/Desktop/Laval_Master_Computer/research/MIMIC/ADMISSIONS.csv.gz')

# %%
adm.columns

# %%
import pandas as pd

# Assuming you have a DataFrame named df with a "subject" column and an "admission" column

# Group the DataFrame by "subject" and count the number of unique "admission" values for each subject
subject_counts = adm.groupby("SUBJECT_ID")["HADM_ID"].nunique()

# Filter the subjects that have more than one admission
subjects_with_multiple_admissions = subject_counts[subject_counts > 1]

# Count the number of subjects with multiple admissions
count = len(subjects_with_multiple_admissions)

# Print the count
print(count)

# %%
7505/53423 

# %%


new_column_names = {'ADMISSION_TYPE': 'Admission type', 'ADMISSION_LOCATION': 'Admission location', 'DISCHARGE_LOCATION': 'Discharge location',
                    'INSURANCE': 'Insurance', 'LANGUAGE': 'Language', 'RELIGION': 'Religion', 'MARITAL_STATUS': 'Marital status',   'ETHNICITY':'Ethnicity',     'HOSPITAL_EXPIRE_FLAG':'Death', 'HAS_CHARTEVENTS_DATA':'Has Chart Events'}

adm = adm.rename(columns=new_column_names)


# %%

adm["Death"] = np.where(adm["Death"]==1, 'Died', 'Survived')


# %%
adm.columns

# %%
adm.columns

# %%
adm["Death"].value_counts()
adm['Has Chart Events'] = np.where(adm['Has Chart Events']==1, 'Yes', 'No')

# %%
adm.drop_duplicates().shape

# %%
adm.ADMITTIME.nunique()

# %%
adm["Ethnicity"].value_counts()
adm["Death"].value_counts()
adm["Ethnicity"] = np.where(adm["Ethnicity"]=='BLACK/AFRICAN AMERICAN', 'AFRICAN AMERICAN', adm["Ethnicity"])

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'adm' is the DataFrame containing the data
# and 'columns' is the list of column names

# Select the top 5 and bottom 5 columns
top_columns = ['Admission type', 'Admission location',
               'Discharge location', 'Insurance', 'Marital status',]
bottom_columns = ['Language','Religion',  'Ethnicity', 'Death', 'Has Chart Events']

# Create subplots for the histograms
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))

# Plot histograms for the top columns
for i, column in enumerate(top_columns):
    ax = axes[0, i]
    adm[column].hist(ax=ax, bins=8, color='skyblue')  # Change the color to 'skyblue'
    ax.set_title(column)
    # Get the 5 most common categories
    if column == "Discharge location":
        top_categories = adm[column].value_counts().head(3)
    else:
        top_categories = adm[column].value_counts().head(5)
    # Set the x-axis labels to the top categories
    ax.set_xticks(top_categories.index)
    ax.set_xticklabels(top_categories.index, rotation=45)

# Plot histograms for the bottom columns
for i, column in enumerate(bottom_columns):
    ax = axes[1, i]
    adm[column].hist(ax=ax, bins=10, color='skyblue')  # Change the color to 'deepskyblue'
    ax.set_title(column)
    # Get the 5 most common categories
    top_categories = adm[column].value_counts().head(2)
    # Set the x-axis labels to the top categories
    ax.set_xticks(top_categories.index)
    ax.set_xticklabels(top_categories.index, rotation=45)

# Adjust the spacing between subplots
plt.tight_layout()

# Display the histograms
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt

top_columns = ['Admission type', 'Admission location',
               'Discharge location', 'Insurance', 'Marital status',]
bottom_columns = ['Language','Religion',  'Ethnicity', 'Death', 'Has Chart Events']

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(30, 30))

for i, column in enumerate(top_columns+bottom_columns):
    if column in ['Death', 'Has Chart Events']:
        ax = axes[i // 2, i % 2]
        top_categories = adm[column].value_counts()
        top_categories.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(column, fontsize=24)
        ax.set_xticklabels(top_categories.index, rotation=45, fontsize=20)
        ax.tick_params(axis='y', labelsize=16)
    else:
        ax = axes[i // 2, i % 2]
        value_counts = adm[column].value_counts()
        top_categories = value_counts.head(5)
        other_count = value_counts.sum() - top_categories.sum()
        categories = top_categories.append(pd.Series([other_count], index=["Other"]))
        categories.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(column, fontsize=24)
        ax.set_xticklabels(categories.index, rotation=45, fontsize=18)
        ax.tick_params(axis='y', labelsize=16)

plt.tight_layout()
plt.show()

# %%
[ 'ADMISSION_TYPE', 'ADMISSION_LOCATION',
       'DISCHARGE_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION',
       'MARITAL_STATUS', 'ETHNICITY',  
       'GENDER', 'EXPIRE_FLAG',]

# %%
df['age'].describe()

# %%
import pandas as pd

# Assuming 'df' is the DataFrame containing the data

# Get the name of all columns
column_names = adm.columns.tolist()

# Get the number of categories in each column
num_categories = [adm[column].nunique() for column in column_names]

# Create a new DataFrame with the column names and number of categories
table_data = {'Column Name': column_names, 'Number of Categories': num_categories}
table_df = pd.DataFrame(table_data)

# Display the table
print(table_df)

# %%
adm.columns

# %%
bins = [0, 18, 30, 40, 50, 60, 250]  # Define the age intervals for bins

# Create labels for the bins
labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '61-100']  # Labels for each bin range

# Use cut to create bins from the 'age' column
df['age_group'] = pd.cut(df['year_age'], bins=bins, labels=labels, right=False) 



#para grupo lenght of stay
bins = [-1,0, 25, 100, 400, 500, 700, 4107]  # Define the age intervals for bins

# Create labels for the bins
labels = ['0','1-25', '26-100', '101-401', '402-500', '501-700', '701-4107']  # Labels for each bin range

# Use cut to create bins from the 'age' column
df['INTERVAL_group'] = pd.cut(df['LOSRD'], bins=bins, labels=labels, right=False)  # Assign each value to a bin
# Assign each value to a bin
#df.loc[df['INTERVAL_group'].isnull(), 'INTERVAL_group'] = 0

# %%
df['INTERVAL_group'].unique()

# %%
df['age_group'].unique()

# %%
df.columns

# %%


# %%
import pandas as pd

# Group the DataFrame by 'ADMISSION_TYPE' and calculate the mode of the variables
grouped_df = df.groupby('SUBJECT_ID')['ADMISSION_TYPE', 'INSURANCE', 'GENDER', 'EXPIRE_FLAG', ].apply(lambda x: x.mode().iloc[0])

# Print the grouped DataFrame


# %%
grouped_df['EXPIRE_FLAG'].value_counts()

# %%
gender_map2 = {1.0: 'DIED', 0.0: 'SURVIVED'}

# Use the map function to create a new column 'gender_label'
grouped_df['EXPIRE_FLAG'] = grouped_df['EXPIRE_FLAG'].map(gender_map2)


df['EXPIRE_FLAG'] = df['EXPIRE_FLAG'].map(gender_map2)

# %%
df.columns

# %%
import matplotlib.pyplot as plt
import seaborn as sns

grouped_df = df
variables = ['ADMISSION_TYPE', 'INSURANCE', 'GENDER', 'EXPIRE_FLAG', 'ADMISSION_LOCATION','DISCHARGE_LOCATION','MARITAL_STATUS','LANGUAGE','RELIGION','ETHNICITY',]
title = ['ADMISSION TYPE', 'INSURANCE', 'GENDER',  'DEATH','ADMISSION LOCATION','DISCHARGE LOCATION','MARITAL STATUS','LANGUAGE','RELIGION','ETHNICITY',]
fig, axes = plt.subplots(5, 2, figsize=(50, 50))

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
    axes[i].set_title(title[i], fontsize=30)  
    axes[i].legend(pie[0], top_2_counts.index, loc='best', fontsize=30)
    axes[i].set_aspect('equal')  # Ensure pie is circular
    axes[i].set_xlabel('')  # Remove x-axis label
    axes[i].set_ylabel('')  # Remove y-axis label

plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

grouped_df = df
variables = ['ADMISSION_TYPE', 'INSURANCE', 'GENDER', 'EXPIRE_FLAG', 'ADMISSION_LOCATION','DISCHARGE_LOCATION','MARITAL_STATUS','LANGUAGE','RELIGION','ETHNICITY',]
title = ['ADMISSION TYPE', 'INSURANCE', 'GENDER',  'DEATH','ADMISSION LOCATION','DISCHARGE LOCATION','MARITAL STATUS','LANGUAGE','RELIGION','ETHNICITY',]
fig, axes = plt.subplots(5, 2, figsize=(30, 20))

fig.patch.set_facecolor('white')

axes = axes.flatten()

for i, variable in enumerate(variables):
    counts = grouped_df[variable].value_counts()
    if variable != 'EXPIRE_FLAG':  
        top_2 = counts.head(2)  
        other_count = counts.shape[0] - 2
        other_label = f'Other ({other_count:,})'
        top_2_counts = top_2.append(pd.Series([counts.sum() - top_2.sum()], index=[other_label]))
        colors = sns.color_palette('Blues', len(top_2_counts))
    else:
        top_2_counts = counts.head(2)  # Select top 2 categories for other variables
        colors = sns.color_palette('Blues', len(top_2_counts))
    
    pie = axes[i].pie(top_2_counts, labels=None, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 18}) 
    axes[i].set_title(title[i], fontsize=20)  
    axes[i].legend(pie[0], top_2_counts.index, loc='best', fontsize=18)
    axes[i].set_aspect('equal')  # Ensure pie is circular
    axes[i].set_xlabel('')  # Remove x-axis label
    axes[i].set_ylabel('')  # Remove y-axis label

plt.tight_layout()
plt.show()

# %%
pie = axes[i].pie(top_2_counts, labels=None, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 24})

# %%
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a DataFrame named 'df' with the required variables
grouped_df = df
# Create a list of variable names
variables = ['ADMISSION_TYPE', 'INSURANCE', 'GENDER', 'EXPIRE_FLAG',  'INTERVAL_group', 'ADMISSION_LOCATION','DISCHARGE_LOCATION','MARITAL_STATUS','LANGUAGE','RELIGION','ETHNICITY','year_age']
title = ['ADMISSION TYPE', 'INSURANCE', 'GENDER', 'DEATH', 'AGE', 'TOTAL LENGTH OF STAY']
# Create a figure with 2 rows and 3 columns to accommodate the 6 pie charts
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Set the background color to white
fig.patch.set_facecolor('white')

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Iterate over the variables and create a pie chart for each
for i, variable in enumerate(variables):
    # Count the occurrences of each category in the variable
    counts = grouped_df[variable].value_counts()
    
    # Create the pie chart with a different shade of blue for each chart
    colors = sns.color_palette('Blues', len(counts))
    pie = axes[i].pie(counts, labels=None, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[i].set_title(title[i])
    
    # Get the top 5 categories and their counts
    if variable == 'LANGUAGE':
        top_2 = counts.head(2)
        labels = [f'{label} ({count:,})' for label, count in zip(top_2.index, top_2.values)]
        legend_labels = [f'{label} ({count:,})' for label, count in zip(counts.index[:5], counts.values[:5])]
    else:
        top_5 = counts.head(5)
        labels = [f'{label} ({count:,})' for label, count in zip(top_5.index, top_5.values)]
        legend_labels = labels
    
    # Set the labels for the pie chart with better positioning
    axes[i].legend(pie[0], legend_labels, loc='best')
    
# Adjust the spacing between subplots
#plt.title("Visit level variables")
plt.tight_layout()

# Show the plot
plt.show()

# %%
df.columns  

# %%
df.columns

# %%
df.age

# %%
def mode(x):
    '''function to obtain the mode'''
    return x.mode()[0]
grouped_df1 = df.groupby('SUBJECT_ID').agg({'age': 'max', 'LOSRD': 'sum','GENDER':mode}).reset_index()


# %%
df2 = pd.read_csv('./data/ICD9_CODE_procedures_outs_visit_non_filtered.csv')

# %%
df2.shape

# %%
aux = df2[df2['LOSRD_sum']<200]
aux.shape

# %%
45956/50565

# %%
df2[df2["Age_max"]<0]

# %%
df["Age_max"] = df["year_age"]

# %%
grouped_df1 = df2.groupby('SUBJECT_ID').agg({'Age_max': 'max', 'LOSRD_sum': 'sum','GENDER':mode,'L_1s_last_p1':mode}).reset_index()



# %%
df2['GENDER']

# %%
import pandas as pd

# Assuming you have a DataFrame named 'df' with a column named 'gender' containing 1s and 0s

# Create a dictionary to map the values
gender_map = {1: 'Male', 0: 'Female'}
gender_map2 = {1.0: 'Male', 0.0: 'Female'}

# Use the map function to create a new column 'gender_label'
#df2['GENDER'] = df2['GENDER'].map(gender_map)
grouped_df1['GENDER'] = grouped_df1['GENDER'].map(gender_map2)
df2['GENDER'] = df2['GENDER'].map(gender_map2)

# %%
df2['GENDER']

# %%
grouped_df1[grouped_df1['GENDER'] == 'Female'] ['Age_max']

# %%
print(grouped_df1['GENDER'].value_counts())
gender_map = {1.0: 'Male', 0.0: 'Female'}

# Use the map function to create a new column 'gender_label'
grouped_df1['Gender'] = grouped_df1['GENDER'].map(gender_map)
grouped_df1['GENDER'] = np.where(grouped_df1['GENDER'] ==1.0, 'Male', 'Female')

# %%
grouped_df1[grouped_df1['GENDER'] ==1.0]

# %%
grouped_df1[grouped_df1['GENDER'] == 'Male']

# %%
grouped_df1["Age_max"]=[int(i) for i in grouped_df1.Age_max]

# %%
df2.iloc[:,-17:]

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Assuming 'grouped_df1' is your DataFrame with 'Age_max', 'LOSRD_sum', and 'GENDER' columns

# Set the background style
plt.style.use("seaborn-white")

# Define two shades of blue
dark_blue = "#00008B"   # A dark blue
light_blue = "#ADD8E6"  # A light blue

# Create two figures and axes side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

# Calculate the kernel density estimate for 'Age_max' for each gender
age_max_male = df2[df2['GENDER'] == 'Male']['Age_max']
age_max_female = df2[df2['GENDER'] == 'Female']['Age_max']
kde_age_max_male = gaussian_kde(age_max_male)
kde_age_max_female = gaussian_kde(age_max_female)

# Generate x-values for the kernel density estimate
x_age_max = np.linspace(age_max_male.min(), age_max_male.max(), 100)

# Plot the kernel density estimate for 'Age_max' with the dark blue for male and light blue for female
ax1.plot(x_age_max, kde_age_max_male(x_age_max), color=dark_blue, label='Male')
ax1.plot(x_age_max, kde_age_max_female(x_age_max), color=light_blue, label='Female')

# Set the labels and title for the 'Age_max' plot
ax1.set_xlabel('Age')
ax1.set_ylabel('Density')
ax1.set_title('Kernel Density Plot - Age')
ax1.set_ylim(bottom=0)  # Set the y-axis limit to start at zero

# Calculate the kernel density estimate for 'LOSRD_sum' for each gender
aux = df2[df2['LOSRD_sum']<200]
losrd_sum_male = aux[aux['GENDER'] == 'Male'] ['LOSRD_sum']
losrd_sum_female = aux[aux['GENDER'] == 'Female'][ 'LOSRD_sum']
kde_losrd_sum_male = gaussian_kde(losrd_sum_male)
kde_losrd_sum_female = gaussian_kde(losrd_sum_female)

# Generate x-values for the kernel density estimate
x_losrd_sum = np.linspace(losrd_sum_male.min(), losrd_sum_male.max(), 100)

# Plot the kernel density estimate for 'LOSRD_sum' with the dark blue for male and light blue for female
ax2.plot(x_losrd_sum, kde_losrd_sum_male(x_losrd_sum), color=dark_blue, label='Male')
ax2.plot(x_losrd_sum, kde_losrd_sum_female(x_losrd_sum), color=light_blue, label='Female')

# Set the labels and title for the 'LOSRD_sum' plot
ax2.set_xlabel('Length of Stay (days)')
ax2.set_ylabel('Density')
ax2.set_title('Kernel Density Plot - Length of Stay')
ax2.set_ylim(bottom=0)  # Set the y-axis limit to start at zero

# Remove the legends from the individual plots
ax1.legend().remove()
ax2.legend().remove()

# Adjust the layout before adding a common legend
plt.subplots_adjust(wspace=0.3, top=0.85)

# Create a common legend for both plots with corrected labels
fig.legend(labels=['Male', 'Female'], loc='upper center', ncol=2)

# Show the plots
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'df' with the 'age' and 'lord' columns

# Set the background color to white
sns.set_style("white")

# Create three subplots in a single figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1
sns.kdeplot(data=grouped_df1, x='Age_max', fill=True, hue='GENDER', color='#ADD8E6', alpha=0.5, ax=ax1)
ax1.set_xlabel('Age')
ax1.set_ylabel('Density')
ax1.set_title('Kernel Density Plot - Age')

# Plot 2
sns.kdeplot(data=grouped_df1, x='LOSRD_sum', fill=True, hue='GENDER', color='#E0FFFF', alpha=0.5, ax=ax2)
ax2.set_xlabel('Length of Stay')
ax2.set_ylabel('Density')
ax2.set_title('Kernel Density Plot - Length of Stay')

# Plot 3
sns.kdeplot(data=grouped_df1, x='L_1s_last_p1', fill=True, hue='GENDER', color='#FFC0CB', alpha=0.5, ax=ax3)
ax3.set_xlabel('Another Column')
ax3.set_ylabel('Density')
ax3.set_title('Kernel Density Plot - Another Column')

# Remove the legends from all plots
ax1.legend().remove()
ax2.legend().remove()
ax3.legend().remove()

# Create a common legend for all plots
legend_labels = ['Male', 'Female']
fig.legend(labels=legend_labels, loc='center')

plt.subplots_adjust(wspace=0.4)

# Show the plots
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Generate some sample data
np.random.seed(0)
data = pd.DataFrame({
    'Variable1': np.random.normal(loc=50, scale=10, size=100),
    'Variable2': np.random.normal(loc=60, scale=15, size=100),
    'Variable3': np.random.normal(loc=55, scale=20, size=100),
    'Time': pd.date_range('20210101', periods=100)
})

data['Variable4'] = data['Variable1'] * 0.5 + data['Variable2'] * 0.3 + np.random.normal(loc=0, scale=5, size=100)

# Create a 2x2 grid of plots
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

# Scatter Plot
sns.scatterplot(x='Variable1', y='Variable2', data=data, ax=ax[0, 0])
ax[0, 0].set_title('Scatter Plot of Variable1 vs Variable2')

# Line Graph
sns.lineplot(x='Time', y='Variable3', data=data, ax=ax[0, 1])
ax[0, 1].set_title('Line Graph of Variable3 over Time')

# Histogram
sns.histplot(data['Variable1'], bins=20, kde=True, ax=ax[1, 0])
ax[1, 0].set_title('Histogram of Variable1')

# Box Plot
sns.boxplot(y=data['Variable4'], ax=ax[1, 1])
ax[1, 1].set_title('Box Plot of Variable4')

plt.tight_layout()
plt.show()


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'df' with the 'age' and 'lord' columns

# Set the background color to white
sns.set_style("white")

# Create two figures and axes side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot the kernel density estimate for 'age' with a green shade
sns.kdeplot(data=grouped_df1, x='Age_max', fill=True, hue='GENDER', color='green', alpha=0.5, ax=ax1)

# Set the labels and title for the 'age' plot
ax1.set_xlabel('Age')
ax1.set_ylabel('Density')
ax1.set_title('Kernel Density Plot - Age')

# Plot the kernel density estimate for 'lord' with a green shade
sns.kdeplot(data=grouped_df1, x='LOSRD_sum', fill=True, hue='GENDER', color='green', alpha=0.5, ax=ax2)

# Set the labels and title for the 'lord' plot
ax2.set_xlabel('Lord')
ax2.set_ylabel('Density')
ax2.set_title('Kernel Density Plot - Lord')

# Change the majority of 'GENDER' labels to 'Male' (1.0) and the minority to 'Female' (0.0)

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.4)

# Show the plots
plt.show()

# %% [markdown]
# # Preds

# %%


days = '30'
ejemplo_dir = './input_model_pred/'
archivo_input_label = 'data_preprocess_non_filtered.csv'


path = "./input_model_pred/" +"images"
days_list = ["90"]
ficheros = read_director(ejemplo_dir)
len(ficheros)

# %%
for i in ficheros:
    print(i)
    
ficheros = [i for i in ficheros if i != 'sin_codigo_non_filtered.csv']
ficheros

# %%
X_aux = pd.read_csv(ejemplo_dir+'cat_threshold .88 most frequent_outs_visit_non_filtered.csv')
X_aux.columns[-17:]

# %%
dataframes = []
for i in ficheros:
        
    if i in ['ICD9_CODE_procedures.csv', 'CCS CODES_proc.csv', 'cat_threshold .999 most frequent']:
        prepo = "max"
    elif i in ['cat_threshold .95 most frequent_proc','cat_threshold .88 most frequent']:
        prepo = "power"
    else:
        prepo = "std"
    print(prepo)
    
    # se obtiene la matriz de features y la variable response
    X_aux = pd.read_csv(ejemplo_dir+i)
    
    # if it's not BERT embedding, we do the preprocessing
    X_aux = X_aux.drop(["HADM_ID","SUBJECT_ID",'Unnamed: 0','L_1s_last','HADM_ID', 'Age_max', 'LOSRD_sum',
       'L_1s_last', 'LOSRD_avg', 'ADMISSION_TYPE', 'ADMISSION_LOCATION',
       'DISCHARGE_LOCATION', 'INSURANCE', 'RELIGION', 'MARITAL_STATUS',
       'ETHNICITY', 'GENDER', 'L_1s_last_p1'], axis=1)

    try:
        X = X_aux.values
        dataframes.append(pd.DataFrame(X))
    except:
        pass


# %%
dataframes[0]

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pacmap

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
plt.show()


# %% [markdown]
# ## preds results

# %%
res = pd.read_csv('./results_pred/results_prediction_30+_realv3.csv')


# %%
res.shape

# %%
res.drop_duplicates()

# %%
res.columns

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming df is your dataframe with the data
# df = pd.read_csv('your_data.csv')  # If your data is in a CSV file

# Set the style
sns.set_style("whitegrid")

# Create a figure for the combined dashboard
fig = plt.figure(figsize=(20, 10))

# Create a grid for the subplots
grid = plt.GridSpec(2, 4, hspace=0.6, wspace=0.4)

# Add a line chart for F1 scores on the top left of the grid
ax1 = fig.add_subplot(grid[0, 0])
sns.lineplot(data=res, x='Classifiers', y='f1_test', marker='o', label='F1 Test', ax=ax1)
sns.lineplot(data=res, x='Classifiers', y='f1_train', marker='o', label='F1 Train', ax=ax1)
ax1.set_title('F1 Scores by Classifier')
ax1.set_xticklabels(res['Classifiers'].unique(), rotation=45, ha='right')

# Add a bar chart for accuracy on the top right of the grid
ax2 = fig.add_subplot(grid[0, 1])
sns.barplot(data=res, x='Classifiers', y='accuracy_test', hue='Sampling', ax=ax2)
ax2.set_title('Accuracy by Classifier and Sampling')
ax2.set_xticklabels(res['Classifiers'].unique(), rotation=45, ha='right')

# Add a box plot for precision on the bottom left of the grid
ax3 = fig.add_subplot(grid[1, 0])
sns.boxplot(data=res, x='Classifiers', y='precision_train', ax=ax3)
ax3.set_title('Precision Distribution by Classifier')
#ax3.set_xticklabels(df['Classifiers'].unique(), rotation=45, ha='right')

# Add a scatter plot for recall vs. precision on the bottom right of the grid
ax4 = fig.add_subplot(grid[1, 1])
sns.scatterplot(data=res, x='precision_train', y='sensitivity_train', hue='Classifiers', style='Sampling', ax=ax4)
ax4.set_title('Recall vs Precision')

# Adjust subplots
plt.subplots_adjust(top=0.9)

# Add an overall title
plt.suptitle('Model Performance Dashboard', fontsize=16)

# Show the plot
plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming df is your dataframe with the data
# df = pd.read_csv('your_data.csv')  # If your data is in a CSV file

# Set the style
sns.set_style("whitegrid")

# Create a figure for the combined dashboard
fig = plt.figure(figsize=(20, 10))

# Create a grid for the subplots
grid = plt.GridSpec(2, 4, hspace=0.6, wspace=0.4)

# Add a scatter plot for recall vs. precision on the top left of the grid
ax1 = fig.add_subplot(grid[0, 0])
sns.scatterplot(data=res, x='precision_train', y='sensitivity_train', hue='Classifiers', style='Sampling', ax=ax1)
ax1.set_title('Recall vs Precision')
ax1.set_xlabel('Precision')
ax1.set_ylabel('Recall')

# Add a box plot for precision on the top right of the grid
ax2 = fig.add_subplot(grid[0, 1])
sns.boxplot(data=res, x='Classifiers', y='precision_train', ax=ax2)
ax2.set_title('Precision Distribution by Classifier')
ax2.set_xticklabels(res['Classifiers'].unique(), rotation=45, ha='right')

# Add a line chart for F1 scores on the bottom left of the grid
ax3 = fig.add_subplot(grid[1, 0])
sns.lineplot(data=res, x='Classifiers', y='f1_test', marker='o', label='F1 Test', ax=ax3)
sns.lineplot(data=res, x='Classifiers', y='f1_train', marker='o', label='F1 Train', ax=ax3)
ax3.set_title('F1 Scores by Classifier')
ax3.set_xticklabels(res['Classifiers'].unique(), rotation=45, ha='right')

# Add a bar chart for accuracy on the bottom right of the grid
ax4 = fig.add_subplot(grid[1, 1])
sns.barplot(data=res, x='Classifiers', y='accuracy_test', ax=ax4)
ax4.set_title('Accuracy by Classifier')
ax4.set_xticklabels(res['Classifiers'].unique(), rotation=45, ha='right')

# Adjust subplots
plt.subplots_adjust(top=0.9)

# Add an overall title
plt.suptitle('Model Performance Dashboard', fontsize=16)

# Show the plot
plt.show()

# %%
res["Classifiers"].unique()

# %%


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming df is your dataframe with the data
# df = pd.read_csv('your_data.csv')  # If your data is in a CSV file

# Set the style
sns.set_style("whitegrid")

# Create a figure for the combined dashboard
fig = plt.figure(figsize=(30, 10))

# Create a grid for the subplots
grid = plt.GridSpec(2, 6, hspace=0.6, wspace=0.4)

# Add a scatter plot for recall vs. precision on the top left of the grid (using test set data)
ax1 = fig.add_subplot(grid[0, 0])
scatterplot = sns.scatterplot(data=res, x='precision_test', y='sensitivity_test', hue='Classifiers',  ax=ax1)
ax1.set_title('Recall vs Precision (Test Set)')
ax1.set_xlabel('Precision')
ax1.set_ylabel('Recall')

# Add a box plot for precision on the top right of the grid (using test set data)
ax2 = fig.add_subplot(grid[0, 1])
sns.boxplot(data=res, x='Classifiers', y='precision_test', ax=ax2)
ax2.set_title('Precision Distribution by Classifier (Test Set)')
ax2.set_xticklabels(res['Classifiers'].unique(), rotation=45, ha='right')

# Add a line chart for F1 scores on the bottom left of the grid
ax3 = fig.add_subplot(grid[1, 0])
sns.boxplot(data=res, x='Classifiers', y='precision_train', ax=ax3)
ax3.set_title('Precision Distribution by Classifier (Train Set)')
ax3.set_xticklabels(res['Classifiers'].unique(), rotation=45, ha='right')

# Add a bar chart for accuracy on the bottom right of the grid
ax4 = fig.add_subplot(grid[1, 1])
sns.scatterplot(data=res, x='precision_train', y='sensitivity_train', hue='Classifiers', ax=ax4)
ax4.set_title('Recall vs Precision (Test Set)')
ax4.set_xlabel('Precision')
ax4.set_ylabel('Recall')

# Add a scatter plot for recall vs. accuracy on the top middle of the grid (using test set data)
ax5 = fig.add_subplot(grid[0, 2])
sns.scatterplot(data=res, x='accuracy_test', y='sensitivity_test', hue='Classifiers', ax=ax5)
ax5.set_title('Recall vs Accuracy (Test Set)')
ax5.set_xlabel('Accuracy')
ax5.set_ylabel('Recall')

# Add a box plot for accuracy on the top right of the grid (using test set data)
ax6 = fig.add_subplot(grid[0, 3])
sns.boxplot(data=res, x='Classifiers', y='accuracy_test', ax=ax6)
ax6.set_title('Accuracy Distribution by Classifier (Test Set)')
ax6.set_xticklabels(res['Classifiers'].unique(), rotation=45, ha='right')

# Add a line chart for F1 scores on the bottom middle of the grid
ax7 = fig.add_subplot(grid[1, 2])
sns.boxplot(data=res, x='Classifiers', y='accuracy_train', ax=ax7)
ax7.set_title('Accuracy Distribution by Classifier (Train Set)')
ax7.set_xticklabels(res['Classifiers'].unique(), rotation=45, ha='right')

# Add a bar chart for accuracy on the bottom right of the grid
ax8 = fig.add_subplot(grid[1, 3])
sns.scatterplot(data=res, x='accuracy_train', y='sensitivity_train', hue='Classifiers', ax=ax8)
ax8.set_title('Recall vs Accuracy (Test Set)')
ax8.set_xlabel('Accuracy')
ax8.set_ylabel('sensitivity')

# Remove the legend from all scatter plots except the first one
for ax in [ax1, ax5, ax8]:
    try:
       ax.get_legend().remove()
    except:
        pass   

# Adjust subplots
plt.subplots_adjust(top=0.9)

# Add an overall title
plt.suptitle('', fontsize=16)

# Show the plot
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming df is your dataframe with the data
# df = pd.read_csv('your_data.csv')  # If your data is in a CSV file

# Set the style and color palette
sns.set_style("whitegrid")
sns.set_palette(["#1f77b4", "#aec7e8"])

# Create a figure for the combined dashboard
fig = plt.figure(figsize=(30, 8))

# Create a grid for the subplots
grid = plt.GridSpec(2, 6, hspace=0.6, wspace=0.4)

# Add a scatter plot for recall vs. precision on the top left of the grid (using test set data)
ax1 = fig.add_subplot(grid[0, 0])
scatterplot = sns.scatterplot(data=res, x='precision_test', y='sensitivity_test', hue='Classifiers',  ax=ax1)
ax1.set_title('Sensitivity  vs Precision (Test Set)')
ax1.set_xlabel('Precision')
ax1.set_ylabel('Sensitivity test')  # Change the y-axis label here

# Add a box plot for precision on the top right of the grid (using test set data)
ax2 = fig.add_subplot(grid[0, 1])
sns.boxplot(data=res, x='Classifiers', y='precision_test', ax=ax2)
ax2.set_title('Precision Distribution by Classifier (Test Set)')
ax2.set_xticklabels(res['Classifiers'].unique(), rotation=45, ha='right')
ax2.set_ylabel('Precision test')
# Add a line chart for F1 scores on the bottom left of the grid
ax3 = fig.add_subplot(grid[1, 0])
sns.boxplot(data=res, x='Classifiers', y='precision_train', ax=ax3)
ax3.set_title('Precision Distribution by Classifier (Train Set)')
ax3.set_xticklabels(res['Classifiers'].unique(), rotation=45, ha='right')
ax3.set_ylabel('Precision train')

# Add a bar chart for accuracy on the bottom right of the grid
ax4 = fig.add_subplot(grid[1, 1])
sns.scatterplot(data=res, x='precision_train', y='sensitivity_train', hue='Classifiers', ax=ax4)
ax4.set_title('Sensitivity vs Precision (Train Set)')
ax4.set_xlabel('Precision')
ax4.set_ylabel('Sensitivity train')  # Change the y-axis label here

# Add a scatter plot for recall vs. accuracy on the top middle of the grid (using test set data)
ax5 = fig.add_subplot(grid[0, 2])
sns.scatterplot(data=res, x='accuracy_test', y='sensitivity_test', hue='Classifiers', ax=ax5)
ax5.set_title('Sensitivity vs Accuracy (Test Set)')
ax5.set_xlabel('Accuracy')
ax5.set_ylabel('Sensitivity test')  # Change the y-axis label here

# Add a box plot for accuracy on the top right of the grid (using test set data)
ax6 = fig.add_subplot(grid[0, 3])
sns.boxplot(data=res, x='Classifiers', y='accuracy_test', ax=ax6)
ax6.set_title('Accuracy Distribution by Classifier (Test Set)')
ax6.set_xticklabels(res['Classifiers'].unique(), rotation=45, ha='right')
ax6.set_ylabel('Accuracy test')
# Add a line chart for F1 scores on the bottom middle of the grid
ax7 = fig.add_subplot(grid[1, 2])
sns.boxplot(data=res, x='Classifiers', y='accuracy_train', ax=ax7)
ax7.set_title('Accuracy Distribution by Classifier (Train Set)')
ax7.set_xticklabels(res['Classifiers'].unique(), rotation=45, ha='right')
ax7.set_ylabel('Accuracy train')

# Add a bar chart for accuracy on the bottom right of the grid
ax8 = fig.add_subplot(grid[1, 3])
sns.scatterplot(data=res, x='accuracy_train', y='sensitivity_train', hue='Classifiers', ax=ax8)
ax8.set_title('Recall vs Accuracy (Train Set)')
ax8.set_xlabel('Accuracy')
ax8.set_ylabel('Sensitivity train')  # Change the y-axis label here

# Remove the legend from all scatter plots except the first one
for ax in [ax1, ax5, ax8]:
    try:
       ax.get_legend().remove()
    except:
        pass   

# Adjust subplots
plt.subplots_adjust(top=0.9)

# Add an overall title
plt.suptitle('', fontsize=16)

# Show the plot
plt.show()

# %%
res.head()

# %%
res.columns

# %%
res["Mapping"] = res["Mapping"].replace('CCS CODES_proc_outs_visit_non_filtered.csv', 'CCS CODES')
res["Mapping"] = res["Mapping"].replace('ICD9_CODE_procedures_outs_visit_non_filtered.csv', 'ICD-9 CODES')
res["Mapping"] = res["Mapping"].replace('cat_threshold .95 most frequent_proc_outs_visit_non_filtered.csv', 'threshold .95')
res["Mapping"] = res["Mapping"].replace('cat_threshold .88 most frequent_outs_visit_non_filtered.csv', 'threshold .88')
res["Mapping"] = res["Mapping"].replace('cat_threshold .999 most frequent_outs_visit_non_filtered.csv', 'threshold .999')
res["Mapping"] = res["Mapping"].replace('cat_threshold .98 most frequent_outs_visit_non_filtered.csv', 'threshold .98')
res["Mapping"] = res["Mapping"].replace('sin_codigo_non_filtered.csv', 'No ICD9-Codes')


# %%
pivot_df

# %%
df = res[res["Feature selection"]==True]

# Pivot the DataFrame for the heatmap
pivot_df = df.pivot_table(index=['Mapping', ], values=[ 'var_change','var_ini'])
pivot_df = pivot_df.rename(columns={'var_change': 'Variables removed', 'var_ini': 'Initial variables'})
plt.figure(figsize=(6, 3))
heatmap = sns.heatmap(pivot_df, cmap='Blues', annot=True, fmt=".0f")  # Change the cmap to 'Blues' for different shades of blue
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=0)  # Change the x-axis labels rotation if needed
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)  # Change the y-axis labels rotation if needed
plt.xlabel('')  # Change the x-axis label here
plt.ylabel('Simplification')  # Change the y-axis label here

# Change the 'var_change' and 'var_ini' labels in the colorbar
colorbar = heatmap.collections[0].colorbar
colorbar.set_label('Number of features')

plt.title('Feature Reduction')
plt.show()

# %% [markdown]
# # demografics

# %%
df = pd.read_csv("./data/data_preprocess_non_filtered.csv")
pd.set_option('display.max_columns', None)




# %%
df.columns

# %%


df = pd.read_csv("./data/procedures_preprocess_threshold_nonfiltered.csv")
df.columns

# %%
import matplotlib.pyplot as plt
import seaborn as sns

result = df.groupby(["'ICD-9-CM CODE DESCRIPTION'"]).size().reset_index(name='Count')
result = result.sort_values(by="Count", ascending=False)[:10]

plt.figure(figsize=(12, 8))
sns.barplot(x="'ICD-9-CM CODE DESCRIPTION'", y='Count', data=result, color='lightblue')
plt.xticks(rotation=45, fontsize=12)  # Increase the font size of x-axis labels
plt.xlabel('Procedure', fontsize=14)  # Increase the font size of x-axis label
plt.ylabel('Count', fontsize=14)  # Increase the font size of y-axis label
plt.title('Top 10 Most Frequent procedures', fontsize=16)  # Increase the font size of the title

plt.show()

# %%
result_subject = df.groupby("SUBJECT_ID").size().reset_index(name='Count')

# Count per admission ID
result_admission = df.groupby("HADM_ID").size().reset_index(name='Count')

# %%
result_subject_1 = result_subject[result_subject["Count"]<100]
result_admission_1 = result_admission[result_admission["Count"]<1000]

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
sns.set(style="darkgrid")

# Load the iris dataset
df = sns.load_dataset("iris")

# Create a figure with two matplotlib.Axes objects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Assigning a graph to each ax
sns.histplot(data=result_subject_1, x="Count", ax=ax1, color='darkblue',bins = 50)
sns.histplot(data=result_admission_1, x="Count", ax=ax2, color='lightblue',bins = 30)

# Set x-axis and y-axis labels for each subplot
ax1.set(xlabel='Count of ICD-9 codes per patient', ylabel='Frequency')
ax2.set(xlabel='Count of ICD-9 codes per admission', ylabel='Frequency')

ax1.text(-0.1, -0.2, '(a)', transform=ax1.transAxes, size=10, )
ax2.text(-0.1, -0.2, '(b)', transform=ax2.transAxes, size=10, )

fig.suptitle('Count of ICD-9 codes procedures', fontsize=14)  # Increase the font size of the title
# Show the plot
plt.show()

# %%
print(df.GENDER.value_counts())
uq = df.GENDER.unique()
[print(i .replace("'", "")) for i in uq]

# %%
print(df.ETHNICITY.value_counts())
uq = df.ETHNICITY.unique()
[print(i .replace("'", "")) for i in uq]

# %%
print(df.ETHNICITY.value_counts())
uq = df.ETHNICITY.unique()
[print(i .replace("'", "")) for i in uq]

# %%



df = pd.read_csv("./data/data_preprocess_non_filtered.csv")
pd.set_option('display.max_columns', None)

#Lectures of dataframe that have the procedures icd-9 codes with different threshold
proc = pd.read_csv("./data/procedures_preprocess_threshold_nonfiltered.csv")
grouped = proc.groupby(['SUBJECT_ID', 'HADM_ID']).agg(lambda x: x.tolist())

# Reset
# the index to make 'SUBJECT_ID' and 'HADM_ID' regular columns
grouped_proc = grouped.reset_index()
df1=grouped_proc.copy()

list_cat = ['CCS CODES_proc', 'cat_threshold .95 most frequent_proc','cat_threshold .88 most frequent', 'cat_threshold .98 most frequent',
              'cat_threshold .999 most frequent']

categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
              'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
              'MARITAL_STATUS', 'ETHNICITY','GENDER']


nam_p_list = ['CCS CODES_proc', 'cat_threshold .95 most frequent_proc','Threshold', 'Threshold',
       'Threshold']

filtered = True
v = "j" 


result_stat  = {'Name' : [],
       'count': [],
'mean': [],
'std': [],
'min': [],
'25%': [],
'50%': [],
'75%': [],
'max': [] ,
"Min >":[],
"Unique_codes":[],
"Unique_codes_icd9":[]}
list_dataframes = []
for i in range(len(list_cat)):
       nam_p = nam_p_list[i]
       real = list_cat[i]
       df_descon = obtener_ghist(i,df,df1,nam_p,v,real,filtered)
       list_dataframes.append(df_descon) 
       

#df_res = pd.DataFrame(result_stat)
#df_res.to_csv("./results_changes/"+v+"_nonfiltered.csv")

# %%
list_dataframes[1].iloc[:,:4]

# %%
new_l = [i.iloc[:,:4] for i in list_dataframes]

# %%
df = list_dataframes[1]
df.head()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Example DataFrame

# Count the number of ICD-9 codes per admission
icd9_counts_a = df.groupby('HADM_ID')['ICD9_CODE_procedures'].count().reset_index(name='count_icd9_codes')

# If you have specific admission IDs you want to plot, filter them here
# For example, if you want to plot for admission IDs 'A1', 'A2', 'A3', 'A4', 'A5'
selected_admissions_a = icd9_counts_a

icd9_counts = df.groupby('SUBJECT_ID')['ICD9_CODE_procedures'].count().reset_index(name='count_icd9_codes')

# If you have specific admission IDs you want to plot, filter them here
# For example, if you want to plot for admission IDs 'A1', 'A2', 'A3', 'A4', 'A5'
selected_admissions = icd9_counts

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Boxplot of ICD-9 Codes Count per admission
boxplot_a = axs[0, 0].boxplot(selected_admissions_a['count_icd9_codes'])
axs[0, 0].set_title('Boxplot of ICD-9 Codes Count per admission')
axs[0, 0].set_ylabel('Count of ICD-9 Codes')


# Histogram of ICD-9 Codes Count per admission
axs[0, 1].hist(selected_admissions_a['count_icd9_codes'], bins=25, color='lightblue')
axs[0, 1].set_title('Histogram of ICD-9 Codes Count')
axs[0, 1].set_xlabel('Count of ICD-9 Codes per admission')
axs[0, 1].set_ylabel('Frequency')

# Boxplot of ICD-9 Codes Count per subject
boxplot = axs[1, 0].boxplot(selected_admissions['count_icd9_codes'])
axs[1, 0].set_title('Boxplot of ICD-9 Codes Count per subject')
axs[1, 0].set_ylabel('Count of ICD-9 Codes')


# Histogram of ICD-9 Codes Count per subject
axs[1, 1].hist(selected_admissions['count_icd9_codes'], bins=25, color='lightblue')
axs[1, 1].set_title('Histogram of ICD-9 Codes Count')
axs[1, 1].set_xlabel('Count of ICD-9 Codes per subject')
axs[1, 1].set_ylabel('Frequency')

# Add legend with median, quartiles, and mean to both boxplots
legend_text_a = f"Median: {medians_a[0]:.2f}\nQ1: {quartiles_a[0]:.2f}\nQ3: {quartiles_a[1]:.2f}\nMean: {selected_admissions_a['count_icd9_codes'].mean():.2f}"
axs[0, 0].legend([legend_text_a], loc='upper left')

legend_text = f"Median: {medians[0]:.2f}\nQ1: {quartiles[0]:.2f}\nQ3: {quartiles[1]:.2f}\nMean: {selected_admissions['count_icd9_codes'].mean():.2f}"
axs[1, 0].legend([legend_text], loc='upper left')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Example DataFrame

# Count the number of ICD-9 codes per admission
icd9_counts_a = df.groupby('HADM_ID')['ICD9_CODE_procedures'].count().reset_index(name='count_icd9_codes')

# If you have specific admission IDs you want to plot, filter them here
# For example, if you want to plot for admission IDs 'A1', 'A2', 'A3', 'A4', 'A5'
selected_admissions_a = icd9_counts_a

icd9_counts = df.groupby('SUBJECT_ID')['ICD9_CODE_procedures'].count().reset_index(name='count_icd9_codes')

# If you have specific admission IDs you want to plot, filter them here
# For example, if you want to plot for admission IDs 'A1', 'A2', 'A3', 'A4', 'A5'
selected_admissions = icd9_counts

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Boxplot of ICD-9 Codes Count per admission
boxplot_a = axs[0, 0].boxplot(selected_admissions_a['count_icd9_codes'])
axs[0, 0].set_title('Boxplot of ICD-9 Codes Count per admission')
axs[0, 0].set_ylabel('Count of ICD-9 Codes')

# Add labels to the boxplot
medians_a = [median.get_ydata()[0] for median in boxplot_a['medians']]
quartiles_a = [q.get_ydata()[1] for q in boxplot_a['whiskers']]
labels_a = [f'Median: {median:.2f}\n Q1: {q1:.2f}\n Q3: {q3:.2f}' for median, q1, q3 in zip(medians_a, quartiles_a[::2], quartiles_a[1::2])]
for label, median in zip(labels_a, medians_a):
    axs[0, 0].text(1, median, label, verticalalignment='top')

# Histogram of ICD-9 Codes Count per admission
axs[0, 1].hist(selected_admissions_a['count_icd9_codes'], bins=25, color='lightblue')
axs[0, 1].set_title('Histogram of ICD-9 Codes Count')
axs[0, 1].set_xlabel('Count of ICD-9 Codes per admission')
axs[0, 1].set_ylabel('Frequency')

# Boxplot of ICD-9 Codes Count per subject
boxplot = axs[1, 0].boxplot(selected_admissions['count_icd9_codes'])
axs[1, 0].set_title('Boxplot of ICD-9 Codes Count per subject')
axs[1, 0].set_ylabel('Count of ICD-9 Codes')

# Add labels to the boxplot
medians = [median.get_ydata()[0] for median in boxplot['medians']]
quartiles = [q.get_ydata()[1] for q in boxplot['whiskers']]
labels = [f'Median: {median:.2f}\n Q1: {q1:.2f}\n Q3: {q3:.2f}' for median, q1, q3 in zip(medians, quartiles[::2], quartiles[1::2])]
for label, median in zip(labels, medians):
    axs[1, 0].text(1, median, label, verticalalignment='top')

# Histogram of ICD-9 Codes Count per subject
axs[1, 1].hist(selected_admissions['count_icd9_codes'], bins=25, color='lightblue')
axs[1, 1].set_title('Histogram of ICD-9 Codes Count')
axs[1, 1].set_xlabel('Count of ICD-9 Codes per subject')
axs[1, 1].set_ylabel('Frequency')

# Add legend with median, quartiles, and mean
legend_text = f"Median: {medians[0]:.2f}\nQ1: {quartiles[0]:.2f}\nQ3: {quartiles[1]:.2f}\nMean: {selected_admissions['count_icd9_codes'].mean():.2f}"
plt.legend([legend_text], loc='upper left')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# %%


# %% [markdown]
# # Diagnosis

# %%
import pandas as pd

nuevo_df = pd.read_csv("data/diagnosis_preprocess_nonfiltered.csv")

# %%
import matplotlib.pyplot as plt
import seaborn as sns

result = nuevo_df.groupby(["'CCS CATEGORY DESCRIPTION'"]).size().reset_index(name='Count')
result = result.sort_values(by="Count", ascending=False)[:10]

plt.figure(figsize=(12, 8))
sns.barplot(x="'CCS CATEGORY DESCRIPTION'", y='Count', data=result, color='lightblue')
plt.xticks(rotation=90, fontsize=12)  # Increase the font size of x-axis labels
plt.xlabel('Diagnosis', fontsize=14)  # Increase the font size of x-axis label
plt.ylabel('Count', fontsize=14)  # Increase the font size of y-axis label
plt.title('Top 10 Most Frequent Diagnoses', fontsize=16)  # Increase the font size of the title

plt.show()

# %%
result_subject = nuevo_df.groupby("SUBJECT_ID").size().reset_index(name='Count')

# Count per admission ID
result_admission = nuevo_df.groupby("HADM_ID").size().reset_index(name='Count')

# %%
result_subject_1 = result_subject[result_subject["Count"]<100]
result_admission_1 = result_admission[result_admission["Count"]<100]

# %%


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
sns.set(style="darkgrid")

# Load the iris dataset
df = sns.load_dataset("iris")

# Create a figure with two matplotlib.Axes objects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Assigning a graph to each ax
sns.histplot(data=result_subject_1, x="Count", ax=ax1, color='darkblue',bins = 50)
sns.histplot(data=result_admission_1, x="Count", ax=ax2, color='lightblue',bins = 30)

# Set x-axis and y-axis labels for each subplot
ax1.set(xlabel='Count of ICD-9 codes per patient', ylabel='Frequency')
ax2.set(xlabel='Count of ICD-9 codes per admission', ylabel='Frequency')



ax1.text(-0.1, -0.2, '(a)', transform=ax1.transAxes, size=10, )
ax2.text(-0.1, -0.2, '(b)', transform=ax2.transAxes, size=10, )

fig.suptitle('Count of ICD-9 codes diagnoses', fontsize=14)  # Increase the font size of the title
# Show the plot
plt.show()

# %% [markdown]
# # Drugs

# %%
nuevo_df = pd.read_csv("data/drugs2_preprosss_non_preprocess.csv")



# %%
nuevo_df.columns

# %%

result = nuevo_df.groupby(["DRUG"]).size().reset_index(name='Count')
result = result.sort_values(by="Count", ascending=False)[:10]

plt.figure(figsize=(14, 9))
sns.barplot(x="DRUG", y='Count', data=result, color='lightblue')
plt.xticks(rotation=45, fontsize=14)  # Increase the font size of x-axis labels
plt.xlabel('Drug', fontsize=16)  # Increase the font size of x-axis label
plt.ylabel('Count', fontsize=16)  # Increase the font size of y-axis label
plt.title('Top 10 Most Frequent Diagnoses', fontsize=18)  # Increase the font size of the title

plt.show()

# %%
result_subject = nuevo_df.groupby("SUBJECT_ID").size().reset_index(name='Count')

# Count per admission ID
result_admission = nuevo_df.groupby("HADM_ID").size().reset_index(name='Count')

# %%
result_subject_1 = result_subject[result_subject["Count"]<300]
result_admission_1 = result_admission[result_admission["Count"]<150]

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
sns.set(style="darkgrid")

# Load the iris dataset
df = sns.load_dataset("iris")

# Create a figure with two matplotlib.Axes objects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Assigning a graph to each ax
sns.histplot(data=result_subject_1, x="Count", ax=ax1, color='darkblue',bins = 50)
sns.histplot(data=result_admission_1, x="Count", ax=ax2, color='lightblue',bins = 30)

# Set x-axis and y-axis labels for each subplot
ax1.set(xlabel='Count of drugs per patient', ylabel='Frequency')
ax2.set(xlabel='Count of drugs per admission', ylabel='Frequency')

ax1.text(-0.1, -0.2, '(a)', transform=ax1.transAxes, size=10, )
ax2.text(-0.1, -0.2, '(b)', transform=ax2.transAxes, size=10, )

fig.suptitle('Count of ICD-9 codes  drugs', fontsize=14)  # Increase the font size of the title
# Show the plot

plt.show()

# %%



