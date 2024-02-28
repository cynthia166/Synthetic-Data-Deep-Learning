from random import randint
from function_pred import *
import pandas as pd
import csv
import json
import yaml

import argparse
import json
import wandb







def train(json_config, readmit_df,fichero,i,project_name):
    # Inicializa WandB
      # Puedes combinar aquí la configuración cargada desde el archivo JSON si es necesario
 
    nom_t = json_config["nom_t"]
    ejemplo_dir = json_config["ejemplo_dir"]
    
    path = json_config["path"]
    days_list = json_config["days_list"]
    ficheros = read_director(ejemplo_dir)

    type_reg = json_config["type_reg"]
    
    # Instantiate the model based on the string in the JSON
    #models_config = config["model"]
    #model = initialize_models(models_config)
 
  
      #list_cat = config["list_cat"]
    prepro = json_config["prepro"]
    #LogisticRegression,"Xgboost"   
    model = json_config["model"]
    splits = json_config["splits"]
    
    #     make_preds(ejemplo_dir, path, days, ficheros, kfolds, type_reg, prepro,
    #               archivo_input_label, nom_t, model, sampling, li_feature_selection,
    #                lw, K,models_config,config)
    days = "30"
    #readmit_df = label_fun(days,archivo_input_label)
    #archivo_input_label = config["archivo_input_label"]
    fichero_y ="label_"+days+"j.csv"
    
    # Se obtiene dataframe que sera el output del perfomance del entrenamiento
    j = i
    # se obtiene el respectivo preprocesing de acuerdo al experimento que se realizo
    
        #concat_var_ = create_var(ejemplo_dir,i,readmit_df)
    #eda_embedding(path,i,concat_var_,i)
    
    print(i)

    prepo = prepro[i]
    
    print(prepo)
    
    X,y ,concat_var  = lectura_variables(readmit_df,fichero,prepo,ejemplo_dir,days)
    try:
        X = X.values
        
    except:
        pass   
    try:
        y = y[days +"_READMIT"].to_numpy()
    except:
        y = y
    ####ENtrenamiento del modelo#####
    # funcion de entrenamiento dem odelo
    #df_res = modelo_df_aux_grid_search(X,y,i,type_reg,model,sampling,li_feature_selection,kfolds,lw,K,models_config,config)
    
    
        
            
    

    result = {   'f1_test':0,
        'f1_train':0,

        'sensitivity_test':0,
        'specificity_test':0,
        'precision_test':0,
        'accuracy_test':0,
        'sensitivity_train':0,
        'specificity_train':0,
        'precision_train':0,
        'accuracy_train':0, 
        'confusion matrix':0,
        'auc_train':0,
        'auc_test':0,
        'Classifiers':0,
        'Mapping':0,
        'mean_test_scores_folds':0,
        'mean_train_scores_folds':0,
        'time_model':0,
        "prepo":0,

    } 

        
             

    timi_ini =time.time()
    #model = LogisticRegression(penalty='l1', solver='saga')
    
    rf_sen,rf_spe,rf_prec,rf_acc,auc_train,auc_test,rf_conf,rf_sen_t,rf_spe_t,rf_prec_t,rf_acc_t,f1,f1_t,    mean_test_scores_folds ,mean_train_scores_folds= function_models2(X,y,model,splits)
   
    
    time_model = timi_ini-time.time()  
    result['sensitivity_test']=rf_sen
    result['specificity_test']=rf_spe
    result['precision_test']=rf_prec
    result['accuracy_test']=rf_acc
    result['sensitivity_train']=rf_sen_t
    result['specificity_train']=rf_spe_t
    result['precision_train']=rf_prec_t
    result['accuracy_train']=rf_acc_t
    result['f1_test']=f1
    result['f1_train']=f1_t
    result['confusion matrix']=rf_conf
    result['auc_train']=auc_train
    result['auc_test']=auc_test
    result['Mapping']=fichero
    result["mean_test_scores_folds"]=mean_test_scores_folds
    result["mean_train_scores_folds"]=mean_train_scores_folds
    result["time_model"]=time_model
    result["prepo"]=str(prepo)
    #result["fichero"]=i
    return result
    
    #df_res = pd.DataFrame(result)
    #df_res_aux = pd.concat([df_res_aux,df_res])
                     
                
# Close the WandB run
        
                
    
    


    #concatenación de dataframes 

# se guarda dataframes    
#df_res_aux.to_csv("./results_pred/results_prediction_"+days+"+non_filtered"+nom_t+".csv")   

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import loguniform


def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)         

def function_models2(X,y,model,splits):
   
    train_size = int(len(X) * 0.75)  # 75% for training
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    #tscv = TimeSeriesSplit(n_splits=splits)
    tscv = KFold(n_splits=splits, shuffle=False)
    if model == "Xgboost":
        model = XGBClassifier()
        param_grid = {
        'criterion': ['f1_t'],  # Asumiendo que tu modelo soporta criterios personalizados
        'learning_rate': [0.00000001, 0.0001, 0.01, 0.1, 1],  # Ejemplo de tasas de aprendizaje
        'max_delta_step': [1, 2, 3, 4, 5],  # Entero entre 0 y 10
        'max_depth': [10, 20, 30, 40, 50],  # Entero entre 1 y 50
        'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # Entero entre 0 y 15
        'n_estimators': [100, 500, 1000, 5000, 10000],  # Ejemplo de números de estimadores
        'reg_alpha': [0.1, 0.5, 1],  # Ejemplo de términos de regularización Alpha
        'reg_lambda': [0.1, 0.75, 1.5],  # Ejemplo de términos de regularización Lambda
        'scale_pos_weight': [0.1, 0.5, 1],  # Ejemplo de balance de pesos
        'subsample': [0.1, 0.5, 1],  # Ejemplo de ratio de submuestreo
        }
    elif model == "LogisticRegression":
        param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Type of regularization to be applied
        'C': np.logspace(-4, 4, 20),  # Inverse of regularization strength
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Algorithm to use in the optimization problem
        'max_iter': [100, 1000, 2500, 5000]  # Maximum number of iterations for the solvers to converge
        }    
    grid_search = RandomizedSearchCV(model, param_grid, cv=tscv,scoring="f1",return_train_score=True)
    grid_search.fit(X_train, y_train)
                                                                    
    y_pred_train = grid_search.predict(X_train)
    y_pred = grid_search.predict(X_test)
    mean_test_scores_folds = grid_search.cv_results_['mean_test_score']

# Get the mean train score for each parameter combination
    mean_train_scores_folds = grid_search.cv_results_['mean_train_score']
    try:
        #it obtain metric considered and confussion matrix, metrics for the test set
        rf_conf = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = rf_conf.ravel()
        rf_sen = tp/(tp+fn)
        rf_spe = tn/(tn+fp)
        rf_prec = tp/(tp+fp)
        rf_acc = (tp+tn)/(tp+tn+fp+fn)
        f1 = f1_score(y_test, y_pred, average='macro')
        #metrics for the training set
        rf_conf_t = confusion_matrix(y_train, y_pred_train)
        tn_t, fp_t, fn_t, tp_t = rf_conf_t.ravel()
        rf_sen_t = tp_t/(tp_t+fn_t)
        rf_spe_t = tn_t/(tn_t+fp_t)
        rf_prec_t = tp_t/(tp_t+fp_t)
        rf_acc_t = (tp_t+tn_t)/(tp_t+tn_t+fp_t+fn_t)
        f1_t = f1_score(y_train, y_pred_train, average='macro')
        auc_test = roc_auc_score(y_test, y_pred)
        auc_train = roc_auc_score(y_train, y_pred_train)
    except:
        rf_conf = 0
        tn, fp, fn, tp = 0,0,0,0
        rf_sen = 0
        rf_spe = 0
        rf_prec = 0
        rf_acc = 0
        f1 = 0
        
        rf_conf_t = 0
        tn_t, fp_t, fn_t, tp_t = 0,0,0,0
        rf_sen_t = 0
        rf_spe_t = 0
        rf_prec_t = 0
        rf_acc_t = 0
        f1_t = 0
        auc_train = 0
        auc_test = 0


    
    return rf_sen,rf_spe,rf_prec,rf_acc,auc_train,auc_test,rf_conf,rf_sen_t,rf_spe_t,rf_prec_t,rf_acc_t,f1,f1_t,    mean_test_scores_folds ,mean_train_scores_folds



    

    




    
 
def main(json_config, readmit_df,fichero,i,project_name):
    wandb.init(project=project_name)
    result = train(json_config, readmit_df,fichero,i,project_name)
    wandb.log(result)
    
    
if __name__ == "__main__":
    
    project_name =   "Predic_Readmission_diagnosis_XGboost_kfolds_preproC"
    # PARAMETRO NO FIJO#######
    arconfig_path = "input_json/config_diagnosis_pred2.json"
    def load_json_config(config_path):
        with open(config_path, 'r') as file:
            return json.load(file)
      # Analizar los argumentos de la línea de comandos
    #sweep_id = wandb.sweep(sweep_config, project="your_project_name")
    json_config = load_json_config(arconfig_path)
    # Run the sweep
    # PARAMETRO NO FIJO#######
    ejemplo_dir ="./input_model_pred_diagnosis_u/"
    ficheros = read_director(ejemplo_dir)
    # PARAMETRO FIJO#######
    archivo_input_label = "data_preprocess_nonfilteres.csv"
    days = "30"
    readmit_df = label_fun(days,archivo_input_label)
    
    
    
    for i,fichero in enumerate(ficheros):
        print(i)
    # This lambda function will be called for each set of parameters
        main(json_config, readmit_df,fichero,i,project_name)
          
    wandb.finish()

    
    
        
            
    




