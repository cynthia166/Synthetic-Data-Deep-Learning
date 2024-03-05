from random import randint, uniform
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
 
  
      #list_cat = config["list_cat"f]
    prepro = json_config[f"prepro"]
    #LogisticRegression,"Xgboost"   
    #model = json_config["model"]
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

        'penalty':0,
        'C':0,

        'solver':0,
        'best_n_estimax_itermators':0,
        "l1_ratio":0,
         
 }
     

        
             

    timi_ini =time.time()
    #model = LogisticRegression(penalty='l1', solver='saga')
    
    rf_sen,rf_spe,rf_prec,rf_acc,auc_train,auc_test,rf_conf,rf_sen_t,rf_spe_t,rf_prec_t,rf_acc_t,f1,f1_t,    mean_test_scores_folds ,mean_train_scores_folds, penalty,C,solver,best_n_estimax_itermators,l1_ratio= function_models2(X,y,model,splits)
   
    
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
    
    result['penalty']=penalty
    result['C']=C
    
    result["solver"]=solver
    result["best_n_estimax_itermators"]=best_n_estimax_itermators
    result["l1_ratio"]=str(l1_ratio)

        #result["fichero"]=i
    wandb.init(project=project_name,name =f"experiment_{fichero}" ,config = param_grid )

    wandb.log(result)
    wandb.finish()    
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
    
    #train_size = int(len(X) * 0.75)  # 75% for training
    #val_size = int(len(X) * 0.15)  # 15% for validation
    #X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    #y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    #tscv = TimeSeriesSplit(n_splits=splits)
    tscv = KFold(n_splits=splits, shuffle=False)

    '''if model != LogisticRegression():
        grid_search = RandomizedSearchCV(model, param_grid, cv=tscv,scoring="f1",return_train_score=True)
        grid_search.fit(X_train, y_train,eval_set=[(X_val, y_val)], early_stopping_rounds=4)
    else:'''
    grid_search = RandomizedSearchCV(model, param_grid, cv=tscv,scoring="f1",return_train_score=True)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

# Accede a los valores en el diccionario 'best_params'
    penalty = best_params['penalty']
    C = best_params['C']
  
    solver = best_params['solver']
    best_n_estimax_itermators = best_params['max_iter']
    l1_ratio = best_params['l1_ratio']





                                                                    
    y_pred_train = grid_search.predict(X_train)
    y_pred = grid_search.predict(X_test)
    mean_test_scores_folds = grid_search.cv_results_['mean_test_score']

# Get the mean train score for each parameter combination
    mean_train_scores_folds = grid_search.cv_results_['mean_train_score']
    results = grid_search.cv_results_

# Registrar los resultados en WandB
  
    for i in range(len(results['params'])):
        

        wandb.init(project=project_name,name =f"experiment_{fichero}" ,config = param_grid  )
 
        wandb.log({
            'mean_test_score': results['mean_test_score'][i],
            'mean_train_score': results['mean_train_score'][i],
            'params': results['params'][i]
        })
        
        wandb.finish()


    
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


    
    return rf_sen,rf_spe,rf_prec,rf_acc,auc_train,auc_test,rf_conf,rf_sen_t,rf_spe_t,rf_prec_t,rf_acc_t,f1,f1_t,    mean_test_scores_folds ,mean_train_scores_folds,  penalty,C,solver,best_n_estimax_itermators,l1_ratio


    

    




    
 
def main(json_config, readmit_df,fichero,i,project_name):

    result = train(json_config, readmit_df,fichero,i,project_name)
      
    
if __name__ == "__main__":
    global days,param_grid,model
    project_name =   "Predic_Readmission_drugs_LR_kfolds_preproC"
    # PARAMETRO NO FIJO#######
    arconfig_path = "input_json/config_drugsLR.json"
    def load_json_config(config_path):
        with open(config_path, 'r') as file:
            return json.load(file)
      # Analizar los argumentos de la línea de comandos
    #sweep_id = wandb.sweep(sweep_config, project="your_project_name")
    json_config = load_json_config(arconfig_path)
    # Run the sweep
    # PARAMETRO NO FIJO#######
    ejemplo_dir ="./input_model_pred_drugs_u/"
    model  = "LogisticRegression"
    print(model)
    if model == "Xgboost":
        model = XGBClassifier()
        '''param_grid = {
            'criterion': ['f1_t'],  # Asumiendo que tu modelo soporta criterios personalizados
            'learning_rate': [0.1,0.8,10],  # Reducido a un solo valor
             # Reducido a un solo valor
            'max_depth': [10, 20,50], 
            'reg_alpha': [0.1,0.5],  # Coeficiente de regularización L1
            'reg_lambda': [1.0],# Reducido a dos valores
            'n_estimators': [100, 1000, 2500, 5000],  # Número de árboles a construir
            
         # Número de rondas sin mejora después de las cuales se detendrá el entrenamiento
    
        }'''
        param_grid = {
        # Para 'learning_rate', una lista de valores posibles en una escala logarítmica desde 10^-8 a 10^0
        'learning_rate': [1e-8, 1e-5,   1e-1, 1e-0],
        
        # Para 'max_delta_step', una lista de valores enteros desde 0 a 10
        'max_delta_step': list(range(0, 10)),
        
        # Para 'max_depth', una lista de valores enteros desde 1 a 30
        'max_depth': list(range(1, 31)),
        
        # Para 'min_child_weight', una lista de valores enteros desde 0 a 15
        'min_child_weight': list(range(0, 16)),
        
        # Para 'n_estimators', una lista de valores enteros representativos desde 1 a 10000
        # Dada la amplia gama, se eligen valores representativos
        'n_estimators': [1, 10, 100, 1000, 5000, 10000],
        
        # Para 'reg_alpha' (alpha region), una lista de valores desde 0.1 a 1 en pasos definidos
        'reg_alpha': [0.1, 0.5, 1],
        
        # Para 'reg_lambda' (lambda region), una lista de valores desde 0.1 a 1.5 en pasos definidos
        'reg_lambda':[1,0.75],
        
        # Para 'scale_pos_weight', una lista de valores desde 0.1 a 1 en pasos definidos
        'scale_pos_weight': [1,3.8,2,5,7,10],
        
        # Para 'subsample', una lista de valores desde 0.1 a 1 en pasos definidos
        'subsample': [0.1, 0.5, 1]
         }

    elif model == "LogisticRegression":
            model = LogisticRegression()
            '''param_grid = {	
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Type of regularization to be applied
            'C': np.logspace(-4, 4, 20),  # Inverse of regularization strength
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Algorithm to use in the optimization problem
            'max_iter': [100, 1000, 2500, 5000]  # Maximum number of iterations for the solvers to converge
            } '''   
            param_grid = {
            'penalty': ['l2', 'elasticnet'],
            'C': np.logspace(-2, 2, 5),
            'solver': ['newton-cg', 'saga'],
            'max_iter': [100, 1000],
            'l1_ratio': np.linspace(0, 1, 5)
            }
            #'penalty': ['l1', 'l2', 'elasticnet'],  # Agregar 'l1' a la lista de penalidades
            #'C': np.concatenate((np.logspace(-2, 4, 7), [90])),  # Expande el rango de 'C' e incluye 90
            #'solver': ['newton-cg', 'saga'],  # 'saga' es compatible con todas las penalidades
            #'max_iter': [100, 1000, 2000],  # Especifica los valores para 'max_iter'
            #'l1_ratio': np.linspace(0, 1, 10)  # Hace el rango de 'l1_ratio' más detallado
            
    # PARAMETRO NO FIJO#######     
    ficheros = read_director(ejemplo_dir)
    # PARAMETRO FIJO#######
    archivo_input_label = "data_preprocess_nonfilteres.csv"
    days = "30"
    readmit_df = label_fun(days,archivo_input_label)
    
    
    
    for i,fichero in enumerate(ficheros):
        print(i)
        print(fichero)
    # This lambda function will be called for each set of parameters
        main(json_config, readmit_df,fichero,i,project_name)
          
   

    
    
        
            
    




