from Preprocessing.function_pred import *
import pandas as pd
import csv
import json
import yaml

import argparse
import json
import wandb



def load_json_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)



def train(config):
    # Inicializa WandB
      # Puedes combinar aquí la configuración cargada desde el archivo JSON si es necesario
 
    nom_t = json_config["nom_t"]
    ejemplo_dir = json_config["ejemplo_dir"]
    
    path = json_config["path"]
    days_list = json_config["days_list"]
    ficheros = read_director(ejemplo_dir)
    ficheros = [i for i in ficheros if i != "sin_codigo.csv"]
    type_reg = json_config["type_reg"]
    
    # Instantiate the model based on the string in the JSON
    #models_config = config["model"]
    #model = initialize_models(models_config)
 
    sampling = json_config["sampling"]
    li_feature_selection = json_config["li_feature_selection"]
    kfolds = json_config["kfolds"]
    lw = json_config["lw"]
    K = json_config["K"]
    #list_cat = config["list_cat"]
    prepro = json_config["prepro"]

    
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
    
    if config.model_type == 'XGBClassifier':
            model = XGBClassifier(
                
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                max_depth=config.max_depth,
                
                
            )
    elif config.model_type == 'LogisticRegression':
        model = LogisticRegression(
            penalty=config.penalty,
            C=config.C,
            solver='saga',
           
        )
    else:
        raise ValueError("Unsupported model type")

        
            
    

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
        'Sampling':0,
        'Feature selection':0,
        'Classifiers':0,
        'Mapping':0,
        'var_change':0,
        'var_ini':0,
        'time_model':0,
        "prepo":0,

    } 

    model_name = i.__class__.__name__
        
    j1 =sampling[0]
    k1 =li_feature_selection[0]
            

    timi_ini =time.time()
    #model = LogisticRegression(penalty='l1', solver='saga')
                
    rf_sen,rf_spe,rf_prec,rf_acc,mean_auc_,rf_conf,rf_sen_t,rf_spe_t,rf_prec_t,rf_acc_t,f1,f1_t ,var_moin,var_ini= function_models2(model, sampling,k1,kfolds,lw,K,type_reg,X,y)
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
    result['Sampling']=j1
    result['Feature selection']=k1
    result['Classifiers']=model_name
    result['Mapping']=fichero
    result["var_change"]=var_moin
    result["var_ini"]=var_ini
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



def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)         

def function_models2():
    y_true = []
    y_pred = []
    y_true_train   =[]
    y_pred_train =[]
    roc_aucs_xgb1 = []

    
 
def main():
    wandb.init(project=project_name)
    result = train(wandb.config)
    wandb.log(result)
    
    
if __name__ == "__main__":
    
    global json_config, readmit_df,fichero,i,project_name
    project_name =   "Predic_Readmission_diagnosis"
    # PARAMETRO NO FIJO#######
    arconfig_path = "input_json/config_diagnosis_pred.json"
    json_config = load_json_config(arconfig_path)
    # Añadir argumentos para cada hiperparámetro
      # Analizar los argumentos de la línea de comandos
    #sweep_id = wandb.sweep(sweep_config, project="your_project_name")
    
    # Run the sweep
    # PARAMETRO NO FIJO#######
    ejemplo_dir ="./input_model_pred_diagnosis_u/"
    ficheros = read_director(ejemplo_dir)
    # PARAMETRO FIJO#######
    archivo_input_label = "data_preprocess_nonfilteres.csv"
    days = "30"
    readmit_df = label_fun(days,archivo_input_label)
    
    
    #fichero = ficheros[i]
    sweep_configuration = {
        "program": "readmission_pred_script_wandb.py",
        "method": "random",
        "metric": {
            "name": "f1_test",
            "goal": "maximize"
        },
        "parameters": {
            "n_estimators": {
                "values": [100, 200, 300]
            },
            "learning_rate": {
                "values": [0.01, 0.1, 0.2]
            },
            "max_depth": {
                "values": [3, 6]
            },
            "penalty": {
                "values": ["l1", "l2"]
            },
            "C": {
                "values": [0.1, 1.0, 10.0]
            },
            "model_type": {
                "values": ["XGBClassifier", "LogisticRegression"]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_configuration, project=project_name)
    i = 0
    for i,fichero in enumerate(ficheros):
        print(i)
    # This lambda function will be called for each set of parameters
        wandb.agent(sweep_id,  main,count=10) 
          
    wandb.finish()

    