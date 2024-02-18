from function_pred import *
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='input_json/config_pred_drugs1.json', help='Path to the configuration JSON file')
    parser.add_argument('--n_estimators', type=int, help='Number of trees in the forest', required=False)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', required=False)
    parser.add_argument('--max_depth', type=int, help='Maximum depth of the trees', required=False)
    parser.add_argument('--penalty', type=str, help='The penalty (l1 or l2)', required=False)
    parser.add_argument('--C', type=float, help='Inverse of regularization strength', required=False)
    parser.add_argument('--model_type', type=str, help='Type of model to train (XGBClassifier or LogisticRegression)', required=False)
    args = parser.parse_args()

    # Inicializa WandB
    wandb.init(project="Predic_Readmission", config=args)
    
    # Puedes combinar aquí la configuración cargada desde el archivo JSON si es necesario
    json_config = load_json_config(args.config_path) if args.config_path else {}
    config = {**json_config, **vars(args)}
    
    wandb.config.update(config)

    config = wandb.config

    nom_t = config["nom_t"]
    ejemplo_dir = config["ejemplo_dir"]
    archivo_input_label = config["archivo_input_label"]
    path = config["path"]
    days_list = config["days_list"]
    ficheros = read_director(ejemplo_dir)
    ficheros = [i for i in ficheros if i != "sin_codigo.csv"]
    type_reg = config["type_reg"]
    
    # Instantiate the model based on the string in the JSON
    models_config = config["model"]
    model = initialize_models(models_config)
 
    sampling = config["sampling"]
    li_feature_selection = config["li_feature_selection"]
    kfolds = config["kfolds"]
    lw = config["lw"]
    K = config["K"]
    #list_cat = config["list_cat"]
    prepro = config["prepro"]

    
    #     make_preds(ejemplo_dir, path, days, ficheros, kfolds, type_reg, prepro,
    #               archivo_input_label, nom_t, model, sampling, li_feature_selection,
    #                lw, K,models_config,config)
    days = "30"
    fichero_y ="label_"+days+"j.csv"
    readmit_df = label_fun(days,archivo_input_label)

    # Se obtiene dataframe que sera el output del perfomance del entrenamiento
    df_res_aux = pd.DataFrame(columns=[ ])
    j = 0
    # se obtiene el respectivo preprocesing de acuerdo al experimento que se realizo
    for i in tqdm(ficheros):
        #concat_var_ = create_var(ejemplo_dir,i,readmit_df)
        #eda_embedding(path,i,concat_var_,i)
        
        print(i)

        prepo = prepro[j]
        print(prepo)
        
        X,y ,concat_var  = lectura_variables(readmit_df,i,fichero_y,prepo,ejemplo_dir,days)
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
       
        if config.model_type == "XGBClassifier":
            model = XGBClassifier(
               # tree_method='gpu_hist',
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                max_depth=config.max_depth
            )
        elif config.model_type == "LogisticRegression":
            model = LogisticRegression(
                penalty=config.penalty,
                C=config.C,
                solver='saga'
            )
        
            
                
        

        result = {    'f1_test':0,
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
            'time_model':0

        } 

        model_name = i.__class__.__name__
            
        j1 =sampling[0]
        k1 =li_feature_selection[0]
                

        timi_ini =time.time()
        #model = LogisticRegression(penalty='l1', solver='saga')
                    
        rf_sen,rf_spe,rf_prec,rf_acc,mean_auc_,rf_conf,rf_sen_t,rf_spe_t,rf_prec_t,rf_acc_t,f1,f1_t ,var_moin,var_ini= function_models(model, sampling,k1,kfolds,lw,K,type_reg,X,y)
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
        result['Mapping']=i
        result["var_change"]=var_moin
        result["var_ini"]=var_ini
        result["time_model"]=time_model

        wandb.log(result)
        #df_res = pd.DataFrame(result)
        #df_res_aux = pd.concat([df_res_aux,df_res])
        j += 1                    
                    
    # Close the WandB run
            
                    
        
  

        #concatenación de dataframes 

# se guarda dataframes    
    #df_res_aux.to_csv("./results_pred/results_prediction_"+days+"+non_filtered"+nom_t+".csv")   


def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)         


if __name__ == "__main__":

    # Añadir argumentos para cada hiperparámetro
      # Analizar los argumentos de la línea de comandos

    main()