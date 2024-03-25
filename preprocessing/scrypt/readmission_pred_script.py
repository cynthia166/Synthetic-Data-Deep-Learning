from preprocessing.function_pred import *
import pandas as pd
import csv
import json

def main(config_path):
    
    with open("input_json/"+config_path, 'r') as f:
        config = json.load(f)

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

    for days in days_list:
         make_preds(ejemplo_dir, path, days, ficheros, kfolds, type_reg, prepro,
                   archivo_input_label, nom_t, model, sampling, li_feature_selection,
                    lw, K,models_config)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', help='Path to the configuration JSON file',default='config_pred_drugs1.json')
    args = parser.parse_args()

    main(args.config_path)