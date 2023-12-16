from function_pred import *
import pandas as pd
import csv



def main():
    days = '30'
    ejemplo_dir = './input_model_pred/'
    archivo_input_label = 'data_preprocess_non_filtered.csv'
    
    
    path = "./input_model_pred/" +"images"
    days_list = ["180"]
    ficheros = read_director(ejemplo_dir)
    fichero = ficheros[2:]
    kfolds = 5
    type_reg = "lasso"
    prepro = "std"
    prepo_li=["max","max","power","power","std","max","power"]
    list_cat = ["ICD9_CODE_procedures",'CCS CODES_proc', 'cat_threshold .95 most frequent_proc','cat_threshold .88 most frequent', 'cat_threshold .98 most frequent',
        'cat_threshold .999 most frequent']


    for days in days_list:
        make_preds(ejemplo_dir,path,days,ficheros,kfolds,type_reg,prepro,archivo_input_label)

    
    
    
    
if __name__ == "__main__":
    main()     
        