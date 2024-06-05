import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from scipy.stats import wasserstein_distance
#os.chdir("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning")
os.chdir("/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/")
import sys
sys.path.append('/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/')
from evaluation.resemb.resemblance.EvaluationResemblance import *
from evaluation.resemb.resemblance.utilsstats import *



    # Save the statistics to a file


if __name__=="__main__":
    import wandb
    import evaluation.resemb.resemblance.config
    from generative_model.SD.constraints import *
    attributes = False
    if attributes:    
        doppleganger_data_synthetic(data, synthetic_data,  attributes,  path_o, attributes_path_train, features_path_train, features_path_valid, attributes_path_valid,synthetic_path_attributes,synthetic_path_features)
    
    #read_dict
    csv_files = glob.glob(path_to_directory + '.pkl')
    
    #obtain dataset admission
    if read_ehr:
        test_ehr_dataset = load_pickle(file_path_dataset + 'test_ehr_dataset.pkl')
        train_ehr_dataset = load_pickle(file_path_dataset + 'train_ehr_dataset.pkl')
        synthetic_ehr_dataset = load_pickle(file_path_dataset + 'synthetic_ehr_dataset.pkl')
        features = load_pickle(file_path_dataset + 'features.pkl')
    else:    
        test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset,features = obtain_dataset_admission_visit_rank(file,file,valid_perc,features_path,'ARFpkl')
    if save_ehr:
        save_pickle(test_ehr_dataset, file_path_dataset + 'test_ehr_dataset.pkl')
        save_pickle(train_ehr_dataset, file_path_dataset + 'train_ehr_dataset.pkl')
        save_pickle(synthetic_ehr_dataset, file_path_dataset + 'synthetic_ehr_dataset.pkl')
        save_pickle(features, file_path_dataset + 'features.pkl')
        

    #constrains    
    
    c = EHRDataConstraints(train_ehr_dataset, test_ehr_dataset, synthetic_ehr_dataset)
    c.print_shapes()
    #cols_accounts = c.handle_categorical_data()
    synthetic_ehr_dataset = c.initiate_processing()
    c.print_shapes()
    #drop column between_cum sum made from constrains       
    synthetic_ehr_dataset = cols_todrop(synthetic_ehr_dataset,[cols_to_drop_syn])
  
    #   get sam num patient in train set as synthetic
    train_ehr_dataset = get_same_numpatient_as_synthetic(   train_ehr_dataset, synthetic_ehr_dataset)      
    print(test_ehr_dataset.shape)
    print(train_ehr_dataset.shape)
    print(synthetic_ehr_dataset.shape)
    print(train_ehr_dataset['id_patient'].nunique())
    print(synthetic_ehr_dataset['id_patient'].nunique())
    
              
     #get cols diagnosis, procedures, drugs   
    columnas_test_ehr_dataset = get_cols_diag_proc_drug(train_ehr_dataset)
    #obtener cols para demosgraphics, contnious, procedures, diagnosis, drugs
    cols_categorical,cols_diagnosis,cols_procedures,cols_drugs = cols_to_filter(     train_ehr_dataset,keywords,categorical_cols,)
   
    #obtener 300 codes
    top_300_codes = obtain_most_freuent(train_ehr_dataset,columnas_test_ehr_dataset,100)

    #drop columns not needed 
    print("cols to drop: " ,columns_to_drop) 
    if all(column in synthetic_ehr_dataset.columns for column in columns_to_drop):
        synthetic_ehr_dataset.drop(columns_to_drop, axis=1, inplace=True)
        train_ehr_dataset.drop(columns_to_drop, axis=1, inplace=True)
        test_ehr_dataset.drop(columns_to_drop, axis=1, inplace=True) 
        
    print(test_ehr_dataset.shape)
    print(train_ehr_dataset.shape)
    print(synthetic_ehr_dataset.shape)
    #lista de metricas
 
    list_metric_resemblance = [
            "outliers_and_histograms_patients_admissions",
            "metric_outliers",
            "get_descriptive_statistics",
            "compare_average_trends_recordlen",
            "plot_first_visit",
            "plot_means_continuos_variables",
            "plot_kerneldis",
            "get_top10_different_proportion",
            "get_visit_value_counts",
            "plot_differen_correlation",
            "plot_pacmap",
            "temporal_histogram_heatmap",
            "compare_maximum_range",
            "plot_dimension_wise",
            "plot_prevalence_wise",
            "top_10diference_absbalute",
            "get_MaximumMeanDiscrepancy",
            "kolmogorov_smirnof_test_chissquare",
            "get_jensenshannon_dist",
            "get_proportion_demos",
            "get_excat_match",
            "get_dupplicates",
            "obtain_least_frequent",
            "plot_acumulated",
            "get_common_proportions"
        ]

    metrics = EHRResemblanceMetrics(test_ehr_dataset, train_ehr_dataset, synthetic_ehr_dataset, columnas_test_ehr_dataset, top_300_codes, list_metric_resemblance, cols_continuous, keywords, categorical_cols, dependant_fist_visit)

    