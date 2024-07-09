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

import evaluation.resemb.resemblance.config
#from generative_model.SD.constraints import *
from evaluation.resemb.resemblance.config import *

    # Save the statistics to a file


if __name__=="__main__":

    attributes = False
    if attributes:    
        doppleganger_data_synthetic(data, synthetic_data,  attributes,  path_o, attributes_path_train, features_path_train, features_path_valid, attributes_path_valid,synthetic_path_attributes,synthetic_path_features)
    
    #read_dict
    csv_files = glob.glob(path_to_directory + '.pkl')
    
       #create electorni 
    test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset,features  = load_create_ehr(read_ehr,save_ehr,file_path_dataset,sample_patients_path,file,valid_perc,features_path,name_file_ehr,type_file='ARFpkl')
    
       #constrains  
    train_ehr_dataset,synthetic_ehr_dataset,test_ehr_dataset = make_read_constraints( make_contrains,
                          save_constrains,
                          train_ehr_dataset,
                          test_ehr_dataset,
                          synthetic_ehr_dataset,
                          columns_to_drop,
                          columns_to_drop_syn,
                          type_archivo,
                          invert_normalize,
                          cols_continuous,
                          create_visit_rank_col,
                            propagate_fistvisit_categoricaldata,
                            adjust_age_and_dates_get,
                            get_remove_duplicates,
                            get_handle_hospital_expire_flag,
                            get_0_first_visit,
                            get_sample_synthetic_similar_real,
                            subject_continous,
                            
                            
                            create_days_between_visits_by_date_var,
                            eliminate_negatives_var,
                            file_path_dataset ,
                            make_read_constraints_name)
                
    #get_columns
    columnas_test_ehr_dataset,cols_categorical,cols_diagnosis,cols_procedures,cols_drugs,top_300_codes        = get_columns_codes_commun(train_ehr_dataset,keywords,categorical_cols)   
      
    #get cols diagnosis, procedures, drugs  
    #synthetic_ehr_dataset, train_ehr_dataset, test_ehr_dataset = cols_drop_ehr(columns_to_drop, synthetic_ehr_dataset, train_ehr_dataset, test_ehr_dataset)
     
     #lista de metricas
    list_metric_resemblance = [
        ##descriptive stadistics
        
        # "get_descriptive_statistics",             #diferencia alarmante en estadiscas descriptivas 
        # "compare_average_trends_recordlen",       #record lenght depende en numero de admissiones
        # #  "get_visit_value_counts",              #analisar numero de admissiones / comparar la estadisticas con media y std normal truncada*
        # "compare_descriptive_stadistics",
        
        # # #plots descriptive stadistics
        
        # "plot_means_continuos_variables",
        # "plot_first_visit",
        "plot_dimension_wise",
        "plot_prevalence_wise",
        
        # #top10 dif or least diff                   #identificar los codigos o variables mas distintas/ identificar que importancia tienen en selccion random fors
        
        # "get_top10_different_mean_difference",         #identificar la frecuencia por alrbol y hoja/ identidicar mas comumnes pro arbol y hoja
        # # "obtain_least_frequent",
        
        # #comparrison                               # notar que el modelo no truna la log normal por lo que es nomal tener valor negativos y mayor al maximo
        
        # "compare_maximum_range",                  #to what percentageis is greater than thereal max
        ## "get_proportion_demos",
        
        # ##scores/metrics
        
        "get_jensenshannon_dist",                   #score to obtain eta match duplicates. alsooo seee summerazed score, all of the score make a score with train/test as baseline
        "get_MaximumMeanDiscrepancy",
        #"get_common_proportions",
        "kolmogorov_smirnof_test_chissquare",
        
        # #exact rcords
        # # "get_excat_match",
        # # "get_dupplicates",
        # #distributions
        # "plot_kerneldis",
        
        # # ##outiers
        
        #"outliers_and_histograms_patients_admissions",    # Todo make a funciton to obtain a dataframe o wanseterin distance to it aumatically
        #"metric_outliers",
        
        # #multivariate  
        
        "plot_differen_correlation",
        # #temporality
        #"temporal_histogram_heatmap",
        # #"plot_acumulated",
        #"analyse_trajectories"
        # #dimensional 
        # ##"plot_pacmap",
        ]
    # iniit metrics
    metrics_e = EHRResemblanceMetrics(test_ehr_dataset, 
                                      train_ehr_dataset, 
                                      synthetic_ehr_dataset,
                                      columnas_test_ehr_dataset, 
                                      top_300_codes,
                                      list_metric_resemblance,
                                      cols_continuous, 
                                      keywords, 
                                      categorical_cols,
                                      dependant_fist_visit,
                                      cols_diagnosis,
                                      cols_procedures,
                                      cols_drugs,
                                      path_img,
                                      columns_to_drop,
                                      
                                    all_data_heatmap_diff=False,
                    corr_plot_syn_real=False,
                    corr_plot_continous = False,
                    corr_plot_categorical =True,
                    corr_plot_codes=True,
                    num_visit_count=num_visit_count,
                    patient_visit=num_visit_count)
    #evaluate
    results = metrics_e.evaluate()
    #combine_svgs(['dimensio_wise_2994.svg', 'dimensio_wise_2994.svg', 'dimensio_wise_2994.svg', 'dimensio_wise_2926.svg'], path_img)

    #print_latex(results)
