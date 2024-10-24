import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from scipy.stats import wasserstein_distance
#os.chdir("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning")
file_principal = os.getcwd()
os.chdir(file_principal)
import sys
sys.path.append(file_principal)
from evaluation.resemb.resemblance.EvaluationResemblance import *
from evaluation.resemb.resemblance.utilsstats import *
#import wandb
import evaluation.resemb.resemblance.config
from generative_model.SD.constraints import *
from evaluation.resemb.resemblance.config import *
#from generative_model.SD.model.coupling_subject import *
    # Save the statistics to a file


if __name__=="__main__":

       #create electorni 
    test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset,features  = load_create_ehr(read_ehr,save_ehr,file_path_dataset,sample_patients_path,file,valid_perc,features_path,name_file_ehr,type_file=type_archivo)
    print(file)



    if exclude_codes:
        test_ehr_dataset= test_ehr_dataset[[i for i in test_ehr_dataset.columns if i not in list_col_exclute['codes_no'].to_list()]]
        train_ehr_dataset= train_ehr_dataset[[i for i in train_ehr_dataset.columns if i not in list_col_exclute['codes_no'].to_list()]]
        synthetic_ehr_dataset= synthetic_ehr_dataset[[i for i in synthetic_ehr_dataset.columns if i not in list_col_exclute['codes_no'].to_list()]]
    
    # the only thing that is done is subject_id is change to patient id and logging and clipping negstive values
    
       #constrains  
         
    train_ehr_dataset,synthetic_ehr_dataset,test_ehr_dataset = make_read_constraints( make_contrains,
                          save_constrains,
                          train_ehr_dataset,
                          test_ehr_dataset,
                          synthetic_ehr_dataset,
                          columns_to_drop,
                          columns_to_drop_syn,
                          type_archivo,
                   
                          cols_continuous,
                           create_visit_rank_col,
                            propagate_fistvisit_categoricaldata,
                            adjust_age_and_dates_get,
                            get_remove_duplicates,
                            get_handle_hospital_expire_flag,
                            get_0_first_visit,
                            get_sample_synthetic_similar_real,
         
                            create_days_between_visits_by_date_var,
                            eliminate_negatives_var,
                            get_days_grom_visit_histogram,
                            get_admitted_time,
                            get_synthetic_subject_clustering,
                            file_path_dataset ,
                            make_read_constraints_name,
                            medication_columns = medication_columns,
                            columns_to_drop_sec = columns_to_drop_sec,
                            encoder =encoder,
                            columnas_demograficas=columnas_demograficas,
                            synthetic_type=synthetic_type)
   
        #get_columns
    columnas_test_ehr_dataset,cols_categorical,cols_diagnosis,cols_procedures,cols_drugs,top_300_codes        = get_columns_codes_commun(train_ehr_dataset,keywords,categorical_cols)   
      
    #get cols diagnosis, procedures, drugs v 
    #synthetic_ehr_dataset, train_ehr_dataset, test_ehr_dataset = cols_drop_ehr(columns_to_drop, synthetic_ehr_dataset, train_ehr_dataset, test_ehr_dataset)
    #datasets para comparar
    d_synthetic_data = {} 
    synthetic_ehr_dataset_same_col ,train_ehr_dataset = common_cols(synthetic_ehr_dataset,train_ehr_dataset)  
    d_synthetic_data["Min_node_15"] = synthetic_ehr_dataset_same_col
    d_synthetic_data["Min_node_15_"] = synthetic_ehr_dataset_same_col
 
     #test_bed
    list_metric_resemblance = [] 
    if visualization_dimension_wise_distribution_similarity:
        list_metric_resemblance.extend([
            #"plot_kerneldis",
            #"plot_dimension_wise",
            "plot_prevalence_wise",
            #"demographics_analysis_medicalcodes",
           # "age_occurance_stat",
        #"plot_genereate_histogramplot_total_unique_stats",
                                   
                                 
        ])
    if metric_dimension_wise_distribution_similarity:
        list_metric_resemblance .extend([
            # "kolmogorov_smirnof_test_chissquare",
            # "get_descriptive_statistics",
            #  "dimenssion_bernoulli",
            #   "get_proportion_demos",
           #  "compare_average_trends_recordlen",
           # "outliers_and_histograms_patients_admissions",
           # "get_patient_stats",
           #"get_analyze_one_hot_encoding_var",
           #"get_get_analyze_continous",
           # "get_domain",
           #"get_proportion_visits"
             ])
    if metric_joint_distribution_similarity_coverage:
        list_metric_resemblance .extend([
            #"crammer_metric",
             "get_evaluate_synthetic_data_wassetein_overall",
            # "get_MaximumMeanDiscrepancy",
           
        ])
    if metric_joint_distribution_similarity_structure:    
        list_metric_resemblance.extend([
           "get_progression_quantity_medicamentpe_visit",
            #"analyse_trajectories",
            "temporal_histogram_heatmap",
           # "plot_pacmap". I
        ])

    if  metric_inter_dimensional_similarity:
        
        list_metric_resemblance .extend([
            #"pairwisecorrelation",
            "plot_differen_correlation",
            #"metric_outliers"
            ])
        
    if consistency_information:  
        list_metric_resemblance .extend([
            "get_excat_match",
            "get_dupplicates",
            "get_common_proportions"
        ]       )
    if other_metrics:
        list_metric_resemblance .extend([
            "plot_means_continuos_variables",
            "plot_first_visit",
            "get_visit_value_counts",
            "compare_descriptive_stadistics",
            "get_top10_different_mean_difference", 
            "obtain_least_frequent", 
            "compare_maximum_range",  
            "get_jensenshannon_dist",
            "kolmogorov_smirnof_test_chissquare"
            
           
        ]    )
               

    #wandb.init(project="Resemblance_metrics", name="DEmographics_runs")
    metrics_e = EHRResemblanceMetrics(
        test_ehr_dataset, 
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
        corr_plot_continous = True,
        corr_plot_categorical =True,
        corr_plot_codes=True,
        num_visit_count=num_visit_count,
        patient_visit=num_visit_count,
        d_synthetic_data = d_synthetic_data,
        eliminate_variables_generadas_post=eliminate_variables_generadas_post,
        variables_generadas_post=variables_generadas_post,
        columnas_demograficas=columnas_demograficas,
        synthetic_type=synthetic_type,
        encoder =encoder)
    results = metrics_e.evaluate()
   
   # for i in results:
   #     log_results_to_wandb(i)
  
    # If you have any plots or figures, you can log them like this:
    # wandb.log({"plot_name": wandb.Image(fig)})

    # Finish the wandb run
    #wandb.finish()
    
    
    #print_latex(results)
    
    if attributes:    
        doppleganger_data_synthetic(data, synthetic_data,  attributes,  path_o, attributes_path_train, features_path_train, features_path_valid, attributes_path_valid,synthetic_path_attributes,synthetic_path_features)
