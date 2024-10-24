import os
import matplotlib.pyplot as plt
import numpy as np
#from pacmap import PaCMAP
import logging
from scipy.stats import wasserstein_distance
import random
from sklearn import preprocessing
from evaluation_framework import EvaluationFramework
#os.chdir("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning")
file_principal = os.getcwd()
os.chdir(file_principal )
import sys    
sys.path.append('evaluation/resemb/resemblance/utils_stats/')
sys.path.append('evaluation')
from  preprocessing.config import diagnosis_columns,procedure_columns,medication_columns,set_graph_settings

sys.path.append(file_principal )
from evaluation.resemb.resemblance.utilsstats import *

ruta_actual = os.getcwd()
print(ruta_actual)

set_graph_settings()

class EHRResemblanceMetrics:
    def __init__(self, test_ehr_dataset, 
                 train_ehr_dataset, 
                 synthetic_ehr_dataset, 
                 columnas_test_ehr_dataset, 
                 top_300_codes, 
               
                 list_metric_resemblance,
                 cols,
                 keywords,
                 categorical_cols,
                 dependant_fist_visit,
                 cols_diagnosis,
                 cols_procedures,
                    cols_drugs,
                    path_img,
                    columns_to_drop,
                    all_data_heatmap_diff,
                    corr_plot_syn_real,
                    corr_plot_continous ,
                    corr_plot_categorical ,
                    corr_plot_codes,
                    num_visit_count,
                    patient_visit,
                    d_synthetic_data,
                    eliminate_variables_generadas_post,
                    variables_generadas_post,
                    columnas_demograficas,
                    synthetic_type,
                    encoder
                   
                    ):
        self.patient_visit = patient_visit
        self.num_visit_count = num_visit_count
        self.test_ehr_dataset = test_ehr_dataset
        self.train_ehr_dataset = train_ehr_dataset
        self.synthetic_ehr_dataset = synthetic_ehr_dataset
        self.columnas_test_ehr_dataset = columnas_test_ehr_dataset
        self.top_300_codes = top_300_codes
        
        self.list_metric_resemblance = list_metric_resemblance
        self.result_resemblance = []
        self.results_final = {}
        self.cols = cols
        self.cols_continuous = cols
        self.keywords = keywords
        self.categorical_cols = categorical_cols
        self.dependant_fist_visit = dependant_fist_visit
        logging.basicConfig(level=logging.INFO)
        self.cols_diagnosis = cols_diagnosis
        self.cols_procedures = cols_procedures
        self.cols_drugs = cols_drugs
        self.path_img  = path_img
        self.columns_to_drop = columns_to_drop
        
        self.all_data_heatmap_diff = all_data_heatmap_diff
        self.corr_plot_syn_real = corr_plot_syn_real    
        self.corr_plot_continous = corr_plot_continous
        self.corr_plot_categotical = corr_plot_categorical
        self.corr_plot_codes = corr_plot_codes
        self.d_synthetic_data = d_synthetic_data
        self.eliminate_variables_generadas_post = eliminate_variables_generadas_post
        self.variables_generadas_post = variables_generadas_post
        self.columnas_demograficas = columnas_demograficas
        self.synthetic_type =synthetic_type
        self.encoder = encoder 
    def evaluate(self):    
        results = {}
        if "crammer_metric" in self.list_metric_resemblance:
            logging.info("Executing method: crammer_metric")
            results.update(self.crammer_fun())
        if "analyse_trajectories" in self.list_metric_resemblance:
            logging.info("Executing method: analyse_trajectories")
            self.plot_count_matrix_for_specific_subject( self.train_ehr_dataset, self.synthetic_ehr_dataset,3)
            
        if "outliers_and_histograms_patients_admissions" in self.list_metric_resemblance:
            logging.info("Executing method: outliers_and_histograms_patients_admissions")
            self.outliers_and_histograms_patients_admissions(self.test_ehr_dataset, self.train_ehr_dataset, self.synthetic_ehr_dataset,self.path_img)

        if "metric_outliers" in self.list_metric_resemblance:
            logging.info("Executing method: metric_outliers")
            results.update(self.metric_outliers(self.train_ehr_dataset, self.synthetic_ehr_dataset, self.cols))

        if "get_descriptive_statistics" in self.list_metric_resemblance:
            logging.info("Executing method: get_descriptive_statistics")
            results.update(self.get_descriptive_statistics(self.synthetic_ehr_dataset, self.train_ehr_dataset, self.cols))

        if "compare_average_trends_recordlen" in self.list_metric_resemblance:
            logging.info("Executing method: compare_average_trends_recordlen")
            results.update(self.compare_average_trends_recordlen(self.train_ehr_dataset, self.columnas_test_ehr_dataset, self.test_ehr_dataset, self.synthetic_ehr_dataset))

        if "plot_first_visit" in self.list_metric_resemblance:
            logging.info("Executing method: plot_first_visit")
            self.plot_first_visit(self.train_ehr_dataset, self.synthetic_ehr_dataset, self.dependant_fist_visit,self.path_img)

        if "plot_means_continuos_variables" in self.list_metric_resemblance:
            logging.info("Executing method: plot_means_continuos_variables")
            self.plot_means_continuos_variables(self.train_ehr_dataset, self.synthetic_ehr_dataset, self.cols,self.path_img)

        if "plot_kerneldis" in self.list_metric_resemblance:
            logging.info("Executing method: plot_kerneldis")
            self.plot_kerneldis(self.train_ehr_dataset, self.synthetic_ehr_dataset, self.cols,self.path_img)

        if "get_top10_different_mean_difference" in self.list_metric_resemblance:
            logging.info("Executing method: get_top10_different_proportion")
            results.update(self.get_top10_different_mean_difference(self.test_ehr_dataset,self.train_ehr_dataset, self.synthetic_ehr_dataset, self.cols))

        if "get_visit_value_counts" in self.list_metric_resemblance:
            logging.info("Executing method: get_visit_value_counts")
            results.update(self.get_visit_value_counts(self.train_ehr_dataset, self.synthetic_ehr_dataset, self.columnas_test_ehr_dataset))

        if "plot_differen_correlation" in self.list_metric_resemblance:
            logging.info("Executing method: plot_differen_correlation")
            self.plot_differen_correlation(self.synthetic_ehr_dataset, self.train_ehr_dataset, self.cols, self.categorical_cols, self.keywords,self.path_img,self.all_data_heatmap_diff,
                    self.corr_plot_syn_real,
                    self.corr_plot_continous ,
                    self.corr_plot_categotical ,
                    self.corr_plot_codes)

        if "plot_pacmap" in self.list_metric_resemblance:
            logging.info("Executing method: plot_pacmap")
            self.plot_pacmap(self.synthetic_ehr_dataset, self.train_ehr_dataset, self.keywords, self.categorical_cols,self.path_img)

        if "temporal_histogram_heatmap" in self.list_metric_resemblance:
            logging.info("Executing method: temporal_histogram_heatmap")
            self.temporal_histogram_heatmap(self.synthetic_ehr_dataset, self.train_ehr_dataset,self.path_img)

        if "compare_maximum_range" in self.list_metric_resemblance:
            logging.info("Executing method: compare_maximum_range")
            results.update(self.compare_maximum_range(self.test_ehr_dataset, self.synthetic_ehr_dataset))

        if "plot_dimension_wise" in self.list_metric_resemblance:
            logging.info("Executing method: plot_dimension_wise")
            self.plot_dimension_wise(self.synthetic_ehr_dataset, self.train_ehr_dataset, self.cols_continuous, self.categorical_cols, self.cols_diagnosis, self.cols_drugs, self.cols_procedures,self.path_img)

        if "plot_prevalence_wise" in self.list_metric_resemblance:
            logging.info("Executing method: plot_prevalence_wise")
            self.plot_prevalence_wise(self.synthetic_ehr_dataset, self.train_ehr_dataset,   self.cols_diagnosis, self.cols_drugs, self.cols_procedures,self.path_img)

        if "compare_descriptive_stadistics" in self.list_metric_resemblance:
            logging.info("Executing method: compare_descriptive_stadistics")
            results.update(self.compare_descriptive_stadistics(self.test_ehr_dataset, self.synthetic_ehr_dataset, self.cols))

        if "get_MaximumMeanDiscrepancy" in self.list_metric_resemblance:
            logging.info("Executing method: get_MaximumMeanDiscrepancy")
            results.update(self.get_MaximumMeanDiscrepancy(self.train_ehr_dataset, self.synthetic_ehr_dataset))

        if "kolmogorov_smirnof_test_chissquare" in self.list_metric_resemblance:
            logging.info("Executing method: kolmogorov_smirnof_test_chissquare")
            results.update(self.kolmogorov_smirnof_test_chissquare(self.train_ehr_dataset, self.synthetic_ehr_dataset, self.cols_continuous))

        if "get_jensenshannon_dist" in self.list_metric_resemblance:
            logging.info("Executing method: get_jensenshannon_dist")
            results.update(self.get_jensenshannon_dist(self.train_ehr_dataset, self.synthetic_ehr_dataset, self.test_ehr_dataset, self.cols_continuous))

        if "get_proportion_demos" in self.list_metric_resemblance:
            logging.info("Executing method: get_proportion_demos")
            results.update(self.get_proportion_demos(self.train_ehr_dataset, self.synthetic_ehr_dataset, self.categorical_cols))

        if "get_excat_match" in self.list_metric_resemblance:
            logging.info("Executing method: get_excat_match")
            results.update(self.get_excat_match(self.train_ehr_dataset, self.synthetic_ehr_dataset))

        if "get_dupplicates" in self.list_metric_resemblance:
            logging.info("Executing method: get_dupplicates")
            results.update(self.get_dupplicates(self.train_ehr_dataset, self.synthetic_ehr_dataset))

        if "obtain_least_frequent" in self.list_metric_resemblance:
            logging.info("Executing method: obtain_least_frequent")
            results.update(self.obtain_least_frequent(self.train_ehr_dataset, self.synthetic_ehr_dataset, self.columnas_test_ehr_dataset, 10))

        if "plot_acumulated" in self.list_metric_resemblance:
            logging.info("Executing method: plot_acumulated")
            self.plot_acumulated(self.train_ehr_dataset, self.synthetic_ehr_dataset, self.cols_continuous, self.categorical_cols,self.path_img)

        if "get_common_proportions" in self.list_metric_resemblance:
            logging.info("Executing method: get_common_proportions")
            results.update(self.get_common_proportions(self.train_ehr_dataset, self.synthetic_ehr_dataset))
        if "dimenssion_bernoulli" in self.list_metric_resemblance:
            logging.info("Executing method: dimenssion_bernoulli")
            self.dimension_bernoulli_metric()    
        if "pairwisecorrelation" in self.list_metric_resemblance:
            logging.info("Executing method: pairwisecorrelation")
            results.update(self.pairwisecorrelation())
        if "get_progression_quantity_medicamentpe_visit" in self.list_metric_resemblance:
            logging.info("Executing method: get_progression_quantity_medicamentpe_visit")
            self.progression_quantity_medicamentpe_visit()  
            results.update(self.progression_quantity_medicamentpe_visit())    
            
        if "get_evaluate_synthetic_data_wassetein_overall" in self.list_metric_resemblance:
            logging.info("Executing method: get_evaluate_synthetic_data_wassetein_overall")
            results.update(self.evaluate_synthetic_data_wassetein_overall())
            
        

        if "get_patient_stats" in self.list_metric_resemblance:
           logging.info("Execition method: get patient codes stats")
           self.get_stat_patients()
        if "get_analyze_one_hot_encoding_var" in self.list_metric_resemblance:
            logging.info("get_analyze_demographics(self)")
            self.get_analyze_demographics()   
        if "get_get_analyze_continous" in self.list_metric_resemblance:
            logging.info("get_analyze_continous")
            self.get_analyze_continous()    
        if "demographics_analysis_medicalcodes"     in self.list_metric_resemblance:
            logging.info("demographics_analysis_medicalcodes")
            results = self.demographics_analysis_medicalcodes()
        if "age_occurance_stat" in self.list_metric_resemblance:
            logging.info("age_occurance_stat") 
            results = self.age_occurance_stat()
        if "plot_genereate_histogramplot_total_unique_stats" in self.list_metric_resemblance:
            logging.info("plot_genereate_histogramplot_total_unique_stats")
            results = self.plot_genereate_histogramplot_total_unique_stats()    
        if "get_domain" in self.list_metric_resemblance:
            self.get_domain()
            logging.info(" get_domain")
        if "get_proportion_visits" in self.list_metric_resemblance:
            self.get_proportion_visits()
            logging.info(" get_domain")    
            
        return results
    def get_proportion_visits(self):
        result = process_data(self.train_ehr_dataset, self.synthetic_ehr_dataset,  "visit_rank")
        print(result.to_latex())
        
    def get_domain(self):
        df = self.train_ehr_dataset
        def get_num_categories(series):
            return len(series.unique())

        # Function to get range for continuous variables
        def get_range(series):
            return f"{series.min()}-{series.max()}"

    
        # Get information about the variables
        lis_cat =list(self.columnas_demograficas )+ ["month","year"]
        info = {
            'Static Categorical': {col: get_num_categories(df[col]) for col in lis_cat},
            'Continuous': {
                'Age': get_range(df['Age']),
                'Last Admission': get_range(df['days from last visit']),
                'Length of Stay': get_range(df['LOSRD_avg'])
            },
                 }

        # Count columns containing drugs, diagnosis, and procedures
        count_columns = {
            'Drugs': sum('drug' in col.lower() for col in df.columns),
            'Diagnosis': sum('diagnosis' in col.lower() for col in df.columns),
            'Procedures': sum('procedures' in col.lower() for col in df.columns)
        }

        # Print the information
        print("Variable Information:")
        for category, vars in info.items():
            print(f"\n{category}:")
            for var, value in vars.items():
                print(f"  {var}: {value}")

        print("\nCount of columns containing:")
        for item, count in count_columns.items():
            print(f"  {item}: {count}")

        # Display the first few rows of the dataframe
        print("\nFirst few rows of the dataframe:")
        print(df.head())

        # Display dataframe info
        print("\nDataframe Info:")
        df.info()
    def plot_genereate_histogramplot_total_unique_stats(self):
        logging.info("admission_level")
        compare_and_plot_datasets_patients_syn(self.train_ehr_dataset, self.synthetic_ehr_dataset, self.cols_diagnosis,self.cols_drugs, self.cols_procedures, self.path_img, truncated=True,type_graph="admissions")
        #compare_and_plot_datasets_patients_diag(self.train_ehr_dataset, self.synthetic_ehr_dataset, self.cols_diagnosis,self.cols_drugs, self.cols_procedures, self.path_img, truncated=True)
        #compare_and_plot_datasets(self.train_ehr_dataset, self.synthetic_ehr_dataset, self.cols_diagnosis,self.cols_drugs, self.cols_procedures, self.path_img)
        # compare_and_plot_datasets(real_data, synthetic_data, diagnosis_cols, drug_cols, procedure_cols, type_level)
        logging.info("patient_leve")
        all_code_cols = self.cols_diagnosis.to_list() + self.cols_drugs.to_list() + self.cols_procedures.to_list()
  
        train_ehr_dataset_grouped = self.train_ehr_dataset.groupby("id_patient").sum().reset_index()
        synthetic_ehr_dataset_grouped =  self.synthetic_ehr_dataset.groupby("id_patient").sum().reset_index()
        compare_and_plot_datasets_patients_syn(train_ehr_dataset_grouped, synthetic_ehr_dataset_grouped, self.cols_diagnosis,self.cols_drugs, self.cols_procedures, self.path_img, truncated=True,type_graph="patients")
        #compare_and_plot_datasets_patients(train_ehr_dataset_grouped, synthetic_ehr_dataset_grouped, self.cols_diagnosis,self.cols_drugs, self.cols_procedures, self.path_img)
        
    def age_occurance_stat(self):
        results = analyze_diagnoses(self.train_ehr_dataset, self.synthetic_ehr_dataset, self.cols_diagnosis)
        
        results.to_csv("generated_synthcity_tabular/ARF"+self.synthetic_type+"edad_ocurrance.csv")
       

        return results

    def demographics_analysis_medicalcodes(self):
        # Example usage:
        synthetic_type = self.synthetic_type
        dem = ['RELIGION_encoded', 'MARITAL_STATUS_encoded',  'ETHNICITY_encoded','GENDER_encoded']
        #dem = ['GENDER']
        make_a =True
        diagnosis_codes = self.cols_diagnosis
        procedure_codes = self.cols_procedures
        drug_codes = self.cols_drugs
        results = pd.DataFrame()
        top_codes = "Other"
        if make_a:
            for i in dem:
                #final_results,top_codes = process_dataframesv2(self.train_ehr_dataset, self.synthetic_ehr_dataset, i, diagnosis_codes, procedure_codes, drug_codes,self.encoder,top_codes= top_codes,)
                final_results, overall_prop, overall_count, drug_codes, diagnosis_codes, procedure_codes = process_dataframesv22(self.train_ehr_dataset, self.synthetic_ehr_dataset, i, diagnosis_codes, procedure_codes, drug_codes, self.encoder, 'id_patient')
                
                #final_results, overall_prop, overall_count, top_drugs, top_diagnoses, top_procedures = process_dataframesv2(self.train_ehr_dataset, self.synthetic_ehr_dataset, i, diagnosis_codes, procedure_codes, drug_codes, self.encoder, "id_patient", path_img=None, top_codes=None)
   
                #final_results.index = [j +"_" +i if j == 'Unknown' or j=='Otra' or j == '0' else j for j in final_results.index]
                #results = pd.concat([final_results,results])
                
                print(final_results.to_latex())
               
                final_results.to_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/ARF_demo/"+synthetic_type+"_"+i+"_.csv")
       

        type_diag = "diagnoses"
        comparison_results =compare_synthetic_real_demographics(self.train_ehr_dataset, self.synthetic_ehr_dataset, dem,diagnosis_codes,self.encoder)   
        comparison_results = consistent_sort_demographic_classes(comparison_results)
        comparison_results.to_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/ARF_demo/"+type_diag+"_"+synthetic_type+".csv")
        
        type_diag = "procedures"
        comparison_results =compare_synthetic_real_demographics(self.train_ehr_dataset, self.synthetic_ehr_dataset, dem,procedure_codes,self.encoder)   
        comparison_results = consistent_sort_demographic_classes(comparison_results)
        comparison_results.to_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/ARF_demo/"+type_diag+"_"+synthetic_type+".csv")
        
        type_diag = "drugs"
        comparison_results =compare_synthetic_real_demographics(self.train_ehr_dataset, self.synthetic_ehr_dataset, dem,drug_codes,self.encoder)   
        comparison_results = consistent_sort_demographic_classes(comparison_results)
        comparison_results.to_csv("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/ARF_demo/"+type_diag+"_"+synthetic_type+".csv")
        
        print(comparison_results.to_latex())
        #overall codes
        # results = pd.DataFrame()      
        # top_codes = None  
        # for i in dem:
        #     final_results,top_codes = process_dataframes(self.train_ehr_dataset, self.synthetic_ehr_dataset, i, diagnosis_codes, procedure_codes, drug_codes,top_codes= top_codes)
        #     results = pd.concat([final_results,results])
        #     print(results.to_latex())    
        return results
      


    def get_analyze_continous(self):
        #print(Admission level)
        
        def get_continous_cosl(synthetic_ehr_dataset,train_ehr_dataset,columns_to_analyze):
            df_comparative = analyze_continuous_variables(synthetic_ehr_dataset,train_ehr_dataset, columns_to_analyze)

            print("LaTeX table for comparative statistics of continuous variables:")
            #print(df_comparative.to_latex(index=False, escape=False))  

            latex_table = latex_general(df_comparative,"",""
                                    ,no_percentage_float=['Visit 1 Real', 'Visit 1 Synthetic','Visit 2 Real','Visit 3 Real',
            'Visit 2 Synthetic',
        'Visit 3 Synthetic', ])
            print(latex_table)

    

            df_comparative_overall = analyze_overall_continuous_variables(synthetic_ehr_dataset, train_ehr_dataset, columns_to_analyze)

            print("LaTeX table for overall comparative statistics of continuous variables:")
            print(df_comparative_overall.to_latex(index=False, escape=False))
            latex_table = format_continuous_variables_to_latex(df_comparative_overall, 
                                            "Statistics of demographic variables with Percentage Differences",
                                            "tab:stats_table_1")
            print(latex_table)



            # Assuming 'synthetic_df' and 'real_df' are your DataFrames with the required column
            df_comparative_admission = analyze_admission_dates(synthetic_ehr_dataset, train_ehr_dataset, 'ADMITTIME')
            print("LaTeX table for comparative statistics of admission dates:")
            print(df_comparative_admission.to_latex(index=False, escape=False))
            
            # patient level stat
            df_results = analyze_patient_level(synthetic_ehr_dataset, train_ehr_dataset, columns_to_analyze)
            print(df_results.to_latex())
        print("print admission level")
        columns_to_analyze = self.cols_continuous
        get_continous_cosl(self.synthetic_ehr_dataset,self.train_ehr_dataset,columns_to_analyze)

        print("print patient level")
        dataframe_train = self.train_ehr_dataset.groupby('id_patient').mean().reset_index()   
        admit_times = self.train_ehr_dataset.groupby('id_patient')['ADMITTIME'].first().reset_index()
        dataframe_train = pd.merge(dataframe_train, admit_times, on='id_patient')

        dataframe_synthetic = self.synthetic_ehr_dataset.groupby('id_patient').mean().reset_index()  
        admit_times = self.synthetic_ehr_dataset.groupby('id_patient')['ADMITTIME'].first().reset_index()
        dataframe_synthetic = pd.merge(dataframe_synthetic, admit_times, on='id_patient')
 
        get_continous_cosl(dataframe_synthetic,dataframe_train,columns_to_analyze) 
            
    def get_analyze_demographics(self):
        
        # static columns
        type_data = "Real"

       


        def generate_demographic_prop(synthetic_ehr_dataset,train_ehr_dataset,dependant_fist_visit):
            cols_to_analyze = [i + '_encoded' for i in self.dependant_fist_visit if i != 'ADMITTIME']
            groups = {col: list(synthetic_ehr_dataset.filter(like=col, axis=1).columns) for col in cols_to_analyze}
        
            synthetic_df = analyze_demographics2(synthetic_ehr_dataset, groups)
            real_df = analyze_demographics2(train_ehr_dataset, groups)
            
            latex_table = generate_demographics_latex_table2(
                synthetic_df, real_df,
                "Statistics of demographic variables with Percentage Differences",
                "tab:stats_table_1"
            )
        
            print(latex_table)

            generate_demographics_latex_table(train_ehr_dataset,synthetic_ehr_dataset,groups,"","")
                    # static columns
            type_data = "Real"
            
            cols_to_analyze = [i + '_encoded' for i in dependant_fist_visit if i != 'ADMITTIME']
            groups = {col: list(synthetic_ehr_dataset.filter(like=col, axis=1).columns) for col in cols_to_analyze}
        
            synthetic_df = analyze_demographics2(synthetic_ehr_dataset, groups)
            real_df = analyze_demographics2(train_ehr_dataset, groups)
            
            latex_table = generate_demographics_latex_table2(
                synthetic_df, real_df,
                "Statistics of demographic variables with Percentage Differences",
                "tab:stats_table_1"
            )
        
            print(latex_table)

            generate_demographics_latex_table(train_ehr_dataset,synthetic_ehr_dataset,groups,"","")
        

        
        print("admissio level")
        dataframe_train = self.train_ehr_dataset
        dataframe_synthetic =self.synthetic_ehr_dataset
        dependant_fist_visit =self.dependant_fist_visit
        generate_demographic_prop(dataframe_synthetic,dataframe_train,dependant_fist_visit)


        print("patient level")
        dataframe_train = self.train_ehr_dataset.groupby('id_patient').first().reset_index()
        dataframe_synthetic =self.synthetic_ehr_dataset.groupby('id_patient').first().reset_index()
        dependant_fist_visit =self.dependant_fist_visit
        generate_demographic_prop(dataframe_synthetic,dataframe_train,dependant_fist_visit)


    def get_stat_patients(self):
        #patient_level
        patient_level_stats = True
        admission_level_stats = False
        if patient_level_stats: 
            all_code_cols = diagnosis_columns + medication_columns + procedure_columns
            columns_for_max = ['visit_rank']
    
            # Group by 'id_patient'
            grouped_data_patietn = self.train_ehr_dataset.groupby('id_patient').agg({
                **{col: 'sum' for col in all_code_cols},  # Sum all columns except 'visit_Rank'
                **{col: 'max' for col in columns_for_max}  # Max for 'visit_Rank'
            })

            grouped_data_patietn_synthetic = self.synthetic_ehr_dataset.groupby('id_patient').agg({
                **{col: 'sum' for col in all_code_cols},  # Sum all columns except 'visit_Rank'
                **{col: 'max' for col in columns_for_max}  # Max for 'visit_Rank'
            })
            print("patient_level")
            result = print_latex_generate_stats2(grouped_data_patietn, grouped_data_patietn_synthetic, diagnosis_columns, medication_columns, procedure_columns, "Admission level")
            #print_latex_genrate_stats(grouped_data_patietn,grouped_data_patietn_synthetic,diagnosis_columns,medication_columns, procedure_columns,"Patient ")



            
            

            grouped_data_patietn = self.train_ehr_dataset.groupby('id_patient').agg({
                **{col: 'sum' for col in all_code_cols},  # Sum all columns except 'visit_Rank'
                **{col: 'max' for col in columns_for_max}  # Max for 'visit_Rank'
            })

            grouped_data_patietn_synthetic = self.synthetic_ehr_dataset.groupby('id_patient').agg({
                **{col: 'sum' for col in all_code_cols},  # Sum all columns except 'visit_Rank'
                **{col: 'max' for col in columns_for_max}  # Max for 'visit_Rank'
            })
            print("patient_leve")
            real_data = get_stat_per_visit3(grouped_data_patietn, diagnosis_columns, medication_columns, procedure_columns, "Real", is_synthetic=False)
            synthetic_data = get_stat_per_visit3(grouped_data_patietn_synthetic, diagnosis_columns, medication_columns, procedure_columns, "Synthetic", is_synthetic=True)

            table1 = generate_latex_table3(real_data, synthetic_data, 
                                            "Statistics for first five admissions, UD - unique diagnosis. TD - total diagnosis. UDr - unique drugs. TDr - total drugs. UP - unique procedures. TP - total procedures.",
                                            "tab:stats_p_5")
            
            print("considering 5 visits for patient data ")
            print(table1)

            table1 = generate_latex_table3_v2(real_data, synthetic_data, 
                                            "Statistics for first five admissions, UD - unique diagnosis. TD - total diagnosis. UDr - unique drugs. TDr - total drugs. UP - unique procedures. TP - total procedures.",
                                            "tab:stats_p_5")
            #print(table1)

            table2 = generate_latex_table_all_codes3(real_data, synthetic_data,
                                                    "Statistics for first five admissions",
                                                    "tab:stats_p_6")
            print(table2)


            #real_stats = ger_stat_per_vist(grouped_data_patietn,diagnosis_columns,medication_columns,procedure_columns,"Real",is_synthetic= False)
            
            #syn_stats = ger_stat_per_vist(grouped_data_patietn_synthetic,diagnosis_columns,medication_columns,procedure_columns,"Synthetic",is_synthetic= True)

            #stats for the first 4 visits:
        if admission_level_stats:        
            print("admission_level")

            grouped_data_admission_leve = self.train_ehr_dataset.groupby(['id_patient', 'visit_rank']).sum().reset_index()
            grouped_data_admission_leve_synthetic = self.synthetic_ehr_dataset.groupby(['id_patient', 'visit_rank']).sum().reset_index()
            #print_latex_genrate_stats(grouped_data_admission_leve,grouped_data_admission_leve_synthetic,diagnosis_columns,medication_columns, procedure_columns,"Admission ")
            #result = print_latex_generate_stats2(grouped_data_admission_leve, grouped_data_admission_leve_synthetic, diagnosis_columns, medication_columns, procedure_columns, "Admission level")
            
            result = print_latex_generate_stats2(self.train_ehr_dataset, self.synthetic_ehr_dataset, diagnosis_columns, medication_columns, procedure_columns, "Admission level")
            
            #print_latex_genrate_stats(self.train_ehr_dataset,self.synthetic_ehr_dataset,diagnosis_columns,medication_columns, procedure_columns,"Admission general")
    
            real_data = get_stat_per_visit3(grouped_data_admission_leve, diagnosis_columns, medication_columns, procedure_columns, "Real", is_synthetic=False)
            synthetic_data = get_stat_per_visit3(grouped_data_admission_leve_synthetic, diagnosis_columns, medication_columns, procedure_columns, "Synthetic", is_synthetic=True)
            print("temporal per admission")
            table1 = generate_latex_table3_v2(real_data, synthetic_data, 
                                            "Statistics for first five admissions, UD - unique diagnosis. TD - total diagnosis. UDr - unique drugs. TDr - total drugs. UP - unique procedures. TP - total procedures.",
                                            "tab:stats_p_5")
            print(table1)


            table1 = generate_latex_table3(real_data, synthetic_data, 
                                            "Statistics for first five admissions, UD - unique diagnosis. TD - total diagnosis. UDr - unique drugs. TDr - total drugs. UP - unique procedures. TP - total procedures.",
                                            "tab:stats_p_5")
 #           print(table1)

            table2 = generate_latex_table_all_codes3(real_data, synthetic_data,
                                                    "Statistics for first five admissions",
                                                    "tab:stats_p_6")
            print(table2)


            real_data = get_stat_per_visit3(self.train_ehr_dataset, diagnosis_columns, medication_columns, procedure_columns, "Real", is_synthetic=False)
            synthetic_data = get_stat_per_visit3(self.synthetic_ehr_dataset, diagnosis_columns, medication_columns, procedure_columns, "Synthetic", is_synthetic=True)
            table1 = generate_latex_table3_v2(real_data, synthetic_data, 
                                            "Statistics for first five admissions, UD - unique diagnosis. TD - total diagnosis. UDr - unique drugs. TDr - total drugs. UP - unique procedures. TP - total procedures.",
                                            "tab:stats_p_5")
 #           print(table1)

            table1 = generate_latex_table3(real_data, synthetic_data, 
                                            "Statistics for first five admissions, UD - unique diagnosis. TD - total diagnosis. UDr - unique drugs. TDr - total drugs. UP - unique procedures. TP - total procedures.",
                                            "tab:stats_p_5")
 #           print(table1)

            table2 = generate_latex_table_all_codes3(real_data, synthetic_data,
                                                    "Statistics for first five admissions",
                                                    "tab:stats_p_6")
            print(table2)


            #real_stats = ger_stat_per_vist(grouped_data_admission_leve,diagnosis_columns,medication_columns,procedure_columns,"Real",is_synthetic= False)
            
            #syn_stats = ger_stat_per_vist(grouped_data_admission_leve_synthetic,diagnosis_columns,medication_columns,procedure_columns,"Synthetic",is_synthetic= True)

            #real_stats = ger_stat_per_vist(self.train_ehr_dataset,diagnosis_columns,medication_columns,procedure_columns,"Real",is_synthetic= False)
            
            #syn_stats = ger_stat_per_vist(self.synthetic_ehr_dataset,diagnosis_columns,medication_columns,procedure_columns,"Synthetic",is_synthetic= True)


    def pairwisecorrelation(self):
        res  = {}
        #pcd = calculate_pcd(self.train_ehr_dataset, self.synthetic_ehr_dataset)
        #res["pair wise correlation overall"] = pcd
        #continous
        pcd = calculate_pcd(self.train_ehr_dataset[self.cols], self.synthetic_ehr_dataset[self.cols])
        res["pair wise correlation continous variables"] = pcd
        print(f"Pairwise Correlation Difference (PCD): {pcd}")
        #categorical
        cols_list = []
        
        for i in self.categorical_cols:
            cols_f = self.synthetic_ehr_dataset.filter(like=i, axis=1).columns
            cols_list.extend(list(cols_f))
        pcd = calculate_pcd(self.train_ehr_dataset[cols_list], self.synthetic_ehr_dataset[cols_list])
        res["pair wise correlation categorical variable"] = pcd
        print(f"Pairwise Correlation Difference (PCD): {pcd}")
        #icd-9codes
        
        for i in self.keywords:
            col_prod = [col for col in self.train_ehr_dataset.columns if any(palabra in col for palabra in [i])]
            pcd = calculate_pcd(self.train_ehr_dataset[col_prod], self.synthetic_ehr_dataset[col_prod])
            res["pair wise correlation icd-9 codes"] = pcd
            print(f"Pairwise Correlation Difference (PCD): {pcd}")
        datafram_res = pd.DataFrame(res, index=[0])    
        print(datafram_res.to_latex())
        return res
           
     
        
    def progression_quantity_medicamentpe_visit(self):
        columns_to_check = ['drug_vs_procedure', 'drug_vs_diagnosis', 'procedure_vs_diagnosis',]
        
        # Asumimos que el dataframe se llama 'df' y tiene las columnas necesarias
        # Creamos las columnas de conteo
        #get the train visit that have more than 100 patients per visit
        #get the ratio for both synthetic and no synthetic
        #for each visit get a threshol list that is quartil of drugs,diagnos and procedure ratios, see if synthetic data has similar vists
        df_filtered = get_proportions_vs_qunatities_drugs(self.train_ehr_dataset,"train",medication_columns,procedure_columns,diagnosis_columns)
        df_synthetic = get_proportions_vs_qunatities_drugs(self.synthetic_ehr_dataset,"synthetic",medication_columns,procedure_columns,diagnosis_columns)

        df_real_reshaped = reshape_data(df_filtered)
        df_synthetic_reshaped = reshape_data(df_synthetic)
        create_ratio_histograms(df_real_reshaped, 'Real Data Ratio Distributions',self.path_img)
        create_ratio_histograms(df_synthetic_reshaped, 'Synthetic Data Ratio Distributions',self.path_img)
        #create_boxplots(df_real_reshaped, df_synthetic_reshaped, 'Box plot of Ratios (Synthetic and Real Data)')
        
        
        

        results = {}
        for visit in range(1,7):
            #for each visit get a threshol list that is quartil of drugs,diagnos and procedure ratios, see if synthetic data has similar vists
            df_filtered_visitunique = df_filtered[df_filtered['visit_rank'] == visit]
            threshold_list = get_threshold(df_filtered_visitunique,quartil = '75%')
       
            df_filtered_visitunique_synthetic = df_synthetic[df_synthetic['visit_rank'] == visit]
            percentages = get_percentage_synthetic(df_filtered_visitunique_synthetic, columns_to_check, threshold_list)
            results[f'Visit_{visit}'] = percentages
        df_results = pd.DataFrame(results).T
        df_results.index.name = "Visits"
        mean_values = df_results.mean()
        df_results.loc['Mean'] = mean_values
        print(df_results.to_latex())# Imprimir resultados
                           
                    

        return df_results.to_dict()
    
    def dimension_bernoulli_metric(self):
        # Identificar columnas categóricas
        # categorical_columns = self.train_ehr_dataset.select_dtypes(include=['object', 'category']).columns

        # # Identificar columnas numéricas discretas (este es un enfoque simple, puede necesitar ajustes)
        # numeric_discrete =  self.train_ehr_dataset.select_dtypes(include=['int64']).columns[ self.train_ehr_dataset.select_dtypes(include=['int64']).nunique() < 10]
        # float_cols = self.train_ehr_dataset.select_dtypes(include=['float64']).columns
        # # Combinar todas las columnas discretas
        # discrete_columns = list(float_cols) + list(numeric_discrete)

        # # Extraer características discretas
        # discrete_x =  self.train_ehr_dataset[discrete_columns]
        # discreat_real =  self.train_ehr_dataset[discrete_columns]
        # synthethic_data = self.synthetic_ehr_dataset[discrete_columns]
        # plo_dimensionwisebernoulli(discreat_real,synthethic_data,self.path_img)
        
        
        #count matrix
        for word in self.keywords:
            cols_sel = self.train_ehr_dataset.filter(like=word).columns
            discreat_real =  self.train_ehr_dataset[cols_sel]
            synthethic_data = self.synthetic_ehr_dataset[cols_sel]
            plo_dimensionwisebernoulli(discreat_real,synthethic_data,str(word),self.path_img)
        
        
        #categorical     
        cols_categorical = []
        for i in self.categorical_cols:
            cols_f = self.synthetic_ehr_dataset.filter(like=i, axis=1).columns
            cols_categorical.extend(list(cols_f))
        discreat_real =  self.train_ehr_dataset[cols_categorical]
        synthethic_data = self.synthetic_ehr_dataset[cols_categorical]
        plo_dimensionwisebernoulli(discreat_real,synthethic_data,"Categorical variables",self.path_img)
        
        
        #continous    
        discreat_real =  self.train_ehr_dataset[self.cols]
        synthethic_data = self.synthetic_ehr_dataset[self.cols]
        plo_dimensionwisebernoulli(discreat_real,synthethic_data,"Continous variables",self.path_img)
             
        
        
         
    def evaluate_synthetic_data_wassetein_overall(self):
        res_metrics = {}
        
        # For keywords (assuming these are continuous)
        for word in self.keywords:
            cols_sel = self.train_ehr_dataset.filter(like=word).columns
            distances = calculate_wasserstein_distance(self.train_ehr_dataset, self.synthetic_ehr_dataset, cols_sel)
            res_metrics["Wasserstein distance " + word] = np.mean(list(distances.values()))
            
            # Create histogram for this keyword group
            df_distances = pd.DataFrame(list(distances.items()), columns=['Column', 'Distance'])
            hist_ori(df_distances, 'Distance', f'Wasserstein Distances for {word}', 'Wasserstein Distance', label=word,path_img=self.path_img)
        
        # For categorical variables
        
      
    
        cols_categorical = []
        self.categorical_cols = [i for i in self.categorical_cols if i!= "visit_rank"]
        for i in self.categorical_cols:
            cols_f = self.synthetic_ehr_dataset.filter(like=i, axis=1).columns
            cols_categorical.extend(list(cols_f))
        results, mean_cramers_v = calculate_cramers_v_with_mean(self.train_ehr_dataset, self.synthetic_ehr_dataset, cols_categorical)
        res_metrics["Wasserstein Distances for demographics and admission variables"] = mean_cramers_v
        
        # Create histogram for Cramer's V values
        #df_cramer_v = pd.DataFrame(list(cramer_v_values.items()), columns=['Column', 'Cramer_V'])
        #hist_ori(df_cramer_v, 'Cramer_V', 'Wasserstein Distances for demographics and admission variables', 'Wasserstein Distance', label='Categorical',path_img=self.path_img)
        
        # For continuous variables
        
        distances_continuous = calculate_wasserstein_distance(self.train_ehr_dataset, self.synthetic_ehr_dataset, self.cols_continuous)
        res_metrics["Wasserstein distance numerical"] = np.mean(list(distances_continuous.values()))
        
        # Create histogram for continuous variables
        df_distances_continuous = pd.DataFrame(list(distances_continuous.items()), columns=['Column', 'Distance'])
        hist_ori(df_distances_continuous, 'Distance', 'Wasserstein Distances for Continuous Variables', 'Wasserstein Distance', label='Continuous',path_img=self.path_img)
        
        print_latex(res_metrics)
        return res_metrics
    def evaluate_synthetic_data_wassetein_overall_(self,):
        res_metrics = {}
        
        # For keywords (assuming these are continuous)
        for word in self.keywords:
            cols_sel = self.train_ehr_dataset.filter(like=word).columns
            res_metrics["Wasserstein distance " + word] = calculate_wasserstein_distance(
                self.train_ehr_dataset, self.synthetic_ehr_dataset, cols_sel
            )
        
        # For categorical variables
        cols_categorical = []
        for i in self.categorical_cols:
            cols_f = self.synthetic_ehr_dataset.filter(like=i, axis=1).columns
            cols_categorical.extend(list(cols_f))
            res_metrics["Wasserstein distance demographics and admission"] = calculate_wasserstein_distance(
            self.train_ehr_dataset, self.synthetic_ehr_dataset, self.cols_categorical)
        
        # For continuous variables
        res_metrics["Wasserstein distance numerical variables"] = calculate_wasserstein_distance(
            self.train_ehr_dataset, self.synthetic_ehr_dataset, self.continuous_cols
        )
        
        print_latex(res_metrics)
        return res_metrics

                    #outliers
    def crammer_fun(self):
   
     
        categorical_features=[i for i in self.synthetic_ehr_dataset if i!= self.cols and i!="month" and i!= "year"]
        
        #categorical_features = identify_categorical_features(self.synthetic_ehr_dataset)
        #common_cols = list(set(self.synthetic_ehr_dataset.columns).intersection(self.train_ehr_dataset[categorical_features].columns))
        #categorical_features = list(set(categorical_features).intersection(common_cols))
        if self.eliminate_variables_generadas_post:
           categorical_features = self.synthetic_ehr_dataset[categorical_features].columns[~self.train_ehr_dataset.columns.isin(self.variables_generadas_post)]
        # categorical_features = [
        #                         feature for feature in self.synthetic_ehr_dataset.select_dtypes(include="object")]

        # # Label encoder
        # label_encoder = preprocessing.LabelEncoder()
        # for feature in categorical_features:
        #     # Real data
        #     self.train_ehr_dataset[feature] = label_encoder.fit_transform(self.train_ehr_dataset[feature])
        #     # Syntheric data
        #     for dataset in self.d_synthetic_data:
        #          self.synthetic_ehr_dataset[dataset][feature] = label_encoder.transform(
        #              self.synthetic_ehr_dataset[dataset][feature]
        #         )}
        
        
        res = {}
        
        # Encuentra columnas en df1 que no están en df2
        evaluation = EvaluationFramework(self.train_ehr_dataset, self.d_synthetic_data, categorical_features, verbose=True) 
        score_wasserstein_cramers_v,df_scores_before_aggregation,scores_aux = evaluation.wasserstein_cramers_v_test()
        
        
        was_cram = pd.DataFrame()
        was_cram["Wasserstein distance"] = scores_aux[scores_aux["Wasserstein distance"].notnull()].mean()
        was_cram["Cramer's V"] = scores_aux["Cramer's V"].T[categorical_features].mean()
        print(was_cram.to_latex())
        hist(scores_aux,"Cramer's V","Distribution of Cramer's V Synthetic vs Real data","Cramer's V",scores_aux["Cramer's V"].describe()[ '75%'])
        print(scores_aux["Cramer's V"].describe().to_latex())
        print(score_wasserstein_cramers_v)
        # Novelty test
        #score_novelty = evaluation.novelty_test()
        #print(score_novelty)
        # Anomaly detection test
        res["wasserstein"] = score_wasserstein_cramers_v['Min_node_15']
        score_anomaly = evaluation.anomaly_detection()   
        print(score_anomaly)
        res["anomaly"] = score_anomaly['Min_node_15_']
        try:
            Ranking = evaluation.get_synthesizers_ranking()
            print(Ranking)
        except:
            pass    
        return res
    def outliers_and_histograms_patients_admissions(self,test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset,path_img ):
               
        combined_df = pd.DataFrame()    
        
        train_ehr_dataset_t = train_ehr_dataset.copy()
        
        #same sie for trainse
        
        if test_ehr_dataset.shape[0] < train_ehr_dataset_t.shape[0]:
            train_ehr_dataset_t = train_ehr_dataset_t[:test_ehr_dataset.shape[0]]
        else:
            test_ehr_dataset = test_ehr_dataset[:train_ehr_dataset_t.shape[0]]
        
        #equalsize
        #train_ehr_dataset, synthetic_ehr_dataset = equalize_length_dataframe(train_ehr_dataset, synthetic_ehr_dataset)    
        
        for i in self.keywords:
                synthetic_train_values = []
                test_train_values = []
                synthetic_train_outliers_values = []
                # It looks like the code snippet you provided is not a valid Python code. It seems to be a
                    # comment or a placeholder text. If you have a specific question or need help with Python
                # code, please provide more details or the actual code snippet for me to assist you better.
                test_train_outliers_values = []
        
                if i == 'diagnosis':
                    title_list = ['Diagnosis per Patient','Diagnosis per Admission','Admission per Diagnosis',  ]
                elif i == 'procedures':
                    title_list = ['Procedures per Patient', 'Procedures per Admission', 'Admission per Procedures']
                elif i == 'drugs':
                    title_list = ['Drugs per Patient', 'Drugs per Admission', 'Admission per Drugs']
        
                    
                    
                # Shynthetic/train
                # get the same  nmebr o patients*
                real_drugs_per_patient_t, real_drugs_per_admission_t, real_admissions_per_drug_t,real_patients_per_drug_t = calculate_counts(train_ehr_dataset_t,i)
                test_drugs_per_patient_t, test_drugs_per_admission_t, test_admissions_per_drug_t,synthetic_patients_per_drug = calculate_counts(test_ehr_dataset,i)
                df_descirb = obtatain_df(real_drugs_per_patient_t[i+'_count'], real_drugs_per_admission_t[i+'_count'],
                            test_drugs_per_patient_t[i+'_count'], test_drugs_per_admission_t[i+'_count'],
                            real_admissions_per_drug_t["admission"+'_count'], test_admissions_per_drug_t["admission"+'_count'],
                            "train "+i+" per patient","train "+i+"  per admission",
                            "test "+i+"  per patient","test "+i+"  per admission",
                            "train admission per "+i,"test admission per "+i)
                print(df_descirb.to_latex())   
                #, 'Patient Count (Synthetic)'
                real_out_t = calculate_outlier_ratios_tout2(real_drugs_per_patient_t,i,)
                test_out_t = calculate_outlier_ratios_tout2(test_drugs_per_patient_t,i)
                #real_data, synthetic_data, column_name, title, xlabel, ylabel_real, ylabel_synthetic=real_out_t, test_out_t, i, 'Outliers of ' +i+ ' per Patient', 'Number of ' + i, 'Patient Count (Train)','Test'
                #plot_outliers(real_out_t, test_out_t, i,
                #                'Outliers of ' +i+ ' per Patient', 'Number of ' + i, 'Patient Count (Train)','Test')
                wd_real_drugs_per_patient_outlier_esttrain = plot_histograms_separate_axes22(real_out_t[i+'_count'], test_out_t[i+'_count'], 
                                'Histogram of ' +i+ ' per patient', 'Number of ' + i, 'Patient Count (Train)','Test',None)
                
                wd_real_drugs_per_patient_esttrain = plot_histograms_separate_axes22(real_drugs_per_patient_t[i+'_count'], test_drugs_per_patient_t[i+'_count'], 
                             'Histogram of ' +i+ ' per patient', 'Number of ' + i, 'Patient Count (Train)','Test',None)
    
                # plot_boxplots(real_drugs_per_patient_t[i+'_count'], test_drugs_per_patient_t[i+'_count'], 
                #                 'Histogram of ' +i+ ' per Patient', 'Number of ' + i, 'Patient Count (Real)',path_img)
                # #, 'Patient Count 
                #, 'ADmission Count 
                real_out_a_t = calculate_outlier_ratios_tout2(real_drugs_per_admission_t,i)
                syn_out_a_t = calculate_outlier_ratios_tout2(test_drugs_per_admission_t,i)
                #plot_outliers(real_out_a_t, syn_out_a_t, i,
                #                'Outliers of ' +i+ ' per Admission', 'Number of ' + i, 'Patient Count (Real)','Test')
                wd_real_drugs_per_admission_outliers_testtrain =  plot_histograms_separate_axes22(real_out_a_t[i+'_count'], syn_out_a_t[i+'_count'], 
                                'Histogram of ' +i+ ' per admission', 'Number of' + i, 'Patient Count (Real)','Test',None)
            
        
                # plot_boxplots(real_drugs_per_admission_t[i+'_count'], test_drugs_per_admission_t[i+'_count'], 
                #                 'Histogram of ' +i+ ' per Admission', 'Number of ' + i, 'Patient Count (Real)',path_img)

                wd_real_drugs_per_admission_testtrain = plot_histograms_separate_axes22(real_drugs_per_admission_t[i+'_count'], test_drugs_per_admission_t[i+ '_count'], 
                                'Histogram of ' +i+ ' per Admission', 'Number of ' + i, 'Admission Count (Real)'
                            ,'Test',None)
                #, 'ADmission Count per drug(train, test)'
                
                real_out_a_a_t = calculate_outlier_ratios_tout2(real_admissions_per_drug_t,"admission")
                test_out_a_a_t = calculate_outlier_ratios_tout2(test_admissions_per_drug_t,"admission")
                    
                #plot_outliers(real_out_a_a_t, test_out_a_a_t, "admission",
                #                'Outliers of   Admission per ' + i, 'Number of ' + i, 'Patient Count (Real)','Test')
                wd_real_admissions_per_drug_outliers_esttrain = plot_histograms_separate_axes22(real_out_a_a_t["admission"+'_count'], test_out_a_a_t["admission"+'_count'], 
                                'Histogram of outliers ' + 'Admission per ' + i, 'Number of ' + i, 'Patient Count (Real)','Test',None)
                

                wd_real_admissions_per_drug_testtrain = plot_histograms_separate_axes22(real_admissions_per_drug_t['admission_count'], test_admissions_per_drug_t['admission_count'], 
                                'Histogram of Admissions per ' + i, 'Number of Admissions ',  i+ ' Count (Real)','Test' ,None)
                
                # plot_boxplots(real_admissions_per_drug_t['admission_count'], test_admissions_per_drug_t['admission_count'], 
                #                 'Histogram of Admissions per ' +i, 'Number of Admissions ',  i+ ' Count (Real)',path_img)

                
                #, 'Patient Count (Synthetic/train)'
                real_drugs_per_patient, real_drugs_per_admission, real_admissions_per_drug,real_patients_per_drug = calculate_counts(train_ehr_dataset,i)
                synthetic_drugs_per_patient, synthetic_drugs_per_admission, synthetic_admissions_per_drug,synthetic_patients_per_drug = calculate_counts(synthetic_ehr_dataset,i)
                #, 'Patient Count (Synthetic)'
                df_descirb = obtatain_df(real_drugs_per_patient[i+'_count'], real_drugs_per_admission[i+'_count'],
                            synthetic_drugs_per_patient[i+'_count'], synthetic_drugs_per_admission[i+'_count'],
                            real_admissions_per_drug['admission_count'], synthetic_admissions_per_drug['admission_count'],
                            "train "+i+" per patient","train "+i+"  per admission",
                            "synthetic "+i+"  per patient", "synthetic "+i+"  per admission",
                            "train admission per "+i, "synthetic admission per "+i)
                print(df_descirb.to_latex())  
                
                real_out = calculate_outlier_ratios_tout2(real_drugs_per_patient,i)
                syn_out = calculate_outlier_ratios_tout2(synthetic_drugs_per_patient,i)
                #plot_outliers(real_drugs_per_patient, synthetic_drugs_per_patient, i,
                #                'Outliers of ' +i+ ' per Patient', 'Number of '+ i, 'Patient Count (Real)','Synthetic')
                wd_real_drugs_per_patient_outlier_syntrain = plot_histograms_separate_axes22(real_out[i+'_count'], syn_out[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of '+ i, 'Patient Count (Real)','Synthetic',path_img)
            
            
                wd_real_drugs_per_patient_syntrain = plot_histograms_separate_axes22(real_drugs_per_patient[i+'_count'], synthetic_drugs_per_patient[i+'_count'], 
                                'Histogram of ' +i+ ' per patient', 'Number of '+ i, 'Patient Count (Real)','Synthetic',path_img)
                
                wd_real_drugs_per_patient_syntrain = plot_histograms_separate_axes22(real_drugs_per_patient[i+'_count'], synthetic_drugs_per_patient[i+'_count'], 
                                'Histogram of ' +i+ ' per patient', 'Number of '+ i, 'Patient Count (Real)','Synthetic',path_img)
                
                
                plot_boxplots(real_drugs_per_patient[i+'_count'], synthetic_drugs_per_patient[i+'_count'], 
                                'Histogram of ' +i+ ' per patient', 'Number of '+ i, 'Patient Count (Real)',path_img)

                #, 'ADmission Count (Synthetic)'
                real_out_a = calculate_outlier_ratios_tout2(real_drugs_per_admission,i)
                syn_out_a = calculate_outlier_ratios_tout2(synthetic_drugs_per_admission,i)
                #plot_outliers(real_drugs_per_admission, synthetic_drugs_per_admission, i,
                #                'Outliers of ' +i+ ' per Admission', 'Number of ' + i, 'Patient Count (Real)','Synthetic')
                wd_real_drugs_per_admission_outliers_syntrain = plot_histograms_separate_axes22(real_out_a[i+'_count'], syn_out_a[i+'_count'], 
                                'Histogram of ' +i+ ' per admission', 'Number of ' + i, 'Admissions Count (Real)','Synthetic',path_img)
            
        
                plot_boxplots(real_drugs_per_admission[i+'_count'], synthetic_drugs_per_admission[i+'_count'], 
                                'Histogram of ' +i+ ' per admission', 'Number of ' + i, 'Admissions Count (Real)',path_img)

                wd_real_drugs_per_admission_syntrain = plot_histograms_separate_axes22(real_drugs_per_admission[i+'_count'], synthetic_drugs_per_admission[i+ '_count'], 
                                'Histogram of ' +i+ ' per admission', 'Number of ' + i, 'Admission Count (Real)','Synthetic',path_img
                            )
                
                        #ADmission per drug #, 
                real_out_a_a = calculate_outlier_ratios_tout2(real_admissions_per_drug,"admission")
                syn_out_a_a = calculate_outlier_ratios_tout2(synthetic_admissions_per_drug,"admission")
                    
                #plot_outliers(real_admissions_per_drug, synthetic_admissions_per_drug, "admission",
                #                'Outliers of   Admission per ' + i, 'Number of ' + i, 'Patient Count (Real)','Synthetic')
                wd_real_admissions_per_drug_outliers_syntrain = plot_histograms_separate_axes22(real_out_a_a["admission"+'_count'], syn_out_a_a["admission"+'_count'], 
                                'Histogram of outliers ' + 'Admission per ' + i, 'Number of ' + i, 'Patient Count (Real)','Synthetic')
                

                wd_real_admissions_per_drug_syntrain = plot_histograms_separate_axes22(real_admissions_per_drug['admission_count'], synthetic_admissions_per_drug['admission_count'], 
                                'Histogram of Admissions per ' + i, 'Number of Admissions ',  i+ ' Count (Real)','Synthetic' )
                
                plot_boxplots(real_admissions_per_drug['admission_count'], synthetic_admissions_per_drug['admission_count'], 
                                'Histogram of Admissions per ' +i, 'Number of Admissions ',  i+ ' Count (Real)' )
                 
                synthetic_train_values.append([wd_real_drugs_per_patient_syntrain,wd_real_drugs_per_admission_syntrain,wd_real_admissions_per_drug_syntrain])
                test_train_values.append([wd_real_drugs_per_patient_esttrain,wd_real_drugs_per_admission_testtrain,wd_real_admissions_per_drug_testtrain])
                synthetic_train_outliers_values.append([wd_real_drugs_per_patient_outlier_syntrain,wd_real_drugs_per_admission_outliers_syntrain,wd_real_admissions_per_drug_outliers_syntrain])
                test_train_outliers_values.append([wd_real_drugs_per_patient_outlier_esttrain,wd_real_drugs_per_admission_outliers_testtrain,wd_real_admissions_per_drug_outliers_esttrain])
        
                aux = pd.DataFrame({
                                'Feature': title_list,
                                'Synthetic/Train': synthetic_train_values[0],
                                'Test/Train': test_train_values[0],
                                'Synthetic/Train (Outliers)': synthetic_train_outliers_values[0],
                                'Test/Train (Outliers)': test_train_outliers_values[0]
                            })
                combined_df = pd.concat([combined_df, aux], ignore_index=True)
                combined_df = combined_df.round(2)
        print(combined_df.to_latex())
       
    def metric_outliers(self,train_ehr_dataset,synthetic_ehr_dataset,cols_sel):
        res_ratio = {}
        #diagnosis
        for word in keywords:
            cols_sel = train_ehr_dataset.filter(like=word).columns
            res_ratio["Ratio outliers "+word] =calculate_outlier_ratios_tout__(train_ehr_dataset, synthetic_ehr_dataset,cols_sel)    
                #categorical var
        
        cols_accounts=[]
        for i in self.categorical_cols:
            cols_f = synthetic_ehr_dataset.filter(like=i, axis=1).columns
            cols_accounts.extend(list(cols_f))
        res_ratio["Ratio outlierrs categorical"] =calculate_outlier_ratios_tout__(train_ehr_dataset, synthetic_ehr_dataset,cols_accounts)    
        
        cols_continuous = self.cols
        res_ratio["Ratio outliers numerica"] =calculate_outlier_ratios_tout__(train_ehr_dataset, synthetic_ehr_dataset,cols_continuous)    
        
        print_latex(res_ratio)
        return res_ratio
    
        if "outliers" in self.list_metric_resemblance:
            outliers_and_histograms_patients_admissions(self.test_ehr_dataset,self.train_ehr_dataset,self.synthetic_ehr_dataset ,self.columnas_test_ehr_dataset,self.top_300_codes,self.synthetic_ehr,self.list_metric_resemblance)
    
    def get_descriptive_statistics(self,synthetic_ehr_dataset,train_ehr_dataset,cols):        
          
        result_syn = descriptive_statistics_matrix(synthetic_ehr_dataset[cols],"")
        result_train = descriptive_statistics_matrix(train_ehr_dataset[cols],"")
        
        print_latex(result_syn)
        print_latex(result_train)
        #result = descriptive_statistics_matrix(test_ehr_dataset[cols])
        result_syn = pd.DataFrame(result_syn,index = [0]) #syntehti
        result_train = pd.DataFrame(result_train,index = [0])
        #result.columns = result1.columns
        result_syn['Source'] = 'Synthetic'
        result_train['Source'] = 'Train'
        data = pd.concat([result_syn, result_train], ignore_index=True)
        # Reemplaza los subrayados en los nombres de las columnas con un espacio vacío
        data.columns = data.columns.str.replace('_', ' ')
        data.set_index('Source', inplace=True) 
        #data = pd.concat([result.reset_index(drop=True), result1.reset_index(drop=True)], axis=1)
        data = data.transpose()
        
        
        data = data.round(2)
        print(data.to_latex())
        return data.to_dict()

    
    def compare_average_trends_recordlen(self,train_ehr_dataset,columnas_test_ehr_dataset,test_ehr_dataset,synthetic_ehr_dataset):
    
        statistics = get_statistics(train_ehr_dataset,columnas_test_ehr_dataset,test_ehr_dataset,synthetic_ehr_dataset)
        print_latex(statistics)
        return statistics
    
    def plot_first_visit(self,train_ehr_dataset,synthetic_ehr_dataset,dependant_fist_visit,path_img):
        cols_accounts = []
        
        for i in dependant_fist_visit:
            cols_f = synthetic_ehr_dataset.filter(like=i, axis=1).columns
            cols_accounts.extend(list(cols_f))
        for i in   cols_accounts:  
             plot_admission_date_histograms(train_ehr_dataset,synthetic_ehr_dataset,i,path_img)
             plot_admission_date_bar_charts2(train_ehr_dataset,synthetic_ehr_dataset,i,path_img)
             

    def plot_means_continuos_variables(self,train_ehr_dataset,synthetic_ehr_dataset,cols,path_img):
        # Assuming real_df and synthetic_df are your dataframes    
        plot_means(train_ehr_dataset, synthetic_ehr_dataset, cols,path_img)

    def plot_kerneldis(self,train_ehr_dataset,synthetic_ehr_dataset,cols,path_img):
            for i in cols: 
                synthetic_desc = synthetic_ehr_dataset[i].describe().T
                train_desc = train_ehr_dataset[i].describe().T
                

                synthetic_desc.columns = [i + "synthetic"]
                train_desc.columns = [i + "train"]
                combined_desc = pd.merge(synthetic_desc, train_desc, left_index=True, right_index=True)
                print(combined_desc.to_latex())

                #plot_kernel_syn(train_ehr_dataset,synthetic_ehr_dataset, i,path_img)
                plot_kernel_wasseteint(train_ehr_dataset,synthetic_ehr_dataset, i,path_img)
                #plot_kde_with_distance(train_ehr_dataset,synthetic_ehr_dataset, i,path_img)
     
    def plot_count_matrix_for_specific_subject(self,train_ehr_dataset,synthetic_ehr_dataset,num_patient):
        #training patietns
        plot_visit_trejetory(   train_ehr_dataset,diagnosis_columns,procedure_columns,medication_columns,"Train",
                             self.num_visit_count 
                             , num_patient ,
                             path_img=self.path_img)
                    
        #Synthetic Dataset
        plot_visit_trejetory(   synthetic_ehr_dataset,diagnosis_columns,procedure_columns,medication_columns,"Synthetic", 
                             self.num_visit_count, 
                              num_patient,
                             path_img=self.path_img)    
        
                

           
    
    def get_top10_different_mean_difference(self,test_ehr_dataset , train_ehr_dataset,synthetic_ehr_dataset,cols):
        cols_ = ['ADMITTIME','HADM_ID','id_patient']
        #synthetic
        train_ehr_dataset_auz = cols_todrop(train_ehr_dataset,cols_)
        synthetic_ehr_dataset_auz = cols_todrop(synthetic_ehr_dataset,cols_)
  
        columns_take_account = [i for i in synthetic_ehr_dataset_auz.columns if i not in self.columns_to_drop+['year', 'month', 'ADMITTIME','id_patient']+cols]
        res = compare_proportions_and_return_dictionary(train_ehr_dataset_auz[columns_take_account], synthetic_ehr_dataset_auz[columns_take_account])    
        top_10_diff_proportions_syn = dict(sorted(res.items(), key=lambda item: item[1], reverse=True)[:10])
        #real
        train_ehr_dataset_auz = train_ehr_dataset_auz[list(top_10_diff_proportions_syn.keys())]
        test_ehr_dataset =test_ehr_dataset[train_ehr_dataset_auz.columns]
        test_ehr_dataset = test_ehr_dataset[:train_ehr_dataset_auz.shape[0]]
        
        res2 = compare_proportions_and_return_dictionary(train_ehr_dataset_auz, test_ehr_dataset)    
        top_10_diff_proportions_test_real = dict(sorted(res2.items(), key=lambda item: item[1], reverse=True)[:10])
     
        
        data = concat_to_dict_tolatex_syn_vs_real(top_10_diff_proportions_syn, top_10_diff_proportions_test_real )
        
        cols_to_consider = list(top_10_diff_proportions_test_real.keys())
        #calcular means
        real_means = calculate_means(train_ehr_dataset_auz, cols_to_consider)
        
        synthetic_means = calculate_means(synthetic_ehr_dataset, cols_to_consider)
        data2 = concat_to_dict_tolatex_syn_vs_real(synthetic_means, real_means )
        
        #calcular no 0
        
        non_zero_counts_real = proportion_non_zeros(train_ehr_dataset, cols_to_consider)
        non_zero_counts_synthetic = proportion_non_zeros(synthetic_ehr_dataset, cols_to_consider)
        data3 = concat_to_dict_tolatex_syn_vs_real(non_zero_counts_synthetic, non_zero_counts_real )
        
        return data

    def get_visit_value_counts(self,train_ehr_dataset,synthetic_ehr_dataset,columnas_test_ehr_dataset):
  
        df = train_ehr_dataset[columnas_test_ehr_dataset+["visit_rank"]]
        type_d = "train"
        res_train = df.groupby(["visit_rank"]).count().iloc[:,0].to_dict()
        res_train = {f"{key} set": value for key, value in res_train.items()}
         
        df = synthetic_ehr_dataset[columnas_test_ehr_dataset+["visit_rank"]]
        type_d = "synthetic"
        res2_synthetic = df.groupby(["visit_rank"]).count().iloc[:,0].to_dict()
        res2_synthetic = {f"{key} set": value for key, value in res2_synthetic.items()}
        
        result_syn = pd.DataFrame(res2_synthetic,index = [0]) #syntehti
        result_train = pd.DataFrame(res_train,index = [0])
        #esult_syn.columns = result_train.columns
        result_syn['Source'] = 'Synthetic'
        result_train['Source'] = 'Train'
        data = pd.concat([result_syn, result_train], axis=0)
        # Reemplaza los subrayados en los nombres de las columnas con un espacio vacío
        #data.columns = data.columns.str.replace('_', ' ')
        data.set_index('Source', inplace=True) 
  
        data = data.transpose()
        la = data.to_latex()
        print(la)
        res_train.update(res2_synthetic)
        return res_train  
    def plot_differen_correlation(self,synthetic_ehr_dataset,train_ehr_dataset,cols,categorical_cols,keywords,path_img,all_data_heatmap_diff=False,corr_plot_syn_real=False,corr_plot_continous = True,corr_plot_categotical =True,corr_plot_codes=True):

        # Example usage:
        if all_data_heatmap_diff:
           heatmap_diff_corr(synthetic_ehr_dataset, train_ehr_dataset,path_img)
        if corr_plot_syn_real:
            corr_plot(synthetic_ehr_dataset,"Syn" ,path_img)
            corr_plot(train_ehr_dataset,"Train" ,path_img)  
            #continous cols
        if corr_plot_continous:
            syn_c = corr_plot(synthetic_ehr_dataset[cols],"Syn" ,path_img)
            real_c = corr_plot(train_ehr_dataset[cols],"Train" ,path_img)
            heatmap_diff_corr(syn_c,real_c,path_img)
        
        if corr_plot_categotical:    
            cols_list = []
            categorical_cols = categorical_cols
            for i in categorical_cols:
                cols_f = synthetic_ehr_dataset.filter(like=i, axis=1).columns
                cols_list.extend(list(cols_f))
            syn = corr_plot(synthetic_ehr_dataset[cols_list],"Syn" ,path_img)
            #cols_with_high_corr = correlacion_otra_col(synthetic_ehr_dataset[cols_list])
            real = corr_plot(train_ehr_dataset[cols_list],"Train" ,path_img   )
            #diferencia de correlationes
            heatmap_diff_corr(syn,real,self.path_img)
            #correlacion_otra_col(synthetic_ehr_dataset[cols_list])
        
        if corr_plot_codes:    
            keywords = keywords
            for i in keywords:
                col_prod = [col for col in train_ehr_dataset.columns if any(palabra in col for palabra in [i])]
                syn1 = corr_plot(synthetic_ehr_dataset[col_prod],"Syn" ,path_img )
                #cols_with_high_corr, cols_with_all_nan =correlacion_total(synthetic_ehr_dataset[col_prod])
                real1 =corr_plot(train_ehr_dataset[col_prod],"Train" ,path_img )    
                #cols_with_high_corr, cols_with_all_nan = correlacion_total(train_ehr_dataset[col_prod])
                y_labels = heatmap_diff_corr(syn1,real1,path_img )
                print(y_labels)
            
    def plot_pacmap(self,synthetic_ehr_dataset,train_ehr_dataset,keywords,categorical_cols,path_img ):    

                   
        keywords = keywords
        for i in keywords:
            col_prod = [col for col in train_ehr_dataset.columns if any(palabra in col for palabra in [i])]
            PACMAP_PLOT(col_prod,synthetic_ehr_dataset,train_ehr_dataset,i,False,path_img )
        cols_list = []
        
        for i in categorical_cols:
            cols_f = train_ehr_dataset.filter(like=i, axis=1).columns
            cols_list.extend(list(cols_f))
        PACMAP_PLOT(cols_list,synthetic_ehr_dataset,train_ehr_dataset,"Demographic and admission",path_img )    

    
        PACMAP_PLOT(cols_list,synthetic_ehr_dataset,train_ehr_dataset,"Continuos variables",path_img )    
    
    def temporal_histogram_heatmap(self,synthetic_ehr_dataset,train_ehr_dataset,path_img ):    
                
        name = "Days between visits"
        col =   'days from last visit_bins'
        synthetic_ehr_dataset['days from last visit_bins'] = pd.qcut(synthetic_ehr_dataset['days from last visit'], q=2, duplicates='drop')
        hist_d("days from last visit",synthetic_ehr_dataset,path_img ) 
        plot_heatmap_(synthetic_ehr_dataset, name,col,"Synthetic",path_img)
        train_ehr_dataset['days from last visit_bins'] = pd.qcut(train_ehr_dataset['days from last visit'], q=2, duplicates='drop')
        hist_d("days from last visit",train_ehr_dataset,path_img) 
        hist_betw_a(train_ehr_dataset,synthetic_ehr_dataset,"days from last visit",path_img)
        hist_betw_a(train_ehr_dataset,synthetic_ehr_dataset,"id_patient",path_img)
        
        plot_heatmap_(train_ehr_dataset, name,col, "Real",path_img)
        # age
        name = "Age interval"
        col =   'Age_bins'
        synthetic_ehr_dataset['Age_bins'] = pd.qcut(synthetic_ehr_dataset['Age'], q=5, duplicates='drop')
        hist_d("Age",synthetic_ehr_dataset) 
        plot_heatmap_(synthetic_ehr_dataset, name,col, "Synthetic",path_img)
        train_ehr_dataset['Age_bins'] = pd.qcut(train_ehr_dataset['Age'], q=5, duplicates='drop')
        plot_heatmap_(train_ehr_dataset, name,col,"Real",path_img)
        hist_d("days from last visit",train_ehr_dataset) 
        
        
        # Histograms
        for col in self.cols_continuous:
            box_pltos(train_ehr_dataset,synthetic_ehr_dataset,col,path_img)
        keywords = self.keywords
        for i in keywords:
            col_prod = [col for col in train_ehr_dataset.columns if any(palabra in col for palabra in [i])]
            plot_heatmap(train_ehr_dataset, i,i, 2,col_prod,"Real",path_img)
            plot_heatmap(synthetic_ehr_dataset, i,i, 2,col_prod,"Synthetic",path_img)
            box_pltos(train_ehr_dataset,synthetic_ehr_dataset,col_prod,path_img)

    
    
    def compare_maximum_range(self,train_ehr_dataset,synthetic_ehr_dataset):
        res_ratio = {}
        for word in self.keywords:
            cols_sel = train_ehr_dataset.filter(like=word).columns
            res =compare_data_ranges(train_ehr_dataset, synthetic_ehr_dataset,cols_sel,word)    
            res_ratio.update(res)
                #categorical var
        
        # cols_accounts=[]
        # for i in self.categorical_cols:
        #     cols_f = train_ehr_dataset.filter(like=i, axis=1).columns
        #     cols_accounts.extend(list(cols_f))
        # res2 = compare_data_ranges(train_ehr_dataset, synthetic_ehr_dataset,cols_accounts,"categorical")  
          
        # res_ratio.update(res2)
        data = pd.DataFrame(res_ratio)
        print(data.to_latex())
        
        return res_ratio

    def plot_dimension_wise(self,synthetic_ehr_dataset,train_ehr_dataset,cols_continuous,categorical_cols,cols_diagnosis,cols_drugs,cols_procedures,path_img ):    
        gen_samples = synthetic_ehr_dataset.select_dtypes(include=['number'])  
        real_samples = train_ehr_dataset.select_dtypes(include=['number']) 
        gen_columns = set(gen_samples.columns)
        real_columns = set(real_samples.columns)

        # Encuentra la intersección de las dos listas de columnas
        common_columns = gen_columns & real_columns

        # Crea nuevos DataFrames que solo contienen las columnas comunes
        gen_samples_common = gen_samples[common_columns]
        real_samples_common = real_samples[common_columns]
        if gen_samples_common.shape[0] > real_samples_common.shape[0]:
            gen_samples_common = gen_samples_common[:real_samples_common.shape[0]]
        else:
            real_samples_common = real_samples_common[:gen_samples_common.shape[0]]
        #categorical
        cols_cat = filter_cols(categorical_cols,real_samples_common)        
        dimensio_wise(gen_samples_common[cols_cat],real_samples_common[cols_cat], " categorical columns",path_img)
        #numerical
               
        dimensio_wise(gen_samples_common[cols_continuous],real_samples_common[cols_continuous], " continous columns",path_img)
       
       #count codes
        cols_s =cols_diagnosis.to_list()+cols_drugs.to_list()+cols_procedures.to_list()
        dimensio_wise(gen_samples_common[cols_s],real_samples_common[cols_s]," all IC9-codes and drugs",path_img)
        dimensio_wise(gen_samples_common[cols_diagnosis],real_samples_common[cols_diagnosis]," diagnosis",path_img)
        dimensio_wise(gen_samples_common[cols_drugs],real_samples_common[cols_drugs]," drugs",path_img)
        dimensio_wise(gen_samples_common[cols_procedures],real_samples_common[cols_procedures]," procedures",path_img)

    def plot_prevalence_wise(self,synthetic_ehr_dataset, train_ehr_dataset,  cols_diagnosis, cols_drugs, cols_procedures, path_img):
        # Calcular MAD para variables de conteo
        diagnosis_features = cols_diagnosis.to_list()
        drug_features = cols_drugs.to_list()
        procedure_features = cols_procedures.to_list()
        
        real_diagnosis_frequencies = calculate_relative_frequencies(train_ehr_dataset[diagnosis_features])
        synthetic_diagnosis_frequencies = calculate_relative_frequencies(synthetic_ehr_dataset[diagnosis_features])
        #mad_diagnosis = calculate_mean_absolute_difference(real_diagnosis_frequencies, synthetic_diagnosis_frequencies)
        
        real_drug_frequencies = calculate_relative_frequencies(train_ehr_dataset[drug_features])
        synthetic_drug_frequencies = calculate_relative_frequencies(synthetic_ehr_dataset[drug_features])
        #mad_drugs = calculate_mean_absolute_difference(real_drug_frequencies, synthetic_drug_frequencies)
        
        real_procedure_frequencies = calculate_relative_frequencies(train_ehr_dataset[procedure_features])
        synthetic_procedure_frequencies = calculate_relative_frequencies(synthetic_ehr_dataset[procedure_features])
        #mad_procedures = calculate_mean_absolute_difference(real_procedure_frequencies, synthetic_procedure_frequencies)
        
        # Crear DataFrame combinado para el gráfico
        plot_data = pd.DataFrame({
            'Real': pd.concat([real_diagnosis_frequencies, real_drug_frequencies, real_procedure_frequencies]),
            'Synthetic': pd.concat([synthetic_diagnosis_frequencies, synthetic_drug_frequencies, synthetic_procedure_frequencies]),
            'Feature': (
                ['Diagnosis'] * len(diagnosis_features) +
                ['Drugs'] * len(drug_features) +
                ['Procedures'] * len(procedure_features)
            )
        })
        
        def find_high_proportion_features(plot_data, threshold=0.6):
            # Filter rows where either Real or Synthetic proportion is greater than the threshold
            high_proportion = plot_data[(plot_data['Real'] > threshold) | (plot_data['Synthetic'] > threshold)]
            
            # Create a list of tuples containing the feature, its type, and proportions
            high_proportion_list = []
            for index, row in high_proportion.iterrows():
                high_proportion_list.append((
                    index,  # This is the feature name
                    row['Feature'],
                    f"Real: {row['Real']:.2f}",
                    f"Synthetic: {row['Synthetic']:.2f}"
                ))
            
            return high_proportion_list
        high_proportion_list = find_high_proportion_features(plot_data, threshold=0.6)
        cols_to_consider =self.cols_diagnosis.to_list() +self.cols_drugs.to_list()
        result =analyze_diagnosis_frequency(self.train_ehr_dataset[cols_to_consider], high_proportion_list)
        for i in high_proportion_list:
            print(i)
        print(result)  
        for drug, data in result.items():
            print(f"Top diagnoses for {drug}:")
            for diagnosis, count in data['top_diagnoses']:
                print(f"  - {diagnosis}: {count}")
      #plot_data = remove_outliers(plot_data, 'Real')
        #plot_data = remove_outliers(plot_data, 'Synthetic')

        # Encontrar los límites máximos para ajustar los ejes
        max_real = plot_data['Real'].max()
        max_synthetic = plot_data['Synthetic'].max()
        axis_limit = max(max_real, max_synthetic) * 1.1  # añadir un 10% de margen

        # Crear el gráfico de dispersión combinado
        plt.figure(figsize=(10, 10))
        sns.scatterplot(data=plot_data, x='Real', y='Synthetic', hue='Feature', style='Feature', s=100)

        # Añadir línea de referencia diagonal
        plt.plot([0, axis_limit], [0, axis_limit], ls="--", c=".3")

        # Añadir los valores de MAD al gráfico
        # plt.text(0.05, axis_limit * 0.95, f'MAD (Diagnosis) = {mad_diagnosis:.2f}', horizontalalignment='left', size='medium', color='black', weight='semibold')
        # plt.text(0.05, axis_limit * 0.90, f'MAD (Drugs) = {mad_drugs:.2f}', horizontalalignment='left', size='medium', color='black', weight='semibold')
        # plt.text(0.05, axis_limit * 0.85, f'MAD (Procedures) = {mad_procedures:.2f}', horizontalalignment='left', size='medium', color='black', weight='semibold')

        # Ajustar los límites de los ejes
        plt.xlim(0, axis_limit)
        plt.ylim(0, axis_limit)

        # Añadir títulos y etiquetas
        plt.title('Comparison of Proportions of Occurrences  in  Real and Synthetic Medical Codes')
        plt.xlabel('Real Proportion of Occurrences in Medical Codes')
        plt.ylabel('Synthetic Proportion of Occurrences in Medical Codes')

        # Ajustar la leyenda para que quede dentro del gráfico
        plt.legend(title='Medical codes', loc='upper left', bbox_to_anchor=(0.75, 0.75))
        plt.tight_layout()
        plt.show()

        if path_img is not None:
            save_plot_as_svg(plt, path_img, "Comparison_prevalence_synthetic")    
       
    # # def plot_prevalence_wise(self,synthetic_ehr_dataset,train_ehr_dataset,cols_continuous,categorical_cols,cols_diagnosis,cols_drugs,cols_procedures,path_img = None):
    # #     # Función para calcular la prevalencia de características binarias
    # #     # Calcular APD para características binarias
    # #     cols_to_filter = categorical_cols +['readmission','HOSPITAL_EXPIRE_FLAG']
    # #     binary_features = filter_cols(cols_to_filter,train_ehr_dataset)
    # #     real_binary_prevalences = calculate_binary_prevalence(train_ehr_dataset[binary_features])
    # #     synthetic_binary_prevalences = calculate_binary_prevalence(synthetic_ehr_dataset[binary_features])
    # #     apd_binary = calculate_mean_absolute_difference(real_binary_prevalences, synthetic_binary_prevalences)

    # #     # Calcular MAD para variables de conteo
    # #     count_features = cols_diagnosis.to_list() +cols_drugs.to_list()+cols_procedures.to_list() +['visit_rank']
    # #     real_count_frequencies = calculate_relative_frequencies(train_ehr_dataset[count_features])
    # #     synthetic_count_frequencies = calculate_relative_frequencies(synthetic_ehr_dataset[count_features])
    # #     mad_count = calculate_mean_absolute_difference(real_count_frequencies, synthetic_count_frequencies)

    # #     # Calcular AWD para variables continuas
    # #     continuous_features = cols_continuous
    # #     normalized_real_data = normalize_features(train_ehr_dataset[continuous_features])
    # #     normalized_synthetic_data = normalize_features(synthetic_ehr_dataset[continuous_features])
    # #     awd_continuous = calculate_awd(normalized_real_data, normalized_synthetic_data)

    # #     # Crear DataFrame combinado para el gráfico
    # #     plot_data = pd.DataFrame({
    # #         'Real': pd.concat([real_binary_prevalences, real_count_frequencies, normalized_real_data.mean()]),
    # #         'Synthetic': pd.concat([synthetic_binary_prevalences, synthetic_count_frequencies, normalized_synthetic_data.mean()]),
    # #         'Feature': ['Binary'] * len(real_binary_prevalences) + ['Count'] * len(real_count_frequencies) + ['Continuous'] * len(normalized_real_data.columns)
    # #     })

    #     plot_data = remove_outliers(plot_data, 'Real')
    #     plot_data = remove_outliers(plot_data, 'Synthetic')
    #     index_to_drop = ["id_patient","HADM_ID","days from last visit","Age","LOSRD_avg"]
        
    #     #plot_data = plot_data.drop(["days from last visit", "Age", "LOSRD_avg"])

        
        
    #     # Encontrar los límites máximos para ajustar los ejes
    #     max_real = plot_data['Real'].max()
    #     max_synthetic = plot_data['Synthetic'].max()
    #     axis_limit = max(max_real, max_synthetic) * 1.1  # añadir un 10% de margen

    #     # Crear el gráfico de dispersión combinado
    #     plt.figure(figsize=(10, 10))
    #     sns.scatterplot(data=plot_data, x='Real', y='Synthetic', hue='Feature', style='Feature', s=100)

    #     # Añadir línea de referencia diagonal
    #     plt.plot([0, 1], [0, 1], ls="--", c=".3")

    #     # Añadir los valores de APD, MAD y AWD al gráfico
    #     plt.text(0.05, axis_limit * 0.95, f'APD (Binary) = {apd_binary:.2f}', horizontalalignment='left', size='medium', color='black', weight='semibold')
    #     plt.text(0.05, axis_limit * 0.90, f'MAD (Count) = {mad_count:.2f}', horizontalalignment='left', size='medium', color='black', weight='semibold')
    #     plt.text(0.05, axis_limit * 0.85, f'AWD (Continuous) = {awd_continuous:.2f}', horizontalalignment='left', size='medium', color='black', weight='semibold')

    #     # Ajustar los límites de los ejes
    #     plt.xlim(0, axis_limit)
    #     plt.ylim(0, axis_limit)

    #     # Añadir títulos y etiquetas
    #     plt.title('Comparison of Feature Prevalence in Real and Synthetic Data')
    #     plt.xlabel('Real Prevalence')
    #     plt.ylabel('Synthetic Prevalence')
    #     plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.show()
    #     if path_img != None:
    #         save_plot_as_svg(plt, path_img, "Comparison_prevalence_synthetic")
        
   

    def compare_descriptive_stadistics(self,train_ehr_dataset,synthetic_ehr_dataset,cols):
    
        dif =  compare_descriptive_statistics_fun(train_ehr_dataset[cols], synthetic_ehr_dataset[cols])
        top_10_differences = sorted(dif.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        top_10_dict = {item[0]: item[1] for item in top_10_differences}
        top_10_dict = pd.DataFrame(top_10_dict,index = [0]).T
        print(top_10_dict.to_latex())
        
        return top_10_dict
    
    def get_MaximumMeanDiscrepancy(self,train_ehr_dataset,synthetic_ehr_dataset):
        
        mmd_evaluator = MaximumMeanDiscrepancy(kernel="rbf")
        train_test = "test"
        cols = ['ADMITTIME','HADM_ID']
        #cols = "days from last visit_cumsum"
        train_ehr_dataset, synthetic_ehr_dataset = filter_and_equalize_datasets(train_ehr_dataset, synthetic_ehr_dataset)
          
        #train_ehr_dataset_a = cols_todrop(train_ehr_dataset,cols)
        
        result =   mmd_evaluator._evaluate(train_ehr_dataset, synthetic_ehr_dataset)
        print("Maximum Mean Discrepancy (flattened):", result)
        print_latex(result)
        return result

    # Example usage:
    # X_gt and X_syn are two numpy arrays representing empirical distributions
    def kolmogorov_smirnof_test_chissquare(self,train_ehr_dataset,synthetic_ehr_dataset,cols_continuous):

        
        continuous_cols = cols_continuous     
        ks_results = {'column': [], 'ks_statistic': [], 'p_value': []}

        for col in continuous_cols:
            ks_stat, p_value = ks_2samp(train_ehr_dataset[col], synthetic_ehr_dataset[col])
            ks_results['column'].append(col)
            ks_results['ks_statistic'].append(ks_stat)
            ks_results['p_value'].append(p_value)

        # Convert results to DataFrame
        ks_df = pd.DataFrame(ks_results)

        # Calculate summary statistics
        summary_stats = {
            'mean_ks_statistic': ks_df['ks_statistic'].mean(),
            'median_ks_statistic': ks_df['ks_statistic'].median(),
            'std_ks_statistic': ks_df['ks_statistic'].std(),
            'mean_p_value': ks_df['p_value'].mean(),
            'median_p_value': ks_df['p_value'].median(),
            'std_p_value': ks_df['p_value'].std(),
            'proportion_significant': (ks_df['p_value'] < 0.05).mean()
        }

        summary_df = pd.DataFrame([summary_stats])

        print(ks_df.to_latex())
        from scipy.stats import chi2_contingency
        real_data = train_ehr_dataset
        synthetic_data = synthetic_ehr_dataset
        real_data,synthetic_data=filter_and_equalize_datasets(real_data,synthetic_data)
        cols_not_con = [i for i in synthetic_data.columns if i not in continuous_cols+['id_patient','ADMITTIME']]
        
        # Generate example one-hot encoded data for illustration purposes

        # Perform Chi-Square test for each column and calculate Total Variation Distance
        chi_results = {'column': [], 'chi2_statistic': [], 'p_value': [], 'tv_distance': []}

        for col in cols_not_con:
            contingency_table = pd.crosstab(real_data[col], synthetic_data[col])
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
            
            # Calculate Total Variation Distance
            real_prob = real_data[col].value_counts(normalize=True)
            synthetic_prob = synthetic_data[col].value_counts(normalize=True)
            tv_distance = np.sum(np.abs(real_prob - synthetic_prob)) / 2
            
            chi_results['column'].append(col)
            chi_results['chi2_statistic'].append(chi2_stat)
            chi_results['p_value'].append(p_value)
            chi_results['tv_distance'].append(tv_distance)

        # Convert results to DataFrame
        chi_df = pd.DataFrame(chi_results)
        # Identify the top 10 most different and least different columns by p-value


        most_different_cols = chi_df.nsmallest(10, 'p_value')
        least_different_cols = chi_df.nlargest(10, 'p_value')
        most_different_cols_tv = chi_df.nlargest(10, 'tv_distance')
        least_different_cols_tv = chi_df.nsmallest(10, 'tv_distance')

        # dic_results = {
        #     'Most Different by p-value': most_different_cols['column'].tolist() + [''] * (10 - len(most_different_cols)),
        #     'Least Different by p-value': least_different_cols['column'].tolist() + [''] * (10 - len(least_different_cols)),
        #     'Most Different by TV Distance': most_different_cols_tv['column'].tolist() + [''] * (10 - len(most_different_cols_tv)),
        #     'Least Different by TV Distance': least_different_cols_tv['column'].tolist() + [''] * (10 - len(least_different_cols_tv)),
        # }
        # # Create the final DataFrame
        
        dic_results = {
             'Most Different by TV Distance': most_different_cols_tv['column'].tolist() + [''] * (10 - len(most_different_cols_tv)),
            'Least Different by TV Distance': least_different_cols_tv['column'].tolist() + [''] * (10 - len(least_different_cols_tv)),
        }
        
        final_df = pd.DataFrame(dic_results)

        # Display the DataFrame
        print(final_df.to_latex())
        # Combine the results
        return  dic_results
    def get_jensenshannon_dist( self,train_ehr_dataset,synthetic_ehr_dataset,test_ehr_dataset,cols_continuous):
        results = pd.DataFrame()
        #Train/Syn

        cols_ = ['ADMITTIME','HADM_ID','id_patient']
        #synthetic
        train_ehr_dataset = cols_todrop(train_ehr_dataset,cols_)
        synthetic_ehr_dataset = cols_todrop(synthetic_ehr_dataset,cols_)
        test_ehr_dataset = cols_todrop(test_ehr_dataset,cols_)
  
    
        train_ehr_dataset, synthetic_ehr_dataset = filter_and_equalize_datasets(train_ehr_dataset, synthetic_ehr_dataset)
        jsd_calculator = JensenShannonDistance2(train_ehr_dataset, synthetic_ehr_dataset, 10)
        jsd_results_train_syn, avg_js_train_syn = jsd_calculator.jensen_shannon()
        
        #results["train_syn"] = jsd_results_train_syn
        print_latex(jsd_results_train_syn)
        jsd_results_train_syn.columns = [f"{col} Train/Syn" for col in jsd_results_train_syn.columns]


        # Test/Syn
        test_ehr_dataset, synthetic_ehr_dataset = filter_and_equalize_datasets(test_ehr_dataset, synthetic_ehr_dataset)
        jsd_calculator = JensenShannonDistance2(test_ehr_dataset, synthetic_ehr_dataset, 10)
        jsd_results_test_syn, avg_js_test_syn = jsd_calculator.jensen_shannon()
        
        #results["test_syn"] = jsd_results_test_syn
        print_latex(jsd_results_test_syn)
        jsd_results_test_syn.columns = [f"{col} Test/Syn" for col in jsd_results_test_syn.columns]

        # Test/Train
        test_ehr_dataset, train_ehr_dataset = filter_and_equalize_datasets(test_ehr_dataset, train_ehr_dataset)
        jsd_calculator = JensenShannonDistance2(test_ehr_dataset, train_ehr_dataset, 10)
        jsd_results_test_train, avg_js_test_train = jsd_calculator.jensen_shannon()
        
        #results["train_test"] = jsd_results_test_train
        print_latex(jsd_results_test_train)
        jsd_results_test_train.columns = [f"{col} Test / Train" for col in jsd_results_test_train.columns]

        df_final = pd.concat([jsd_results_train_syn, jsd_results_test_syn, jsd_results_test_train], axis=1)
        print_latex(df_final)
        print(df_final.to_latex())
        return df_final.to_dict()
        
        



    # Example usage with two dataframes, `X_gt_df` and `X_syn_df`

                    
    
   
    def get_proportion_demos(self,train_ehr_dataset,synthetic_ehr_dataset,categorical_cols):    
        prop_datafram = []
        categorical_cols = categorical_cols
        for i in categorical_cols:
            cols_f = synthetic_ehr_dataset.filter(like=i, axis=1).columns
            #col_name = limpiar_col(i)
            
            tabla_proporciones_final = get_proportions(train_ehr_dataset[cols_f],"train "+ i)
            tabla_proporciones_final_2= get_proportions(synthetic_ehr_dataset[cols_f],"synthetic "+i)
            #plot_pie_proportions(cols_f, tabla_proporciones_final, "Train data")
            #plot_pie_proportions(cols_f, tabla_proporciones_final_2, "Synthetic data")
            total = pd.merge(tabla_proporciones_final,tabla_proporciones_final_2,  on=["Variable","Category "], how='inner') 
            #total = total['Variable','Category ]+test_ehr_dataset[cols_f],"test "+ i]
            prop_datafram.append(total)
            total = total.round(5)
            latex_code = total.to_latex( index = "Variable")
            if i == "visit_rank":
               plot_vistis(    total )
               total.drop(columns= "Variable",inplace = True)
               total.columns = [col.replace('visit_rank', '') for col in total.columns]
               general_l = latex_general(total,"Count number of visist and proportioncs between real and synthetic data","tab:visit_counts_1",integer_columns = ["Category ","Count train ","Count synthetic "],num_decimales =5 )
               print(general_l)
            print(latex_code)
            general_l = latex_general(total,"Count number of visist and proportioncs between real and synthetic data","tab:visit_counts_1",num_decimales =5)
            
            print(general_l)
        return total.to_dict()  
    def get_excat_match(self,train_ehr_dataset,synthetic_ehr_dataset):
        train_ehr_dataset, synthetic_ehr_dataset =filter_and_equalize_datasets(train_ehr_dataset, synthetic_ehr_dataset)
        df1_sorted = synthetic_ehr_dataset.sort_values(by=synthetic_ehr_dataset.columns.tolist()).reset_index(drop=True)  
        df2_sorted = train_ehr_dataset.sort_values(by=train_ehr_dataset.columns.tolist()).reset_index(drop=True)
        exact_matches = (df1_sorted == df2_sorted).all(axis=1).sum()
        print("Number of exact matching rows:", exact_matches)
        print_latex({"exact_matches" : exact_matches} )
        return {"exact_matches" : exact_matches} 
    
    def get_dupplicates(self,   train_ehr_dataset,synthetic_ehr_dataset):    
        combined = pd.concat([train_ehr_dataset, synthetic_ehr_dataset])
        synthetic_ehr_dataset['is_duplicate'] = combined.duplicated(keep=False).iloc[len(train_ehr_dataset):]

        print("Test data with duplicate flag:")
        value_counts = synthetic_ehr_dataset['is_duplicate'].value_counts()
        new_dict = {"Duplicates - " + str(key): value for key, value in value_counts.items()}
                     
        print_latex(new_dict)
        return new_dict

    def obtain_least_frequent(self,train_ehr_dataset,synthetic_ehr_dataset, columnas_test_ehr_dataset, n):    
        least_cols =  obtain_least_frequent_fun(train_ehr_dataset, columnas_test_ehr_dataset, 10)
        

# Columns to analyze
        columns_to_analyze =least_cols

        # Function to calculate proportions for a given column in a DataFrame
        
        # Step 3: Create a DataFrame for plotting
        plot_data = {}

        for column in columns_to_analyze:
            real_props = calculate_proportions(train_ehr_dataset, column)
            synth_props = calculate_proportions(synthetic_ehr_dataset, column)
            # Prepare data for plotting
            for category in set(real_props.index).union(set(synth_props.index)):
                real_prop = real_props.get(category, 0)
                synth_prop = synth_props.get(category, 0)
                plot_data[(column, category)] = [real_prop, synth_prop]

        # Convert plot data into a DataFrame
        plot_df = pd.DataFrame.from_dict(plot_data, orient='index', columns=['Real', 'Synthetic']).reset_index()
        print(plot_df.to_latex())
        return plot_df.to_dict()
    ### TODO debugerar col     
    def plot_acumulated(self,   train_ehr_dataset,synthetic_ehr_dataset,cols_continuous,categorical_cols,path_img):    
        import matplotlib.pyplot as plt

        categorilca_cols_fun(train_ehr_dataset,synthetic_ehr_dataset,categorical_cols,False,path_img)
        categorilca_cols_fun(train_ehr_dataset,synthetic_ehr_dataset,cols_continuous,False,path_img)
        keywords = self.keywords
        categorilca_cols_fun(train_ehr_dataset,synthetic_ehr_dataset,keywords,False,path_img)
        
    def get_common_proportions(self,train_ehr_dataset,synthetic_ehr_dataset):
        train_ehr_dataset,synthetic_ehr_dataset = same_size_synthetic(    train_ehr_dataset,synthetic_ehr_dataset)
        common_columns = set(train_ehr_dataset.columns).intersection(set(synthetic_ehr_dataset.columns))

        train_ehr_dataset = train_ehr_dataset[common_columns]
        synthetic_ehr_dataset = synthetic_ehr_dataset[common_columns]   
        cp = CommonRowsProportion()
        dict_s = cp.evaluate(train_ehr_dataset, synthetic_ehr_dataset)
        print_latex(dict_s)
        return dict_s

