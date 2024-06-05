import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from pacmap import PaCMAP

from scipy.stats import wasserstein_distance
#os.chdir("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning")
os.chdir("/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/")
import sys
sys.path.append('evaluation/resemb/resemblance/utils_stats/')
sys.path.append('evaluation')

sys.path.append('/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/')
from evaluation.resemb.resemblance.utilsstats import *

ruta_actual = os.getcwd()
print(ruta_actual)

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
                 dependant_fist_visit):
        
        self.test_ehr_dataset = test_ehr_dataset
        self.train_ehr_dataset = train_ehr_dataset
        self.synthetic_ehr_dataset = synthetic_ehr_dataset
        self.columnas_test_ehr_dataset = columnas_test_ehr_dataset
        self.top_300_codes = top_300_codes
        
        self.list_metric_resemblance = list_metric_resemblance
        self.result_resemblance = []
        self.results_final = {}
        self.cols = cols
        self.keywords = keywords
        self.categorical_cols = categorical_cols
        self.dependant_fist_visit = dependant_fist_visit
        if "outliers" in self.list_metric_resemblance:
            self.outliers_and_histograms_patients_admissions()

        if "descriptive_statistics" in self.list_metric_resemblance:
            self.get_descriptive_statistics()

        if "Record_lengh" in self.list_metric_resemblance:
            self.compare_average_trends_recordlen()

        if "plt_hist_first_visits" in self.list_metric_resemblance:
            self.plot_first_visit()

        if "cols_plot_mean" in self.list_metric_resemblance:
            self.plot_means_continuos_variables()

        if "plot_kernel_vssyn" in self.list_metric_resemblance:
            self.plot_kerneldis()

        if "frequency_categorical_10" in self.list_metric_resemblance:
            self.get_top10_different_proportion()

        if "visit_counts2" in self.list_metric_resemblance:
            self.get_visit_value_counts()

        if "corr" in self.list_metric_resemblance:
            self.plot_differen_correlation()

        if "pacmap" in self.list_metric_resemblance:
            self.plot_pacmap()

        if "temporal_time_line" in self.list_metric_resemblance:
            self.temporal_histogram_heatmap()

        if "compare_ranges" in self.list_metric_resemblance:
            self.compare_maximum_range()

        if "dimension_wise" in self.list_metric_resemblance:
            self.plot_dimension_wise()

        if "prevalence_wise" in self.list_metric_resemblance:
            self.plot_prevalence_wise()

        if "diference_decriptives" in self.list_metric_resemblance:
            self.top_10diference_absbalute()

        if "mmd" in self.list_metric_resemblance:
            self.get_MaximumMeanDiscrepancy()

        if "ks_test" in self.list_metric_resemblance:
            self.kolmogorov_smirnof_test_chissquare()

        if "jensenshannon_dist" in self.list_metric_resemblance:
            self.get_jensenshannon_dist()

        if "proportion_demos" in self.list_metric_resemblance:
            self.get_proportion_demos()

        if "exact_match" in self.list_metric_resemblance:
            self.get_excat_match()

        if "duplicates" in self.list_metric_resemblance:
            self.get_dupplicates()

        if "compare_least_codes" in self.list_metric_resemblance:
            self.obtain_least_frequent()

        if "gradico_acum" in self.list_metric_resemblance:
            self.plot_acumulated()

        if "common_proportoins" in self.list_metric_resemblance:
            self.get_common_proportions()


        
        #outliers
        
        

    def outliers_and_histograms_patients_admissions(self,test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset ,columnas_test_ehr_dataset,top_300_codes,synthetic_ehr,list_metric_resemblance):
        for i in keywords[:1]:
                # Shynthetic/train
                
                test_ehr_dataset_t = train_ehr_dataset[:test_ehr_dataset.shape[0]]
                real_drugs_per_patient_t, real_drugs_per_admission_t, real_admissions_per_drug_t,real_patients_per_drug_t = calculate_counts(train_ehr_dataset,i)
                synthetic_drugs_per_patient_t, synthetic_drugs_per_admission_t, synthetic_admissions_per_drug_t,synthetic_patients_per_drug = calculate_counts(test_ehr_dataset_t,i)
                #, 'Patient Count (Synthetic)'
                real_out_t = calculate_outlier_ratios_tout2(real_drugs_per_patient_t,i)
                syn_out_t = calculate_outlier_ratios_tout2(synthetic_drugs_per_patient_t,i)
                plot_outliers(real_drugs_per_patient_t, synthetic_drugs_per_patient_t, i,
                                'Outliers of ' +i+ ' per Patient', 'Number of ' + i, 'Patient Count (Train)','Test')
                plot_histograms_separate_axes22(real_out_t[i+'_count'], syn_out_t[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of ' + i, 'Patient Count (Train)','Test')
                
                plot_histograms_separate_axes22(real_drugs_per_patient_t[i+'_count'], synthetic_drugs_per_patient_t[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of ' + i, 'Patient Count (Train)','Test')
    
                plot_boxplots(real_drugs_per_patient_t[i+'_count'], synthetic_drugs_per_patient_t[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of ' + i, 'Patient Count (Real)')
                #, 'Patient Count (Synthetic)'
                #, 'ADmission Count (Synthetic)'
                real_out_a_t = calculate_outlier_ratios_tout2(real_drugs_per_admission_t,i)
                syn_out_a_t = calculate_outlier_ratios_tout2(synthetic_drugs_per_admission_t,i)
                plot_outliers(real_drugs_per_admission_t, synthetic_drugs_per_admission_t, i,
                                'Outliers of ' +i+ ' per Admission', 'Number of ' + i, 'Patient Count (Real)','Test')
                plot_histograms_separate_axes22(real_out_a_t[i+'_count'], syn_out_a_t[i+'_count'], 
                                'Histogram of ' +i+ ' per Admission', 'Number of' + i, 'Patient Count (Real)','Test')
            
        
                plot_boxplots(real_drugs_per_admission_t[i+'_count'], synthetic_drugs_per_admission_t[i+'_count'], 
                                'Histogram of ' +i+ ' per Admission', 'Number of ' + i, 'Patient Count (Real)')

                plot_histograms_separate_axes22(real_drugs_per_admission_t[i+'_count'], synthetic_drugs_per_admission_t[i+ '_count'], 
                                'Histogram of ' +i+ ' per Admission', 'Number of ' + i, 'Admission Count (Real)'
                            ,'Test')
                #, 'ADmission Count per drug(train, test)'
                
                real_out_a_a_t = calculate_outlier_ratios_tout2(real_admissions_per_drug_t,"admission")
                syn_out_a_a_t = calculate_outlier_ratios_tout2(synthetic_admissions_per_drug_t,"admission")
                    
                plot_outliers(real_admissions_per_drug_t, synthetic_admissions_per_drug_t, "admission",
                                'Outliers of   Admission per ' + i, 'Number of ' + i, 'Patient Count (Real)','Test')
                plot_histograms_separate_axes22(real_out_a_a_t["admission"+'_count'], syn_out_a_a_t["admission"+'_count'], 
                                'Histogram of otliers ' + 'Admission per ' + i, 'Number of ' + i, 'Patient Count (Real)','Test')
                

                plot_histograms_separate_axes22(real_admissions_per_drug_t['admission_count'], synthetic_admissions_per_drug_t['admission_count'], 
                                'Histogram of Admissions per ' + i, 'Number of Admissions ',  i+ ' Count (Real)','Test' )
                
                plot_boxplots(real_admissions_per_drug_t['admission_count'], synthetic_admissions_per_drug_t['admission_count'], 
                                'Histogram of Admissions per ' +i, 'Number of Admissions ',  i+ ' Count (Real)', )

                
                #, 'Patient Count (Synthetic/train)'
                real_drugs_per_patient, real_drugs_per_admission, real_admissions_per_drug,real_patients_per_drug = calculate_counts(train_ehr_dataset,i)
                synthetic_drugs_per_patient, synthetic_drugs_per_admission, synthetic_admissions_per_drug,synthetic_patients_per_drug = calculate_counts(synthetic_ehr_dataset,i)
                #, 'Patient Count (Synthetic)'
                real_out = calculate_outlier_ratios_tout2(real_drugs_per_patient,i)
                syn_out = calculate_outlier_ratios_tout2(synthetic_drugs_per_patient,i)
                plot_outliers(real_drugs_per_patient, synthetic_drugs_per_patient, i,
                                'Outliers of ' +i+ ' per Patient', 'Number of '+ i, 'Patient Count (Real)','Synthetic')
                plot_histograms_separate_axes22(real_out[i+'_count'], syn_out[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of '+ i, 'Patient Count (Real)','Synthetic')
            
            
                plot_histograms_separate_axes22(real_drugs_per_patient[i+'_count'], synthetic_drugs_per_patient[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of '+ i, 'Patient Count (Real)','Synthetic')
                
                plot_histograms_separate_axes22(real_drugs_per_patient[i+'_count'], synthetic_drugs_per_patient[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of '+ i, 'Patient Count (Real)','Synthetic')
                
                
                plot_boxplots(real_drugs_per_patient[i+'_count'], synthetic_drugs_per_patient[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of '+ i, 'Patient Count (Real)')

                #, 'ADmission Count (Synthetic)'
                real_out_a = calculate_outlier_ratios_tout2(real_drugs_per_admission,i)
                syn_out_a = calculate_outlier_ratios_tout2(synthetic_drugs_per_admission,i)
                plot_outliers(real_drugs_per_admission, synthetic_drugs_per_admission, i,
                                'Outliers of ' +i+ ' per Admission', 'Number of ' + i, 'Patient Count (Real)','Synthetic')
                plot_histograms_separate_axes22(real_out_a[i+'_count'], syn_out_a[i+'_count'], 
                                'Histogram of ' +i+ ' per Admission', 'Number of ' + i, 'Patient Count (Real)','Synthetic')
            
        
                plot_boxplots(real_drugs_per_admission[i+'_count'], synthetic_drugs_per_admission[i+'_count'], 
                                'Histogram of ' +i+ ' per Admission', 'Number of ' + i, 'Patient Count (Real)')

                plot_histograms_separate_axes22(real_drugs_per_admission[i+'_count'], synthetic_drugs_per_admission[i+ '_count'], 
                                'Histogram of ' +i+ ' per Admission', 'Number of ' + i, 'Admission Count (Real)','Synthetic'
                            )
                
                        #ADmission per drug #, 
                real_out_a_a = calculate_outlier_ratios_tout2(real_admissions_per_drug,"admission")
                syn_out_a_a = calculate_outlier_ratios_tout2(synthetic_admissions_per_drug,"admission")
                    
                plot_outliers(real_admissions_per_drug, synthetic_admissions_per_drug, "admission",
                                'Outliers of   Admission per ' + i, 'Number of ' + i, 'Patient Count (Real)','Synthetic')
                plot_histograms_separate_axes22(real_out_a_a["admission"+'_count'], syn_out_a_a["admission"+'_count'], 
                                'Histogram of otliers ' + 'Admission per ' + i, 'Number of ' + i, 'Patient Count (Real)','Synthetic')
                

                plot_histograms_separate_axes22(real_admissions_per_drug['admission_count'], synthetic_admissions_per_drug['admission_count'], 
                                'Histogram of Admissions per ' + i, 'Number of Admissions ',  i+ ' Count (Real)','Synthetic' )
                
                plot_boxplots(real_admissions_per_drug['admission_count'], synthetic_admissions_per_drug['admission_count'], 
                                'Histogram of Admissions per ' +i, 'Number of Admissions ',  i+ ' Count (Real)', )
            
    def metric_outliers(self,train_ehr_dataset,synthetic_ehr_dataset,cols_sel):
        res_ratio = {}
        #diagnosis
        for word in keywords:
            cols_sel = train_ehr_dataset.filter(like=word).columns
            res_ratio["Ratio_outlierrs_"+word] =calculate_outlier_ratios_tout(train_ehr_dataset, synthetic_ehr_dataset,cols_sel)
            
        #categorical var
        categorical_cols = self.dependant_fist_visit
        cols_accounts=[]
        for i in categorical_cols:
            cols_f = train_ehr_dataset.filter(like=i, axis=1).columns
            cols_accounts.extend(list(cols_f))
            
        res_ratio["Ratio_outlierrs_categorical"] =calculate_outlier_ratios_tout(train_ehr_dataset, synthetic_ehr_dataset,cols_accounts)
        
        cols = [ 'Age_max', 'LOSRD_avg','visit_rank','days_between_visits']
        res_ratio["Ratio_outlierrs_numerical"] =calculate_outlier_ratios_tout(train_ehr_dataset, synthetic_ehr_dataset,cols)

        if "outliers" in self.list_metric_resemblance:
            outliers_and_histograms_patients_admissions(self.test_ehr_dataset,self.train_ehr_dataset,self.synthetic_ehr_dataset ,self.columnas_test_ehr_dataset,self.top_300_codes,self.synthetic_ehr,self.list_metric_resemblance)
    
    def get_descriptive_statistics(self,synthetic_ehr_dataset,train_ehr_dataset,cols):        
          
        result = descriptive_statistics_matrix(synthetic_ehr_dataset[cols])
        result1 = descriptive_statistics_matrix(train_ehr_dataset[cols])
        
        #result = descriptive_statistics_matrix(test_ehr_dataset[cols])
        result_resemblence.append(result)
        data = pd.concat([pd.DataFrame(result,index = [0]),pd.DataFrame(result1,index = [0])])
        x1 = data[[ 'mean_visit_rank', 
        'mode_visit_rank', 'minimum_visit_rank', 'maximum_visit_rank',
        'range_visit_rank', 'mean_days_between_visits',
        'mode_days_between_visits',
        'minimum_days_between_visits', 'maximum_days_between_visits' ]].to_latex()
        print(x1)

    
    def compare_average_trends_recordlen(self,train_ehr_dataset,columnas_test_ehr_dataset,test_ehr_dataset,synthetic_ehr,synthetic_ehr_dataset):
    
        statistics = get_statistics(train_ehr_dataset,columnas_test_ehr_dataset,test_ehr_dataset,synthetic_ehr,synthetic_ehr_dataset)
        print_latex(statistics)
        return statistics
    
    def plot_first_visit(self,train_ehr_dataset,synthetic_ehr_dataset,dependant_fist_visit):
        cols_accounts = []
        
        for i in dependant_fist_visit:
            cols_f = train_ehr_dataset.filter(like=i, axis=1).columns
            cols_accounts.extend(list(cols_f))
        for i in   cols_accounts:  
             plot_admission_date_histograms(train_ehr_dataset,synthetic_ehr_dataset,i)
             plot_admission_date_bar_charts(train_ehr_dataset,synthetic_ehr_dataset,i)
             

    def plot_means_continuos_variables(self,train_ehr_dataset,synthetic_ehr_dataset,cols):
        # Assuming real_df and synthetic_df are your dataframes    
        plot_means(train_ehr_dataset, synthetic_ehr_dataset, cols)

    def plot_kerneldis(self,train_ehr_dataset,synthetic_ehr_dataset,cols):
            for i in cols: 
                plot_kernel_syn(train_ehr_dataset, synthetic_ehr_dataset, i, "Marginal_distribution")
     
    
    
    def get_top10_different_proportion(self,train_ehr_dataset,synthetic_ehr_dataset,cols):
        cols_ = ['ADMITTIME','HADM_ID']
     
        train_ehr_dataset_auz = cols_todrop(train_ehr_dataset,cols_)
        synthetic_ehr_dataset_auz = cols_todrop(synthetic_ehr_dataset,cols_)
  
        columns_take_account = [i for i in train_ehr_dataset_auz.columns if i not in cols]
        res = compare_proportions_and_return_dictionary(train_ehr_dataset_auz[columns_take_account], synthetic_ehr_dataset_auz[columns_take_account])    
        top_10_diff_proportions = dict(sorted(res.items(), key=lambda item: item[1], reverse=True)[:10])
        df_top = pd.DataFrame(top_10_diff_proportions,index = [0])
        la = df_top.to_latex()
        print(la)
        return top_10_diff_proportions

    def get_visit_value_counts(self,train_ehr_dataset,synthetic_ehr_dataset,columnas_test_ehr_dataset):
  
        df = train_ehr_dataset[columnas_test_ehr_dataset+["visit_rank"]]
        type_d = "train"
        res = df.groupby(["visit_rank"]).count().iloc[:,0].head().to_dict()
        res = {f"{key} test set": value for key, value in res.items()}

        df = synthetic_ehr_dataset[columnas_test_ehr_dataset+["visit_rank"]]
        type_d = "synthetic"
        res2 = df.groupby(["visit_rank"]).count().iloc[:,0].head().to_dict()
        res2 = {f"{key} synthetic set": value for key, value in res2.items()}
        aux_s = pd.concat([pd.DataFrame(res,index = [0]),pd.DataFrame(res2,index = [0])],axis = 1)
        la = aux_s.to_latex()
        print(la)
        res.update(res2)
        return res  
    def plot_differen_correlation(self,synthetic_ehr_dataset,train_ehr_dataset,cols,categorical_cols,keywords):

        # Example usage:
        heatmap_diff_corr(df1, df2)

        corr_plot(synthetic_ehr_dataset,"Syn" )
        corr_plot(train_ehr_dataset,"Train" )  
     
        syn_c = corr_plot(synthetic_ehr_dataset[cols],"Syn" )
        real_c = corr_plot(train_ehr_dataset[cols],"Train" )
        heatmap_digbet_corr(syn_c,real_c)
        cols_list = []
        categorical_cols = categorical_cols
        for i in categorical_cols:
            cols_f = train_ehr_dataset.filter(like=i, axis=1).columns
            cols_list.extend(list(cols_f))
        syn = corr_plot(synthetic_ehr_dataset[cols_list],"Syn" )
        cols_with_high_corr = correlacion_otra_col(synthetic_ehr_dataset[cols_list])
        real = corr_plot(train_ehr_dataset[cols_list],"Train" )
        #diferencia de correlationes
        heatmap_diff_corr(syn,real)
        correlacion_otra_col(synthetic_ehr_dataset[cols_list])
        keywords = keywords
        for i in keywords:
            col_prod = [col for col in train_ehr_dataset.columns if any(palabra in col for palabra in [i])]
            syn1 = corr_plot(synthetic_ehr_dataset[col_prod],"Syn" )
            #cols_with_high_corr, cols_with_all_nan =correlacion_total(synthetic_ehr_dataset[col_prod])
            real1 =corr_plot(train_ehr_dataset[col_prod],"Train" )    
            #cols_with_high_corr, cols_with_all_nan = correlacion_total(train_ehr_dataset[col_prod])
            y_labels = heatmap_diff_corr(syn1,real1)
            print(y_labels)
        
    def plot_pacmap(self,synthetic_ehr_dataset,train_ehr_dataset,keywords,categorical_cols):    

                   
        keywords = keywords
        for i in keywords:
            col_prod = [col for col in train_ehr_dataset.columns if any(palabra in col for palabra in [i])]
            PACMAP_PLOT(col_prod,synthetic_ehr_dataset,train_ehr_dataset,i,save=False)
        cols_list = []
        
        for i in categorical_cols:
            cols_f = train_ehr_dataset.filter(like=i, axis=1).columns
            cols_list.extend(list(cols_f))
        PACMAP_PLOT(cols_list,synthetic_ehr_dataset,train_ehr_dataset,"Demographic and admission",save=False)    

        cols = [ 'Age_max', 'LOSRD_sum','LOSRD_avg',
        'visit_rank',
        'days_between_visits']
        PACMAP_PLOT(cols_list,synthetic_ehr_dataset,train_ehr_dataset,"Continuos variables",save=False)    
    
    def temporal_histogram_heatmap(self,synthetic_ehr_dataset,train_ehr_dataset):    
                
        name = "Days between visits"
        col =   'days_between_visits_bins'
        synthetic_ehr_dataset['days_between_visits_bins'] = pd.qcut(synthetic_ehr_dataset['days_between_visits'], q=10, duplicates='drop')
        hist_d("days_between_visits",synthetic_ehr_dataset) 
        plot_heatmap_(synthetic_ehr_dataset, name,col,"Synthetic")
        train_ehr_dataset['days_between_visits_bins'] = pd.qcut(train_ehr_dataset['days_between_visits'], q=10, duplicates='drop')
        hist_d("days_between_visits",train_ehr_dataset) 
        hist_betw_a(train_ehr_dataset,synthetic_ehr_dataset,"days_between_visits")
        hist_betw_a(train_ehr_dataset,synthetic_ehr_dataset,"id_patient")
        
        plot_heatmap_(train_ehr_dataset, name,col, "Real")
        # age
        name = "Age interval"
        col =   'Age_max_bins'
        synthetic_ehr_dataset['Age_max_bins'] = pd.qcut(synthetic_ehr_dataset['Age_max'], q=5, duplicates='drop')
        hist_d("Age_max",synthetic_ehr_dataset) 
        plot_heatmap_(synthetic_ehr_dataset, name,col, "Synthetic")
        train_ehr_dataset['Age_max_bins'] = pd.qcut(train_ehr_dataset['Age_max'], q=5, duplicates='drop')
        plot_heatmap_(train_ehr_dataset, name,col,"Real")
        hist_d("days_between_visits",train_ehr_dataset) 
        
        
        # Histograms
        box_pltos(train_ehr_dataset,synthetic_ehr_dataset,'days_between_visits')
        box_pltos(train_ehr_dataset,synthetic_ehr_dataset,'Age_max')
        keywords = keywords
        for i in keywords:
            col_prod = [col for col in train_ehr_dataset.columns if any(palabra in col for palabra in [i])]
            plot_heatmap(train_ehr_dataset, i,i, 2,col_prod,type_c="Real")
            plot_heatmap(synthetic_ehr_dataset, i,i, 2,col_prod,type_c="Synthetic")
        

    
    
    def compare_maximum_range(self,test_ehr_dataset,synthetic_ehr_dataset,cols):     
        result = compare_data_ranges(test_ehr_dataset, synthetic_ehr_dataset,cols)
        print("Feature Ranges:\n", result)
        aux_s =pd.DataFrame(result,index = [0])
        la = aux_s.to_latex()
        print(la)
      
        return result

    def plot_dimension_wise(self,synthetic_ehr_dataset,train_ehr_dataset,cols_continuous,cols_categorical,cols_diagnosis,cols_drugs,cols_procedures):    
        gen_samples = synthetic_ehr_dataset.select_dtypes(include=['number'])    
        real_samples = train_ehr_dataset.select_dtypes(include=['number']) 
        dimensio_wise(real_samples,gen_samples)
        cols_to_drop = ["id_patient","HADM_ID","days_between_visits","Age_max","LOSRD_avg"]
        gen_samples.drop(columns = cols_to_drop,inplace = True)
        real_samples.drop(columns = cols_to_drop, inplace = True)
        dimensio_wise(real_samples,gen_samples)
        
       
    def plot_prevalence_wise(self,synthetic_ehr_dataset,train_ehr_dataset,cols_continuous,cols_categorical,cols_diagnosis,cols_drugs,cols_procedures):
        # Función para calcular la prevalencia de características binarias
        # Calcular APD para características binarias
        binary_features = cols_categorical +['readmission','HOSPITAL_EXPIRE_FLAG']
        real_binary_prevalences = calculate_binary_prevalence(train_ehr_dataset[binary_features])
        synthetic_binary_prevalences = calculate_binary_prevalence(synthetic_ehr_dataset[binary_features])
        apd_binary = calculate_mean_absolute_difference(real_binary_prevalences, synthetic_binary_prevalences)

        # Calcular MAD para variables de conteo
        count_features = cols_diagnosis.to_list() +cols_drugs.to_list()+cols_procedures.to_list() +['visit_rank']
        real_count_frequencies = calculate_relative_frequencies(train_ehr_dataset[count_features])
        synthetic_count_frequencies = calculate_relative_frequencies(synthetic_ehr_dataset[count_features])
        mad_count = calculate_mean_absolute_difference(real_count_frequencies, synthetic_count_frequencies)

        # Calcular AWD para variables continuas
        continuous_features = cols_continuous
        normalized_real_data = normalize_features(train_ehr_dataset[continuous_features])
        normalized_synthetic_data = normalize_features(synthetic_ehr_dataset[continuous_features])
        awd_continuous = calculate_awd(normalized_real_data, normalized_synthetic_data)

        # Crear DataFrame combinado para el gráfico
        plot_data = pd.DataFrame({
            'Real': pd.concat([real_binary_prevalences, real_count_frequencies, normalized_real_data.mean()]),
            'Synthetic': pd.concat([synthetic_binary_prevalences, synthetic_count_frequencies, normalized_synthetic_data.mean()]),
            'Feature': ['Binary'] * len(real_binary_prevalences) + ['Count'] * len(real_count_frequencies) + ['Continuous'] * len(normalized_real_data.columns)
        })

        plot_data = remove_outliers(plot_data, 'Real')
        plot_data = remove_outliers(plot_data, 'Synthetic')
        index_to_drop = ["id_patient","HADM_ID","days_between_visits","Age_max","LOSRD_avg"]
        
        plot_data = plot_data.drop(["days_between_visits", "Age_max", "LOSRD_avg"])

        
        
        # Encontrar los límites máximos para ajustar los ejes
        max_real = plot_data['Real'].max()
        max_synthetic = plot_data['Synthetic'].max()
        axis_limit = max(max_real, max_synthetic) * 1.1  # añadir un 10% de margen

        # Crear el gráfico de dispersión combinado
        plt.figure(figsize=(10, 10))
        sns.scatterplot(data=plot_data, x='Real', y='Synthetic', hue='Feature', style='Feature', s=100)

        # Añadir línea de referencia diagonal
        plt.plot([0, 1], [0, 1], ls="--", c=".3")

        # Añadir los valores de APD, MAD y AWD al gráfico
        plt.text(0.05, axis_limit * 0.95, f'APD (Binary) = {apd_binary:.2f}', horizontalalignment='left', size='medium', color='black', weight='semibold')
        plt.text(0.05, axis_limit * 0.90, f'MAD (Count) = {mad_count:.2f}', horizontalalignment='left', size='medium', color='black', weight='semibold')
        plt.text(0.05, axis_limit * 0.85, f'AWD (Continuous) = {awd_continuous:.2f}', horizontalalignment='left', size='medium', color='black', weight='semibold')

        # Ajustar los límites de los ejes
        plt.xlim(0, axis_limit)
        plt.ylim(0, axis_limit)

        # Añadir títulos y etiquetas
        plt.title('Comparison of Feature Prevalence in Real and Synthetic Data')
        plt.xlabel('Real Prevalence')
        plt.ylabel('Synthetic Prevalence')
        plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        
        
   

    def top_10diference_absbalute(self,test_ehr_dataset,synthetic_ehr_dataset,cols):
    
        dif =  compare_descriptive_statistics(test_ehr_dataset[cols], synthetic_ehr_dataset[cols])
        top_10_differences = sorted(dif.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_10_dict = {item[0]: item[1] for item in top_10_differences}
        print_latex(top_10_dict)
        return top_10_dict
    
    def get_MaximumMeanDiscrepancy(self):
        
        mmd_evaluator = MaximumMeanDiscrepancy(kernel="rbf")
        train_test = "test"
        cols = ['ADMITTIME','HADM_ID']
        #cols = "days_between_visits_cumsum"
        train_ehr_dataset_a = cols_todrop(train_ehr_dataset,cols)
        #train_ehr_dataset_a = cols_todrop(train_ehr_dataset,cols)
        synthetic_ehr_a = cols_todrop(synthetic_ehr_dataset,cols)
        result =   mmd_evaluator._evaluate(train_ehr_dataset_a.iloc[:,:synthetic_ehr_a.shape[1]], synthetic_ehr_a)
        print("MaximumMeanDiscrepancy (flattened):", result)
        print_latex(result)
            

    # Example usage:
    # X_gt and X_syn are two numpy arrays representing empirical distributions
    def kolmogorov_smirnof_test_chissquare(self,train_ehr_dataset,synthetic_ehr_dataset,cols_continuous,cols_not_con):

        
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
        cols_not_con = [i for i in train_ehr_dataset.columns if i not in continuous_cols+['id_patient','ADMITTIME']]
        real_data = train_ehr_dataset
        synthetic_data = synthetic_ehr_dataset
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


        # Create the final DataFrame
        final_df = pd.DataFrame({
            'Most Different by p-value': most_different_cols['column'].tolist() + [''] * (10 - len(most_different_cols)),
            'Least Different by p-value': least_different_cols['column'].tolist() + [''] * (10 - len(least_different_cols)),
            'Most Different by TV Distance': most_different_cols_tv['column'].tolist() + [''] * (10 - len(most_different_cols_tv)),
            'Least Different by TV Distance': least_different_cols_tv['column'].tolist() + [''] * (10 - len(least_different_cols_tv)),
        })

        # Display the DataFrame
        print(final_df.to_latex())
        # Combine the results
            
    def get_jensenshannon_dist( self,train_ehr_dataset,synthetic_ehr_dataset,test_ehr_dataset,cols_continuous):
        cols = cols_continuous
        #cols = "days_between_visits_cumsum"
        test_ehr_dataset_a = cols_todrop(train_ehr_dataset,cols)
        #train_ehr_dataset_a = cols_todrop(train_ehr_dataset,cols)
        synthetic_ehr_dataset_a = cols_todrop(synthetic_ehr_dataset,cols)
        test_ehr_dataset_a2 = cols_todrop(test_ehr_dataset,cols)
        train_ehr_dataset_a2 = test_ehr_dataset_a[:test_ehr_dataset_a.shape[0]]
        jsd_calculator = JensenShannonDistance2(test_ehr_dataset_a, synthetic_ehr_dataset_a,10)
        jsd_results, avg_js = jsd_calculator.jensen_shannon()
        print("Jensen-Shannon Distance:", score)
        
        
        jsd_calculator = JensenShannonDistance2(test_ehr_dataset_a, synthetic_ehr_dataset_a,10)
        jsd_results, avg_js = jsd_calculator.jensen_shannon()
        print_latex(jsd_results)
        
        
        



    # Example usage with two dataframes, `X_gt_df` and `X_syn_df`

                    
    
   
    def get_proportion_demos(self,train_ehr_dataset,synthetic_ehr_dataset,categorical_cols):    
        prop_datafram = []
        categorical_cols = categorical_cols
        for i in categorical_cols:
            cols_f = train_ehr_dataset.filter(like=i, axis=1).columns
            tabla_proporciones_final = get_proportions(train_ehr_dataset[cols_f],"train "+ i)
            tabla_proporciones_final_2= get_proportions(synthetic_ehr_dataset[cols_f],"synthetic"+ i)
            
            total = pd.merge(tabla_proporciones_final,tabla_proporciones_final_2,  on=["Variable","Category "], how='inner') 
            #total = total['Variable','Category ]+test_ehr_dataset[cols_f],"test "+ i]
            prop_datafram.append(total)
            latex_code = total.to_latex( index = "Variable")
            print(latex_code)
    def  get_excat_match(self,train_ehr_dataset,synthetic_ehr_dataset):
        df1_sorted = synthetic_ehr_dataset.sort_values(by=synthetic_ehr_dataset.columns.tolist()).reset_index(drop=True)  
        df2_sorted = train_ehr_dataset.sort_values(by=train_ehr_dataset.columns.tolist()).reset_index(drop=True)
        exact_matches = (df1_sorted == df2_sorted).all(axis=1).sum()
        print("Number of exact matching rows:", exact_matches)
        print_latex(exact_matches)
        return {"exact_matches" : exact_matches} 
    
    def get_dupplicates(self,   train_ehr_dataset,synthetic_ehr_dataset):    
        combined = pd.concat([train_ehr_dataset, synthetic_ehr_dataset])
        synthetic_ehr_dataset['is_duplicate'] = combined.duplicated(keep=False).iloc[len(train_ehr_dataset):]

        print("Test data with duplicate flag:")
        value_counts = synthetic_ehr_dataset['is_duplicate'].value_counts()
        new_dict = {"Duplicates - " + str(key): value for key, value in value_counts.items()}
        result_resemblence.append(new_dict)               
        print_latex(new_dict)
        

    def obtain_least_frequent(self,train_ehr_dataset, columnas_test_ehr_dataset, n):    
        least_cols =  obtain_least_frequent(train_ehr_dataset, columnas_test_ehr_dataset, 10)
        

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
            plot_df_filtered = plot_df[plot_df['index'].apply(lambda x: x[1] == 1)]

        # Imprime el DataFrame filtrado

            latex_table = plot_df_filtered.to_latex(index=False)
            print(latex_table)
   
    
    ### TODO debugerar col     
    def plot_acumulated(self,   train_ehr_dataset,synthetic_ehr_dataset,cols):    
        import matplotlib.pyplot as plt

        categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                    'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                    'MARITAL_STATUS',  'ETHNICITY','GENDER',"visit_rank","HOSPITAL_EXPIRE_FLAG"  ]
        categorilca_cols_fun(train_ehr_dataset,synthetic_ehr_dataset,categorical_cols,save=False)
        cols = [ 'Age_max', 'LOSRD_sum','LOSRD_avg','id_patient','L_1s_last_p1','days_between_visits']
        categorilca_cols_fun(train_ehr_dataset,synthetic_ehr_dataset,cols,save=False)
        keywords = keywords
        categorilca_cols_fun(train_ehr_dataset,synthetic_ehr_dataset,keywords,save=False)
        
    def get_common_proportions(self,train_ehr_dataset,synthetic_ehr_dataset):

    
        cp = CommonRowsProportion()
        dict_s = cp.evaluate(train_ehr_dataset, synthetic_ehr_dataset)
        print_latex(dict_s)
        return dict_s
