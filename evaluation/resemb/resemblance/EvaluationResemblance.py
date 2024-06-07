import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from pacmap import PaCMAP
import logging
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
                 dependant_fist_visit,
                 cols_diagnosis,
                 cols_procedures,
                    cols_drugs,
                    path_img):
        
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
    def evaluate(self):    
        results = {}

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

        if "get_top10_different_proportion" in self.list_metric_resemblance:
            logging.info("Executing method: get_top10_different_proportion")
            results.update(self.get_top10_different_proportion(self.train_ehr_dataset, self.synthetic_ehr_dataset, self.cols))

        if "get_visit_value_counts" in self.list_metric_resemblance:
            logging.info("Executing method: get_visit_value_counts")
            results.update(self.get_visit_value_counts(self.train_ehr_dataset, self.synthetic_ehr_dataset, self.columnas_test_ehr_dataset))

        if "plot_differen_correlation" in self.list_metric_resemblance:
            logging.info("Executing method: plot_differen_correlation")
            self.plot_differen_correlation(self.synthetic_ehr_dataset, self.train_ehr_dataset, self.cols, self.categorical_cols, self.keywords,self.path_img)

        if "plot_pacmap" in self.list_metric_resemblance:
            logging.info("Executing method: plot_pacmap")
            self.plot_pacmap(self.synthetic_ehr_dataset, self.train_ehr_dataset, self.keywords, self.categorical_cols,self.path_img)

        if "temporal_histogram_heatmap" in self.list_metric_resemblance:
            logging.info("Executing method: temporal_histogram_heatmap")
            self.temporal_histogram_heatmap(self.synthetic_ehr_dataset, self.train_ehr_dataset,self.path_img)

        if "compare_maximum_range" in self.list_metric_resemblance:
            logging.info("Executing method: compare_maximum_range")
            results.update(self.compare_maximum_range(self.test_ehr_dataset, self.synthetic_ehr_dataset, self.cols))

        if "plot_dimension_wise" in self.list_metric_resemblance:
            logging.info("Executing method: plot_dimension_wise")
            self.plot_dimension_wise(self.synthetic_ehr_dataset, self.train_ehr_dataset, self.cols_continuous, self.categorical_cols, self.cols_diagnosis, self.cols_drugs, self.cols_procedures,self.path_img)

        if "plot_prevalence_wise" in self.list_metric_resemblance:
            logging.info("Executing method: plot_prevalence_wise")
            self.plot_prevalence_wise(self.synthetic_ehr_dataset, self.train_ehr_dataset, self.cols_continuous, self.categorical_cols, self.cols_diagnosis, self.cols_drugs, self.cols_procedures,self.path_img)

        if "top_10diference_absbalute" in self.list_metric_resemblance:
            logging.info("Executing method: top_10diference_absbalute")
            results.update(self.top_10diference_absbalute(self.test_ehr_dataset, self.synthetic_ehr_dataset, self.cols))

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

        return results

        
        #outliers
        
        

    def outliers_and_histograms_patients_admissions(self,test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset ):
        for i in keywords:
                # Shynthetic/train
                
                test_ehr_dataset_t = train_ehr_dataset[:test_ehr_dataset.shape[0]]
                real_drugs_per_patient_t, real_drugs_per_admission_t, real_admissions_per_drug_t,real_patients_per_drug_t = calculate_counts(train_ehr_dataset,i)
                test_drugs_per_patient_t, test_drugs_per_admission_t, test_admissions_per_drug_t,synthetic_patients_per_drug = calculate_counts(test_ehr_dataset_t,i)
                #, 'Patient Count (Synthetic)'
                real_out_t = calculate_outlier_ratios_tout2(real_drugs_per_patient_t,i)
                test_out_t = calculate_outlier_ratios_tout2(test_drugs_per_patient_t,i)
                #real_data, synthetic_data, column_name, title, xlabel, ylabel_real, ylabel_synthetic=real_out_t, test_out_t, i, 'Outliers of ' +i+ ' per Patient', 'Number of ' + i, 'Patient Count (Train)','Test'
                #plot_outliers(real_out_t, test_out_t, i,
                #                'Outliers of ' +i+ ' per Patient', 'Number of ' + i, 'Patient Count (Train)','Test')
                plot_histograms_separate_axes22(real_out_t[i+'_count'], test_out_t[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of ' + i, 'Patient Count (Train)','Test')
                
                plot_histograms_separate_axes22(real_drugs_per_patient_t[i+'_count'], test_drugs_per_patient_t[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of ' + i, 'Patient Count (Train)','Test')
    
                plot_boxplots(real_drugs_per_patient_t[i+'_count'], test_drugs_per_patient_t[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of ' + i, 'Patient Count (Real)')
                #, 'Patient Count (Synthetic)'
                #, 'ADmission Count (Synthetic)'
                real_out_a_t = calculate_outlier_ratios_tout2(real_drugs_per_admission_t,i)
                syn_out_a_t = calculate_outlier_ratios_tout2(test_drugs_per_admission_t,i)
                #plot_outliers(real_out_a_t, syn_out_a_t, i,
                #                'Outliers of ' +i+ ' per Admission', 'Number of ' + i, 'Patient Count (Real)','Test')
                plot_histograms_separate_axes22(real_out_a_t[i+'_count'], syn_out_a_t[i+'_count'], 
                                'Histogram of ' +i+ ' per Admission', 'Number of' + i, 'Patient Count (Real)','Test')
            
        
                plot_boxplots(real_drugs_per_admission_t[i+'_count'], test_drugs_per_admission_t[i+'_count'], 
                                'Histogram of ' +i+ ' per Admission', 'Number of ' + i, 'Patient Count (Real)')

                plot_histograms_separate_axes22(real_drugs_per_admission_t[i+'_count'], test_drugs_per_admission_t[i+ '_count'], 
                                'Histogram of ' +i+ ' per Admission', 'Number of ' + i, 'Admission Count (Real)'
                            ,'Test')
                #, 'ADmission Count per drug(train, test)'
                
                real_out_a_a_t = calculate_outlier_ratios_tout2(real_admissions_per_drug_t,"admission")
                test_out_a_a_t = calculate_outlier_ratios_tout2(test_admissions_per_drug_t,"admission")
                    
                #plot_outliers(real_out_a_a_t, test_out_a_a_t, "admission",
                #                'Outliers of   Admission per ' + i, 'Number of ' + i, 'Patient Count (Real)','Test')
                plot_histograms_separate_axes22(real_out_a_a_t["admission"+'_count'], test_out_a_a_t["admission"+'_count'], 
                                'Histogram of otliers ' + 'Admission per ' + i, 'Number of ' + i, 'Patient Count (Real)','Test')
                

                plot_histograms_separate_axes22(real_admissions_per_drug_t['admission_count'], test_admissions_per_drug_t['admission_count'], 
                                'Histogram of Admissions per ' + i, 'Number of Admissions ',  i+ ' Count (Real)','Test' )
                
                plot_boxplots(real_admissions_per_drug_t['admission_count'], test_admissions_per_drug_t['admission_count'], 
                                'Histogram of Admissions per ' +i, 'Number of Admissions ',  i+ ' Count (Real)', )

                
                #, 'Patient Count (Synthetic/train)'
                real_drugs_per_patient, real_drugs_per_admission, real_admissions_per_drug,real_patients_per_drug = calculate_counts(train_ehr_dataset,i)
                synthetic_drugs_per_patient, synthetic_drugs_per_admission, synthetic_admissions_per_drug,synthetic_patients_per_drug = calculate_counts(synthetic_ehr_dataset,i)
                #, 'Patient Count (Synthetic)'
                real_out = calculate_outlier_ratios_tout2(real_drugs_per_patient,i)
                syn_out = calculate_outlier_ratios_tout2(synthetic_drugs_per_patient,i)
                #plot_outliers(real_drugs_per_patient, synthetic_drugs_per_patient, i,
                #                'Outliers of ' +i+ ' per Patient', 'Number of '+ i, 'Patient Count (Real)','Synthetic')
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
                #plot_outliers(real_drugs_per_admission, synthetic_drugs_per_admission, i,
                #                'Outliers of ' +i+ ' per Admission', 'Number of ' + i, 'Patient Count (Real)','Synthetic')
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
                    
                #plot_outliers(real_admissions_per_drug, synthetic_admissions_per_drug, "admission",
                #                'Outliers of   Admission per ' + i, 'Number of ' + i, 'Patient Count (Real)','Synthetic')
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
            res_ratio["Ratio outliers "+word] =calculate_outlier_ratios_tout__(train_ehr_dataset, synthetic_ehr_dataset,cols_sel)    
                #categorical var
        
        cols_accounts=[]
        for i in self.categorical_cols:
            cols_f = train_ehr_dataset.filter(like=i, axis=1).columns
            cols_accounts.extend(list(cols_f))
        res_ratio["Ratio outlierrs categorical"] =calculate_outlier_ratios_tout__(train_ehr_dataset, synthetic_ehr_dataset,cols_accounts)    
        
        cols_continuous = self.cols
        res_ratio["Ratio outliers numerica"] =calculate_outlier_ratios_tout__(train_ehr_dataset, synthetic_ehr_dataset,cols_continuous)    
        
        print_latex(res_ratio)
        return res_ratio.to_dict()
    
        if "outliers" in self.list_metric_resemblance:
            outliers_and_histograms_patients_admissions(self.test_ehr_dataset,self.train_ehr_dataset,self.synthetic_ehr_dataset ,self.columnas_test_ehr_dataset,self.top_300_codes,self.synthetic_ehr,self.list_metric_resemblance)
    
    def get_descriptive_statistics(self,synthetic_ehr_dataset,train_ehr_dataset,cols):        
          
        result_syn = descriptive_statistics_matrix(synthetic_ehr_dataset[cols],"")
        result_train = descriptive_statistics_matrix(train_ehr_dataset[cols],"")
        result_concat_dict = result.update(result1)
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
        data.to_latex()
        return data.to_dict()

    
    def compare_average_trends_recordlen(self,train_ehr_dataset,columnas_test_ehr_dataset,test_ehr_dataset,synthetic_ehr_dataset):
    
        statistics = get_statistics(train_ehr_dataset,columnas_test_ehr_dataset,test_ehr_dataset,synthetic_ehr_dataset)
        print_latex(statistics)
        return statistics
    
    def plot_first_visit(self,train_ehr_dataset,synthetic_ehr_dataset,dependant_fist_visit,path_img = None):
        cols_accounts = []
        
        for i in dependant_fist_visit:
            cols_f = train_ehr_dataset.filter(like=i, axis=1).columns
            cols_accounts.extend(list(cols_f))
        for i in   cols_accounts:  
             plot_admission_date_histograms(train_ehr_dataset,synthetic_ehr_dataset,i,path_img)
             plot_admission_date_bar_charts2(train_ehr_dataset,synthetic_ehr_dataset,i,path_img)
             

    def plot_means_continuos_variables(self,train_ehr_dataset,synthetic_ehr_dataset,cols,path_img):
        # Assuming real_df and synthetic_df are your dataframes    
        plot_means(train_ehr_dataset, synthetic_ehr_dataset, cols,path_img)

    def plot_kerneldis(self,train_ehr_dataset,synthetic_ehr_dataset,cols,path_img):
            for i in cols: 
                plot_kernel_syn(train_ehr_dataset, synthetic_ehr_dataset, i, "Marginal_distribution",path_img)
     
    
    
    def get_top10_different_proportion(self,train_ehr_dataset,synthetic_ehr_dataset,cols):
        cols_ = ['ADMITTIME','HADM_ID']
     
        train_ehr_dataset_auz = cols_todrop(train_ehr_dataset,cols_)
        synthetic_ehr_dataset_auz = cols_todrop(synthetic_ehr_dataset,cols_)
  
        columns_take_account = [i for i in synthetic_ehr_dataset.columns if i not in cols+['year', 'month', 'ADMITTIME']]
        res = compare_proportions_and_return_dictionary(train_ehr_dataset_auz[columns_take_account], synthetic_ehr_dataset_auz[columns_take_account])    
        top_10_diff_proportions = dict(sorted(res.items(), key=lambda item: item[1], reverse=True)[:10])
        df_top = pd.DataFrame(top_10_diff_proportions,index = [0])
        la = df_top.to_latex()
        print(la)
        return top_10_diff_proportions

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
    def plot_differen_correlation(self,synthetic_ehr_dataset,train_ehr_dataset,cols,categorical_cols,keywords,path_img):

        # Example usage:
        heatmap_diff_corr(synthetic_ehr_dataset, train_ehr_dataset,path_img)

        corr_plot(synthetic_ehr_dataset,"Syn" ,path_img)
        corr_plot(train_ehr_dataset,"Train" ,path_img)  
        #continous cols
        syn_c = corr_plot(synthetic_ehr_dataset[cols],"Syn" ,path_img)
        real_c = corr_plot(train_ehr_dataset[cols],"Train" ,path_img)
        heatmap_diff_corr(syn_c,real_c,path_img)
        cols_list = []
        categorical_cols = categorical_cols
        for i in categorical_cols:
            cols_f = train_ehr_dataset.filter(like=i, axis=1).columns
            cols_list.extend(list(cols_f))
        syn = corr_plot(synthetic_ehr_dataset[cols_list],"Syn" ,path_img)
        cols_with_high_corr = correlacion_otra_col(synthetic_ehr_dataset[cols_list])
        real = corr_plot(train_ehr_dataset[cols_list],"Train" ,path_img   )
        #diferencia de correlationes
        heatmap_diff_corr(syn,real,self.path_img,path_img)
        #correlacion_otra_col(synthetic_ehr_dataset[cols_list])
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

        cols = [ 'Age_max', 'LOSRD_sum','LOSRD_avg',
        'visit_rank',
        'days_between_visits']
        PACMAP_PLOT(cols_list,synthetic_ehr_dataset,train_ehr_dataset,"Continuos variables",path_img )    
    
    def temporal_histogram_heatmap(self,synthetic_ehr_dataset,train_ehr_dataset,path_img ):    
                
        name = "Days between visits"
        col =   'days_between_visits_bins'
        synthetic_ehr_dataset['days_between_visits_bins'] = pd.qcut(synthetic_ehr_dataset['days_between_visits'], q=10, duplicates='drop')
        hist_d("days_between_visits",synthetic_ehr_dataset,path_img ) 
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
        keywords = self.keywords
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

    def plot_dimension_wise(self,synthetic_ehr_dataset,train_ehr_dataset,cols_continuous,categorical_cols,cols_diagnosis,cols_drugs,cols_procedures,path_img = None):    
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

        
       
    def plot_prevalence_wise(self,synthetic_ehr_dataset,train_ehr_dataset,cols_continuous,categorical_cols,cols_diagnosis,cols_drugs,cols_procedures):
        # Función para calcular la prevalencia de características binarias
        # Calcular APD para características binarias
        cols_to_filter = categorical_cols +['readmission','HOSPITAL_EXPIRE_FLAG']
        binary_features = filter_cols(cols_to_filter,train_ehr_dataset)
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
        
        #plot_data = plot_data.drop(["days_between_visits", "Age_max", "LOSRD_avg"])

        
        
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
    
    def get_MaximumMeanDiscrepancy(self,train_ehr_dataset,synthetic_ehr_dataset):
        
        mmd_evaluator = MaximumMeanDiscrepancy(kernel="rbf")
        train_test = "test"
        cols = ['ADMITTIME','HADM_ID']
        #cols = "days_between_visits_cumsum"
        train_ehr_dataset, synthetic_ehr_dataset = filter_and_equalize_datasets(train_ehr_dataset, synthetic_ehr_dataset)
          
        #train_ehr_dataset_a = cols_todrop(train_ehr_dataset,cols)
        
        result =   mmd_evaluator._evaluate(train_ehr_dataset, synthetic_ehr_dataset)
        print("MaximumMeanDiscrepancy (flattened):", result)
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

        dic_results = {
            'Most Different by p-value': most_different_cols['column'].tolist() + [''] * (10 - len(most_different_cols)),
            'Least Different by p-value': least_different_cols['column'].tolist() + [''] * (10 - len(least_different_cols)),
            'Most Different by TV Distance': most_different_cols_tv['column'].tolist() + [''] * (10 - len(most_different_cols_tv)),
            'Least Different by TV Distance': least_different_cols_tv['column'].tolist() + [''] * (10 - len(least_different_cols_tv)),
        }
        # Create the final DataFrame
        final_df = pd.DataFrame(dic_results)

        # Display the DataFrame
        print(final_df.to_latex())
        # Combine the results
        return  dic_results
    def get_jensenshannon_dist( self,train_ehr_dataset,synthetic_ehr_dataset,test_ehr_dataset,cols_continuous):
       
        #Train/Syn
        train_ehr_dataset, synthetic_ehr_dataset = filter_and_equalize_datasets(train_ehr_dataset, synthetic_ehr_dataset)
        jsd_calculator = JensenShannonDistance2(train_ehr_dataset, synthetic_ehr_dataset, 10)
        jsd_results_train_syn, avg_js_train_syn = jsd_calculator.jensen_shannon()
        
        results["train_syn"] = jsd_results_train_syn
        print_latex(jsd_results_train_syn)
        jsd_results_train_syn.columns = [f"{col} Train/Syn" for col in jsd_results_train_syn.columns]


        # Test/Syn
        test_ehr_dataset, synthetic_ehr_dataset = filter_and_equalize_datasets(test_ehr_dataset, synthetic_ehr_dataset)
        jsd_calculator = JensenShannonDistance2(test_ehr_dataset, synthetic_ehr_dataset, 10)
        jsd_results_test_syn, avg_js_test_syn = jsd_calculator.jensen_shannon()
        
        results["test_syn"] = jsd_results_test_syn
        print_latex(jsd_results_test_syn)
        jsd_results_test_syn.columns = [f"{col} Test/Syn" for col in jsd_results_test_syn.columns]

        # Test/Train
        test_ehr_dataset, train_ehr_dataset = filter_and_equalize_datasets(test_ehr_dataset, train_ehr_dataset)
        jsd_calculator = JensenShannonDistance2(test_ehr_dataset, train_ehr_dataset, 10)
        jsd_results_test_train, avg_js_test_train = jsd_calculator.jensen_shannon()
        
        results["train_test"] = jsd_results_test_train
        print_latex(jsd_results_test_train)
        jsd_results_test_train.columns = [f"{col} Test / Train" for col in jsd_results_test_train.columns]

        df_final = pd.concat([jsd_results_train_syn, jsd_results_test_syn, jsd_results_test_train], axis=1)
        print_latex(df_final)
        return df_final.to_dict()
        
        



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

    def obtain_least_frequent_m(self,train_ehr_dataset,synthetic_ehr_dataset, columnas_test_ehr_dataset, n):    
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
    def plot_acumulated(self,   train_ehr_dataset,synthetic_ehr_dataset,cols_continuous,categorical_cols):    
        import matplotlib.pyplot as plt

        categorilca_cols_fun(train_ehr_dataset,synthetic_ehr_dataset,categorical_cols,save=False)
        categorilca_cols_fun(train_ehr_dataset,synthetic_ehr_dataset,cols_continuous,save=False)
        keywords = self.keywords
        categorilca_cols_fun(train_ehr_dataset,synthetic_ehr_dataset,keywords,save=False)
        
    def get_common_proportions(self,train_ehr_dataset,synthetic_ehr_dataset):

    
        cp = CommonRowsProportion()
        dict_s = cp.evaluate(train_ehr_dataset, synthetic_ehr_dataset)
        print_latex(dict_s)
        return dict_s
