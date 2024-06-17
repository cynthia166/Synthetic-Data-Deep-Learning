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
from generative_model.SD.constraints import *
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
    train_ehr_dataset,synthetic_ehr_dataset,test_ehr_dataset = make_read_constraints(make_contrains,save_constrains,train_ehr_dataset,test_ehr_dataset,synthetic_ehr_dataset,columns_to_drop,columns_to_drop_syn,type_archivo,make_read_constraints_name)
    
    #get_columns
    columnas_test_ehr_dataset,cols_categorical,cols_diagnosis,cols_procedures,cols_drugs,top_300_codes        = get_columns_codes_commun(train_ehr_dataset,keywords)   
      
    #get cols diagnosis, procedures, drugs  
    #synthetic_ehr_dataset, train_ehr_dataset, test_ehr_dataset = cols_drop_ehr(columns_to_drop, synthetic_ehr_dataset, train_ehr_dataset, test_ehr_dataset)
     
     #lista de metricas
    list_metric_resemblance = [
       ##descriptive stadistics
        
        "get_descriptive_statistics",             #diferencia alarmante en estadiscas descriptivas 
        "compare_average_trends_recordlen",       #record lenght depende en numero de admissiones
        "get_visit_value_counts",              #analisar numero de admissiones / comparar la estadisticas con media y std normal truncada*
        "compare_descriptive_stadistics",
        
        #plots descriptive stadistics
        "plot_means_continuos_variables",
        "plot_first_visit",
        "plot_dimension_wise",
        "plot_prevalence_wise",
     
        #top10 dif or least diff                   #identificar los codigos o variables mas distintas/ identificar que importancia tienen en selccion random fors
        "get_top10_different_mean_difference",         #identificar la frecuencia por alrbol y hoja/ identidicar mas comumnes pro arbol y hoja
       "obtain_least_frequent",
        
        
        #comparrison                               # notar que el modelo no truna la log normal por lo que es nomal tener valor negativos y mayor al maximo
        "compare_maximum_range",                  #to what percentageis is greater than thereal max
        "get_proportion_demos",
       
        
        ##scores/metrics
        "get_jensenshannon_dist",                   #score to obtain eta match duplicates. alsooo seee summerazed score, all of the score make a score with train/test as baseline
        "get_MaximumMeanDiscrepancy",
        "get_common_proportions",
        "kolmogorov_smirnof_test_chissquare",
        
        #exact rcords
        
        "get_excat_match",
        "get_dupplicates",
        
            
        
        #distributions
        "plot_kerneldis",
            
              
        

        ##outiers
        "outliers_and_histograms_patients_admissions",    # Todo make a funciton to obtain a dataframe o wanseterin distance to it aumatically
        "metric_outliers",
        
            
        #multivariate  
        #"plot_differen_correlation",
                
        
        #temporality
        "temporal_histogram_heatmap",
        "plot_acumulated",
            
    
        #dimensional 
        ##"plot_pacmap",
        
    
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
                    corr_plot_continous = True,
                    corr_plot_categorical =True,
                    corr_plot_codes=True)
    #evaluate
    results = metrics_e.evaluate()
    #combine_svgs(['dimensio_wise_2994.svg', 'dimensio_wise_2994.svg', 'dimensio_wise_2994.svg', 'dimensio_wise_2926.svg'], path_img)

    #print_latex(results)
# cols_continuous = ['Age_max', 'LOSRD_avg','days_between_visits']     
# fored = load_pkl("/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/FORED")
# trnorm_modified = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/ARF_tnorm/synthetic_data_generative_model_arf_adjs_per_0.7.pkl"
# initial_tnorm = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/synthetic_data_generative_model_arf_per_0.7.pkl"
fixed_norm = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/prueba_synthetic_data_generative_model_arf_per_fixed0.7.pkl"
synthetic_raw_data =  load_pickle(fixed_norm)
fored_fixed = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/FORED_fixed"
FORED =  load_pkl(fored_fixed)
#feature importance
clf = FORED["forest"]
feature_importances = clf.feature_importances_
print("Feature Importances:", feature_importances)
df = pd.DataFrame({
    'importance': feature_importances,
    'name': synthetic_raw_data.columns
})

# Sort by importance
df_sorted = df.sort_values(by='importance',ascending = False)

# Get the names of the lowest and highest 20 variables
print(df_sorted.head(20).to_latex())
print(df_sorted.tail(20).to_latex())

# Swap their names
df_sorted.head(20)['name'], df_sorted.tail(20)['name'] = high_20_names, low_20_names


# synthetic_ehr_dataset_constrains = load_pickle("generated_synthcity_tabular/ARF/" + 'synthetic_ehr_dataset_contrainst_tnorm.pkl')
# real_training_data =load_pickle("generated_synthcity_tabular/ARF/" + 'train_ehr_dataset.pkl')

def load_pkl(name):
    with open(name+'.pkl', 'rb') as f:
        data = pickle.load(f)
    return data        

aux = FORED['cat']
aux2 = aux[aux['variable'] == 'SUBJECT_ID']
aggregated_stats = aux.groupby('variable').agg({
    'prob': ['mean'],
   
}).reset_index()
aggregated_stats.sort_values(ascending=False, by=(    'prob', 'mean'))
# Aplanar el MultiIndex de las columnas
# aggregated_stats = df.groupby('age').agg({
#     'variable1': ['mean', 'std', 'min', 'max'],
#     'variable2': ['mean', 'std', 'min', 'max']
# }).reset_index()


#aggregated_stats.columns = ['_'.join(col).strip() for col in aggregated_stats.columns.values]
#print(aggregated_stats.to_latex())
print(aggregated_stats.sort_values(ascending=True, by=(    'prob', 'mean'))[:20].to_latex())
print(aggregated_stats.sort_values(ascending=False, by=(    'prob', 'mean'))[:20].to_latex())
# fored_aux = fored["cnt"][fored["cnt"]["variable"]=="days_between_visits"]

# import matplotlib.pyplot as plt

# min_vals = real_training_data["days_between_visits"].min(axis=0)
# max_vals = real_training_data["days_between_visits"].max(axis=0)



# ['readmission','HADM_ID',"ADMITTIME",'GENDER_0']
# real_data = real_training_data["days_between_visits"]
# synthetic_data = synthetic_ehr_dataset["days_between_visits"]
# synthetic_data = np.clip(synthetic_data, min_vals, max_vals)
# plot_hist(real_data,synthetic_ehr_dataset_constrains["days_between_visits"])

# import numpy as np
# import seaborn as sns
# # Supongamos que estas son tus dos series de datos

# def visualize_data(data):
#     for column in data.columns:
#         plt.figure(figsize=(10, 5))
#         sns.histplot(data[column], kde=True)
#         plt.title(f'Distribution of {column}')
#         plt.xlabel(column)
#         plt.ylabel('Frequency')
#         plt.show()

# # Supongamos que `data_real` es tu conjunto de datos real
# visualize_data(real_training_data[cols_continuous])


# from scipy.stats import beta, gamma, norm, lognorm, truncnorm, kstest
# import pandas as pd

# def fit_distributions(data):
#     distributions = {
#         'beta': beta,
#         'gamma': gamma,
#         'norm': norm,
#         'lognorm': lognorm,
#         'truncnorm': truncnorm,
#     }
#     results = []
#     for column in data.columns:
#         column_results = {'feature': column}
#         feature_data = data[column]
#         for name, distribution in distributions.items():
#             if name == 'truncnorm':
#                 min_val, max_val = feature_data.min(), feature_data.max()
#                 mean, std = feature_data.mean(), feature_data.std()
#                 a, b = (min_val - mean) / std, (max_val - mean) / std
#                 params = (a, b, mean, std)  # Pass (a, b, loc, scale) for truncnorm
#             else:
#                 params = distribution.fit(feature_data)
#             # Use Kolmogorov-Smirnov test for goodness of fit
#             if name == 'truncnorm':
#                 ks_stat, ks_p_value = kstest(feature_data, name, args=params)
#             else:
#                 ks_stat, ks_p_value = kstest(feature_data, name, args=params)
#             column_results[name + '_params'] = params
#             column_results[name + '_ks_stat'] = ks_stat
#             column_results[name + '_ks_p_value'] = ks_p_value
#         results.append(column_results)
#     return pd.DataFrame(results)
# # Ajustar distribuciones a los datos reales

# def select_best_distribution(fit_results):
#     best_distributions = {}
#     for index, row in fit_results.iterrows():
#         best_dist = None
#         best_stat = float('inf')
#         for dist in ['beta', 'gamma', 'norm', 'lognorm', 'truncnorm']:
#             if row[dist + '_ks_stat'] < best_stat:
#                 best_stat = row[dist + '_ks_stat']
#                 best_dist = dist
#         best_distributions[row['feature']] = best_dist
#     return best_distributions


# def validate_best_distributions(data, best_distributions, fit_results):
#     for column in data.columns:
#         best_dist = best_distributions[column]
#         params = fit_results.loc[fit_results['feature'] == column, best_dist + '_params'].values[0]
#         plt.figure(figsize=(10, 5))
#         sns.histplot(data[column], kde=True, label='Real Data', color='blue')
#         if best_dist == 'truncnorm':
#             a, b, mean, std = params
#             synthetic_data = truncnorm.rvs(a, b, loc=mean, scale=std, size=1000)
#         else:
#             synthetic_data = eval(f"{best_dist}.rvs(*params, size=1000)")
#         sns.histplot(synthetic_data, kde=True, label='Synthetic Data', color='red', alpha=0.5)
#         plt.title(f'Validation for {column} using {best_dist} distribution')
#         plt.xlabel(column)
#         plt.ylabel('Frequency')
#         plt.legend()
#         plt.show()

# # Validar visualmente las mejores distribuciones seleccionadas

# fit_results = fit_distributions(real_training_data[cols_continuous])
# print(fit_results)
# fit_results.to_csv("fit_results.csv")
# best_distributions = select_best_distribution(fit_results)
# #{'Age_max': 'beta', 'LOSRD_avg': 'lognorm', 'days_between_visits': 'norm'}
# print("Mejores distribuciones por característica:")
# print(best_distributions)

# validate_best_distributions(real_training_data[cols_continuous], best_distributions, fit_results)



# def plot_hist(real_data,synthetic_data):
#     # Calcular los límites combinados de los datos reales y sintéticos
#     data_min = min(real_data.min(), synthetic_data.min())
#     data_max = max(real_data.max(), synthetic_data.max())
#     # Generar los bins en función del rango combinado
#     bins = np.linspace(data_min, data_max, 80)  # 30 bins
#     # Crear los histogramas usando los mismos bins
#     plt.figure(figsize=(10, 6))
#     sns.histplot(real_data, bins=bins, color='blue', alpha=0.5, label='Real data', kde=False)
#     sns.histplot(synthetic_data, bins=bins, color='orange', alpha=0.5, label='Synthetic data', kde=False)
#     plt.legend(loc='upper right')
#     plt.title('Days between visits')
#     plt.xlabel('Days')
#     plt.ylabel('Frequency')
#     plt.show()
