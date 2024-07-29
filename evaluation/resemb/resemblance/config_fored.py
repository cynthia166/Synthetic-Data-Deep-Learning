#create feature with random sampled pkl
from scipy.stats import beta, uniform, triang, truncnorm, expon, gamma, lognorm, weibull_min, chi2, f, t
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
create_features = False
get_fit_dist = False
anal_categotical = False
#a nalyze the feature Subject_I
continous_variable_ks_test = False
if continous_variable_ks_test:
    perform_ks_test = True 
else:
    perform_ks_test = False    
#get continous values count and parameters leaf (mean std deviation)
get_count_variables_per_node_tree_cont = False
#get aras nodes descete
get_long_params_per_node_tree_distcre = False
get_analysis_continous_fun = True
#threshol for values in leaves
threshold = 0
#obtain mean_per node
obtain_promedio_por_nodo_ks_dist = False
#perform ks test for each node

visualization_days_bewteen = False

#"id subject if"
col_cat = "SUBJECT_ID"
#threshold que se analiza
threshold_values = 30
threshold_continous = 30
col_continous = "days_between_visits"

columns_to_drop = ['LOSRD_sum', 'L_1s_last_p1','HADM_ID',"ADMITTIME",'GENDER_0']
cols_continuous = ['Age', 'LOSRD_avg','days_between_visits'] 
keywords = ['diagnosis', 'procedures', 'drugs']

folder_arf = "ARF_fixed_postpros/"
fored_fixed = "C:/Users/cyn_n/Desktop/Synthetic-Data-Deep-Learning/
generated_synthcity_tabular/ARF/"+folder_arf + "FORED_fixedr"

sample_patients_path = "generated_synthcity_tabular/ARF/"+folder_arf+"sample_patients_fixed_v"
original_data_path = "data/intermedi/SD/inpput/entire_ceros_tabular_data.pkl"
ruta_continous_observations = "C:/Users/cyn_n/Desktop/Synthetic-Data-Deep-Learning/
generated_synthcity_tabular/ARF/"+folder_arf+"result_test_concat"
#save path
save_path_features = "C:/Users/cyn_n/Desktop/Synthetic-Data-Deep-Learning/
generated_synthcity_tabular/ARF/"+folder_arf+"features"
save_path__leaf_coverages = "C:/Users/cyn_n/Desktop/Synthetic-Data-Deep-Learning/
generated_synthcity_tabular/ARF/"+folder_arf+"leaf_coveerage.pkl"
save_path_count_features = "C:/Users/cyn_n/Desktop/Synthetic-Data-Deep-Learning/
generated_synthcity_tabular/ARF/"+folder_arf+"result_test_concat_v2"
save_path_leaf_coverages = "C:/Users/cyn_n/Desktop/Synthetic-Data-Deep-Learning/
generated_synthcity_tabular/ARF/"+folder_arf+"result_test_concat_coverage"
save_path_dist = "C:/Users/cyn_n/Desktop/Synthetic-Data-Deep-Learning/
generated_synthcity_tabular/ARF/"+folder_arf+"result_test_concat.pkl"
read_path_dist = "C:/Users/cyn_n/Desktop/Synthetic-Data-Deep-Learning/
generated_synthcity_tabular/ARF/"+folder_arf+"result_test_concat"
read_path_dist2 = "C:/Users/cyn_n/Desktop/Synthetic-Data-Deep-Learning/
generated_synthcity_tabular/ARF/"+folder_arf+"result_test_concat_dist"

distributions = {
    
        'truncnorm',
        'expon',
        'lognorm',
       
    
        
    }
   