#create feature with random sampled pkl
create_features = True
#analyze the feature Subject_I
continous_variable_ks_test = False
#get continous values count and parameters leaf (mean std deviation)
get_count_variables_per_node_tree_cont = True
#get aras nodes descete
get_long_params_per_node_tree_distcre = False
#threshol for values in leaves
threshold = 30
#obtain mean_per node
obtain_promedio_por_nodo_ks_dist = False
#perform ks test for each node
perform_ks_test = False 
visualization_days_bewteen = False

"id subject if"
col_cat = "SUBJECT_ID"
#threshold que se analiza
threshold_values = 30
threshold_continous = 30
col_continous = "days_between_visits"

columns_to_drop = ['LOSRD_sum', 'L_1s_last_p1','HADM_ID',"ADMITTIME",'GENDER_0']
cols_continuous = ['Age_max', 'LOSRD_avg','days_between_visits'] 
keywords = ['diagnosis', 'procedures', 'drugs']


fored_fixed = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/FORED_fixed"
path_arf = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF_local/"
sample_patients_path = "generated_synthcity_tabular/ARF/sample_patients"
original_data_path = "data/intermedi/SD/inpput/entire_ceros_tabular_data.pkl"
ruta_continous_observations = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/ARF_gmm/result_test_concat"
#save path
save_path_features = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/x_real"
save_path_count_features = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/ARF_gmm/result_test_concat.pkl"
save_path_dist = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/ARF_gmm/result_test_concat.pkl"


