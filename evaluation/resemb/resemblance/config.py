import pandas as pd
# save ehr en la primera corrida
save_ehr = True
# en la segunda solo se leen
read_ehr = False
#si se tiene dopplegange con attraibutes y features
attributes = False


if attributes:
    path_o = "train_sp/"    
    attributes_path_train= "non_prepo/DATASET_NAME_non_preprotrain_data_attributes.pkl"
    features_path_train = "non_prepo/DATASET_NAME_non_preprotrain_data_features.pkl"
    features_path_valid = "non_prepo/DATASET_NAME_non_preprovalid_data_features.pkl"
    attributes_path_valid = "non_prepo/DATASET_NAME_non_preprovalid_data_attributes.pkl"
    synthetic_path_attributes = 'non_prepo/DATASET_NAME_non_prepronon_prepo_synthetic_attributes_10.pkl'
    synthetic_path_features = 'non_prepo/DATASET_NAME_non_prepronon_prepo_synthetic_features_10.pkl'
    
    # esta es para agregar la columnas
    dataset_name = 'DATASET_NAME_non_prepo'
    file_name = "train_sp/non_prepo/DATASET_NAME_non_prepo_non_preprocess.pkl"

#cols to drop        
columns_to_drop = ['LOSRD_sum', 'L_1s_last_p1','HADM_ID']   
cols_to_drop_syn = "days_between_visits_cumsum"
#cols continous
cols_continuous = [ 'Age_max', 'LOSRD_sum','visit_rank','days_between_visits']
#categorical cols
categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                        'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                        'MARITAL_STATUS',  'ETHNICITY','GENDER',"visit_rank","HOSPITAL_EXPIRE_FLAG"  ]
#dependant visit , static information
dependant_fist_visit = ['ADMITTIME',  'RELIGION',
                        'MARITAL_STATUS',  'ETHNICITY','GENDER'] 
# codes icd9 and drugs
keywords = ['diagnosis', 'procedures', 'drugs']
#path to synthetic data
path_to_directory = 'generated_synthcity_tabular/*'  # Asegúrate de incluir el asterisco al final
valid_perc = 0.3 # 30 por ciento de los clientes
results_df = pd.DataFrame()
    
#csv_files = ['generated_synthcity_tabular/adsgantotal_0.2_epochs.pkl','generated_synthcity_tabular/pategantotal_0.2_epochs.pkl']
#file = 'generated_synthcity_tabular/arftotal_0.2_epochs.pkl'
#diel to analys
file = '/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/synthetic_data_generative_model_arf_per_0.7.pkl'
#features path
features_path = "data/intermedi/SD/inpput/entire_ceros_tabular_data.pkl"

file_path_dataset =     "generated_synthcity_tabular/ARF/"