import pandas as pd
from utilsstats import load_data
# save ehr en la primera corrida

save_ehr = True
#si se tiene dopplegange con attraibutes y features

attributes = False
if save_ehr:
    read_ehr = False
    # to  make post-processing
    make_contrains = True
    #save contrains
    save_constrains = True
else: 
    read_ehr = True
    # to  make post-processing
    make_contrains = False
    #save contrains
    save_constrains = False

    

# si los datos son normalizados continuos
inver_normalize = False
# si subject continuo was modeled continous; subject_continous
subject_continous = True
# this is only for Dopplganger
valid_perc = 0.3 # 30 por ciento de los clientes
results_df = pd.DataFrame()
#trajwctories analys
num_patient=3
num_visit_count = 5
    
#csv_files = ['generated_synthcity_tabular/adsgantotal_0.2_epochs.pkl','generated_synthcity_tabular/pategantotal_0.2_epochs.pkl']
#name of synthetic data
file = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF_local/synthetic_data_continous_id_0.4.pkl"
#file = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF_local/synthetic_data_continous_id_0.4.pkl"
#file = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/ARF_norm/synthetic_data_generative_model_arf_per_norm0.7.pkl"
#name  of constraints file
name_file_ehr = 'ARF_local'
make_read_constraints_name = 'synthetic_ehr_dataset_contrainst_ARF_'+name_file_ehr+'.pkl'
#ame of file to be save ehr
 # distribution limits initially modified before the desition tree
#type of model
type_archivo = 'ARFpkl'
#features path/ original data path
folder = "ARF_local"
file_path_dataset =     "generated_synthcity_tabular/ARF/"+folder+"/"
file_path_dataset =     "generated_synthcity_tabular/"+folder+"/"
#path of the patients
sample_patients_path ="generated_synthcity_tabular/"+folder+"/sample_patients"

#image of path
path_img = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/"+folder+"/img/"


## columns
#cols to drop        
columns_to_drop = ['LOSRD_sum', 'L_1s_last_p1','HADM_ID']   
columns_to_drop_syn = ['days_between_visits_cumsum']
#cols continous
cols_continuous = [ 'Age_max', 'LOSRD_avg','days_between_visits','id_patient']
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
#paths
features_path = "data/intermedi/SD/inpput/entire_ceros_tabular_data.pkl"

path_to_directory = 'generated_synthcity_tabular/*'  # Asegúrate de incluir el asterisco al final

train_ehr_dataset = load_data(features_path)
diagnosis_columns = list(train_ehr_dataset.filter(like="diagnosis").columns)  # Ajusta los índices según corresponda
procedure_columns = list(train_ehr_dataset.filter(like="procedures").columns)
medication_columns = list(train_ehr_dataset.filter(like="drugs").columns)

# Dooppleganger
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
    file_name = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/train_sp/non_prepo/DATASET_NAME_non_prepo_non_preprocess.pkl"
