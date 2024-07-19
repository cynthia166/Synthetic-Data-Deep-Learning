
import pandas as pd
import pickle 
import gzip
import glob
# The comment `# save ehr en la primera corrida` is indicating that the variable `save_ehr` is used to
# determine whether to save Electronic Health Record (EHR) data during the first run of the program.
# If `save_ehr` is set to `True`, then the EHR data will be saved. Otherwise, if it is set to `False`,
# the EHR data will not be saved during the first run. This variable controls the behavior related to
# saving EHR data based on its value.
# save ehr en la primera corrida
# TODO Modify*

save_ehr = True
#si se tiene doppleganger con attributes y features


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

    


create_visit_rank_col = True       
post_processing = False
if post_processing:
    
    #create subject id from admission
    
    get_sample_synthetic_similar_real = True
    get_admitted_time = True
    propagate_fistvisit_categoricaldata = True
    #considerin th date as principa pivot
    create_days_between_visits_by_date_var = False
    adjust_age_and_dates_get = True
    get_handle_hospital_expire_flag = True
    eliminate_negatives_var = True
      
    get_days_grom_visit_histogram = True
    if get_days_grom_visit_histogram:
        get_remove_duplicates = False
        get_0_first_visit = False
    else:
        get_remove_duplicates = True
        get_0_first_visit = True        
        
         
else:    
    get_sample_synthetic_similar_real = False
    get_admitted_time = False
    create_days_between_visits_by_date_var = False
    propagate_fistvisit_categoricaldata = False
    adjust_age_and_dates_get = False
    get_handle_hospital_expire_flag = False
    eliminate_negatives_var = True
    #create visits from histogram
    get_days_grom_visit_histogram = False
    if get_days_grom_visit_histogram:
        get_remove_duplicates = False
        get_0_first_visit = False
    else:
        if get_synthetic_subject_clustering:
            get_remove_duplicates = False
            get_0_first_visit = False    
        else:
            get_remove_duplicates = True
            get_0_first_visit = True    
        
    #eliminate_negatives_var = False
# this is only for Dopplganger
#EVALUATION
visualization_dimension_wise_distribution_similarity = False
metric_dimension_wise_distribution_similarity = False
metric_joint_distribution_similarity_coverage=False
metric_joint_distribution_similarity_structure = True
metric_inter_dimensional_similarity=False
consistency_information = False
other_metrics = False

valid_perc = 0.7 # 30 por ciento de los clientes
num_patient=3
num_visit_count = 5

file_data =     "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/"
name_file_ehr = "ARF_fixedv"
#folder = "ARF_fixed_v_sin_subject_id/"
#folder = "ARF_fixed_postpros/"
#folder = "ARF_fixed_v/"
folder = "ARF_fixed_rnn/"
path_to_folder_syn = file_data+folder
file =path_to_folder_syn+ "synthetic_data_generative_model_arf_per_fixed_v0.7.pkl"
#type of model
type_archivo = 'ARFpkl'
sample_patients_path =path_to_folder_syn + "sample_patients_fixed_v"
#read_dict


results_df = pd.DataFrame()
#trajectories analys

#name of synthetic data

#constraints name

features_path = "data/intermedi/SD/inpput/entire_ceros_tabular_data.pkl"
make_read_constraints_name = 'synthetic_ehr_dataset_contrainst_ARF_'+name_file_ehr+'.pkl'
path_to_directory = 'generated_synthcity_tabular/*'  # Asegúrate de incluir el asterisco al final
csv_files = glob.glob(path_to_directory + '.pkl')
    

#name of file to be save ehr

# Not modif
#features path/ original data path
file_path_dataset =   path_to_folder_syn
#path of the patients
#image of path
path_img = path_to_folder_syn+"img/"
## columns
eliminate_variables_generadas_post = True
variables_generadas_post = [ 'id_patient', 'ADMITTIME', 'visit_rank','days from last visit']
#cols to drop        
columns_to_drop = ['LOSRD_sum', 'L_1s_last_p1','HADM_ID']   
if adjust_age_and_dates_get:
   columns_to_drop_syn = ['days from last visit_cumsum']
else:
    columns_to_drop_syn = []   
#cols continous
cols_continuous = [ 'Age', 'LOSRD_avg','days from last visit']
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
def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

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
