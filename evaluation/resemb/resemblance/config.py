
import pandas as pd
import pickle 
import gzip
import glob
import os
import joblib

    
valid_perc = 0.7 # 30 por ciento de los clientes
num_patient=3
num_visit_count = 5
name_file_similaritymatrixcos = "cosine_matrix_similarity"
file_principal = os.getcwd() + "/"
file_data =     file_principal + "generated_synthcity_tabular/ARF/"
name_file_ehr = "ARF_fixed_v"


def load_data(file_path):
    try:
        with gzip.open(file_path, 'rb') as f:
             return pickle.load(f)
    except:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                return data
            
            
#file ="/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/ARF_conditioned/results/synthetic_d/demos_prob_2forgemodified.pkl"
       
#bnds = joblib.load( file )
# The comment `# save ehr en la primera corrida` is indicating that the variable `save_ehr` is used to


# save ehr en la primera corrida
label_encoded = True
post_processing = True
save_ehr = True
#creating subjec

get_sample_synthetic_similar_real = True
#create subject id from admission
get_days_grom_visit_histogram = True


#si se tiene doppleganger con attributes y features
#synthetic_type = "Gaussian"
#synthetic_type = "Logistic"
#synthetic_type ="Bosting"
#synthetic_type = "Random"

#CONTINOUS GENERATION OF ARF
#synthetic_type = "sin_var_con"
#synthetic_type = "Otro"
synthetic_type ="label_decoder"
#type of odel
#type_archivo = 'demo_Arf'

type_archivo= 'ARFpkl'

#type_archivo= 'gru_Arf'

get_synthetic_subject_clustering = False
make_cosin_sim = False
#EVALUATION
visualization_dimension_wise_distribution_similarity = True
metric_dimension_wise_distribution_similarity = False
metric_joint_distribution_similarity_coverage=False
metric_joint_distribution_similarity_structure = False
metric_inter_dimensional_similarity=False
consistency_information = False
other_metrics = False

if type_archivo == "ARFpkl":
    folder = "ARF_fixed_sansvar/"
    #folder = "ARF_fixed_postpros/"
   # folder ="ARF_2_approach/"
    path_to_folder_syn = file_data+folder
    #file =path_to_folder_syn+ "synnthetic_datalog_ARF_1_approach.pkl"
    file = path_to_folder_syn +"synnthetic_datalog_arf_sin_var2.pkl"
    encoder_path =file_principal+"generative_input/input/encoders_entire_ceros_tabular_data_demos2.pkl"
    encoder = load_data(encoder_path)
    #data = load_data(file)
elif type_archivo == "gru_Arf":
    folder = "ARF_GRU/"
    #folder = "ARF_fixed_postpros/"
    path_to_folder_syn = file_data+folder
    file =path_to_folder_syn+ "synthetic_results2_more_subset_visits.pkl"

        
elif type_archivo =='demo_Arf': 
    folder =    'ARF_demo/'
    folder1 = file_principal+ "generative_input/input/ARF_conditioned/results/synthetic_d/"
    if synthetic_type == "label_decoder":
        encoder_path =file_principal+"generative_input/input/encoders_entire_ceros_tabular_data_demos2.pkl"
        encoder = load_data(encoder_path)
        file =file_principal+"generative_input/input/ARF_conditioned/results/synthetic_d/demos_prob_2forgemodified.pkl"


if post_processing:
    #changig existing values
    #cremmer function
    eliminate_variables_generadas_post = True
    create_visit_rank_col = True   
    get_admitted_time = True
    propagate_fistvisit_categoricaldata = True
    adjust_age_and_dates_get = True
    get_handle_hospital_expire_flag = True
    eliminate_negatives_var = True  
    create_days_between_visits_by_date_var = False  
    #creating  features
    if get_days_grom_visit_histogram:
        get_remove_duplicates = False
        get_0_first_visit = False
    else:
        get_remove_duplicates = True
        get_0_first_visit = True        
        

          
else:    
    eliminate_variables_generadas_post = False
    #changig existing values
    if type_archivo == "demo_Arf":
        create_visit_rank_col = False
    else:    
        create_visit_rank_col = True 
    get_admitted_time = True
    propagate_fistvisit_categoricaldata = False
    adjust_age_and_dates_get = False
    get_handle_hospital_expire_flag = False
    create_days_between_visits_by_date_var = False
    eliminate_negatives_var = False
    #creating  features
    if get_days_grom_visit_histogram:
        get_remove_duplicates = False
        get_0_first_visit = False
    
    get_remove_duplicates = False
    get_0_first_visit = False    
    
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
    



        



#path_to_folder_syn = file_data+folder
#folder = "ARF_fixed_v_sin_subject_id/"
#folder = "ARF_fixed_postpros/"
#folder = "ARF_fixed_v/"
#folder = "ARF_demo/"
#folder = "ARF_fixed_sansvar/cosine_sim_subj/"
#folder = "ARF_GRU/"




#file = path_to_folder_syn + "synthetic_data_generative_model_arf_per_arf_acumulative0.7.pkl"
#file = path_to_folder_syn + "synthetic_results2_more_subset_visits.pkl"
#file = path_to_folder_syn +"synthetic_datalog_5.pkl"
sample_patients_path =path_to_folder_syn + "sample_patients_fixed_v"
#read_dict


results_df = pd.DataFrame()
#trajectories analys

#name of synthetic data

#constraints name
remote = True
if remote:
    if label_encoded:
        exclude_codes = True
        if metric_dimension_wise_distribution_similarity:
            exclude_codes = False
            pass
        #features_path = "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/entire_ceros_tabular_data_demos2.pkl"
        features_path =file_principal+"generative_input/input/entire_ceros_tabular_data_demos2_whole.pkl"  
 
    else:
        exclude_codes = False

        features_path = file_principal+"generative_input/input/entire_ceros_tabular_data.pkl"

else:
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

variables_generadas_post = [ 'id_patient', 'ADMITTIME', 'visit_rank','days from last visit']
#cols to drop        
 
if adjust_age_and_dates_get:
    if get_synthetic_subject_clustering:
       columns_to_drop_syn = ['days from last visit_cumsum','similarity_score']
    else: 
        columns_to_drop_syn = ['days from last visit_cumsum']   
else:
    if get_synthetic_subject_clustering:
         columns_to_drop_syn = ['similarity_score']   
    else:
        columns_to_drop_syn = [] 
             
#cols continous
cols_continuous = [ 'Age', 'LOSRD_avg','days from last visit']
#categorical cols
columns_to_drop_sec = ['GENDER_M', 'GENDER_F', 'RELIGION_CATHOLIC', 'RELIGION_Otra', 'RELIGION_Unknown',
                          'MARITAL_STATUS_0', 'MARITAL_STATUS_DIVORCED', 'MARITAL_STATUS_LIFE PARTNER',
                          'MARITAL_STATUS_MARRIED', 'MARITAL_STATUS_SEPARATED', 'MARITAL_STATUS_SINGLE',
                          'MARITAL_STATUS_Unknown', 'MARITAL_STATUS_WIDOWED',
                          'ETHNICITY_Otra', 'ETHNICITY_Unknown', 'ETHNICITY_WHITE','GENDER_0'] 
#dependant visit , static information
dependant_fist_visit = ['ADMITTIME',  'RELIGION',

                        'MARITAL_STATUS',  'ETHNICITY','GENDER']

if label_encoded:
    columnas_demograficas = ['RELIGION_encoded', 'MARITAL_STATUS_encoded',  'ETHNICITY_encoded','GENDER_encoded']
    categorical_cols =columnas_demograficas +['visit_rank']
    columns_to_drop = []  
    get_handle_hospital_expire_flag = False
else:     
    columns_to_drop = ['LOSRD_sum', 'L_1s_last_p1','HADM_ID']  
    categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                        'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                        'MARITAL_STATUS',  'ETHNICITY','GENDER',"visit_rank","HOSPITAL_EXPIRE_FLAG"  ]

    columnas_demograficas =  ['Age',
                                
            'GENDER_M',
            'GENDER_F',
            'RELIGION_CATHOLIC',
            'RELIGION_Otra',
            'RELIGION_Unknown',
            'MARITAL_STATUS_0',
            'MARITAL_STATUS_DIVORCED',
            'MARITAL_STATUS_LIFE PARTNER',
            'MARITAL_STATUS_MARRIED',
            'MARITAL_STATUS_SEPARATED',
            'MARITAL_STATUS_SINGLE',
            'MARITAL_STATUS_Unknown',
            'MARITAL_STATUS_WIDOWED',
            'ETHNICITY_Otra',
            'ETHNICITY_Unknown',
            'ETHNICITY_WHITE'
        ]

# codes icd9 and drugs
keywords = ['diagnosis', 'procedures', 'drugs']
#path to synthetic data
#paths


train_ehr_dataset = load_data(features_path)

diagnosis_columns = list(train_ehr_dataset.filter(like="diagnosis").columns)  # Ajusta los índices según corresponda
procedure_columns = list(train_ehr_dataset.filter(like="procedures").columns)
medication_columns = list(train_ehr_dataset.filter(like="drugs").columns)


for i in dependant_fist_visit:
    print(i)
    print( list(train_ehr_dataset.filter(like=i).columns) )
#not used

 # TODO Modify*
attributes = False


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
    file_name = file_principal + "/train_sp/non_prepo/DATASET_NAME_non_prepo_non_preprocess.pkl"



#color pallet

# graph_settings.py

import matplotlib.pyplot as plt
import seaborn as sns

def set_graph_settings():
    # Set default color palette
    #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    #sns.set_palette(colors)
    sns.color_palette("Blues", as_cmap=True)

    # Set default plot style
    plt.style.use('seaborn-whitegrid')

    # Set default figure size
    plt.rcParams['figure.figsize'] = (10, 6)

    # Set default font size
    plt.rcParams['font.size'] = 12

    # Set default line width
    plt.rcParams['lines.linewidth'] = 2

    # Set default grid style
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.7

    # Set default legend settings
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.fontsize'] = 'medium'
    plt.rcParams['legend.loc'] = 'best'

if exclude_codes:
    list_col_exclute_path = file_principal+"generative_input/ARF_conditioned/results/no_code_list.pkl"
    list_col_exclute = joblib.load( list_col_exclute_path)
