import gzip
import pickle
import joblib

#traindemograohci codes prectors
save_model = True
train_model = False
train_path_forde = False

train_arf =False
demos_probs = True
remote = True
remote_can = False
train_DemographicPredictor = False
type_model = "Logistic2"

columns_to_drop = ['HOSPITAL_EXPIRE_FLAG','LOSRD_sum', 'L_1s_last_p1',
                   'HADM_ID',
                   'ADMISSION_TYPE_EMERGENCY', 'ADMISSION_TYPE_Otra',   
                           'ADMISSION_LOCATION_EMERGENCY ROOM ADMIT', 'ADMISSION_LOCATION_Otra',      
                             'ADMISSION_LOCATION_PHYS REFERRAL/NORMAL DELI',   
                                     'DISCHARGE_LOCATION_HOME', 'DISCHARGE_LOCATION_HOME HEALTH CARE',    
                                           'DISCHARGE_LOCATION_Otra', 'DISCHARGE_LOCATION_SNF',      
                                             'INSURANCE_Medicare', 'INSURANCE_Otra',]

continuous_cols=['Age_max']
columns_to_drop_sec = [
                       'GENDER_M', 'GENDER_F', 'RELIGION_CATHOLIC', 'RELIGION_Otra', 'RELIGION_Unknown',
                          'MARITAL_STATUS_0', 'MARITAL_STATUS_DIVORCED', 'MARITAL_STATUS_LIFE PARTNER',
                          'MARITAL_STATUS_MARRIED', 'MARITAL_STATUS_SEPARATED', 'MARITAL_STATUS_SINGLE',
                          'MARITAL_STATUS_Unknown', 'MARITAL_STATUS_WIDOWED',
                          'ETHNICITY_Otra', 'ETHNICITY_Unknown', 'ETHNICITY_WHITE','LOSRD_sum', 'L_1s_last_p1','HADM_ID','GENDER_0']
cols_continuous = ['Age_max', 'LOSRD_avg','days_between_visits'] 
categorical_cols=['GENDER_M', 'GENDER_F', 'RELIGION_CATHOLIC', 'RELIGION_Otra', 'RELIGION_Unknown',
                          'MARITAL_STATUS_0', 'MARITAL_STATUS_DIVORCED', 'MARITAL_STATUS_LIFE PARTNER',
                          'MARITAL_STATUS_MARRIED', 'MARITAL_STATUS_SEPARATED', 'MARITAL_STATUS_SINGLE',
                          'MARITAL_STATUS_Unknown', 'MARITAL_STATUS_WIDOWED',
                          'ETHNICITY_Otra', 'ETHNICITY_Unknown', 'ETHNICITY_WHITE']

#columns_to_drop_arf = ['days_between_visits','SUBJECT_ID']
columns_to_drop_arf = []
cols_continuous_d = ['Age_max', 'LOSRD_avg'] 
    #categorical olumn are the onts not in continuous and count_matrix_cols

#features_path = "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/entire_ceros_tabular_data.pkl"

#paths
static_features_size = len(continuous_cols) + len(categorical_cols)
hidden_size = 64

demographiccols = [  'RELIGION',

                        'MARITAL_STATUS',  'ETHNICITY','GENDER']

base_demographic_columns = [col +"_encoded" for col in demographiccols] +["Age_max"]
if remote:
    path_sec = "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/ARF_conditioned/results/"
    save_path_arf = "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/ARF_conditioned/"
    data_path =  "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/entire_ceros_tabular_data.pkl" 
    model_path =  'generative_input/ARF_conditioned/results/modelLogistic2.pkl'
    path_result = save_path_arf + "results/predictors_/"
    scaler_path = 'generative_input/ARF_conditioned/results/' + 'scaler.pkl'
    ruta_patients = "generated_synthcity_tabular/ARF/ARF_fixed_postpros/sample_patients_fixed_v.pkl"
    directory_predictors = "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/ARF_conditioned/results/predictors_"
    path_result_arf = path_result = save_path_arf + "results/"
elif remote_can:    
     save_path_arf = "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/ARF_conditioned/"
     data_path =  "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/entire_ceros_tabular_data.pkl" 
     model_path = save_path_arf + 'gru_conditional_probability_model.pth'
     path_result = save_path_arf + "results/"
     scaler_path = save_path_arf + 'scalers.pkl'
     ruta_patients = "generated_synthcity_tabular/ARF/ARF_fixed_postpros/sample_patients_fixed_v.pkl"
     path_sec =  "generative_input/ARF_conditioned/results/"
else:
    save_path_arf = "D:\\Synthetic-Data-Deep-Learning\\generated_synthcity_tabular\\ARF\\Bayes_prob\\" 
    data_path = "D:\\Synthetic-Data-Deep-Learning\\data\\intermedi\\SD\\inpput\\entire_ceros_tabular_data.pkl"
    synthetic_data = "D:\Synthetic-Data-Deep-Learning\generated_synthcity_tabular\ARF\ARF_fixed_postpros\synthetic_ehr_datasetARF_fixed_v.pkl"
    model_path = save_path_arf + 'gru_conditional_probability_model.pth'
    path_result = "generative_model\\ARF_conditioned\\results\\predictors"
    scaler_path = save_path_arf + 'scalers.pkl'


def load_data(file_path):
    try:
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        with open(file_path, 'rb') as f:
             data = pickle.load(f)    
             return data



input_size = 100  # Size of your static input
  # Or whatever size you used


sequence_length = 1  # Or whatever length you used

data =  load_data(data_path)

drug_columns = list(data.filter(like="drugs").columns)
diagnosis_columns = list(data.filter(like="diagnosis").columns)
procedure_columns = list(data.filter(like="procedures").columns)
medical_factors = drug_columns + diagnosis_columns + procedure_columns
num_classes = len(medical_factors)    
output_size = len(medical_factors)

demographic_cols = categorical_cols +continuous_cols  # Add your demographic columns

#list_col_exclute = joblib.load( 'generative_input/ARF_conditioned/results/predictors_/no_code_list_trained100.pkl') 
list_col_exclute = joblib.load( "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/ARF_conditioned/results/no_code_list.pkl")
medical_code_cols = [i for i in medical_factors if i not in list_col_exclute['codes_no'].to_list()] 


#FORDE parameters
#ARF
name_run = "arf_sin_var"
num_trees = 30
save_path_forde_demos = True        
train_arf_demos = True
create_demos_var_dataset = True
arf_demos = "ARF_sin_"+name_run+"2.pkl"
name_fored_output="density_fored_"+name_run+"2.pkl"
synthetic_data_demos = "synthetic_d/"+"synnthetic_datalog_"+name_run+"2.pkl"


if remote:
   xc ="/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/encoders_entire_ceros_tabular_data_demos2_whole.pkl"  
   path_dataset_demos_whole ="/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/entire_ceros_tabular_data_demos2_whole.pkl"  
   path_dataset_demos ="/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/entire_ceros_tabular_data_demos2.pkl"  
   encoders_demos = "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/encoders_entire_ceros_tabular_data_demos2.pkl"  


#shap valur
arf_path_file = "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/ARF_fixed_sansvar/arf_fixed_v.pkl"
path_img_shap = "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/ARF_conditioned/results/shap_values/"
#file_path="/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/ARF_conditioned/results/model_XGBoost.pkl"

#load_df = load_data(path_result+"synthetic_d/"+"demos_prob_"+file_path[-10:-5]+".pkl") ]