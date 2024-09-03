from deepecho import PARModel
from deepecho.demo import load_demo
from  config_conditonalarf import *
import joblib

# Load demo data


def sample_patients_list(ruta_patients, x_train):
    sample_patients = load_pickle(ruta_patients)
    sample_df = x_train[x_train['SUBJECT_ID'].isin(sample_patients)]
    return sample_df

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)    

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data



# Define data types for all the columns
# data_types = {
#     # Continuous columns
#     'Age_max': 'continuous',
#     'LOSRD_avg': 'continuous',
#     'days_between_visits': 'continuous',

   
#     'year': 'categorical',
#     'month': 'categorical',

#     # Medical factors (drugs, diagnosis, procedures)
#     **{col: 'count' for col in drug_columns + diagnosis_columns + procedure_columns}
# }

model = PARModel(epochs=128,verbose=True, cuda=True)

#prepare data
x_train = data# Your training data =
x_train["year"] = x_train['ADMITTIME'].dt.year
x_train['month'] = x_train['ADMITTIME'].dt.month

# Convert visit_date to datetime
# Count visits per patient

patients_multiple_visits = x_train[x_train["visit_rank"] > 1]["SUBJECT_ID"]
x_train = x_train[x_train['SUBJECT_ID'].isin(patients_multiple_visits[:3000])]
x_train = x_train.sort_values(['SUBJECT_ID', 'visit_rank'])

#x_train= sample_patients_list(ruta_patients, x_train)[:100]    
    
    
    # se quitan columnas que no se utilizan y se convierte en categoricas, la matrix de conteo, subject id, admission dat

#columns_to_drop = ['LOSRD_sum', 'L_1s_last_p1','HADM_ID',"ADMITTIME",'GENDER_0']
print("Cols to eliminate",columns_to_drop_sec)
x_train = x_train.drop(columns=columns_to_drop_sec)  


model.fit(
    data=x_train,
    entity_columns=['SUBJECT_ID'],
    context_columns=[],
    
    sequence_index='visit_rank'
)
joblib.dump(model, path_sec+"deepecho_model.pkl")

#model.save( path_sec+"deepecho_model.pkl")
# Sample new data
synthethic_data = model.sample(num_entities=len(x_train))
save_pickle(synthethic_data,path_sec+"synthetic_results.py")
