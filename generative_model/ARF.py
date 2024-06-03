import sys
import os
os.chdir('/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/')
sys.path.append('/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning')
from arfpy import arf
from generative_model.utils import *
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices
import scipy
import pandas as pd
import shap

remote = False
path_arf = "generative_model/ARF/"
percentage_to_sample = 0.03

if remote:
    
    train_data_features = load_data("generative_input/entire_ceros_tabular_data.pkl")
else:
    train_data_features = load_data("data/intermedi/SD/inpput/entire_ceros_tabular_data.pkl")

train_data_features["year"] = train_data_features['ADMITTIME'].dt.year
train_data_features['month'] = train_data_features['ADMITTIME'].dt.month
      
    
    
# se quitan columnas que no se utilizan y se convierte en categoricas, la matrix de conteo, subject id, admission data
columns_to_drop = ['LOSRD_sum', 'L_1s_last_p1','HADM_ID',"ADMITTIME"]

train_data_features = train_data_features.drop(columns=columns_to_drop)  

cols_continuous = ['Age_max', 'LOSRD_avg','days_between_visits',"visit_rank"] 
keywords = ['diagnosis', 'procedures', 'drugs']
# se filtras las columns con keywords
count_matrix_cols = filter_keywords(train_data_features,keywords) 
#categorical olumn are the onts not in continuous and count_matrix_cols
categorical_cols =[col for col in train_data_features.columns if col not in cols_continuous+count_matrix_cols]


#change to categorilcal 
train_data_features = convertir_categoricas(train_data_features,categorical_cols)
print(train_data_features.dtypes)
# samplear 3# fre los pacientes
sample_df, sample_patients_r = sample_patients(train_data_features,percentage_to_sample)

#guardar la muestra de pacientes
save_load_numpy(sample_patients_r,save=True,load=False,name=path_arf +'sample_patients.npy')

# train random adversarial forest
my_arf = arf.arf(x = sample_df) 



#se obtiene parametro por hoja y por arbol
FORDE = my_arf.forde()

# guradar FRODE
save_pkl(FORDE,path_arf+"FORED")

#obtener 0 coverage variables
categorical_columns = sample_df.select_dtypes(include=['object', 'category']).columns.tolist()
x_real = sample_df.to_numpy()
orig_colnames = sample_df.columns.tolist()
#clf
clf = FORDE['forest']



# Create factor_cols: Boolean array indicating categorical columns
factor_cols = sample_df.columns.isin(categorical_columns)
#obetener variables con coverage = 0
zero_coverage_vars = identify_zero_coverage_variables(clf, x_real, orig_colnames, factor_cols, clf.n_estimators)
print("Zero Coverage Variables:", zero_coverage_vars)
save_pkl(zero_coverage_vars,path_arf+"zero_coverage_vars")


#obtener importancia shap_values del modelo
#filtrar el test set 
X_test_v = train_data_features[~train_data_features['SUBJECT_ID'].isin(sample_patients_r)]
#obtener los shap values
df_shap_values = shap_values(clf, X_test_v[:300])
save_pkl(df_shap_values,path_arf+"shap_values")

#generate synthetic data
df_syn = my_arf.forge(n = df_sample.shape[0])
# save synthetic data
save_pkl(df_syn,path_arf+"synthetic_data_generative_model_arf_per_"+percentage_to_sample)
