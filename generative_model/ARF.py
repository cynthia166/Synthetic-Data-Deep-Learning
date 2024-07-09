import sys
import os
os.chdir('/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/')
sys.path.append('/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning')
from arfpy import arf
from generative_model.utils_arf import *
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices
import pandas as pd
from utils_arf import *
import scipy.stats
from ARF_GMM  import arfgmm

remote = False
percentage_to_sample = 0.7
train = True
coverage_vars = False
normalize_continous = False
type_arf = "gmm"

if remote:
    path_arf  = "ARF/"
    train_data_features = load_data("generative_input/entire_ceros_tabular_data.pkl")
else:
    train_data_features = load_data("data/intermedi/SD/inpput/entire_ceros_tabular_data.pkl")
    path_arf = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF_local/todas_var_cat/"



train_data_features["year"] = train_data_features['ADMITTIME'].dt.year
train_data_features['month'] = train_data_features['ADMITTIME'].dt.month
      
    
# se quitan columnas que no se utilizan y se convierte en categoricas, la matrix de conteo, subject id, admission data
columns_to_drop = ['LOSRD_sum', 'L_1s_last_p1','HADM_ID',"ADMITTIME",'GENDER_0']

train_data_features = train_data_features.drop(columns=columns_to_drop)  

#cols_continuous = ['Age_max', 'LOSRD_avg','days from last visit',"SUBJECT_ID"] 
cols_continuous = []
keywords = ['diagnosis', 'procedures', 'drugs']
# se filtras las columns con keywords
#count_matrix_cols = filter_keywords(train_data_features,keywords) 
#categorical olumn are the onts not in continuous and count_matrix_cols
categorical_cols =[col for col in train_data_features.columns if col not in cols_continuous]


#change to categorilcal 
train_data_features = convertir_categoricas(train_data_features,categorical_cols)
print(train_data_features.dtypes)
# samplear 3# fre los pacientes
sample_df, sample_patients = sample_patients(train_data_features,percentage_to_sample)

#guardar la muestra de pacientes
#save_load_numpy(sample_patients_r,save=True,load=False,name=path_arf +'sample_patients.npy')
save_pkl(sample_patients,path_arf +'sample_patients')
if train:
    # train random adversarial forest
    if type_arf =="gmm":
        my_arf = arfgmm(x=sample_df)
        FORDE = my_arf.forde()
        clf = FORDE["forest"] 
        save_pkl(FORDE,path_arf+"FORDE")
    else:    
        my_arf = arf.arf(x = sample_df) 
        dic_train_arf  = train_arf2(train_data_features, num_trees=30, delta=0, max_iters=10, early_stop=True, verbose=True, min_node_size=5)
        clf = dic_train_arf["clf"] ,



    #se obtiene parametro por hoja y por arbol
    #FORDE = dic_train_arf.forde()
    # guradar FRODE
    #save_pkl(FORDE,path_arf+"FORED2")
    save_pkl(clf,path_arf+"train_arf")
else:
   #load model
    FORDE = load_pkl(path_arf+"FORED2")
#

##clf
#clf = FORDE['forest']



#obetener variables con coverage = 0
if coverage_vars:
    # Create factor_cols: Boolean array indicating categorical columns
    #obtener 0 coverage variables
    categorical_columns = sample_df.select_dtypes(include=['object', 'category']).columns.tolist()
    x_real = sample_df.to_numpy()
    orig_colnames = sample_df.columns.tolist()
    factor_cols = sample_df.columns.isin(categorical_columns)
    zero_coverage_vars = identify_zero_coverage_variables(clf, x_real, orig_colnames, factor_cols, clf.n_estimators)
    print("Zero Coverage Variables:", zero_coverage_vars)
    save_pkl(zero_coverage_vars,path_arf+"zero_coverage_vars")



synthetic_data = my_arf.forge(
    n=sample_df.shape[0],
 )

save_pkl(synthetic_data,path_arf+"synthetic_data_continous_id_"+str(percentage_to_sample))


   
