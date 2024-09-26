
# stdlib
import sys
import traceback
import warnings
from tqdm import tqdm
# synthcity absolute
import os
from synthcity.utils.serialization import save_to_file, load_from_file, save, load
import wandb
os.chdir('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning')
# synthcity absolute
import synthcity.logger as log
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import TimeSeriesSurvivalDataLoader
import sys
import warnings
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, PowerTransformer

warnings.filterwarnings("ignore")

# third party
from sklearn.datasets import load_iris

# synthcity absolute
import synthcity.logger as log
#from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader


# synthcity absolute
from synthcity.benchmark import Benchmarks
import numpy as np 



log.add(sink=sys.stderr, level="INFO")
warnings.filterwarnings("ignore")
import pandas as pd

import gzip
import pickle
def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)
    
SD = "generative_model/"
 
train_split = False
if train_split == True:
    print(os.getcwd())
        
        

    
    df = load_data("generative_input/entire_ceros_tabular_data.pkl")
    le = LabelEncoder()

    # Ajustar y transformar la columna 'subject_id'
    df['subject_id'] = le.fit_transform(df['SUBJECT_ID'])
    #inverso
    #df['subject_id'] = le.inverse_transform(df['subject_id'])
    cols_to_drop = df.filter(like='Unnamed', axis=1).columns

    df.drop(cols_to_drop, axis=1, inplace=True)
    cols_to_drop1 = ['HADM_ID','SUBJECT_ID']
    df.drop(cols_to_drop1, axis=1, inplace=True)
    
    # standarise; 
    def preprocess(X, prep, columns_to_normalize):
        global transformers
        transformers = {}
        
        if prep == "std":
            scaler = StandardScaler()
            X[columns_to_normalize] = scaler.fit_transform(X[columns_to_normalize])
            transformers = {col: scaler for col in columns_to_normalize}
            
        elif prep == "max":
            transformer = MaxAbsScaler()
            X[columns_to_normalize] = transformer.fit_transform(X[columns_to_normalize])
            transformers = {col: transformer for col in columns_to_normalize}
            
        elif prep == "power":
            pt = PowerTransformer()
            X[columns_to_normalize] = pt.fit_transform(X[columns_to_normalize])
            transformers = {col: pt for col in columns_to_normalize}
        
        return X
    
    columns_to_normalize = [i for i in df.columns if i!='ADMITTIME' and i!='HOSPITAL_EXPIRE_FLAG']
    

    def inverse_preprocess(X, columns_to_normalize):
        for col in columns_to_normalize:
            X[col] = transformers[col].inverse_transform(X[[col]])
        return X
    columns_to_normalize = [i for i in df.columns if i != "ADMITTIME" and i!='HOSPITAL_EXPIRE_FLAG']
    
    def standardize_data_types(data):
            """
            Convert all numerical data to float64.
            
            Parameters:
                data (numpy.ndarray): Input data array.

            Returns:
                numpy.ndarray: An array with all data converted to float64.
            """
            return data.astype(np.float64)


# Converting and checking the types
    df[columns_to_normalize] = standardize_data_types(df[columns_to_normalize])
    df = preprocess(df, "std", columns_to_normalize)
    # QUITAR ENEL REAL DATASET
  
    valid_perc = 0.5
    dataset_name = 'tabular_'
    
    #split_dataset
    N = df.shape[0]
    N_train = int(N * (1 - valid_perc))
    N_valid = N - N_train
    train_data_features = df[:N_train]
    valid_data_features = df[N_train:]   

    print("train/valid shapes: ", train_data_features.shape, valid_data_features.shape)    
    SD_DATA_split = "train_sp/"
    with gzip.open(SD_DATA_split + dataset_name + 'train_data_features.pkl', 'wb') as f:
        pickle.dump(train_data_features, f)
    with gzip.open(SD_DATA_split+ dataset_name + 'valid_data_features.pkl', 'wb') as f:
        pickle.dump(valid_data_features, f)


        
    
##from synthcity.utils.serialization import load, load_from_file, save, save_to_file


train_data_features = load_data("generative_input/entire_ceros_tabular_data.pkl")

#reloaded = load_from_file('./adsgan_10_epochs.pkl')


#df = df.iloc[:2000,1:]

# stdlib
# Benchmarking the synthetic data generation process
print(train_data_features.shape)
import os


ruta = "train_sp/generated/"  # Reemplaza esto con la ruta que quieres verificar

if os.path.exists(ruta):
    print("La ruta existe.")
else:
    print("La ruta no existe.")
        
loader = GenericDataLoader(train_data_features, target_column="HOSPITAL_EXPIRE_FLAG", sensitive_columns=[])
#[ 'uniform_sampler','ctgan','ddpm', 'nflow', 'marginal_distributions', 'tvae', 'rtvae', 'bayesian_network', 'arf']
for i in tqdm([ 'ctgan']):
    try: 
        print(i)
         
        syn_model = Plugins().get( i)
        syn_model.fit(loader)
        try:
           syn_model.generate(count=train_data_features.shape[0]).dataframe().to_csv("train_sp/generated/" +i+"_100_epochs.pkl")
        except  Exception as e:
            print("Ha ocurrido una excepción:")
            print(e)
        try:    
            syn_model.save_to_file('train_sp/output/' +i+'_100_epochs.pkl')
            buff = save(syn_model)
        except  Exception as e:
            print("Ha ocurrido una excepción:")
            print(e)
        

    except:
       print(i)
 
#Plugins(categories=["generic"]).list()
#['ctgan',  'ddpm', 'great', 'nflow', 'marginal_distributions', 'tvae', 'rtvae', 'bayesian_network', 'uniform_sampler', 'arf']
# , 'great'
'''
score = Benchmarks.evaluate(
    [
        (f"example_{model}", model, {})  # testname, plugin name, plugin args
        for model in ['ctgan','ddpm', 'nflow', 'marginal_distributions', 'tvae', 'rtvae', 'bayesian_network', 'uniform_sampler', 'arf']
    ],
    loader,
    synthetic_size=10,
    task_type="classification",
    metrics = {'sanity': ['data_mismatch', 'common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
               'stats': ['jensenshannon_dist', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist'], 
               'performance' : [ 'mlp', 'xgb', 'feat_rank_distance'] ,
                 'privacy': [ 'k-anonymization', 'k-map', 'distinct l-diversity', 'identifiability_score', ]
               },   
    repeats=1,
)

Benchmarks.print(score)

Benchmarks.highlight(score)
# third party
import numpy as np
import pandas as pd

means = []
for plugin in score:
    data = score[plugin]["mean"]
    res = score[plugin]["mean"].to_dict()
    directions = score[plugin]["direction"].to_dict()
    means.append(data)
    
    run = wandb.init(project="my_project", name=plugin)

    # Registrar los valores en wandb
    wandb.log(res)

    # Finalizar el run
    run.finish()




out = pd.concat(means, axis=1)
out.set_axis(score.keys(), axis=1, inplace=True)

bad_highlight = "background-color: lightcoral;"
ok_highlight = "background-color: green;"
default = ""


def highlights(row):
    metric = row.name
    if directions[metric] == "minimize":
        best_val = np.min(row.values)
        worst_val = np.max(row)
    else:
        best_val = np.max(row.values)
        worst_val = np.min(row)

    styles = []
    for val in row.values:
        if val == best_val:
            styles.append(ok_highlight)
        elif val == worst_val:
            styles.append(bad_highlight)
        else:
            styles.append(default)

    return styles


out.style.apply(highlights, axis=1)

'''