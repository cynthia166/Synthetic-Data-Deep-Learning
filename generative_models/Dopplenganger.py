import pandas as pd 
import sys
import os
import os

# Imprimir la ruta del directorio actual
print(os.getcwd())

# Cambiar el directorio de trabajo actual a 'nueva/ruta'

print(os.getcwd())

sys.path.append('../')
#sys.path.append('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning')
from preprocessing.preprocess_input_SDmodel import *
import numpy as np
#from evaluation.Resemblance.metric_stat import *
#from synthcity.metrics.eval_statistical import AlphaPrecision
from preprocessing.config import *
# Instanciar la clase
#alpha_precision = AlphaPrecision(**kwargs)
# stdlib
import sys
import warnings
#from evaluation.Resemblance.metric_stat import KolmogorovSmirnovTest
# synthcity absolute

warnings.filterwarnings("ignore")

import numpy as np
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig

import pandas as pd
#from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
     
from preprocessing.preprocess_input_SDmodel import *

import os
os.chdir('./')

# Imprimir la ruta del directorio actual
print(os.getcwd())
SD = "generative_models/SD_results_"
ruta = SD  # Reemplaza esto con la ruta que quieres verificar

if os.path.exists(ruta):
    print('La ruta existe.')
    print(ruta)
else:
    print('La ruta no existe.')
    print(ruta)
    
    

import pickle
import gzip


def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)
dataset_name = 'DATASET_NAME_prepo'
features = load_data("train_sp/train_splitDATASET_NAME_preprotrain_data_features.pkl")
attributes=load_data("train_sp/train_splitDATASET_NAME_preprotrain_data_attributes.pkl")
#outcomes=pd.read_csv(DARTA_INTERM_intput+"outcomes_preprocess.csv")     
#####################################################################################



#numpy_array_t = np.stack([df.to_numpy() for df in temporal_dataframes])
#features = numpy_array_t.reshape(numpy_array_t.shape[0],numpy_array_t.shape[2],numpy_array_t.shape[1])
#attributes = static_data.to_numpy()


#full_train_data = np.array(data)
N, T, D = features.shape   
print('data shape:', N, T, D) 
print('attributes shape:', attributes.shape)
valid_perc = 0.1
dataset_name = 'DATASET_NAME_prepro'
name_file = "Dopplenganger_epochocs_120.pth"  
#realizar train and test split
#split(valid_perc,dataset_name):


   
######


#attributes = np.random.rand(10000, 3)
#features = np.random.rand(10000, 20, 2)
 
config = DGANConfig(
    max_sequence_len=20,
    sample_len=2,
    batch_size=1000,
    epochs=10
)
config2 = DGANConfig(
    max_sequence_len=666,
    sample_len=111,
    batch_size=100,
    epochs=120
)

model = DGAN(config2)

model.train_numpy(  features=features,
        attributes= attributes)


model.save(SD+name_file )
#synthetic_attributes, synthetic_features = model.generate_numpy(n =len(sample_patients))





#m = model.load("aux/model_name.pth"  )
#synthetic_attributes, synthetic_features = m.generate_numpy(n =len(sample_patients))


