import pandas as pd 
import sys
import os
import os

# Imprimir la ruta del directorio actual
print(os.getcwd())

# Cambiar el directorio de trabajo actual a 'nueva/ruta'

print(os.getcwd())

sys.path.append('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning')
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
import pickle
import gzip
import os
import config
#os.chdir('./')

# Imprimir la ruta del directorio actual

def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)
    
SD = "generative_models/"
 
train_split = False
if train_split == True:
    print(os.getcwd())
        
        

    
    features = load_data("train_sp/non_prepo/DATASET_NAME_non_prepo_non_preprocess.pkl")
    attributes=pd.read_csv("train_sp/non_prepo/static_data_non_preprocess.csv")
    numpy_array_t = np.stack([df.to_numpy() for df in features])
    features = numpy_array_t.reshape(numpy_array_t.shape[0],numpy_array_t.shape[2],numpy_array_t.shape[1])
    attributes = attributes.to_numpy()

    N, T, D = features.shape   
    print('data shape:', N, T, D) 
    print('attributes shape:', attributes.shape)
    valid_perc = 0.1
    dataset_name = '/non_prepo/DATASET_NAME_non_prepro'
    
    split(valid_perc,dataset_name,features,attributes)

#outcomes=pd.read_csv(DARTA_INTERM_intput+"outcomes_preprocess.csv")     
#########################################################################
#########################################################################
#########################################################################
##########################LOAD DATA##########################
#########################################################################
#########################################################################
dataset_name = SD
ruta = SD  # Reemplaza esto con la ruta que quieres verificar

if os.path.exists(ruta):
    print('La ruta existe.')
    print(ruta)
else:
    print('La ruta no existe.')
    print(ruta)
   
dataset_name = '/non_prepo/DATASET_NAME_non_prepro'
features_train = load_data('train_sp' + dataset_name + 'train_data_features.pkl')
attribute_train = load_data('train_sp'  + dataset_name + 'train_data_attributes.pkl')
#full_train_data = np.array(data)


name_file = "Dopplenganger_nonprepo_epochocs_120.pth"  
#realizar train and test split
#split(valid_perc,dataset_name):


   
######


#attributes = np.random.rand(10000, 3)
#features = np.random.rand(10000, 20, 2)
 
#config = DGANConfig(    max_sequence_len=20,sample_len=2, batch_size=1000,epochs=10)

config2 = DGANConfig(
    max_sequence_len=features_train.shape[1],
    sample_len=661,
    batch_size=100,
    epochs=60
)

model = DGAN(config2)

model.train_numpy(  features=features_train,
        attributes= attribute_train)


model.save(SD+name_file )
#synthetic_attributes, synthetic_features = model.generate_numpy(n =len(sample_patients))





#m = model.load("aux/model_name.pth"  )
#synthetic_attributes, synthetic_features = m.generate_numpy(n =len(sample_patients))


