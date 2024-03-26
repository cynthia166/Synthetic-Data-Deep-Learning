import pandas as pd 
import sys
import os
import os

# Imprimir la ruta del directorio actual
print(os.getcwd())

# Cambiar el directorio de trabajo actual a 'nueva/ruta'

sys.path.append('../')
#sys.path.append('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning')
from preprocessing.preprocess_input_SDmodel import *
import numpy as np
from evaluation.Resemblance.metric_stat import *
#from synthcity.metrics.eval_statistical import AlphaPrecision
from preprocessing.config import *
# Instanciar la clase
#alpha_precision = AlphaPrecision(**kwargs)
# stdlib
import sys
import warnings
from evaluation.Resemblance.metric_stat import KolmogorovSmirnovTest
# synthcity absolute

warnings.filterwarnings("ignore")

import numpy as np
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig

import pandas as pd
#from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
     
from preprocessing.preprocess_input_SDmodel import *
os.chdir('./')
import os

# Imprimir la ruta del directorio actual
print(os.getcwd())

ruta = SD_DATA_split  # Reemplaza esto con la ruta que quieres verificar

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
temporal_dataframes = load_data(DARTA_INTERM_intput + dataset_name + '_preprocess.pkl')
#observation_data=pd.read_csv(DARTA_INTERM_intput+"observation_data_preprocess.csv")
static_data=pd.read_csv(DARTA_INTERM_intput+"static_data_preprocess.csv")
#outcomes=pd.read_csv(DARTA_INTERM_intput+"outcomes_preprocess.csv")     
#####################################################################################



numpy_array_t = np.stack([df.to_numpy() for df in temporal_dataframes])
features = numpy_array_t.reshape(numpy_array_t.shape[0],numpy_array_t.shape[2],numpy_array_t.shape[1])
attributes = static_data.to_numpy()


#full_train_data = np.array(data)
N, T, D = features.shape   
print('data shape:', N, T, D) 

valid_perc = 0.1

# further split the training data into train and validation set - same thing done in forecasting task
N_train = int(N * (1 - valid_perc))
N_valid = N - N_train

# Shuffle data
#np.random.shuffle(full_train_data)

train_data_features = features[:N_train]
valid_data_features = features[N_train:]   

train_data_attributes = attributes[:N_train]
valid_data_attributes = attributes[N_train:]   
print("train/valid shapes: ", train_data_features.shape, valid_data_features.shape)    
dataset_name = 'DATASET_NAME_prepro'
with gzip.open(SD_DATA_split + dataset_name + 'train_data_features.pkl', 'wb') as f:
    pickle.dump(train_data_features, f)
with gzip.open(SD_DATA_split+ dataset_name + 'valid_data_features.pkl', 'wb') as f:
    pickle.dump(valid_data_features, f)

with gzip.open(SD_DATA_split + dataset_name + 'train_data_attributes.pkl', 'wb') as f:
    pickle.dump(train_data_attributes, f)
with gzip.open(SD_DATA_split+ dataset_name + 'valid_data_attributes.pkl', 'wb') as f:
    pickle.dump(valid_data_attributes, f)


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
    max_sequence_len=603,
    sample_len=201,
    batch_size=1000,
    epochs=10
)
model = DGAN(config2)

model.train_numpy(  features=features,
        attributes= attributes,)



synthetic_attributes, synthetic_features = model.generate_numpy(n =len(sample_patients))




model.save(SD_DATA+"Dopplenganger_epochocs.pth"  )
#m = model.load("aux/model_name.pth"  )
#synthetic_attributes, synthetic_features = m.generate_numpy(n =len(sample_patients))


 #Synthetic Data Evaluation    #lpha-precision, beta-recall, and authenticity #Alaa, Ahmed, Boris Van Breugel, Evgeny S. Saveliev, and Mihaela van der Schaar. “How faithful is your synthetic data? 
 # sample-level metrics for evaluating and auditing generative models.” In International Conference on Machine Learning, pp. 290-306. PMLR, 2022.
#alpha_precision = AlphaPrecision()
#result = alpha_precision.metrics(features, synthetic_features)
    

# Ejemplo de cómo llamar a la función plot_tsne
# Supongamos que X_gt y X_syn son tus arrays de NumPy con los datos.
# Deberás reemplazar estas líneas con tu carga de datos real.
#synthetic_features_2d = synthetic_features.reshape(synthetic_features.shape[0], -1)
#features_2d = features.reshape(features.shape[0], -1)

   
    
#mmd_evaluator = MaximumMeanDiscrepancy(kernel="rbf")
#result =   mmd_evaluator._evaluate(features_2d, synthetic_features_2d)
#print("MaximumMeanDiscrepancy Test:", result)

# Example usage:
# X_gt and X_syn are two numpy arrays representing empirical distributions
#features_1d = features.flatten()
#synthetic_features_1d = synthetic_features.flatten()


#ks_test = KolmogorovSmirnovTest()
#result = ks_test._evaluate(features_1d, synthetic_features_1d)
#print("Kolmog orov-Smirnov Test:", result)




#score = JensenShannonDistance()._evaluate(features, synthetic_features)
#print("Jensen-Shannon Distance:", score)

#plot_tsne(plt, features_2d, synthetic_features_2d)
    
 
#plot_marginal_comparison(plt, features_2d, synthetic_features_2d, n_histogram_bins=10, normalize=True)

    
    
