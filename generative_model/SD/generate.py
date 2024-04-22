import os 
import sys
import gzip
import pickle
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig
#from synthcity.metrics.eval_statistical import AlphaPrecision

import numpy as np
# Añadir el directorio evaluacion al sys.path
ruta_evaluacion = os.path.dirname( '/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_models/SD/')
print(os.getcwd())
sys.path.append(ruta_evaluacion)

# Añadir los directorios específicos



# Ahora se pueden importar los módulos como si estuvieran en el mismo directorio

# Cambiar el directorio de trabajo actual a 'nueva/ruta'
os.environ['OPENBLAS_VERBOSE'] = '0'

# Uso de las funciones importadas
# Load data
name_attributes_syn = 'synthetic_attributes.pkl'
name_features_syn ='synthetic_features.pkl'


SD = "modelgenerative_models/SD/model"
name_file = "Dopplenganger_nonprepo_epochocs_10.pth"  

ruta = SD+name_file


def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)
    
def concat_attributes(a,b):

    # Redimensiona 'b' para que tenga la misma forma que 'a' en los dos primeros ejes
    #b is attributes
    #a = np.random.rand(100, 244, 5)
    #b = np.random.rand(100, 2)

    b = np.repeat(b[:, :, np.newaxis], a.shape[2], axis=2)
    print(b.shape)
    print(a.shape)
    # Ahora 'b' es un array de forma (100, 244, 2)
    print(b.shape)  # Impri   me: (100, 244, 2)

    # Concatena 'a' y 'b' a lo largo del tercer eje
    c = np.concatenate((a, b), axis=1)

# Ahora 'c' es un array de forma (100, 244, 7)
    print(c.shape) 
    return c    


def genera_data(N, ruta, path_features, dataset_name, name_attributes_syn, name_features_syn):
    config2 = DGANConfig(
    max_sequence_len=666,
    sample_len=111,
    batch_size=100,
    epochs=120
    )

    model = DGAN(config2)
    model_r = model.load(ruta)
    synthetic_attributes, synthetic_features = model_r.generate_numpy(n =N)

    synthetic_attributes.shape
    synthetic_features.shape
    with gzip.open(path_features + dataset_name + name_attributes_syn, 'wb') as f:
            pickle.dump(synthetic_attributes, f)
        
    with gzip.open(path_features+ dataset_name + name_features_syn, 'wb') as f:
            pickle.dump(synthetic_features, f)
            
    return synthetic_attributes, synthetic_features        

def main():
    name_features_syn = 'non_prepo_synthetic_features_10.pkl'
    name_attributes_syn = 'non_prepo_synthetic_attributes_10.pkl'
    dataset_name = 'DATASET_NAME_non_prepro'
    path_features = "train_sp/non_prepo/"
    ruta_test = "train_sp/non_prepo/DATASET_NAME_non_preprovalid_data_attributes.pkl"
    ruta = ruta_evaluacion + "/model/" + "Dopplenganger_nonprepo_epochocs_10.pth"
    X_test = load_data(ruta_test)
    N = len(X_test)
    genera_data(N, ruta, path_features, dataset_name, name_attributes_syn, name_features_syn)


if __name__ == '__main__':
    main()