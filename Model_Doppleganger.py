import pandas as pd 
from preprocess_input_SDmodel import *
import numpy as np

from synthcity.metrics.eval_statistical import AlphaPrecision

# Instanciar la clase
alpha_precision = AlphaPrecision(**kwargs)
# stdlib
import sys
import warnings
from evaluation_SD import KolmogorovSmirnovTest
# synthcity absolute

warnings.filterwarnings("ignore")

import numpy as np
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig

import pandas as pd
from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
from gretel_client.config import RunnerMode
from gretel_client.evaluation.quality_report import QualityReport
from gretel_client import configure_session
from gretel_client.projects import create_or_get_unique_project
     
from preprocess_input_SDmodel import *
type_df = "entire"
s = .7
type_df = "entire"
arhivo = 'aux/raw_input.csv'
name  = "input_generative_g.csv"
#res = obtener_added_cols_targer_visitrank_(arhivo,name)
cols_to_drop1 = ['ADMITTIME','HADM_ID']
res = pd.read_csv("aux/full_input.csv")
keywords = ['INSURANCE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY','GENDER','SUBJECT']
static_data, temporal_dataframes, observation_data,outcomes,sample_patients = get_input_time( type_df,arhivo,name,cols_to_drop1,res,keywords,s)
pd.DataFrame(sample_patients).to_csv("aux/train_patients.csv")
numpy_array_t = np.stack([df.to_numpy() for df in temporal_dataframes])
features = numpy_array_t.reshape(numpy_array_t.shape[0],numpy_array_t.shape[2],numpy_array_t.shape[1])
attributes = static_data.to_numpy()

loader_real = TimeSeriesDataLoader(
    temporal_data=temporal_dataframes,
    observation_times=observation_data,
    static_data=static_data,
    outcome=outcomes,
)

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

model.train_numpy(    features=features,
        attributes= attributes,)



synthetic_attributes, synthetic_features = model.generate_numpy(n =len(sample_patients))


### add the to data loader fo evaluations###
lista_dataframes = [pd.DataFrame(mi_array[i]) for i in range(mi_array.shape[0])]

# Agregar una nueva columna a cada DataFrame
for i, df in enumerate(lista_dataframes):
    df.index = np.arange(1, df.shape[1] + 1)
for df in temporal_dataframes:
    df.columns = temporal_dataframes.columns

# Ahora 'lista_dataframes' es una lista de DataFrames, cada uno con una nueva columna que va de 1 a 10
loader_synthetic = TimeSeriesDataLoader(
    temporal_data=synthetic_features,
    observation_times=observation_data,
    static_data=pd.DataFrame(synthetic_attributes),
    outcome=outcomes,
)


pd.DataFrame(synthetic_attributes).to_csv("aux/synthetic_attributes.csv")
#lista_dataframes = [pd.DataFrame(mi_array[i]) for i in range(mi_array.shape[0])]
pd.DataFrame(synthetic_features).to_csv("aux/synthetic_features.csv")
model.save("aux/model_name.pth"  )
m = model.load("aux/model_name.pth"  )
synthetic_attributes, synthetic_features = m.generate_numpy(n =len(sample_patients))




# Guardar datos generados;
import pandas as pd
#synthetic_features
lista_dataframes = [pd.DataFrame(features[i]) for i in range(features.shape[0])]
df_total_synthetic_features = pd.concat(lista_dataframes, keys=[f'dataframe_{i}' for i in range(len(lista_dataframes))])
df_total_synthetic_features.to_csv('aux/features.csv')

#synthetic_features
lista_dataframes = [pd.DataFrame(synthetic_features[i]) for i in range(synthetic_features.shape[0])]
df_total_synthetic_features = pd.concat(lista_dataframes, keys=[f'dataframe_{i}' for i in range(len(lista_dataframes))])
df_total_synthetic_features.to_csv('aux/synthetic_features.csv')

pd.DataFrame(synthetic_attributes).to_csv("aux/synthetic_attributes.csv")
pd.DataFrame(attributes).to_csv("aux/attributes.csv")
#sfeatures
df = pd.read_csv('aux/synthetic_features.csv', header=[0,1], index_col=0)
cols_to_drop = df.filter(like='Unnamed', axis=1).columns
df.drop(cols_to_drop, axis=1, inplace=True)
synthetic_features = df.to_numpy()

#same dimension as before
grouped = df.groupby(df.index)
synthetic_features = [group.values for _, group in grouped]
synthetic_features = [arr for arr in synthetic_features if arr.shape == synthetic_features[1].shape]
synthetic_features = np.stack(synthetic_features)


df = pd.read_csv('aux/features.csv', header=[0,1], index_col=0)
cols_to_drop = df.filter(like='Unnamed', axis=1).columns
df.drop(cols_to_drop, axis=1, inplace=True)
features = df.to_numpy()

#otra dimensinb is la linea anterio
grouped = df.groupby(df.index)
features = [group.values for _, group in grouped]
features = [arr for arr in features if arr.shape == features[1].shape]
features = np.stack(features)


synthetic_attributes = pd.read_csv('aux/synthetic_attributes.csv', header=[0,1], index_col=0)
attributes = pd.read_csv('aux/attributes.csv', header=[0,1], index_col=0)
cols_to_drop = synthetic_attributes.filter(like='Unnamed', axis=1).columns
synthetic_attributes.drop(cols_to_drop, axis=1, inplace=True)
cols_to_drop = attributes.filter(like='Unnamed', axis=1).columns
attributes.drop(cols_to_drop, axis=1, inplace=True)


 #Synthetic Data Evaluation    #lpha-precision, beta-recall, and authenticity #Alaa, Ahmed, Boris Van Breugel, Evgeny S. Saveliev, and Mihaela van der Schaar. “How faithful is your synthetic data? 
 # sample-level metrics for evaluating and auditing generative models.” In International Conference on Machine Learning, pp. 290-306. PMLR, 2022.
alpha_precision = AlphaPrecision()
result = alpha_precision.metrics(features, synthetic_features)
    
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Suponiendo que 'features' y 'synthetic_features' son tus arrays de NumPy de dimensiones (31466, 603, 42)
# Remodelar los arrays a dos dimensiones
features_2d = features.reshape(-1, features.shape[-1])
synthetic_features_2d = synthetic_features.reshape(-1, synthetic_features.shape[-1])

# Concatenar los dos conjuntos de datos
data = np.concatenate((features_2d, synthetic_features_2d))

# Aplicar t-SNE
import pacmap

# Assuming 'data' is your high-dimensional data.
# Instantiate PaCMAP with two components, similar to t-SNE
pacmap_instance = pacmap.PaCMAP(n_components=2, random_state=0)

# Fit and transform the data
data_2d = pacmap_instance.fit_transform(data)

# Separate the results for the two datasets
features_2d = data[:len(features)]
synthetic_data_2d = data[len(features):]

features_pacmap = data_2d[:len(features_2d)]
synthetic_features_pacmap = data_2d[len(features_2d):]

# Create a plot
plt.figure(figsize=(10, 10))
plt.scatter(features_pacmap[:, 0], features_pacmap[:, 1], c='blue', label='Features', alpha=0.5)
plt.scatter(synthetic_features_pacmap[:, 0], synthetic_features_pacmap[:, 1], c='red', label='Synthetic Features', alpha=0.5)
plt.title('PaCMAP Dimensionality Reduction')
plt.legend()
plt.show()
