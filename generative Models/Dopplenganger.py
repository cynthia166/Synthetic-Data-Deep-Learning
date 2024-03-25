import pandas as pd 
from Preprocessing.preprocess_input_SDmodel import *
import numpy as np
from metric_stat import *
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
     
from Preprocessing.preprocess_input_SDmodel import *
type_df = "entire"
s = .2
type_df = "entire"
arhivo = 'aux/raw_input.csv'
name  = "input_generative_g.csv"
#res = obtener_added_cols_targer_visitrank_(arhivo,name)
cols_to_drop1 = ['ADMITTIME','HADM_ID']
res = pd.read_csv("aux/full_input.csv")
keywords = ['INSURANCE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY','GENDER','SUBJECT']
static_data, temporal_dataframes, observation_data,outcomes,sample_patients = get_input_time( type_df,arhivo,name,cols_to_drop1,res,keywords,s)
dataset_name = 'DATASET_NAME'
import pickle
import mgzip
with mgzip.open('./data/' + dataset_name + '_train.pkl', 'wb') as f:
    pickle.dump(temporal_dataframes, f)

pd.DataFrame(sample_patients).to_csv("aux/train_patients.csv")
numpy_array_t = np.stack([df.to_numpy() for df in temporal_dataframes])
features = numpy_array_t.reshape(numpy_array_t.shape[0],numpy_array_t.shape[2],numpy_array_t.shape[1])
attributes = static_data.to_numpy()


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




model.save("aux/model_name.pth"  )
m = model.load("aux/model_name.pth"  )
synthetic_attributes, synthetic_features = m.generate_numpy(n =len(sample_patients))


 #Synthetic Data Evaluation    #lpha-precision, beta-recall, and authenticity #Alaa, Ahmed, Boris Van Breugel, Evgeny S. Saveliev, and Mihaela van der Schaar. “How faithful is your synthetic data? 
 # sample-level metrics for evaluating and auditing generative models.” In International Conference on Machine Learning, pp. 290-306. PMLR, 2022.
alpha_precision = AlphaPrecision()
result = alpha_precision.metrics(features, synthetic_features)
    

# Ejemplo de cómo llamar a la función plot_tsne
# Supongamos que X_gt y X_syn son tus arrays de NumPy con los datos.
# Deberás reemplazar estas líneas con tu carga de datos real.
synthetic_features_2d = synthetic_features.reshape(synthetic_features.shape[0], -1)
features_2d = features.reshape(features.shape[0], -1)

   
    
mmd_evaluator = MaximumMeanDiscrepancy(kernel="rbf")
result =   mmd_evaluator._evaluate(features_2d, synthetic_features_2d)
print("MaximumMeanDiscrepancy Test:", result)

# Example usage:
# X_gt and X_syn are two numpy arrays representing empirical distributions
features_1d = features.flatten()
synthetic_features_1d = synthetic_features.flatten()


ks_test = KolmogorovSmirnovTest()
result = ks_test._evaluate(features_1d, synthetic_features_1d)
print("Kolmog orov-Smirnov Test:", result)




score = JensenShannonDistance()._evaluate(features, synthetic_features)
print("Jensen-Shannon Distance:", score)

#plot_tsne(plt, features_2d, synthetic_features_2d)
    
 
#plot_marginal_comparison(plt, features_2d, synthetic_features_2d, n_histogram_bins=10, normalize=True)

    
    
