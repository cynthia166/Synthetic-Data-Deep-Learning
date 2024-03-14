import pandas as pd 
from preprocess_input_SDmodel import *
import numpy as np

# stdlib
import sys
import warnings

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import TimeSeriesDataLoader

log.add(sink=sys.stderr, level="INFO")
warnings.filterwarnings("ignore")

import numpy as np
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig
from preprocess_input_SDmodel import *
type_df = "entire"
s = .2
plugin = TimeGANPlugin
type_df = "entire"
arhivo = 'aux/raw_input.csv'
name  = "input_generative_g.csv"
#res = obtener_added_cols_targer_visitrank_(arhivo,name)
cols_to_drop1 = ['ADMITTIME','HADM_ID']
res = pd.read_csv("aux/full_input.csv")
keywords = ['INSURANCE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY','GENDER','SUBJECT']
static_data, temporal_dataframes, observation_data,outcomes = get_input_time( type_df,arhivo,name,cols_to_drop1,res,keywords,s)

numpy_array_t = np.stack([df.to_numpy() for df in temporal_dataframes])
features = numpy_array_t.reshape(numpy_array_t.shape[0],numpy_array_t.shape[2],numpy_array_t.shape[1])
attributes = static_data.to_numpy()

 attributes = np.random.rand(10000, 3)
 features = np.random.rand(10000, 20, 2)
 
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
model.generate_numpy(n=10)

model.train_dataframe( 
        features=features,
        attributes= attributes,
        )

synthetic_attributes, synthetic_features = model.generate(1000)
'''
synth_data = model_dop_gan.sample(n_samples=600)
synth_df = pd.concat(synth_data, axis=0)

# Create a plot for each measurement column
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(mba_data['traffic_byte_counter'].reset_index(drop=True), label='Real Traffic')
plt.plot(synth_df['traffic_byte_counter'].reset_index(drop=True), label='Synthetic Traffic', alpha=0.7)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Traffic Comparison')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(mba_data['ping_loss_rate'].reset_index(drop=True), label='Real Ping')
plt.plot(synth_df['ping_loss_rate'].reset_index(drop=True), label='Synthetic Ping', alpha=0.7)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Ping Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()'''