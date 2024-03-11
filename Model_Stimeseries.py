import pandas as pd 
from preprocess_input import *
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


type_df = "entire"
arhivo = 'generative_input/input_onehot_encoding.csv'
name  = "input_generative_g.csv"

res = obtener_added_cols_targer_visitrank_(arhivo,name)
cols_to_drop1 = ['ADMITTIME','HADM_ID','L_1s_last']

#res.drop(cols_to_drop1, axis=1, inplace=True)

#res.to_csv(".csv")
res = pd.read_csv("generative_input/full_input.csv")
# Obtener una lista de pacientes únicos
unique_patients = res['SUBJECT_ID'].unique()

# Calcular el 20% del total de pacientes únicos
sample_size = int(0.20 * len(unique_patients))

# Obtener una muestra aleatoria del 20% de los pacientes únicos
sample_patients = np.random.choice(unique_patients, size=sample_size, replace=False)

# Filtrar el DataFrame para incluir solo los registros de los pacientes en la muestra
sample_df = res[res['SUBJECT_ID'].isin(sample_patients)]

dfs = obtener_entire(sample_df,type_df)

# Lista de nombres de columnas a buscar
static_data = [
    'SUBJECT_ID',

    'INSURANCE_Medicare',
    'INSURANCE_Otra',
    'RELIGION_CATHOLIC',
    'RELIGION_NOT SPECIFIED',
    'RELIGION_Otra',
    'RELIGION_UNOBTAINABLE',
    'MARITAL_STATUS_MARRIED',
    'MARITAL_STATUS_Otra',
    'MARITAL_STATUS_SINGLE',
    'ETHNICITY_BLACK/AFRICAN AMERICAN',
    'ETHNICITY_Otra',
    'ETHNICITY_WHITE',
    'GENDER_M',
    'GENDER_Otra',


]

not_considet_temporal = [
    'HOSPITAL_EXPIRE_FLAG',
 
    'INSURANCE_Medicare',
    'INSURANCE_Otra',
    'RELIGION_CATHOLIC',
    'RELIGION_NOT SPECIFIED',
    'RELIGION_Otra',
    'RELIGION_UNOBTAINABLE',
    'MARITAL_STATUS_MARRIED',
    'MARITAL_STATUS_Otra',
    'MARITAL_STATUS_SINGLE',
    'ETHNICITY_BLACK/AFRICAN AMERICAN',
    'ETHNICITY_Otra',
    'ETHNICITY_WHITE',
    'GENDER_M',
    'GENDER_Otra',
   # 'visit_rank',

]

static_data = dfs[static_data]
outcomes = dfs[['HOSPITAL_EXPIRE_FLAG']]
temporal_data = dfs.drop(columns=not_considet_temporal)
cols_to_drop = temporal_data.filter(like='Unnamed', axis=1).columns
temporal_data.drop(cols_to_drop, axis=1, inplace=True)


observation_data, temporal_dataframes = ([] for i in range(2))
for id in static_data["id"].unique():
    temp_df = temporal_data[temporal_data["id"] == id]
    observations = temp_df["timepoint"].tolist()
    temp_df.set_index("timepoint", inplace=True)
    temp_df = temp_df.drop(columns=["id"])
    # add each to list
    observation_data.append(observations)
    temporal_dataframes.append(temp_df)


# Establecer 'SUBJECT_ID' y 'visit_rank' como índices
temporal_data.set_index(['SUBJECT_ID', 'visit_rank'], inplace=True)

# Ahora se agrupa por 'SUBJECT_ID' y se recoge la información necesaria.
grouped = temporal_data.groupby(level='SUBJECT_ID')

# Se obtiene 'observations' como los índices de segundo nivel para cada grupo.
observation_data = [group.index.get_level_values('visit_rank').tolist() for _, group in grouped]

# Se obtiene los dataframes temporales por grupo, reseteando el índice 'SUBJECT_ID'.
temporal_dataframes = [group.reset_index(level=0, drop=True) for _, group in grouped]



loader = TimeSeriesDataLoader(
    temporal_data=temporal_dataframes,
    observation_times=observation_data,
    static_data=static_data,
    outcome=outcome_data,
)