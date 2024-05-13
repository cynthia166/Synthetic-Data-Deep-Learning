import pandas as pd 

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
arhivo = 'aux/raw_input.csv'
name  = "input_generative_g.csv"

#res = obtener_added_cols_targer_visitrank_(arhivo,name)
cols_to_drop1 = ['ADMITTIME','HADM_ID']

#res.drop(cols_to_drop1, axis=1, inplace=True)

#res.to_csv("aux/full_input.csv")
res = pd.read_csv("aux/full_input.csv")
# Obtener una lista de pacientes únicos
unique_patients = res['SUBJECT_ID'].unique()

# Calcular el 20% del total de pacientes únicos
sample_size = int(0.03 * len(unique_patients))
# Obtener una muestra aleatoria del 20% de los pacientes únicos
sample_patients = np.random.choice(unique_patients, size=sample_size, replace=False)
# Filtrar el DataFrame para incluir solo los registros de los pacientes en la muestra
sample_df = res[res['SUBJECT_ID'].isin(sample_patients)]

dfs = obtener_entire(sample_df,type_df)

# Lista de nombres de columnas a buscar
# Supongamos que type_df es tu DataFrame
keywords = ['INSURANCE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY','GENDER','SUBJECT']
# Filtrar las columnas de type_df que contienen las palabras clave
static_data_cols = [col for col in dfs.columns if any(keyword in col for keyword in keywords)]
# Crear un nuevo DataFrame que solo incluye las columnas filtradas


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
    'GENDER_Otra',]




not_considet_temporal = [
   i for i in static_data_cols if i != 'SUBJECT_ID'
]

static_data = dfs[static_data_cols].groupby('SUBJECT_ID').max().reset_index(drop=True)
outcomes = dfs[['HOSPITAL_EXPIRE_FLAG','SUBJECT_ID']].groupby('SUBJECT_ID').max().reset_index(drop=True)
temporal_data = dfs.drop(columns=not_considet_temporal+['HOSPITAL_EXPIRE_FLAG'])
cols_to_drop = list(temporal_data.filter(like='Unnamed', axis=1).columns) + ['ADMITTIME','HADM_ID']
temporal_data.drop(cols_to_drop, axis=1, inplace=True)




# Establecer 'SUBJECT_ID' y 'visit_rank' como índices
temporal_data.set_index(['SUBJECT_ID', 'visit_rank'], inplace=True)
# Ordenar temporal_data por el índice 'visit_rank' en orden ascendente
temporal_data.sort_index(level='visit_rank', 
inplace=True)
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
    outcome=outcomes,
)

#syn_model = Plugins().get("timegan")

#syn_model.fit(loader)

#syn_model.generate(count=10).dataframe()

from synthcity.benchmark import Benchmarks

score = Benchmarks.evaluate(
    [
        (f"test_{model}", model, {})
        for model in ["timevae"]
    ],
    loader,
    synthetic_size=len(temporal_dataframes),
    repeats=2,
    task_type="time_series",  # time_series_survival or time_series
)

Benchmarks.print(score)

import matplotlib.pyplot as plt

#syn_model.plot(plt, loader)
#plt.savefig('aux/timegan_output.png')
#plt.show()

means = []
for plugin in score:
    data = score[plugin]["mean"]
    directions = score[plugin]["direction"].to_dict()
    means.append(data)

out = pd.concat(means, axis=1)
out.set_axis(score.keys(), axis=1, inplace=True)
out.to_csv("aux/timegan_output.csv")

