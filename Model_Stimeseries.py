import pandas as pd 
from concat_input import *

type_df = "entire"
arhivo = 'generative_input/input_onehot_encoding.csv'
name  = "input_generative_g.csv"

res = obtener_added_cols_targer_visitrank_(arhivo,name)
cols_to_drop1 = ['ADMITTIME','HADM_ID','DOB','L_1s_last']

res.drop(cols_to_drop1, axis=1, inplace=True)


dfs = obtener_entire(res,type_df)

