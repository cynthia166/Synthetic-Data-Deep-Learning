import sys
sys.path.append('')
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing.function_pred import *
import ast

# %%
import numpy as np

import seaborn as sns
import importlib
import os
from preprocessing.config import *
# Obtener el directorio actual
directorio_actual = os.getcwd()
print(directorio_actual)
from preprocessing.config import *
importlib.reload(sys.modules['preprocessing.config'])
#run  2_input_2


def  demos_pies():
    import pandas as pd
    archivo_input_label = DARTA_INTERM/ 'data_preprocess_non_filtered.csv'
    df  = pd.read_csv(archivo_input_label)
    df.shape
    adm = pd.read_csv(MIMIC / 'ADMISSIONS.csv.gz')
    adm.columns


    # Assuming you have a DataFrame named df with a "subject" column and an "admission" column

    # Group the DataFrame by "subject" and count the number of unique "admission" values for each subject
    subject_counts = adm.groupby("SUBJECT_ID")["HADM_ID"].nunique()

    # Filter the subjects that have more than one admission
    subjects_with_multiple_admissions = subject_counts[subject_counts > 1]

    # Count the number of subjects with multiple admissions
    count = len(subjects_with_multiple_admissions)

    # Print the count
    print(count)


    new_column_names = {'ADMISSION_TYPE': 'Admission type', 'ADMISSION_LOCATION': 'Admission location', 'DISCHARGE_LOCATION': 'Discharge location',
                        'INSURANCE': 'Insurance', 'LANGUAGE': 'Language', 'RELIGION': 'Religion', 'MARITAL_STATUS': 'Marital status',   'ETHNICITY':'Ethnicity',     'HOSPITAL_EXPIRE_FLAG':'Death', 'HAS_CHARTEVENTS_DATA':'Has Chart Events'}

    adm = adm.rename(columns=new_column_names)


    adm["Death"] = np.where(adm["Death"]==1, 'Died', 'Survived')


    adm['Has Chart Events'] = np.where(adm['Has Chart Events']==1, 'Yes', 'No')
    adm["Ethnicity"].value_counts()
    adm["Death"].value_counts()
    adm["Ethnicity"] = np.where(adm["Ethnicity"]=='BLACK/AFRICAN AMERICAN', 'AFRICAN AMERICAN', adm["Ethnicity"])
    # create batches
    bins = [0, 18, 30, 40, 50, 60, 250]  # Define the age intervals for bins

    # Create labels for the bins
    labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '61-100']  # Labels for each bin range

    # Use cut to create bins from the 'age' column
    df['age_group'] = pd.cut(df['year_age'], bins=bins, labels=labels, right=False) 



    #para grupo lenght of stay
    bins = [-1,0, 25, 100, 400, 500, 700, 4107]  # Define the age intervals for bins

    # Create labels for the bins
    labels = ['0','1-25', '26-100', '101-401', '402-500', '501-700', '701-4107']  # Labels for each bin range

    # Use cut to create bins from the 'age' column
    df['INTERVAL_group'] = pd.cut(df['LOSRD'], bins=bins, labels=labels, right=False)  # Assign each value to a bin
    # Assign each value to a bin
    #df.loc[df['INTERVAL_group'].isnull(), 'INTERVAL_group'] = 0
    import pandas as pd

    # Group the DataFrame by 'ADMISSION_TYPE' and calculate the mode of the variables
    grouped_df = df.groupby('SUBJECT_ID')['ADMISSION_TYPE', 'INSURANCE', 'GENDER', 'EXPIRE_FLAG', ].apply(lambda x: x.mode().iloc[0])

    # Print the grouped DataFrame

    gender_map2 = {1.0: 'DIED', 0.0: 'SURVIVED'}

    # Use the map function to create a new column 'gender_label'
    grouped_df['EXPIRE_FLAG'] = grouped_df['EXPIRE_FLAG'].map(gender_map2)


    df['EXPIRE_FLAG'] = df['EXPIRE_FLAG'].map(gender_map2)
    
    return df, grouped_df

def agregar_anterior_hp_S(df_sin_duplicados_columnas_especificas_,lista):
    s = "f1_test f1_train sensitivity_test specificity_test precision_test accuracy_test sensitivity_train specificity_train precision_train accuracy_train"

    # Convertir la cadena en una lista de strings
    lista = s.split()

    print(lista)
    
    df_sin_duplicados_columnas_especificas_.columns = [col if col not in lista else f"{col}_a" for col in df_sin_duplicados_columnas_especificas_.columns]
    return df_sin_duplicados_columnas_especificas_


def crear_tabla_results(df_sin_duplicados_columnas_especificas_,df_sin_duplicados_columnas_especificas,title_,metric_list,lista,name_p):
    aux = pd.merge(df_sin_duplicados_columnas_especificas_,df_sin_duplicados_columnas_especificas[lista+["Mapping"]], how="right", on="Mapping")
    pivot_table = [aux.pivot_table(index=['Mapping'], values=[metric +'_test_a', metric +'_train_a',metric +'_test', metric +'_train']) for metric in metric_list]
    pivot_table_final = pd.concat(pivot_table, axis=1)
    # Supongamos que 'aux' es tu DataFrame y 'metric_list' es tu lista de métricas

    print(pivot_table_final)
    # Supongamos que 'pivot_table' es tu DataFrame
    pivot_table_final = pivot_table_final.round(3)

    # Supongamos que 'pivot_table' es tu DataFrame y 'metric_list' es tu lista de métricas
    for metric in metric_list:
        pivot_table_final = pivot_table_final.rename(columns={
            metric +'_test': metric +' test HP*',
            metric +'_train': metric +' train HP*',
            metric +'_test_a': metric +' test',
            metric +'_train_a': metric +' train'
        })


    print(pivot_table_final.T.to_latex())
    # Supongamos que 'pivot_table_final' es tu DataFrame
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table_final.T, annot=True, cmap='Blues', fmt=".3f")

    plt.xlabel('Simplification')
    plt.title(title_)
    plt.savefig(IMAGES_PRED_DICT+name_p+'.svg')
    plt.show()


# Supongamos que 'df' es tu DataFrame
#df = df.rename(columns=lambda x: x.replace('_', ' '))
def crear_datafram_hyperparametros(df_sin_duplicados_columnas_especificas):
    import ast
    cols_p = ['learning_rate', 'max_delta_step', 'max_depth', 'min_child_weight', 'n_estimators', 'reg_alpha', 'reg_lambda', 'scale_pos_weight', 'subsample']
    cols_hp = ['best_learning_rate',
       'best_max_delta_step', 'best_max_depth', 'best_min_child_weight',
       'best_n_estimators', 'best_reg_alpha', 'best_reg_lambda',
       'best_scale_pos_weight', 'best_subsample']


    for i in cols_p:
        try:
           df_sin_duplicados_columnas_especificas[i] = df_sin_duplicados_columnas_especificas[i].apply(ast.literal_eval)
        except:
            pass
    df =df_sin_duplicados_columnas_especificas[['learning_rate',
            'max_delta_step', 'max_depth', 'min_child_weight','n_estimators','reg_alpha',
        'reg_lambda', 'scale_pos_weight',  'subsample','Mapping' ]]

    # Using dictionary comprehension to get min and max of each list in the dataframe
    min_max_values = {col: [min(df[col].explode()), max(df[col].explode())] for col in cols_p}

    # Creating a new dataframe with min and max values
    df_min_max = pd.DataFrame(min_max_values, index=['min', 'max'])

    aux = df_min_max.T
    #aux.to_csv("aux/X_Gbost_best.csv")

    aux_b = df_sin_duplicados_columnas_especificas.pivot_table(index=['Mapping'], values=cols_hp).T 
    aux_b.index = aux.index


    df_concatenado = pd.concat([aux_b, aux], axis=1)

    print(df_concatenado.to_latex())

def create_heatmap(df, name,name_im):
    metrics = ['sensitivity', 'f1', 'accuracy', 'precision']
    # Configurar la figura y los ejes
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))
    fig.suptitle('Performance of the prediction task according to the preprocess of the input (model ' + name + ')', fontsize=20)

    # Iterar sobre cada métrica y su eje correspondiente
    for metric, ax in zip(metrics, axes.flatten()):
        # Datos para la métrica actual
        train_values = df[metric + '_train'].values
        test_values = df[metric + '_test'].values
        
        # Preparar los datos para el mapa de calor
        heatmap_data = np.array([train_values, test_values])
        
        # Crear el mapa de calor
        sns_heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap='coolwarm', ax=ax, cbar=True)
        ax.set_title(metric.capitalize(), fontsize=18)
        ax.set_ylabel('Train/Test', fontsize=16)
        ax.set_yticklabels(['Train', 'Test'], rotation=0, fontsize=16)
        
        # Set the font size of the x-axis tick labels
        ax.set_xticklabels(df["Mapping"].tolist(), rotation=45, fontsize=14)
        
        # Set the font size of the annotations
        for text in sns_heatmap.texts:
            text.set_fontsize(14)

    # Ajustar el layout y mostrar la figura
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(IMAGES_PRED_DICT+name_im+'readm.svg')
    plt.show()

import ast
def create_best_lg(aux_2,reemplazos):
      #aux_2 = nomb_(aux_2,reemplazos)

      cols_hp = ['params.C', 'params.l1_ratio', 'params.max_iter', 'params.penalty', 'params.solver']

      # Quitar 'params.' de cada elemento
      cols_p = [x.replace('params.', '') for x in cols_hp]

      
      for i in cols_p:
            try:
               aux_2[i] = aux_2[i].apply(ast.literal_eval)
            except:
                  pass
      aux_b =aux_2[['params.C', 'params.l1_ratio', 'params.max_iter', 'params.penalty', 'params.solver', "Mapping"]]
      aux_b.index = aux_b["Mapping"]
      aux_b.drop(columns="Mapping",inplace = True)
      # OTRO DATAFRAME
      min_max_values = {col: [min(aux_2[col].explode()), max(aux_2[col].explode())] for col in cols_p}

      # Creating a new dataframe with min and max values
      df_min_max = pd.DataFrame(min_max_values, index=['min', 'max'])

      aux = df_min_max.T
      #aux.to_csv("aux/X_Gbost_best.csv")

      aux_b = aux_b.T
      aux_b.index = aux.index

      df_concatenado = pd.concat([aux_b, aux], axis=1)

      print(df_concatenado.to_latex())

def nomb_(df_sin_duplicados_columnas_especificas,reemplazos):

    # Aplicar los reemplazos
    df_sin_duplicados_columnas_especificas["Mapping"] = df_sin_duplicados_columnas_especificas["Mapping"].replace(reemplazos)

    print(df_sin_duplicados_columnas_especificas)
    return(df_sin_duplicados_columnas_especificas)


def string_list(x):
    '''funcion que se convierte una lista de string a una lista normal'''
    '''Input '["\'44\'", "\'44\'", "\'50\'", "\'193\'", "\'222\'", "\'222\'", "\'222\'"]'
    x: string as list
    Output
    list: list'''

    try:
        lista = ast.literal_eval(x)
    except:
        lista = np.nan
    return lista


def plot_readmission(o,df_sin_duplicados_columnas_especificas_):
    highest_sensitivity_idx =df_sin_duplicados_columnas_especificas_.groupby('Mapping')[o+'_train'].idxmax()
    highest_sensitivity_rows = df_sin_duplicados_columnas_especificas_.loc[highest_sensitivity_idx]
    highest_sensitivity_rows

    positions = range(len(highest_sensitivity_rows))
    plt.figure(figsize=(10, 6))
    bars_train = plt.bar(positions, highest_sensitivity_rows[o+'_train'], width=0.4, label=o+" score train")
    bars_test = plt.bar([pos + 0.4 for pos in positions], highest_sensitivity_rows[o+'_test'], width=0.4, label=o+" score test", alpha=0.5)

    # Aumentar el tamaño de las etiquetas y el título
    plt.xticks([pos + 0.2 for pos in positions], highest_sensitivity_rows["Mapping"], rotation=90, fontsize=12)
    plt.xlabel('Simplification', fontsize=14)
    plt.ylabel(o + " score", fontsize=14)
    plt.title('Performance of the prediction task according to the preprocess of the input.' , fontsize=14)

    # Colocar la leyenda fuera del gráfico
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

    # Añadir el valor de F1 a cada barra
    for bars in [bars_train, bars_test]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

def assign_batch_numbers(ids, batch_size):
    # Crear una lista para almacenar los pares (id, batch_number)
    id_batches = []

    # Calcular el número de lote para cada id
    for i, id in enumerate(ids):
        batch_number = i // batch_size
        id_batches.append((id, batch_number))

    # Crear un DataFrame de polars con los resultados
    batch_df = pl.DataFrame(id_batches, schema=['ID', 'Batch_Number'])

    return batch_df    
def create_graph(df_sin_duplicados_columnas_especificas_,name,name_im):
    metrics = ['sensitivity', 'f1', 'accuracy', 'precision']
    # Configurar la figura y los ejes
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))
    fig.suptitle('Performance of the prediction task according to the preprocess of the input (model '+ name+')', fontsize=20)

    # Crear un objeto de leyenda manualmente
    handles, labels = [], []

    # Iterar sobre cada métrica y su eje correspondiente
    for metric, ax in zip(metrics, axes.flatten()):
        # Generar índices de las filas con el valor más alto para la métrica actual
        highest_metric_idx = df_sin_duplicados_columnas_especificas_.groupby('Mapping')[metric + '_train'].idxmax()
        highest_metric_rows = df_sin_duplicados_columnas_especificas_.loc[highest_metric_idx]
        
        # Generar las posiciones de las barras
        positions = range(len(highest_metric_rows))
        
        # Graficar las barras para train y test
        bars_train = ax.bar(positions, highest_metric_rows[metric + '_train'], width=0.4, label=metric + " score train")
        bars_test = ax.bar([p + 0.4 for p in positions], highest_metric_rows[metric + '_test'], width=0.4, label=metric + " score test", alpha=0.5)
        
        # Añadir el valor de la métrica a cada barra
        for bars in [bars_train, bars_test]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.3f}', ha='center', va='bottom', fontsize=12)
        
        # Configurar las etiquetas de los ejes
        ax.set_xticks([p + 0.2 for p in positions])
        ax.set_xticklabels(highest_metric_rows["Mapping"], rotation=90, fontsize=14)
        ax.set_ylabel(metric + " score", fontsize=14)
        
        # Actualizar los manejadores de la leyenda para la figura
        if metric == 'sensitivity':  # Solo agregar una vez
            handles, labels = ax.get_legend_handles_labels()

    # Ajustar layout y añadir una leyenda global
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajusta el rect para dar espacio al título general
    # Colocar la leyenda en medio a la derecha de la figura
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    plt.savefig(IMAGES_PRED_DICT+name_im+'readm.svg')
    plt.show()

def create_graph(df_sin_duplicados_columnas_especificas_,name,name_im):
    metrics = ['sensitivity', 'f1', 'accuracy', 'precision']
    # Configurar la figura y los ejes
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))
    fig.suptitle('Performance of the prediction task according to the preprocess of the input (model '+ name+')', fontsize=20)

    # Crear un objeto de leyenda manualmente
    handles, labels = [], []

    # Iterar sobre cada métrica y su eje correspondiente
    for metric, ax in zip(metrics, axes.flatten()):
        # Generar índices de las filas con el valor más alto para la métrica actual
        highest_metric_idx = df_sin_duplicados_columnas_especificas_.groupby('Mapping')[metric + '_train'].idxmax()
        highest_metric_rows = df_sin_duplicados_columnas_especificas_.loc[highest_metric_idx]
        
        # Generar las posiciones de las barras
        positions = range(len(highest_metric_rows))
        
        # Graficar las barras para train y test
        bars_train = ax.bar(positions, highest_metric_rows[metric + '_train'], width=0.4, label=metric + " score train")
        bars_test = ax.bar([p + 0.4 for p in positions], highest_metric_rows[metric + '_test'], width=0.4, label=metric + " score test", alpha=0.5)
        
        # Añadir el valor de la métrica a cada barra
        for bars in [bars_train, bars_test]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.3f}', ha='center', va='bottom', fontsize=12)
        
        # Configurar las etiquetas de los ejes
        ax.set_xticks([p + 0.2 for p in positions])
        ax.set_xticklabels(highest_metric_rows["Mapping"], rotation=90, fontsize=14)
        ax.set_ylabel(metric + " score", fontsize=14)
        
        # Actualizar los manejadores de la leyenda para la figura
        if metric == 'sensitivity':  # Solo agregar una vez
            handles, labels = ax.get_legend_handles_labels()

    # Ajustar layout y añadir una leyenda global
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajusta el rect para dar espacio al título general
    # Colocar la leyenda en medio a la derecha de la figura
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    plt.savefig(IMAGES_PRED_DICT+name_im+'readm.svg')
    plt.show()


def plot_readmission(o,df_sin_duplicados_columnas_especificas_):
    highest_sensitivity_idx =df_sin_duplicados_columnas_especificas_.groupby('Mapping')[o+'_train'].idxmax()
    highest_sensitivity_rows = df_sin_duplicados_columnas_especificas_.loc[highest_sensitivity_idx]
    highest_sensitivity_rows

    positions = range(len(highest_sensitivity_rows))
    plt.figure(figsize=(10, 6))
    bars_train = plt.bar(positions, highest_sensitivity_rows[o+'_train'], width=0.4, label=o+" score train")
    bars_test = plt.bar([pos + 0.4 for pos in positions], highest_sensitivity_rows[o+'_test'], width=0.4, label=o+" score test", alpha=0.5)

    # Aumentar el tamaño de las etiquetas y el título
    plt.xticks([pos + 0.2 for pos in positions], highest_sensitivity_rows["Mapping"], rotation=90, fontsize=12)
    plt.xlabel('Simplification', fontsize=14)
    plt.ylabel(o + " score", fontsize=14)
    plt.title('Performance of the prediction task according to the preprocess of the input.' , fontsize=14)

    

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

    # Añadir el valor de F1 a cada barra
    for bars in [bars_train, bars_test]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()
    
import ast
def create_best_lg(aux_2,reemplazos):
      #aux_2 = nomb_(aux_2,reemplazos)

      cols_hp = ['params.C', 'params.l1_ratio', 'params.max_iter', 'params.penalty', 'params.solver']

      # Quitar 'params.' de cada elemento
      cols_p = [x.replace('params.', '') for x in cols_hp]

      
      for i in cols_p:
            try:
               aux_2[i] = aux_2[i].apply(ast.literal_eval)
            except:
                  pass
      aux_b =aux_2[['params.C', 'params.l1_ratio', 'params.max_iter', 'params.penalty', 'params.solver', "Mapping"]]
      aux_b.index = aux_b["Mapping"]
      aux_b.drop(columns="Mapping",inplace = True)
      # OTRO DATAFRAME
      min_max_values = {col: [min(aux_2[col].explode()), max(aux_2[col].explode())] for col in cols_p}

      # Creating a new dataframe with min and max values
      df_min_max = pd.DataFrame(min_max_values, index=['min', 'max'])

      aux = df_min_max.T
      #aux.to_csv("aux/X_Gbost_best.csv")

      aux_b = aux_b.T
      aux_b.index = aux.index

      df_concatenado = pd.concat([aux_b, aux], axis=1)

      print(df_concatenado.to_latex())
#create_best_lg(aux_2,reemplazos)

# %%
# Supongamos que 'df' es tu DataFrame
#df = df.rename(columns=lambda x: x.replace('_', ' '))
def crear_datafram_hyperparametros(df_sin_duplicados_columnas_especificas):
    import ast
    cols_p = ['learning_rate', 'max_delta_step', 'max_depth', 'min_child_weight', 'n_estimators', 'reg_alpha', 'reg_lambda', 'scale_pos_weight', 'subsample']
    cols_hp = ['best_learning_rate',
       'best_max_delta_step', 'best_max_depth', 'best_min_child_weight',
       'best_n_estimators', 'best_reg_alpha', 'best_reg_lambda',
       'best_scale_pos_weight', 'best_subsample']


    for i in cols_p:
        try:
           df_sin_duplicados_columnas_especificas[i] = df_sin_duplicados_columnas_especificas[i].apply(ast.literal_eval)
        except:
            pass
    df =df_sin_duplicados_columnas_especificas[['learning_rate',
            'max_delta_step', 'max_depth', 'min_child_weight','n_estimators','reg_alpha',
        'reg_lambda', 'scale_pos_weight',  'subsample','Mapping' ]]

    # Using dictionary comprehension to get min and max of each list in the dataframe
    min_max_values = {col: [min(df[col].explode()), max(df[col].explode())] for col in cols_p}

    # Creating a new dataframe with min and max values
    df_min_max = pd.DataFrame(min_max_values, index=['min', 'max'])

    aux = df_min_max.T
    #aux.to_csv("aux/X_Gbost_best.csv")

    aux_b = df_sin_duplicados_columnas_especificas.pivot_table(index=['Mapping'], values=cols_hp).T 
    aux_b.index = aux.index


    df_concatenado = pd.concat([aux_b, aux], axis=1)

    print(df_concatenado.to_latex())
   
def agregar_anterior_hp_S(df_sin_duplicados_columnas_especificas_,lista):
    s = "f1_test f1_train sensitivity_test specificity_test precision_test accuracy_test sensitivity_train specificity_train precision_train accuracy_train"

    # Convertir la cadena en una lista de strings
    lista = s.split()

    print(lista)
    
    df_sin_duplicados_columnas_especificas_.columns = [col if col not in lista else f"{col}_a" for col in df_sin_duplicados_columnas_especificas_.columns]
    return df_sin_duplicados_columnas_especificas_


def crear_tabla_results(df_sin_duplicados_columnas_especificas_,df_sin_duplicados_columnas_especificas,title_,metric_list,lista,name_p):
    aux = pd.merge(df_sin_duplicados_columnas_especificas_,df_sin_duplicados_columnas_especificas[lista+["Mapping"]], how="right", on="Mapping")
    pivot_table = [aux.pivot_table(index=['Mapping'], values=[metric +'_test_a', metric +'_train_a',metric +'_test', metric +'_train']) for metric in metric_list]
    pivot_table_final = pd.concat(pivot_table, axis=1)
    # Supongamos que 'aux' es tu DataFrame y 'metric_list' es tu lista de métricas

    print(pivot_table_final)
    # Supongamos que 'pivot_table' es tu DataFrame
    pivot_table_final = pivot_table_final.round(3)

    # Supongamos que 'pivot_table' es tu DataFrame y 'metric_list' es tu lista de métricas
    for metric in metric_list:
        pivot_table_final = pivot_table_final.rename(columns={
            metric +'_test': metric +' test HP*',
            metric +'_train': metric +' train HP*',
            metric +'_test_a': metric +' test',
            metric +'_train_a': metric +' train'
        })


    print(pivot_table_final.T.to_latex())
    # Supongamos que 'pivot_table_final' es tu DataFrame
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table_final.T, annot=True, cmap='Blues', fmt=".3f")

    plt.xlabel('Simplification')
    plt.title(title_)
    plt.savefig(IMAGES_PRED_DICT+name_p+'.svg')
    plt.show()
    
def get_dft():
    import pandas as pd
    df2 = pd.read_csv(input_model_pred / 'ICD9_CODE_procedures_outs_visit_non_filtered.csv')
    df2.shape
    aux = df2[df2['LOSRD_sum']<200]
    aux.shape
    
    grouped_df1 = df2.groupby('SUBJECT_ID').agg({'Age_max': 'max', 'LOSRD_sum': 'sum','GENDER':mode,'L_1s_last_p1':mode}).reset_index()


    df2['GENDER']
    

    # Assuming you have a DataFrame named 'df' with a column named 'gender' containing 1s and 0s

    # Create a dictionary to map the values
    gender_map = {1: 'Male', 0: 'Female'}
    gender_map2 = {1.0: 'Male', 0.0: 'Female'}

    # Use the map function to create a new column 'gender_label'
    #df2['GENDER'] = df2['GENDER'].map(gender_map)
    grouped_df1['GENDER'] = grouped_df1['GENDER'].map(gender_map2)
    df2['GENDER'] = df2['GENDER'].map(gender_map2)

    print(grouped_df1['GENDER'].value_counts())
    gender_map = {1.0: 'Male', 0.0: 'Female'}

    # Use the map function to create a new column 'gender_label'
    grouped_df1['Gender'] = grouped_df1['GENDER'].map(gender_map)
    grouped_df1['GENDER'] = np.where(grouped_df1['GENDER'] ==1.0, 'Male', 'Female')
    grouped_df1["Age_max"]=[int(i) for i in grouped_df1.Age_max]
    return grouped_df1,df2
    
# %%
def mode(x):
    '''function to obtain the mode'''
    return x.mode()[0]





def plot_age(df):
    # Set the background style
    from scipy.stats import gaussian_kde
    plt.style.use("seaborn-white")

    # Define two shades of blue
    dark_blue = "#00008B"   # A dark blue
    light_blue = "#ADD8E6"  # A light blue

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(7, 4))

    # Calculate the kernel density estimate for 'Age_max' for each gender
    age_max_male = df[df['GENDER'] == 'Male']['Age_max']
    age_max_female = df[df['GENDER'] == 'Female']['Age_max']
    kde_age_max_male = gaussian_kde(age_max_male)
    kde_age_max_female = gaussian_kde(age_max_female)

    # Generate x-values for the kernel density estimate
    x_age_max = np.linspace(age_max_male.min(), age_max_male.max(), 100)

    # Plot the kernel density estimate for 'Age_max' with the dark blue for male and light blue for female
    ax.plot(x_age_max, kde_age_max_male(x_age_max), color=dark_blue, label='Male')
    ax.plot(x_age_max, kde_age_max_female(x_age_max), color=light_blue, label='Female')

    # Set the labels and title for the 'Age_max' plot
    ax.set_xlabel('Age')
    ax.set_ylabel('Density')
    ax.set_title('Kernel Density Plot - Age')
    ax.set_ylim(bottom=0)  # Set the y-axis limit to start at zero
    plt.savefig(IMAGES_Demo+'age_kernelplot.svg')
    # Show the plot
    plt.show()

def plot_los(df):
    from scipy.stats import gaussian_kde
    # Set the background style
    plt.style.use("seaborn-white")

    # Define two shades of blue
    dark_blue = "#00008B"   # A dark blue
    light_blue = "#ADD8E6"  # A light blue

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(7, 4))

    # Calculate the kernel density estimate for 'LOSRD_sum' for each gender
    aux = df[df['LOSRD_sum']<200]
    losrd_sum_male = aux[aux['GENDER'] == 'Male']['LOSRD_sum']
    losrd_sum_female = aux[aux['GENDER'] == 'Female']['LOSRD_sum']
    kde_losrd_sum_male = gaussian_kde(losrd_sum_male)
    kde_losrd_sum_female = gaussian_kde(losrd_sum_female)

    # Generate x-values for the kernel density estimate
    x_losrd_sum = np.linspace(losrd_sum_male.min(), losrd_sum_male.max(), 100)

    # Plot the kernel density estimate for 'LOSRD_sum' with the dark blue for male and light blue for female
    ax.plot(x_losrd_sum, kde_losrd_sum_male(x_losrd_sum), color=dark_blue, label='Male')
    ax.plot(x_losrd_sum, kde_losrd_sum_female(x_losrd_sum), color=light_blue, label='Female')

    # Set the labels and title for the 'LOSRD_sum' plot
    ax.set_xlabel('Length of Stay (days)')
    ax.set_ylabel('Density')
    ax.set_title('Kernel Density Plot - Length of Stay')
    ax.set_ylim(bottom=0)  # Set the y-axis limit to start at zero
    plt.savefig(IMAGES_Demo+'los_kernelplot.svg')
    # Show the plot
    plt.show()


def get_dataframes():
    
    days = '30'
    archivo_input_label = 'data_preprocess_non_filtered.csv'

    ejemplo_dir = str(PROCEDURES_DIRECTORY)+"/input_model_visit_procedures/"
    days_list = ["90"]
    ficheros = read_director(ejemplo_dir)
    len(ficheros)
    for i in ficheros:
        print(i)
        
    ficheros = [i for i in ficheros if i != 'sin_codigo_non_filtered.csv']
    #X_aux = pd.read_csv(ejemplo_dir+'cat_threshold .88 most frequent_outs_visit_non_filtered.csv')

    
    dataframes = []
    for i in ficheros:
            
        if i in ['ICD9_CODE_procedures.csv', 'CCS CODES_proc.csv', 'cat_threshold .999 most frequent']:
            prepo = "max"
        elif i in ['cat_threshold .95 most frequent_proc','cat_threshold .88 most frequent']:
            prepo = "power"
        else:
            prepo = "std"
        print(prepo)
        
        # se obtiene la matriz de features y la variable response
        X_aux = pd.read_csv(ejemplo_dir+i)
        
        # if it's not BERT embedding, we do the preprocessing
        X_aux = X_aux.drop(["HADM_ID","SUBJECT_ID",'Unnamed: 0','L_1s_last','HADM_ID', 'Age_max', 'LOSRD_sum',
        'L_1s_last', 'LOSRD_avg', 'ADMISSION_TYPE', 'ADMISSION_LOCATION',
        'DISCHARGE_LOCATION', 'INSURANCE', 'RELIGION', 'MARITAL_STATUS',
        'ETHNICITY', 'GENDER', 'L_1s_last_p1'], axis=1)

        try:
            X = X_aux.values
            dataframes.append(pd.DataFrame(X))
        except:
            print("not_found")
            pass
    return dataframes        
        