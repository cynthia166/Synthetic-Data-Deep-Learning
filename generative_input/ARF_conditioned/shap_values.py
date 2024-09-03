from config_conditonalarf import *
from utils import *
import joblib
from ARF_log import arf2
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
import shap

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import LabelEncoder

def get_column_indices(df, column_names):
    """
    Obtiene los índices numéricos de las columnas especificadas en un DataFrame.
    
    :param df: DataFrame de pandas
    :param column_names: Lista de nombres de columnas
    :return: Lista de índices numéricos correspondientes a las columnas
    """
    return [df.columns.get_loc(col) for col in column_names if col in df.columns]


def plot_top_20_rf_importance(rf,feature_names,path_img_shap=None):
    # Create and train the Random Forest model

    # Get feature importances
    importances = rf.feature_importances_

    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Get the top 20 feature importances
    top_20_indices = indices[:20]
    top_20_importances = importances[top_20_indices]
    top_20_features = [feature_names[i] for i in top_20_indices]
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.title("Top 20 Feature Importances in Random Forest")
    plt.bar(range(20), top_20_importances)
    plt.xticks(range(20), top_20_features, rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()
    if path_img_shap is not None:
       save_plot_as_svg(plt, path_img_shap, "concatenated_images")
    return top_20_features



def get_column_indices(df, column_names):
    return [df.columns.get_loc(col) for col in column_names if col in df.columns]



def robust_preprocess_dataframe(df):
    """
    Preprocesa el DataFrame de manera robusta, manejando varios tipos de datos.
    """
    df_processed = df.copy()
    
    for column in df_processed.columns:
        if pd.api.types.is_datetime64_any_dtype(df_processed[column]):
            # Convertir Timestamp a número de días desde una fecha de referencia
            df_processed[column] = (df_processed[column] - pd.Timestamp("1970-01-01")).dt.total_seconds() / (24 * 60 * 60)
        elif df_processed[column].dtype == 'object':
            # Codificar variables categóricas
            le = LabelEncoder()
            df_processed[column] = le.fit_transform(df_processed[column].astype(str))
        elif not pd.api.types.is_numeric_dtype(df_processed[column]):
            # Convertir cualquier otro tipo no numérico a numérico
            df_processed[column] = pd.to_numeric(df_processed[column], errors='coerce')
    
    # Manejar valores NaN
    df_processed = df_processed.fillna(df_processed.mean())
    
    return df_processed

def plot_partial_dependence_top_features(rf_model, X_df, top_n=5,path_img=None):
    # Preprocesar el DataFrame
    X_df_processed = robust_preprocess_dataframe(X_df)
    
    # Asegurarse de que estamos usando las mismas características que en el entrenamiento
    if hasattr(rf_model, 'feature_names_in_'):
        feature_names = rf_model.feature_names_in_
        X_df_processed = X_df_processed[feature_names]
    else:
        feature_names = X_df_processed.columns.tolist()
    
    # Verificar que todas las columnas sean numéricas
    for col in X_df_processed.columns:
        if not pd.api.types.is_numeric_dtype(X_df_processed[col]):
            raise ValueError(f"La columna '{col}' no es numérica después del preprocesamiento.")
    
    # Obtener las importancias de las características
    importances = rf_model.feature_importances_
    
    # Crear un DataFrame con las importancias
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Ordenar por importancia y obtener los top_n
    top_features = feature_importance_df.sort_values('importance', ascending=False).head(top_n)
    
    # Obtener los índices de las características más importantes
    top_indices = [list(feature_names).index(feature) for feature in top_features['feature']]
    
    # Crear gráficos de dependencia parcial
    fig, axes = plt.subplots(nrows=(top_n+1)//2, ncols=2, figsize=(12, 4*((top_n+1)//2)))
    axes = axes.flatten()

    for i, (idx, feature) in enumerate(zip(top_indices, top_features['feature'])):
        ax = axes[i]
        
        # Determinar si la característica es categórica o continua
        if X_df[feature].dtype == 'object' or X_df[feature].nunique() < 10:
            kind = "average"
            f_type = "Categórica"
        else:
            kind = "average"
            f_type = "Continua"
        
        try:
            PartialDependenceDisplay.from_estimator(rf_model, X_df_processed, [idx], 
                                                    feature_names=feature_names,
                                                    ax=ax, kind=kind)
        except Exception as e:
            print(f"Error al procesar la característica '{feature}': {str(e)}")
            continue
        
        ax.set_title(f"{feature} ({f_type})")
        ax.set_ylabel("Partial Dependence")
    
    # Eliminar subplots vacíos si los hay
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
    if path_img_shap is not None:
       save_plot_as_svg(plt, path_img_shap, "concatenated_images")
    plt.close()

# Ejemplo de uso
# Asumiendo que ya tienes tu X_train y tu modelo rf entrenado
# plot_partial_dependence_top_features(rf, X_train, top_n=5)

def show_non_common_columns(set1, set2, name1="Set 1", name2="Set 2"):
    """
    Muestra las columnas que no son comunes entre dos conjuntos de datos.
    
    :param set1: Primer conjunto de columnas (puede ser DataFrame, lista, o array)
    :param set2: Segundo conjunto de columnas (puede ser DataFrame, lista, o array)
    :param name1: Nombre para identificar el primer conjunto
    :param name2: Nombre para identificar el segundo conjunto
    :return: Lista de columnas no comunes
    """
    # Convertir a conjuntos para facilitar la comparación
    if isinstance(set1, pd.DataFrame):
        set1 = set(set1.columns)
    elif isinstance(set1, np.ndarray):
        set1 = set(set1)
    else:
        set1 = set(set1)
    
    if isinstance(set2, pd.DataFrame):
        set2 = set(set2.columns)
    elif isinstance(set2, np.ndarray):
        set2 = set(set2)
    else:
        set2 = set(set2)
    
    # Encontrar las columnas no comunes
    non_common = set1.symmetric_difference(set2)
    
    # Mostrar resultados
    print(f"Columnas que no son comunes entre {name1} y {name2}:")
    for col in sorted(non_common):
        if col in set1:
            print(f"  - {col} (solo en {name1})")
        else:
            print(f"  - {col} (solo en {name2})")
    
    print(f"\nNúmero total de columnas no comunes: {len(non_common)}")
    print(f"Número total de columnas en {name1}: {len(set1)}")
    print(f"Número total de columnas en {name2}: {len(set2)}")
    print(f"Número de columnas en común: {len(set1.intersection(set2))}")
    
    return list(non_common)


def  preprocess_original_input( x_train, demographiccols,encoder_dict):
    x_train, encoders =group_and_encode_demographics(    x_train, demographiccols,encoder_dict)
    try: 
        x_train["year"] = x_train['ADMITTIME'].dt.year
        x_train['month'] = x_train['ADMITTIME'].dt.month
    except:
        pass
    #x_train = x_train.drop(columns=columns_to_drop) 
    
    x_train= sample_patients_list(ruta_patients, x_train)    
    x_train =  change_dtypes(x_train,cols_continuous_d)   

    return x_train, encoders


def prepare_data_for_existing_arf(df, arf_model):
    """
    Prepara los datos para ser compatibles con un modelo ARF existente.
    
    :param df: DataFrame original
    :param arf_model: Modelo ARF ya entrenado
    :return: DataFrame preparado
    """
    df_prepared = df.copy()
    
    # Obtener las características que el modelo ARF está utilizando
    if hasattr(arf_model, 'feature_names_in_'):
        model_features = arf_model.feature_names_in_
    else:
        # Si el modelo no tiene feature_names_in_, asumimos que usa todas las columnas
        model_features = df.columns
    
    for column in model_features:
        if column not in df_prepared.columns:
            raise ValueError(f"La columna '{column}' no está presente en los datos.")
        
        if pd.api.types.is_datetime64_any_dtype(df_prepared[column]):
            # Convertir Timestamp a número de días desde una fecha de referencia
            df_prepared[column] = (df_prepared[column] - pd.Timestamp("1970-01-01")).dt.total_seconds() / (24 * 60 * 60)
        elif df_prepared[column].dtype == 'object' or (df_prepared[column].dtype.name == 'category'):
            # Para columnas categóricas, convertimos a cadena y luego a categórica
            # Esto preserva las categorías existentes y maneja nuevas categorías como NaN
            df_prepared[column] = pd.Categorical(df_prepared[column].astype(str))
        elif not pd.api.types.is_numeric_dtype(df_prepared[column]):
            # Convertir cualquier otro tipo no numérico a numérico
            df_prepared[column] = pd.to_numeric(df_prepared[column], errors='coerce')
    
    # Seleccionar solo las características que el modelo usa
    df_prepared = df_prepared[model_features]
    
    return df_prepared

def create_and_save_shap_summary_plot(arf_model, X, output_file='shap_summary_plot.png', max_display=20):
    """
    Crea un gráfico de resumen SHAP para un modelo ARF existente y lo guarda como imagen.
    
    :param arf_model: Modelo ARF ya entrenado
    :param X: Conjunto de datos preparado
    :param output_file: Nombre del archivo de salida para el gráfico
    :param max_display: Número máximo de características a mostrar en el gráfico
    """
    # Crear el explainer SHAP
    explainer = shap.TreeExplainer(arf_model)
    
    # Calcular los valores SHAP
    samples = X.sample(min(1000, len(X)), random_state=42)
    samples =samples.drop(columns ="ADMITTIME")
    shap_values = explainer.shap_values(samples)
    
    # Crear el gráfico de resumen SHAP
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, 
                      samples, 
                      plot_type="bar", 
                      max_display=max_display, 
                      show=False)
    
    # Ajustar el diseño y guardar el gráfico
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico de resumen SHAP guardado en '{output_file}'")


def convert_to_string(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object' or is_datetime(df[col]):
            df[col] = df[col].astype(str)
    return df

def is_datetime(series):
    return pd.api.types.is_datetime64_any_dtype(series)

features_path_arf = "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/ARF_fixed_sansvar/train_ehr_synthetic_ehr_dataset_contrainst_ARF_ARF_fixed_v.pkl"
x_train = load_data(features_path_arf)[:1000]

file_synthethic ="/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/ARF_fixed_sansvar/synthetic_data_generative_model_arf_per_fixed_v0.7.pkl"


encoder_dict = load_data(encoders_demos)
#x_train, encoders = preprocess_original_input( x_train, demographiccols,encoder_dict)
X_synthetic =load_data(file_synthethic)[:len(x_train)]
#X_synthetic, encoders = preprocess_original_input( X_synthetic, demographiccols,encoder_dict)
#X_synthetic =X_synthetic.drop(columns ="ADMITTIME")
#x_train =x_train.drop(columns ="ADMITTIME")

X_syntheticrf = joblib.load(arf_path_file)
print(rf)
feature_names =X_synthetic.columns

#only seen variables from arf
#x_train=x_train[[i for i in feature_names if i!="ADMITTIME"]]
x_train=x_train[[i for i in feature_names ]]
X_synthetic=X_synthetic[[i for i in feature_names ]]

x_train = convert_to_string(x_train)
X_synthetic = convert_to_string(X_synthetic)

x_train = convert_to_string(x_train)
X_synthetic = convert_to_string(X_synthetic)
#print importance 


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def get_arf_feature_importance(arf_model, X, y, n_repeats=10, random_state=42):
    """
    Calculate feature importance for ARF model using multiple methods.
    
    :param arf_model: Trained ARF model
    :param X: Feature matrix (pandas DataFrame)
    :param y: Target vector
    :param n_repeats: Number of times to repeat permutation importance calculation
    :param random_state: Random state for reproducibility
    :return: DataFrame with feature importances
    """
    feature_names = X.columns.tolist()
    
    importances = {}
    
    # Method 1: Use built-in feature importance if available
    if hasattr(arf_model, 'feature_importances_'):
        importances['built_in'] = arf_model.feature_importances_
    
    # Method 2: Calculate permutation importance
    #X["ADMITTIME"] = X["ADMITTIME"].astype("object") 
    #X_processed[column] = X[column].astype(int) // 10**9

    perm_importance = permutation_importance(arf_model, X, y, n_repeats=n_repeats, random_state=random_state)

    importances['permutation'] = perm_importance.importances_mean
    
    # Method 3: Calculate mean decrease in impurity across all trees
    if hasattr(arf_model, 'estimators_'):
        tree_importances = []
        for tree in arf_model.estimators_:
            tree_importances.append(tree.feature_importances_)
        importances['mean_decrease_impurity'] = np.mean(tree_importances, axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({'feature': feature_names})
    
    for method, imp in importances.items():
        importance_df[f'{method}_importance'] = imp
    
    # Sort by the first available importance method
    sort_column = [col for col in importance_df.columns if col.endswith('_importance')][0]
    importance_df = importance_df.sort_values(sort_column, ascending=False).reset_index(drop=True)
    
    return importance_df

def plot_feature_importance(importance_df, method='permutation', top_n=20):
    """
    Plot top N important features.
    
    :param importance_df: DataFrame with feature importances
    :param method: The importance method to plot ('built_in', 'permutation', or 'mean_decrease_impurity')
    :param top_n: Number of top features to plot
    """
    importance_column = f'{method}_importance'
    if importance_column not in importance_df.columns:
        raise ValueError(f"Importance method '{method}' not found in the DataFrame")
    
    plt.figure(figsize=(12, 6))
    plt.bar(importance_df['feature'][:top_n], importance_df[importance_column][:top_n])
    plt.xticks(rotation=90)
    plt.title(f'Top {top_n} Most Important Features (ARF - {method})')
    plt.tight_layout()
    plt.show()


x = pd.concat([x_train, X_synthetic])
y = np.concatenate([np.zeros(x_train.shape[0]), np.ones(x_train.shape[0])])
      
# Example usage:
# arf_model = joblib.load(path_result_arf + arf_demos)
feature_importance = get_arf_feature_importance(rf,x , y)
print(feature_importance.head(10))  # Print top 10 most important features
plot_feature_importance(feature_importance, method='permutation')
plot_feature_importance(feature_importance, method='mean_decrease_impurity')





feature_importance.to_csv(path_img_shap+"_permuation.csv", index=False)


#print importances tree bases

cols = show_non_common_columns(x_train.columns, feature_names, name1="train", name2="feature")
print(f'lista para excluir {len(cols)}')
x_train = x_train[[i for i in x_train.columns if i not in cols]]  

importances = rf.feature_importances_
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("Feature importances:")
print(feature_importances)

feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Ordenar por importancia en orden descendente
feature_importances = feature_importances.sort_values('Importance', ascending=False)
    
    # Guardar en CSV
feature_importances.to_csv(path_img_shap+"importance_featirs.csv", index=False)
#plot importance for 10 top 20 codes.
top_20_features = plot_top_20_rf_importance(rf,feature_names,path_img_shap)
# Add this method to your MultiModelDemographicPredictor class


#shhap values
explainer = shap.TreeExplainer(rf)
samples =x_train.drop(columns ="ADMITTIME")
shap_values = explainer.shap_values(samples)
shap.summary_plot(shap_values, x_train)
plt.savefig(path_img_shap='my_shap_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
#create_and_save_shap_summary_plot(rf, X_prepared,path_img_shap+ 'mi_shap_summary_plot.png')

print("finish")