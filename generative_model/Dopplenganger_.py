import pandas as pd 
import sys
import os
import os
from gretel_synthetics.timeseries_dgan.config import DfStyle
# Imprimir la ruta del directorio actual
print(os.getcwd())

# Cambiar el directorio de trabajo actual a 'nueva/ruta'

print(os.getcwd())

sys.path.append('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning')
#sys.path.append('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning')
from preprocessing.preprocess_input_SDmodel import *
import numpy as np
#from evaluation.Resemblance.metric_stat import *
#from synthcity.metrics.eval_statistical import AlphaPrecision
from preprocessing.config import *
# Instanciar la clase
#alpha_precision = AlphaPrecision(**kwargs)
# stdlib
import sys
import warnings
#from evaluation.Resemblance.metric_stat import KolmogorovSmirnovTest
# synthcity absolute

warnings.filterwarnings("ignore")

import numpy as np
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig

import pandas as pd
#from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
     
from preprocessing.preprocess_input_SDmodel import *
import pickle
import gzip
import os
import config
#os.chdir('./')
import pandas as pd
import numpy as np

def pad_patient_sequences(df, max_sequence_length, constant_columns):
    # Make a copy of the input DataFrame
    df = df.copy()
    
    # Ensure visit_rank is integer type
    df['visit_rank'] = df['visit_rank'].astype(np.int32)
    df['SUBJECT_ID'] =df['SUBJECT_ID'].astype(np.int32)
    
    # Get unique patient IDs
    unique_patients = df['SUBJECT_ID'].unique()
    
    # Create all possible combinations of patient IDs and visit ranks
    all_combinations = pd.MultiIndex.from_product(
        [unique_patients, range(1, max_sequence_length + 1)],
        names=['SUBJECT_ID', 'visit_rank']
    ).to_frame(index=False)
    
    # Ensure 'SUBJECT_ID' and 'visit_rank' are not in constant_columns
    constant_columns = [col for col in constant_columns if col not in ['SUBJECT_ID', 'visit_rank']]
    
    # For each constant column, get the first non-null value for each patient
    constant_values = df.groupby('SUBJECT_ID')[constant_columns].first().reset_index()
    
    # Merge the constant values with all combinations
    padded_df = pd.merge(all_combinations, constant_values, on='SUBJECT_ID', how='left')
    
    # Merge with the original data
    padded_df = pd.merge(padded_df, df, on=['SUBJECT_ID', 'visit_rank'], how='left', suffixes=('', '_y'))
    
    # For constant columns, use the values from constant_values where the original data is null
    for col in constant_columns:
        padded_df[col] = padded_df[col + '_y'].fillna(padded_df[col])
        padded_df = padded_df.drop(columns=[col + '_y'])
    
    # Identify columns to fill with zeros (excluding SUBJECT_ID, visit_rank, and constant columns)
    columns_to_fill = padded_df.columns.difference(['SUBJECT_ID', 'visit_rank'] + constant_columns)
    
    # Fill NaN values with 0 for the identified columns
    padded_df[columns_to_fill] = padded_df[columns_to_fill].fillna(0)
    
    # Sort the DataFrame by SUBJECT_ID and visit_rank
    padded_df = padded_df.sort_values(['SUBJECT_ID', 'visit_rank'])
    
    return padded_df

# Example usage:
# df = ... # Your input DataFrame
# max_seq_length = 42  # Maximum sequence length
# constant_cols = ['SAMPLE_NAME', 'GENDER', 'AGE_AT_BASELINE']  # Example constant columns
# padded_result = pad_patient_sequences(df, max_seq_length, constant_cols)# Imprimir la ruta del directorio actual
def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)    


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def count_inconsistencies(df, id_column, attribute_column):
    """
    Count the number of inconsistencies in an attribute within each example.
    
    Args:
    df (pd.DataFrame): The input DataFrame
    id_column (str): The name of the column containing unique identifiers
    attribute_column (str): The name of the column to check for consistency
    
    Returns:
    int: The number of unique IDs with inconsistent attribute values
    pd.DataFrame: A DataFrame containing inconsistent entries
    """
    # Group by ID and check if the attribute has more than one unique value
    inconsistent = df.groupby(id_column).filter(
        lambda x: x[attribute_column].nunique() > 1
    )
    
    num_inconsistent = inconsistent[id_column].nunique()
    
    return num_inconsistent, inconsistent.sort_values(by=[id_column, attribute_column])

def fix_inconsistencies(df, id_column, attribute_column, method='mode'):
    """
    Fix inconsistencies in an attribute within each example.
    
    Args:
    df (pd.DataFrame): The input DataFrame
    id_column (str): The name of the column containing unique identifiers
    attribute_column (str): The name of the column to fix
    method (str): The method to use for fixing ('mode', 'first', or 'nan')
    
    Returns:
    pd.DataFrame: The DataFrame with fixed attribute values
    """
    if method == 'mode':
        # Use the most common value for each ID
        mode_values = df.groupby(id_column)[attribute_column].agg(lambda x: x.mode().iloc[0])
        df[attribute_column] = df[id_column].map(mode_values).fillna(df[attribute_column])
    elif method == 'first':
        # Use the first non-null value for each ID
        first_values = df.groupby(id_column)[attribute_column].first()
        df[attribute_column] = df[id_column].map(first_values).fillna(df[attribute_column])
    elif method == 'nan':
        # Set inconsistent values to NaN
        inconsistent_ids = df.groupby(id_column).filter(lambda x: x[attribute_column].nunique() > 1)[id_column].unique()
        df.loc[df[id_column].isin(inconsistent_ids), attribute_column] = np.nan
    else:
        raise ValueError("Invalid method. Choose 'mode', 'first', or 'nan'.")
    
    return df    
    
def filter_columns_by_partial_name(df: pd.DataFrame, partial_names: list):
    """
    Filtra las columnas de un DataFrame que contienen alguno de los nombres parciales especificados.

    :param df: DataFrame de pandas
    :param partial_names: Lista de nombres parciales para buscar en las columnas
    :return: Lista de nombres de columnas que contienen alguno de los nombres parciales
    """
    filtered_columns = []
    for col in df.columns:
        if any(partial_name.upper() in col.upper() for partial_name in partial_names):
            filtered_columns.append(col)
    return filtered_columns    
    
SD = "generative_model/Doopleganger/"
synthetic_file = "synthetic_data_doopleganger.pkl"
train_split = False
if train_split == True:
    print(os.getcwd())
        
        

    
    features = load_pickle("train_sp/non_prepo/DATASET_NAME_non_prepo_non_preprocess.pkl")
    attributes=pd.read_csv("train_sp/non_prepo/static_data_non_preprocess.csv")
    numpy_array_t = np.stack([df.to_numpy() for df in features])
    features = numpy_array_t.reshape(numpy_array_t.shape[0],numpy_array_t.shape[2],numpy_array_t.shape[1])
    attributes = attributes.to_numpy()[:19]

    N, T, D = features.shape   
    print('data shape:', N, T, D) 
    print('attributes shape:', attributes.shape)
    valid_perc = 0.1
    dataset_name = '/non_prepo/DATASET_NAME_non_prepro'
    
    split(valid_perc,dataset_name,features,attributes)

#outcomes=pd.read_csv(DARTA_INTERM_intput+"outcomes_preprocess.csv")     
#########################################################################
#########################################################################
#########################################################################
##########################LOAD DATA##########################
#########################################################################
#########################################################################

ruta = SD  # Reemplaza esto con la ruta que quieres verificar

if os.path.exists(ruta):
    print('La ruta existe.')
    print(ruta)
else:
    print('La ruta no existe.')
    print(ruta)

#prepare dataframe    
   

# Define column types
path_dataset_demos ="/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/entire_ceros_tabular_data_demos2.pkl"  

columnas_demograficas= ['RELIGION_encoded', 'MARITAL_STATUS_encoded',  'ETHNICITY_encoded','GENDER_encoded']

data = load_pickle(path_dataset_demos)

non_datetime_cols = data.select_dtypes(exclude=['datetime64']).columns
non_datetime_cols = [i for i in non_datetime_cols if i!= "ADMITTIME" ]

for col in non_datetime_cols:
    data[col] = data[col].astype(np.int32)

id_column = 'SUBJECT_ID'
attribute_column = 'RELIGION_encoded'
for attribute_column in columnas_demograficas:
    # # Count inconsistencies
    num_inconsistent, inconsistent_data = count_inconsistencies(data, id_column, attribute_column)
    print(f"Number of subjects with inconsistent {attribute_column}: {num_inconsistent}")
    print("Sample of inconsistent data:")
    print(inconsistent_data.head())
    data = fix_inconsistencies(data, id_column, attribute_column, method='mode')


#data = data.drop(columns=columnas_demograficas) 
  # Your input DataFrame
max_seq_length = data.visit_rank.max()  # Or whatever your maximum sequence length is


demographiccols = [  'RELIGION',

                        'MARITAL_STATUS',  'ETHNICITY','GENDER']
attribute_columns = filter_columns_by_partial_name(data, demographiccols)

feature_columns = [i for i in data.columns if i not in attribute_columns + [ 'visit_rank','SUBJECT_ID',] ]
example_id_column = 'SUBJECT_ID'
time_column = 'visit_rank'
discrete_columns = [i for i in data.columns if i not in attribute_columns + [ 'visit_rank','SUBJECT_ID','Age', 'LOSRD_avg','days from last visit']]
#padded sequene
data = pad_patient_sequences(data, max_seq_length,attribute_columns)


name_file = "Dopplenganger_nonprepo_epochocs_120.pth"  
config2 = DGANConfig(
    max_sequence_len=max_seq_length,
    sample_len=2,
    batch_size=32,
    epochs=1
    
    
)

model = DGAN(config2)



# Create a sample DataFrame


attributes = data[columnas_demograficas+['SUBJECT_ID']].groupby('SUBJECT_ID').first().values
def dataframe_to_3d_array(data, subject_col, step_col, feature_cols):
    # Get unique subjects and maximum number of steps
    subjects = data[subject_col].unique()
    max_steps = int(data[step_col].max())  # Ensure max_steps is an integer
    
    # Create the 3D numpy array
    array_3d = np.zeros((len(subjects), max_steps, len(feature_cols)))
    
    # Fill the array
    for i, subject in enumerate(subjects):
        subject_data = data[data[subject_col] == subject]
        for _, row in subject_data.iterrows():
            try:
                step = int(row[step_col]) - 1  # Convert to integer and adjust for 0-based indexing
                if step < 0 or step >= max_steps:
                    print(f"Warning: Step {step+1} out of range for subject {subject}. Skipping.")
                    continue
                array_3d[i, step, :] = row[feature_cols].values
            except ValueError as e:
                print(f"Error processing step for subject {subject}: {e}")
                print(f"Row data: {row}")
            except IndexError as e:
                print(f"IndexError for subject {subject}, step {step+1}: {e}")
                print(f"Array shape: {array_3d.shape}")
                print(f"Attempted indices: i={i}, step={step}")
                print(f"Row data: {row}")
                raise  # Re-raise the exception after printing debug info
    
    return array_3d, subjects
# Usage example
subject_column = 'SUBJECT_ID'
step_column = 'visit_rank'
feature_columns = [i for i in data.columns if i not in columnas_demograficas]

features, subject_labels = dataframe_to_3d_array(data, subject_column, step_column, feature_columns)

print(f"Shape of 3D array: {features.shape}")
print(f"Number of unique subjects: {len(subject_labels)}")


model.train_numpy(  features=features,
       attributes= attributes)

model.generate_numpy(len(data))
model.save(SD+name_file )

data_syn = model.generate_dataframe()
save_pickle(data_syn,SD + synthetic_file)

model.train_dataframe(data,
        attribute_columns,
        feature_columns,
        example_id_column,
        time_column,
        discrete_columns,
        DfStyle.LONG )


model.save(SD+name_file )

data_syn = model.generate_dataframe()
save_pickle(data_syn,SD + synthetic_file)
# dataset_name = '/non_prepo/DATASET_NAME_non_prepro'
# features_train = load_data('train_sp' + dataset_name + 'train_data_features.pkl')
# attribute_train = load_data('train_sp'  + dataset_name + 'train_data_attributes.pkl')
# #full_train_data = np.array(data)

# # Para eliminar la fila i, puedes hacer lo siguiente:
# i = 0  # Cambia esto al índice de la fila que quieres eliminar
# features_train = np.delete(features_train, i, axis=1)
# print(features_train.shape)

# #realizar train and test split
# #split(valid_perc,dataset_name):


   
# ######


# #attributes = np.random.rand(10000, 3)
# #features = np.random.rand(10000, 20, 2)
 
# #config = DGANConfig(    max_sequence_len=20,sample_len=2, batch_size=1000,epochs=10)

# config2 = DGANConfig(
#     max_sequence_len=features_train.shape[1],
#     sample_len=110,
#     batch_size=100,
#     epochs=10
# )

# model = DGAN(config2)

# model.train_numpy(  features=features_train,
#         attributes= attribute_train)


# model.save(SD+name_file )
# #synthetic_attributes, synthetic_features = model.generate_numpy(n =len(sample_patients))





# #m = model.load("aux/model_name.pth"  )
# #synthetic_attributes, synthetic_features = m.generate_numpy(n =len(sample_patients))


# def inverse_transform_attributes(
#     transformed_data: np.ndarray,
#     outputs: list[Output],
# ) -> Optional[np.ndarray]:
#     """Inverse of transform_attributes to map back to original space.

#     Args:
#         transformed_data: 2d numpy array of internal representation
#         outputs: Output metadata for each variable
#     """
#     # TODO: we should not use nans as an indicator and just not call this
#     # method, or use a zero sized numpy array, to indicate no attributes.
#     if np.isnan(transformed_data).any():
#         return None
#     parts = []
#     transformed_index = 0
#     for output in outputs:
#         original = output.inverse_transform(
#             transformed_data[:, transformed_index : (transformed_index + output.dim)]
#         )
#         parts.append(original.reshape((-1, 1)))
#         transformed_index += output.dim

#     return np.hstack(parts)


# def inverse_transform_features(
#     transformed_data: np.ndarray,
#     outputs: List[Output],
#     additional_attributes: Optional[np.ndarray] = None,
# ) -> np.ndarray:
#     """Inverse of transform_features to map back to original space.

#     Args:
#         transformed_data: 3d numpy array of internal representation data
#         outputs: Output metadata for each variable
#         additional_attributes: midpoint and half-ranges for outputs with
#             apply_example_scaling=True

#     Returns:
#         List of numpy arrays, each element corresponds to one sequence with 2d
#         array of (time x variables).
#     """
#     transformed_index = 0
#     additional_attribute_index = 0
#     parts = []
#     for output in outputs:
#         if "OneHotEncodedOutput" in str(
#             output.__class__
#         ) or "BinaryEncodedOutput" in str(output.__class__):

#             v = transformed_data[
#                 :, :, transformed_index : (transformed_index + output.dim)
#             ]
#             target_shape = (transformed_data.shape[0], transformed_data.shape[1], 1)

#             original = output.inverse_transform(v.reshape((-1, v.shape[-1])))

#             parts.append(original.reshape(target_shape))
#             transformed_index += output.dim
#         elif "ContinuousOutput" in str(output.__class__):
#             output = cast(ContinuousOutput, output)

#             transformed = transformed_data[:, :, transformed_index]

#             if output.apply_example_scaling:
#                 if additional_attributes is None:
#                     raise ValueError(
#                         "Must provide additional_attributes if apply_example_scaling=True"
#                     )

#                 midpoint = additional_attributes[:, additional_attribute_index]
#                 half_range = additional_attributes[:, additional_attribute_index + 1]
#                 additional_attribute_index += 2

#                 mins = midpoint - half_range
#                 maxes = midpoint + half_range
#                 mins = np.expand_dims(mins, 1)
#                 maxes = np.expand_dims(maxes, 1)

#                 example_scaled = rescale_inverse(
#                     transformed,
#                     normalization=output.normalization,
#                     global_min=mins,
#                     global_max=maxes,
#                 )
#             else:
#                 example_scaled = transformed

#             original = output.inverse_transform(example_scaled)

#             target_shape = list(transformed_data.shape)
#             target_shape[-1] = 1
#             original = original.reshape(target_shape)

#             parts.append(original)
#             transformed_index += 1
#         else:
#             raise RuntimeError(f"Unsupported output type, class={type(output)}'")

#     return np.concatenate(parts, axis=2)

