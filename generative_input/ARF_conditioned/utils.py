import pandas as pd
import numpy as np
import gzip
import pickle
import random


def group_and_encode_demographics(df, demographic_categories,encoder_dict):
    """
    Group columns by demographic categories and apply label encoding.
    
    :param df: pandas DataFrame containing the data
    :param demographic_categories: list of demographic category names (e.g., ['gender', 'age_group', 'ethnicity'])
    :return: tuple containing the modified DataFrame and a dictionary of LabelEncoders
    """
    encoders = {}

    for category in demographic_categories:
        # Find all columns that contain the category name
        encoder = encoder_dict[category]
        category_cols = [col for col in df.columns if category in col]
        
        if category_cols:
            # Group the columns and create a new column with the category name
            df[category] = df[category_cols].idxmax(axis=1).str.replace(category + '_', '')
            
            # Drop the original columns
            df = df.drop(columns=category_cols)
            
            # Apply label encoding
            
            df[category + '_encoded'] = encoder.fit_transform(df[category])
            
            # Store the encoder for later use
            encoders[category] = encoder
            df = df.drop(columns=category)
    return df, encoders


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def convertir_categoricas(df,categorical_cols):
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    return df  


def sample_patients_list(ruta_patients, x_train):
    sample_patients = load_pickle(ruta_patients)
    sample_df = x_train[x_train['SUBJECT_ID'].isin(sample_patients)]
    return sample_df

def change_dtypes(x_train,cols_continuous_d):
   

    categorical_cols_d =[col for col in x_train.columns if col not in cols_continuous_d]
    #change to categorilcal 
    train_data_features = convertir_categoricas(x_train,categorical_cols_d)
    print(train_data_features.dtypes)
            # samplear 3# fre los pacientes
    return train_data_features


def load_data(file_path):
    try:
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        with open(file_path, 'rb') as f:
             data = pickle.load(f)    
             return data



def save_plot_as_svg(plt, path_img, filename):
    random_number = random.randint(1000, 9999)  # Genera un número aleatorio entre 1000 y 9999
    plt.savefig(f"{path_img}{filename}_{random_number}.svg", format='svg')