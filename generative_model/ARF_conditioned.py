import os
file_principal = os.getcwd()
os.chdir(file_principal)
import sys
sys.path.append(file_principal)

import pickle
from generative_model.gru_ap import GRUModel

import numpy as np
import torch

import numpy as np
import torch
import pandas as pd

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_model_and_scaler(model_path, scaler_path, model_class, device,input_size, hidden_size, num_classes):
    """
    Load the saved model and scaler.
    
    :param model_path: Path to the saved model file
    :param scaler_path: Path to the saved scaler file
    :param model_class: The class of the model to be loaded
    :param device: The device to load the model onto
    :return: Loaded model and scaler
    """
    # Load the model
    model = model_class(input_size, hidden_size, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def load_data(file_path):
    try:
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        with open(file_path, 'rb') as f:
             data = pickle.load(f)    
             return data

# Adjust medical code quantities
data = data= load_data("data/intermedi/SD/inpput/entire_ceros_tabular_data.pkl")[:100]
drug_columns = list(data.filter(like="drugs").columns)
diagnosis_columns = list(data.filter(like="diagnosis").columns)
procedure_columns = list(data.filter(like="procedures").columns)
medical_factors = drug_columns + diagnosis_columns + procedure_columns

save_path_arf = "D:\\Synthetic-Data-Deep-Learning\\generative_model\\"

model_path = save_path_arf + 'gru_conditional_probability_model.pth'
scaler_path = save_path_arf + 'scaler.pkl'
synthetic_data = "D:\Synthetic-Data-Deep-Learning\generated_synthcity_tabular\ARF\ARF_fixed_postpros\synthetic_ehr_datasetARF_fixed_v.pkl"
synthetic_data_df = load_data(synthetic_data)
# Assuming you have these variables defined
input_size = 100  # Size of your static input
hidden_size = 64  # Or whatever size you used
num_classes = len(medical_factors)
sequence_length = 5  # Or whatever length you used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and scaler
model, scaler = load_model_and_scaler(model_path, scaler_path, GRUModel, device)


def prepare_predictive_model_input(data, continuous_cols, categorical_cols, medical_code_cols, scaler, sequence_length, device):
    """
    Prepare input for the predictive model based on generated data, 
    with continuous variables and pre-encoded categorical variables.
    
    :param data: Generated data from the forge method
    :type data: pandas.DataFrame
    :param continuous_cols: List of continuous variable column names
    :type continuous_cols: list
    :param categorical_cols: List of pre-encoded categorical variable column names
    :type categorical_cols: list
    :param medical_code_cols: List of medical code column names
    :type medical_code_cols: list
    :param scaler: Scaler used to normalize the continuous features
    :type scaler: sklearn.preprocessing.StandardScaler
    :param sequence_length: Length of the sequence for medical history
    :type sequence_length: int
    :param device: PyTorch device (cpu or cuda)
    :type device: torch.device
    :return: Tuple of static and sequence tensors ready for the predictive model
    :rtype: tuple(torch.FloatTensor, torch.FloatTensor)
    """
    # Prepare continuous features
    continuous_features = data[continuous_cols].values
    
    # Scale continuous features
    continuous_scaled = scaler.transform(continuous_features)
    
    # Get categorical features (already one-hot encoded)
    categorical_features = data[categorical_cols].values
    
    # Combine continuous and categorical features
    static_features = np.hstack([continuous_scaled, categorical_features])
    
    # Prepare sequence data (assuming no previous medical history)
    sequence_data = np.zeros((len(data), sequence_length, len(medical_code_cols)))
    
    # Convert to PyTorch tensors
    static_tensor = torch.FloatTensor(static_features).to(device)
    sequence_tensor = torch.FloatTensor(sequence_data).to(device)
    
    return static_tensor, sequence_tensor

# Usage in the forge method:
def forge(self, n, predictive_model, continuous_cols, categorical_cols, medical_code_cols, scaler, sequence_length, device,data_new):
    # ... [previous parts of the method remain the same]
    
    # Generate initial data
    # Prepare input for the predictive model
    static_tensor, sequence_tensor = prepare_predictive_model_input(
        data_new, 
        continuous_cols=['Age_max', 'days_between_visits'],
        categorical_cols=['GENDER_M', 'GENDER_F', 'RELIGION_CATHOLIC', 'RELIGION_Otra', 'RELIGION_Unknown',
                          'MARITAL_STATUS_0', 'MARITAL_STATUS_DIVORCED', 'MARITAL_STATUS_LIFE PARTNER',
                          'MARITAL_STATUS_MARRIED', 'MARITAL_STATUS_SEPARATED', 'MARITAL_STATUS_SINGLE',
                          'MARITAL_STATUS_Unknown', 'MARITAL_STATUS_WIDOWED',
                          'ETHNICITY_Otra', 'ETHNICITY_Unknown', 'ETHNICITY_WHITE'],
        medical_code_cols=medical_code_cols,
        scaler=scaler, 
        sequence_length=sequence_length, 
        device=device
    )
    
    # Get probabilities from the predictive model
    predictive_model.eval()
    with torch.no_grad():
        probabilities = predictive_model(static_tensor, sequence_tensor).cpu().numpy()
    
    # Apply evidence
    random_numbers = np.random.random(probabilities.shape)
    occurrences = random_numbers < probabilities
    
    # Adjust medical code quantities
    
    data_new[medical_code_cols] = np.where(
    (data_new[medical_code_cols] == 0) & (occurrences == 1),
            1,
            data_new[medical_code_cols] * occurrences
        )
    
    return data_new