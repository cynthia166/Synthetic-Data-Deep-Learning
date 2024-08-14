

import numpy as np
import gzip
import pickle
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from deepecho import PARModel
import torch

import logging
import pandas as pd
from pgmpy.models import BayesianNetwork
import numpy as np
import pandas as pd

from pgmpy.estimators import BayesianEstimator, K2Score
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split

 
make_patients = False
percent_patient  = 0.8
#funcitons

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)    
# Usage

def load_data(file_path):
    try:
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        with open(file_path, 'rb') as f:
             data = pickle.load(f)    
             return data
    
def sample_patients(res,percent_patient):
    unique_patients = res['SUBJECT_ID'].unique()

    # Calcular el 20% del total de pacientes únicos
    sample_size = int(percent_patient * len(unique_patients))
    # Obtener una muestra aleatoria del 20% de los pacientes únicos
    sample_patients = np.random.choice(unique_patients, size=sample_size, replace=False)
    # Filtrar el DataFrame para incluir solo los registros de los pacientes en la muestra
    sample_df = res[res['SUBJECT_ID'].isin(sample_patients)]
    
    return sample_df, sample_patients   

def convert_to_binary(data, columns):
    for col in columns:
        data[col] = (data[col] > 0).astype(int)
    return data

def generate_bayesian_network_structure(data,drug_columns,diagnosis_columns,procedure_columns):
    # Identify column types
    age_column = 'Age_max'
 
    days_from_last_visit_column ='days_between_visits'
    gender_columns = list(data.filter(like="GENDER").columns)
    
    
    # Generate edges
    edges = []
    predictors = [age_column, days_from_last_visit_column] + gender_columns
    medical_factors = drug_columns + diagnosis_columns + procedure_columns

    for predictor in predictors:
        for factor in medical_factors:
            edges.append((predictor, factor))

    for i, factor1 in enumerate(medical_factors):
        for factor2 in medical_factors[i+1:]:
            edges.append((factor1, factor2))

    return BayesianNetwork(edges)


def predict_medical_factors(model, evidence):
    inference = VariableElimination(model)
    medical_factors = [node for node in model.nodes() if node.startswith(('drugs', 'diagnosis', 'procedures'))]
    
    predictions = {}
    for factor in medical_factors:
        pred = inference.query([factor], evidence=evidence)
        predictions[factor] = pred.values[1]  # Probability of occurrence
    return predictions

def calculate_bayesian_score(model, data):
    scorer = K2Score(data)
    return scorer.score(model)

def calculate_negative_log_likelihood(model, data):
    log_likelihood = 0
    for _, row in data.iterrows():
        evidence = row.to_dict()
        for node in model.nodes():
            cpd = model.get_cpds(node)
            node_value = evidence[node]
            parents = model.get_parents(node)
            if parents:
                parent_values = [evidence[parent] for parent in parents]
                prob = cpd.get_value(**{node: node_value, **dict(zip(parents, parent_values))})
            else:
                prob = cpd.get_value(node_value)
            log_likelihood += np.log(prob)
    return -log_likelihood

def evaluate_model(model, test_data):
    bayesian_score = calculate_bayesian_score(model, test_data)
    neg_log_likelihood = calculate_negative_log_likelihood(model, test_data)
    return bayesian_score, neg_log_likelihood

# Main execution
ruta_modelo = "generated_synthcity_tabular/ARF/Bayes_prob/"
data= load_data("data/intermedi/SD/inpput/entire_ceros_tabular_data.pkl")[:1000]
columns_to_drop = ['GENDER_0','ADMITTIME']
#columns_to_drop = ['LOSRD_sum', 'L_1s_last_p1','HADM_ID',"ADMITTIME",'GENDER_0','days_between_visits']
print("Cols to eliminate",columns_to_drop)
data = data.drop(columns=columns_to_drop) 

drug_columns = list(data.filter(like="drugs").columns)
diagnosis_columns = list(data.filter(like="diagnosis").columns)
procedure_columns = list(data.filter(like="procedures").columns)


logging.info("Generating bayesian network")

model = generate_bayesian_network_structure(data,drug_columns,diagnosis_columns,procedure_columns)

# Split data into train and test sets

logging.info("Creating patients dataset!")
if make_patients:
    train_data, sample_patients = sample_patients(data,percent_patient)
    test_data = data[~data['SUBJECT_ID'].isin(sample_patients)]
    train_data = data[data['SUBJECT_ID'].isin(sample_patients)]
else:
    sample_patients= load_data("generated_synthcity_tabular/ARF/ARF_fixed_sansvar/sample_patients_fixed_v.pkl")
    test_data = data[~data['SUBJECT_ID'].isin(sample_patients)]
    train_data = data[data['SUBJECT_ID'].isin(sample_patients)]

# Fit the model on training data
count_columns = drug_columns+diagnosis_columns+procedure_columns
train_data = convert_to_binary(train_data, count_columns)
model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")


# Evaluate the model on test data
bayesian_score, neg_log_likelihood = evaluate_model(model, test_data)

print(f"Bayesian Score: {bayesian_score}")
print(f"Negative Log-Likelihood: {neg_log_likelihood}")

# Example prediction
sample_evidence = {
    'Age_max': 65,
    
   'days_between_visits': 30,
    'GENDER_M': 1,
    'GENDER_F': 0
}
predictions = predict_medical_factors(model, sample_evidence)

print("\nSample Predictions:")
for factor, prob in predictions.items():
    print(f"{factor}: Probability of occurrence = {prob:.2f}")


save_pickle(model, ruta_modelo+'bayesian_network_model.pkl')

# # Load the model and perform inference
# loaded_model = load_data('bayesian_network_model.pkl')

# # Example prediction using the loaded model
# sample_evidence = {
#     'Age_max': 65,
#     'LOSRD_avg': 5,
#     'days from last visit': 30,
#     'GENDER_M': 1,
#     'GENDER_F': 0
# }
# predictions = inference_from_loaded_model(loaded_model, sample_evidence)

# print("\nSample Predictions from Loaded Model:")
# for factor, prob in predictions.items():
#     print(f"{factor}: Probability of occurrence = {prob:.2f}")


# Split the data into train, validation, and test sets
    

