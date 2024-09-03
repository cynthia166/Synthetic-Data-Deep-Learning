
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from config_conditonalarf import *

class MultiModelDemographicPredictor:
    def __init__(self, data, demographic_cols, medical_code_cols):
        self.data = data
        self.demographic_cols = demographic_cols
        self.medical_code_cols = medical_code_cols
        self.models = {
            'LogisticRegression': {},
            'RandomForest': {},
            'GradientBoosting': {},
            'SVC': {},
            'GaussianNB': {}
        }
        self.scalers = {}
        self.data = self.convert_to_binary(data, medical_code_cols)
    
    def prepare_data(self):
        X = self.data[self.demographic_cols]
        y = self.data[self.medical_code_cols]
        
        for code in self.medical_code_cols:
            scaler = StandardScaler()
            X_scaled_continuous = scaler.fit_transform(X[continuous_cols])
            X_scaled_continuous = pd.DataFrame(X_scaled_continuous, columns=continuous_cols, index=X.index)
            X_scaled = X.drop(columns=continuous_cols).join(X_scaled_continuous)
            self.scalers[code] = scaler
        
        return X_scaled, y
    
    def convert_to_binary(self, data, columns):
        for col in columns:
            data[col] = (data[col] > 0).astype(int)
        return data
    
    def train_models(self):
        self.list_cols = []

        X, y = self.prepare_data()
        
        for code in self.medical_code_cols:
            X_train, X_test, y_train, y_test = train_test_split(X, y[code], test_size=0.3, random_state=42)
            
            if y_train.sum() == 0:
                self.list_cols.append(code)
                self.models[code] = model
                continue
            
            model_classes = {
                'LogisticRegression': LogisticRegression(random_state=42),
                'RandomForest': RandomForestClassifier(random_state=42),
                'GradientBoosting': GradientBoostingClassifier(random_state=42),
                'SVC': SVC(probability=True, random_state=42),
                'GaussianNB': GaussianNB()
            }
            
            for model_name, model_class in model_classes.items():
                model = model_class
                
                # Perform 10-fold cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc')
                print(f"Cross-validation AUC scores for {model_name} - {code}: {cv_scores}")
                print(f"Mean CV AUC score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
                
                # Train the model on the full training set
                model.fit(X_train, y_train)
                
                self.models[model_name][code] = model
                
                # # Calculate final AUC on the test set
                # y_pred_proba = model.predict_proba(X_test)[:, 1]
                # auc = roc_auc_score(y_test, y_pred_proba)
                # print(f"Final AUC for {model_name} - {code} on test set: {auc:.4f}")
                
                # # Plot ROC curve
                # self.plot_roc_curve(y_test, y_pred_proba, f"{model_name} - {code}")
                
                # Plot confusion matrix
                y_pred = model.predict(X_test)
                self.plot_confusion_matrix(y_test, y_pred, f"{model_name} - {code}")
                
                # Print classification report
                print(f"\nClassification Report for {model_name} - {code}:")
                print(classification_report(y_test, y_pred))

    def predict_probabilities(self, new_data, model_name):
        probabilities = pd.DataFrame()
        for code, model in self.models[model_name].items():
            scaler = self.scalers[code]
            X_scaled_continuous = scaler.transform(new_data[continuous_cols])
            X_scaled_continuous = pd.DataFrame(X_scaled_continuous, columns=continuous_cols, index=new_data.index)
            new_data_scaled = new_data.drop(columns=continuous_cols).join(X_scaled_continuous)
            probabilities[code] = model.predict_proba(new_data_scaled)[:, 1]
        return probabilities

    def plot_roc_curve(self, y_true, y_pred_proba, title):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {title}')
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {title}')
        plt.show()

    def save_models(self):
        for model_name in self.models:
            joblib.dump(self.models[model_name], f'{path_result}model_{model_name}.pkl')
        joblib.dump(self.scalers, f'{path_result}scalers.pkl')
        return self.list_cols
    
    def load_models(self):
        for model_name in self.models:
            self.models[model_name] = joblib.load(f'{path_result}model_{model_name}.pkl')
        self.scalers = joblib.load(f'{path_result}scalers.pkl')

# Usage
if __name__ == "__main__":
    data = load_data(path_dataset_demos_whole)[:20000]

    if train_model:
        predictor = MultiModelDemographicPredictor(data, base_demographic_columns, medical_factors)
        predictor.train_models()
        if save_model:
           lis_to = predictor.save_models()
           aux = pd.DataFrame()
        aux["codes_no"] = lis_to
        joblib.dump(aux, f'{path_result}no_code_list.pkl')
           
 


    else:
        predictor = MultiModelDemographicPredictor(data, base_demographic_columns, medical_factors)
        predictor.load_models()

    # To get probabilities for new patients:
    sample_static = np.array([[
        65,  # Age_max
        1,   # GENDER_M
        0,   # GENDER_F
        1,   # RELIGION_CATHOLIC
        0,   # RELIGION_Otra
        0,   # RELIGION_Unknown
        0,   # MARITAL_STATUS_0
        0,   # MARITAL_STATUS_DIVORCED
        0,   # MARITAL_STATUS_LIFE PARTNER
        1,   # MARITAL_STATUS_MARRIED
        0,   # MARITAL_STATUS_SEPARATED
        0,   # MARITAL_STATUS_SINGLE
        0,   # MARITAL_STATUS_Unknown
        0,   # MARITAL_STATUS_WIDOWED
        0,   # ETHNICITY_Otra
        0,   # ETHNICITY_Unknown
        1    # ETHNICITY_WHITE
    ]])

    columns = [
        'Age_max',
        'GENDER_M',
        'GENDER_F',
        'RELIGION_CATHOLIC',
        'RELIGION_Otra',
        'RELIGION_Unknown',
        'MARITAL_STATUS_0',
        'MARITAL_STATUS_DIVORCED',
        'MARITAL_STATUS_LIFE PARTNER',
        'MARITAL_STATUS_MARRIED',
        'MARITAL_STATUS_SEPARATED',
        'MARITAL_STATUS_SINGLE',
        'MARITAL_STATUS_Unknown',
        'MARITAL_STATUS_WIDOWED',
        'ETHNICITY_Otra',
        'ETHNICITY_Unknown',
        'ETHNICITY_WHITE'
    ]

    new_patient_data = pd.DataFrame(sample_static, columns=columns)

    for model_name in predictor.models:
        probabilities = predictor.predict_probabilities(new_patient_data, model_name)
        print(f"\nProbabilities for {model_name}:")
        print(probabilities)

    # Calculate average AUC for each model
    print("\nAverage AUC for each model:")
    for model_name in predictor.models:
        auc_scores = []
        for code in predictor.medical_code_cols:
            _, X_test, _, y_test = train_test_split(X, y[code], test_size=0.2, random_state=42)
            y_pred_proba = predictor.models[model_name][code].predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            auc_scores.append(auc)
        avg_auc = np.mean(auc_scores)
        print(f"{model_name}: {avg_auc:.4f}")

# import os
# file_principal = os.getcwd()
# os.chdir(file_principal)
# import sys
# sys.path.append(file_principal)

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import (roc_auc_score, confusion_matrix, classification_report, 
#                              roc_curve, accuracy_score, f1_score, balanced_accuracy_score)
# import matplotlib.pyplot as plt
# import seaborn as sns
# import logging

# import pickle


# import numpy as np
# #import torch

# import numpy as np

# import pandas as pd
# from config_conditonalarf import *


# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble._forest import _generate_unsampled_indices
# import scipy


# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split


# from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# from config_conditonalarf import *

# class DemographicPredictor:
#     def __init__(self, data, demographic_cols, medical_code_cols,type_model):
#         self.data = data
#         self.demographic_cols = demographic_cols
#         self.medical_code_cols = medical_code_cols
#         self.models = {}
#         self.scaler = StandardScaler()
#         self.type_model = type_model
#         self.data = self.convert_to_binary(data, medical_code_cols)
#     def prepare_data(self):
#         X = self.data[self.demographic_cols]
#         y = self.data[self.medical_code_cols]
        
#         # Scale the features

#         X_scaled_continuous = self.scaler.fit_transform(X[continuous_cols])
#         # Convert the scaled array back to a DataFrame
#         X_scaled_continuous = pd.DataFrame(X_scaled_continuous, columns=continuous_cols, index=X.index)

#         # Drop the original continuous columns and concatenate with the scaled DataFrame
#         X_scaled = X.drop(columns=continuous_cols).join(X_scaled_continuous)
        
#         return X_scaled, y
    
#     def convert_to_binary(self,data, columns):
#         for col in columns:
#             data[col] = (data[col] > 0).astype(int)
#         return data
    
#     def train_models(self):
#         X, y = self.prepare_data()
        
#         for code in self.medical_code_cols:
            

               
#             X_train, X_test, y_train, y_test = train_test_split(X, y[code], test_size=0.2, random_state=42)
#             if y_train.sum() == 0:
#                 self.models[code] = model
#                 continue   
#             model = LogisticRegression(random_state=42)
            
#             # Perform 10-fold cross-validation
#             cv_scores = cross_val_score(model, X_train, y_train, cv=10)
#             print(f"Cross-validation scores for {code}: {cv_scores}")
#             print(f"Mean CV score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
            
#             # Train the model on the full training set
#             model.fit(X_train, y_train)
            
#             self.models[code] = model
            
#             # Calculate final accuracy on the test set
#             accuracy = model.score(X_test, y_test)
#             print(f"Final accuracy for {code} on test set: {accuracy:.4f}")
            
#             # Generate predictions and probabilities
#             y_pred = model.predict(X_test)
#             y_pred_proba = model.predict_proba(X_test)[:, 1]
            
#             # Plot ROC curve
#             #self.plot_roc_curve(y_test, y_pred_proba, code)
            
#             # Plot confusion matrix
#             #self.plot_confusion_matrix(y_test, y_pred, code)
            
#             # Print classification report
#             print(f"\nClassification Report for {code}:")
#             print(classification_report(y_test, y_pred))

#     def predict_probabilities(self, new_data):
#         X_scaled_continuous = self.scaler.fit_transform(new_data[continuous_cols])
#         # Convert the scaled array back to a DataFrame
#         X_scaled_continuous = pd.DataFrame(X_scaled_continuous, columns=continuous_cols, index=new_data.index)

#         # Drop the original continuous columns and concatenate with the scaled DataFrame
#         new_data_scaled = new_data.drop(columns=continuous_cols).join(X_scaled_continuous)
   
  
        
#         probabilities = pd.DataFrame()
#         for code, model in self.models.items():
#             probabilities[code] = model.predict_proba(new_data_scaled)[:, 1]
        
#         return probabilities

#     def plot_roc_curve(self, y_true, y_pred_proba, code):
#         fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
#         auc = roc_auc_score(y_true, y_pred_proba)
        
#         plt.figure(figsize=(8, 6))
#         plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
#         plt.plot([0, 1], [0, 1], linestyle='--', label='Random classifier')
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title(f'ROC Curve for {code}')
#         plt.legend()
#         plt.show()

#     def plot_confusion_matrix(self, y_true, y_pred, code):
#         cm = confusion_matrix(y_true, y_pred)
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.title(f'Confusion Matrix for {code}')
#         plt.show()

#     def save_model(self):
        
#         joblib.dump(self.models,path_result+ 'model'+self.type_model+'.pkl')
#         joblib.dump(self.scaler,path_result+ 'scaler.pkl')
                    

# # Usage
# if __name__ == "__main__":
#      # Add your medical code columns
#     if  train_model:

#         predictor = DemographicPredictor(data, demographic_cols, medical_code_cols,type_model)
#         predictor.train_models()
#         if save_model:
#             predictor.save_model()
#     else:    
#         models = joblib.load(path_result +'model'+type_model+'.pkl')
#         scaler = joblib.load(path_result +'scaler.pkl')
        


#     # To get probabilities for new patients:
#     sample_static = np.array([[
#         65,  # Age_max
#         1,   # GENDER_M
#         0,   # GENDER_F
#         1,   # RELIGION_CATHOLIC
#         0,   # RELIGION_Otra
#         0,   # RELIGION_Unknown
#         0,   # MARITAL_STATUS_0
#         0,   # MARITAL_STATUS_DIVORCED
#         0,   # MARITAL_STATUS_LIFE PARTNER
#         1,   # MARITAL_STATUS_MARRIED
#         0,   # MARITAL_STATUS_SEPARATED
#         0,   # MARITAL_STATUS_SINGLE
#         0,   # MARITAL_STATUS_Unknown
#         0,   # MARITAL_STATUS_WIDOWED
#         0,   # ETHNICITY_Otra
#         0,   # ETHNICITY_Unknown
#         1    # ETHNICITY_WHITE
#     ]])

#     # Define the column names
#     columns = [
#         'Age_max',
#         'GENDER_M',
#         'GENDER_F',
#         'RELIGION_CATHOLIC',
#         'RELIGION_Otra',
#         'RELIGION_Unknown',
#         'MARITAL_STATUS_0',
#         'MARITAL_STATUS_DIVORCED',
#         'MARITAL_STATUS_LIFE PARTNER',
#         'MARITAL_STATUS_MARRIED',
#         'MARITAL_STATUS_SEPARATED',
#         'MARITAL_STATUS_SINGLE',
#         'MARITAL_STATUS_Unknown',
#         'MARITAL_STATUS_WIDOWED',
#         'ETHNICITY_Otra',
#         'ETHNICITY_Unknown',
#         'ETHNICITY_WHITE'
#     ]

#     # Create the DataFrame
#     new_patient_data  = pd.DataFrame(sample_static, columns=columns)

#     def predict_probabilities_dun(models,scaler ,new_data):
#             X_scaled_continuous = scaler.fit_transform(new_data[continuous_cols])
#             # Convert the scaled array back to a DataFrame
#             X_scaled_continuous = pd.DataFrame(X_scaled_continuous, columns=continuous_cols, index=new_data.index)

#             # Drop the original continuous columns and concatenate with the scaled DataFrame
#             new_data_scaled = new_data.drop(columns=continuous_cols).join(X_scaled_continuous)
    
    
            
#             probabilities = pd.DataFrame()
#             for code, model in models.items():
#                 probabilities[code] = model.predict_proba(new_data_scaled)[:, 1]
            
#             return probabilities



#     probabilities = predict_probabilities_dun(models,scaler, new_patient_data)
#     print(probabilities)

