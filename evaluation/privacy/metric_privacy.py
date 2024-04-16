
import pandas as pd 
import sys
import os
import os
import pickle
import gzip
print(os.getcwd())
sys.path.append('../../')
import os
sys.path.append('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing')
sys.path.append('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/evaluation/evaluation/utility')
sys.path.append('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/evaluation/evaluation/functions')
sys.path.append('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/evaluation')

from evaluation.functions import *
# stdlib
import platform
from abc import abstractmethod
from collections import Counter
from typing import Any, Dict, Tuple, Union

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from scipy import stats
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

sys.path.append('../preprocessing')
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from typing import Any, Dict
from sklearn.decomposition import PCA
from config import *
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.append('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/')
#os.chdir('../')
print(os.getcwd())
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig

print(os.getcwd())

# y 'b' es tu array de forma (100, 2)
# y 'b' es tu array de forma (100, 2)

def concat_attributes(a,b):

    # Redimensiona 'b' para que tenga la misma forma que 'a' en los dos primeros ejes
    #b is attributes
    #a = np.random.rand(100, 244, 5)
    #b = np.random.rand(100, 2)

    b = np.repeat(b[:, :, np.newaxis], a.shape[2], axis=2)
    print(b.shape)
    print(a.shape)
    # Ahora 'b' es un array de forma (100, 244, 2)
    print(b.shape)  # Impri   me: (100, 244, 2)

    # Concatena 'a' y 'b' a lo largo del tercer eje
    c = np.concatenate((a, b), axis=1)

# Ahora 'c' es un array de forma (100, 244, 7)
    print(c.shape) 
    return c    


class IdentifiabilityScore:
    """
    A class to compute the identifiability score between real and synthetic datasets.
    The score is based on the concept of Wasserstein distance and is aimed to evaluate
    the re-identification risk of real data points using synthetic data.

    Attributes:
    -----------
    No explicit attributes defined outside of methods.

    Methods:
    --------
    compute_score(X_gt: np.ndarray, X_syn: np.ndarray) -> float:
        Computes the identifiability score between the real dataset `X_gt` and the synthetic dataset `X_syn`.
    """



    @staticmethod
    def compute_score(X_gt: np.ndarray, X_syn: np.ndarray) -> float:
        """Compute the identifiability score based on nearest neighbor distances."""
        # Flatten the data if it's not already 1D.
        X_gt_ = X_gt.reshape(X_gt.shape[0], -1)
        X_syn_ = X_syn.reshape(X_syn.shape[0], -1)

        # Compute weights based on entropy; here simplified to uniform weights
        x_dim = X_gt_.shape[1]
        W = np.ones(x_dim)  # Simplified; ideally, use entropy for weights

        # Normalize data (assuming the simplification of weights to 1)
        eps = 1e-16
        X_hat = X_gt_ / (W + eps)
        X_syn_hat = X_syn_ / (W + eps)

        # Compute distances in original space
        nbrs = NearestNeighbors(n_neighbors=2).fit(X_hat)
        distance, _ = nbrs.kneighbors(X_hat)

        # Compute distances in synthetic space
        nbrs_syn = NearestNeighbors(n_neighbors=1).fit(X_syn_hat)
        distance_syn, _ = nbrs_syn.kneighbors(X_hat)

        # Calculate identifiability score
        R_diff = distance_syn[:, 0] - distance[:, 1]
        identifiability_value = np.mean(R_diff < 0)

        return identifiability_value



class DeltaPresence:
    """
    Clase para calcular la probabilidad máxima de re-identificación en el conjunto
    de datos real a partir del conjunto de datos sintético, utilizando arrays de numpy.
    """
    

   
    def evaluate(self, X_gt: np.ndarray, X_syn: np.ndarray) -> Dict:
        # Verificar que ambos arrays tienen la misma segunda dimensión (número de características)
        if X_gt.shape[1] != X_syn.shape[1]:
            raise ValueError("Real and synthetic datasets must have the same number of features.")

        values = []
        for n_clusters in [2, 5, 10, 15]:
            if len(X_gt) / n_clusters < 10:
                continue
            model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0).fit(X_gt)
            clusters = model.predict(X_syn)
            synth_counts = Counter(clusters)
            gt_counts = Counter(model.labels_)

            for key in gt_counts:
                if key not in synth_counts:
                    continue
                gt_cnt = gt_counts[key]
                synth_cnt = synth_counts[key]

                delta = gt_cnt / (synth_cnt + 1e-8)  # Asegurar no división por cero
                values.append(delta)

        max_value = float(np.max(values)) if values else 0.0  # Asegurar que haya valores antes de tomar el máximo

        return {"score": max_value}





class DetectionEvaluator:
    """
    Base class for training classifiers to detect synthetic data from real data.

    Synthetic and real data are combined to form a new dataset.
    K-fold cross validation is performed to see how well a classifier can distinguish real from synthetic.
    """

    def __init__(self, n_folds=5, random_state=None):
        self.n_folds = n_folds
        self.random_state = random_state

    def evaluate(self, X_gt: np.ndarray, X_syn: np.ndarray, **model_args) -> float:
        """
        Evaluates the detection model.

        Parameters:
        - X_gt: Ground truth (real) data as a numpy array.
        - X_syn: Synthetic data as a numpy array.
        - model_args: Arguments specific to the model template used in subclass.

        Returns:
        - The average AUCROC score for detecting synthetic data.
        """
        raise NotImplementedError("Subclasses should implement this!")

    def _evaluate_detection_generic(self, model_template, X_gt: np.ndarray, X_syn: np.ndarray, **model_args) -> float:
        """
        Generic method to evaluate detection using a provided model template.
        """
        # Concatenate real and synthetic data
        data = np.concatenate([X_gt, X_syn])
        labels = np.concatenate([np.zeros(len(X_gt)), np.ones(len(X_syn))])

        res = []
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for train_idx, test_idx in skf.split(data, labels):
            train_data, train_labels = data[train_idx], labels[train_idx]
            test_data, test_labels = data[test_idx], labels[test_idx]

            model = model_template(**model_args).fit(train_data, train_labels)
            test_pred = model.predict_proba(test_data)[:, 1]

            res.append(roc_auc_score(test_labels, test_pred))

        return float(np.mean(res))

class SyntheticDetectionXGB(DetectionEvaluator):
    """
    Train a XGBoost classifier to detect the synthetic data.
    """

    def __init__(self, n_folds=5, random_state=None):
        super().__init__(n_folds, random_state)

    def evaluate(self, X_gt: np.ndarray, X_syn: np.ndarray) -> float:
        """
        Evaluates the XGBoost model for detection.

        Parameters:
        - X_gt: Ground truth (real) data as a numpy array.
        - X_syn: Synthetic data as a numpy array.

        Returns:
        - The average AUCROC score for detecting synthetic data.
        """
        model_args = {
            "n_jobs": 1,  # Adjust according to your resources
            "verbosity": 0,
            "use_label_encoder": False,  # For newer versions of XGB
            "eval_metric": "logloss",  # To silence warning in newer versions
            "random_state": self.random_state,
        }
        return self._evaluate_detection_generic(XGBClassifier, X_gt, X_syn, **model_args)

# Example usage
# Assuming X_gt and X_syn are your numpy arrays for real and synthetic data respectively:
def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

#codigo para attribute attack    
path_o = "train_sp/"    
attributes_path_train= "non_prepo/DATASET_NAME_non_preprotrain_data_attributes.pkl"
features_path_train = "non_prepo/DATASET_NAME_non_preprotrain_data_features.pkl"
features_path_valid = "non_prepo/DATASET_NAME_non_preprovalid_data_features.pkl"
attributes_path_valid = "non_prepo/DATASET_NAME_non_preprovalid_data_attributes.pkl"
synthetic_path_attributes = 'non_prepo/DATASET_NAME_non_prepronon_prepo_synthetic_attributes_10.pkl'
synthetic_path_features = 'non_prepo/DATASET_NAME_non_prepronon_prepo_synthetic_features_10.pkl'

# se lee los archivos y se obtiene del la longitude de valid
total_features_synthethic, total_fetura_valid,total_features_train,attributes =  get_valid_train_synthetic (path_o, attributes_path_train, features_path_train, features_path_valid, attributes_path_valid,synthetic_path_attributes,synthetic_path_features)
# se transforma de numpy array 3 dimensiones a dataframe
total_features_synthethic,total_fetura_valid,total_features_train = obtener_dataframe_inicial_denumpyarrray(total_features_synthethic, total_fetura_valid,total_features_train )

# esta es para agregar la columnas
dataset_name = 'DATASET_NAME_non_prepo'
file_name = "train_sp/non_prepo/DATASET_NAME_non_prepo_non_preprocess.pkl"
aux = load_data(file_name)
#aux = load_data(DARTA_INTERM_intput + dataset_name + '_non_preprocess.pkl')
con_cols = list(aux[0].columns)
static = pd.read_csv("train_sp/non_prepo/static_data_non_preprocess.csv")
# Suponiendo que 'total_features_synthethic' es tu DataFrame
if 'Unnamed' in static.columns:
    static = static.drop(columns=['Unnamed'])
cat = list(static.columns[2:]) +["visit_rank","id_patient"  ]
del aux
total_cols =  con_cols+cat 
cat1 = list(static.columns[2:]) +["visit_rank","id_patient","max_consultas"     ]
total_cols1 =  con_cols+cat1 

    
test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset = preprocess_data(total_cols,total_features_synthethic,total_cols1,total_fetura_valid,total_features_train)


import pandas as pd

# Assuming your DataFrame is correctly set up and you've identified the relevant code columns

keywords = ['diagnosis', 'procedures', 'drugs']


# Obtener los nombres de las columnas que contienen alguna de las palabras
columnas_test_ehr_dataset = [col for col in train_ehr_dataset.columns if any(palabra in col for palabra in keywords)]

# Filtrar las columnas que contienen las palabras clave

code_sums = train_ehr_dataset[columnas_test_ehr_dataset].sum(axis=0).sort_values(ascending=False)

# Step 2: Identify the top 300 most common codes
top_300_codes = code_sums.head(100).index.tolist()



keywords = ['diagnosis', 'procedures', 'drugs']
code_sums = train_ehr_dataset[columnas_test_ehr_dataset].sum(axis=0).sort_values(ascending=False)
syn_codes = code_sums.head(100).index.tolist()


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
scaler.fit(synthetic_ehr_dataset[columnas_test_ehr_dataset])
df_normalized = scaler.transform( synthetic_ehr_dataset[columnas_test_ehr_dataset])
df_inversed = scaler.inverse_transform(df_normalized)
synthetic_ehr = pd.DataFrame(df_inversed, columns=synthetic_ehr_dataset[columnas_test_ehr_dataset].columns)

    
percentil_10 = synthetic_ehr.quantile(0.05)

# Establecer los valores en cada columna a 1 o 0 dependiendo de si son mayores que el promedio
synthetic_ehr = (synthetic_ehr > percentil_10).astype(int)

import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

# Assume 'df' is your pandas DataFrame with 'id_patient', 'id_visit', and ICD-9 code columns

import pandas as pd

# Assuming 'df' is your pandas DataFrame
# Let's say df looks like this:
#    code1  code2  code3  code4  code5
# 0      1      0      1      0      0
# 1      0      1      0      1      1
# ...

# Initialize an empty list to hold our dataset
def obtetenr_dataset(train_ehr_dataset,top_300_codes,columnas_test_ehr_dataset):
    test_ehr_dataset = []


    # Iterate over each row in the DataFrame
    for index, row in train_ehr_dataset[columnas_test_ehr_dataset].iterrows():
        # Get a set of column names where the value is 1
        codes = set(row.index[row > 0].tolist())
        # Append the dictionary to our list
        test_ehr_dataset.append({'codes': codes})

    # Now test_ehr_dataset is a list of dictionaries in the format you described

    for visit in test_ehr_dataset:
        visit['labels'] = {code for code in visit['codes'] if code in top_300_codes}
        visit['codes'] = {code for code in visit['codes'] if code not in top_300_codes}

    # After running this, each dictionary will now have 'labels' and 'codes' appropriately divided
    print(test_ehr_dataset)
    return test_ehr_dataset

num_filas = {
        "synthethic": synthetic_ehr_dataset.shape[0],
        "valid": test_ehr_dataset.shape[0],
        "train": train_ehr_dataset.shape[0]
    }

    # Encontrar el nombre del DataFrame con el mínimo número de filas
min_filas_df = min(num_filas, key=num_filas.get)

print(min_filas_df)
synthetic_ehr_dataset = synthetic_ehr[:num_filas[min_filas_df]]
test_ehr_dataset = test_ehr_dataset[:num_filas[min_filas_df]]
train_ehr_dataset = train_ehr_dataset[:num_filas[min_filas_df]]

train_ehr_dataset = obtetenr_dataset(train_ehr_dataset,top_300_codes,columnas_test_ehr_dataset)
test_ehr_dataset = obtetenr_dataset(test_ehr_dataset,top_300_codes,columnas_test_ehr_dataset)
synthetic_ehr_dataset = obtetenr_dataset(synthetic_ehr_dataset,top_300_codes,columnas_test_ehr_dataset)




def calc_dist(lab1, lab2):
    return len(lab1.union(lab2)) - len(lab1.intersection(lab2))

def find_closest(patient, data, k):
    cond = patient['labels']
    dists = [(calc_dist(cond, ehr['labels']), ehr['codes']) for ehr in data]
    dists.sort(key= lambda x: x[0], reverse=False)
    dists = [tupla for tupla in dists if tupla[1]]
    options = [o[1] for o in dists[:k]]
    return options

def calc_attribute_risk(train_dataset, reference_dataset, k):
    tp = 0
    fp = 0
    fn = 0
    for p in tqdm(train_dataset):
        closest_k = find_closest(p, reference_dataset, k)
        pred_codes = set([cd for cd, cnt in Counter([c for p in closest_k for c in p]).items() if cnt > k/2])
        true_pos = len(pred_codes.intersection(p['codes']))
        false_pos = len(pred_codes) - true_pos 
        false_neg = len(p['codes']) - true_pos
        tp += true_pos
        fp += false_pos
        fn += false_neg
    try:    
        f1 = tp / (tp + (0.5 * (fp + fn)))
    except:
        print("Error")
        f1 = 0
   
       
    return f1

K = 1

# Filtrar los elementos donde tanto 'codes' como 'labels' son conjuntos vacíos

train = [item for item in train_ehr_dataset if item['codes']]
test = [item for item in test_ehr_dataset if item['codes']]
syn = [item for item in synthetic_ehr_dataset if item['codes']]

train_ehr_dataset = [element for element in train_ehr_dataset if element['codes'] or element['labels']]
test_ehr_dataset = [element for element in test_ehr_dataset if element['codes'] or element['labels']]
synthetic_ehr_dataset = [element for element in synthetic_ehr_dataset if element['codes'] or element['labels']]

att_risk = calc_attribute_risk(train_ehr_dataset, synthetic_ehr_dataset, K)
att_risk = calc_attribute_risk(train, syn, K)
baseline_risk = calc_attribute_risk(train_ehr_dataset, test_ehr_dataset, K)
baseline_risk = calc_attribute_risk(train, test, 1)
results = {
    "Attribute Attack F1 Score": att_risk,
    "Baseline Attack F1 Score": baseline_risk
}