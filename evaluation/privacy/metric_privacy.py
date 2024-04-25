
import pandas as pd 
import random
import sys
import os

import pickle
import gzip
print(os.getcwd())
sys.path.append('../../')
import os
#sys.path.append('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing')
#sys.path.append('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/evaluation/evaluation/utility')
#sys.path.append('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/evaluation/evaluation/functions')
#sys.path.append('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/evaluation')
from sklearn import metrics
from evaluation.functions import *
# stdlib
import platform
from abc import abstractmethod
from collections import Counter
from typing import Any, Dict, Tuple, Union
from sklearn.preprocessing import MinMaxScaler  
# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from scipy import stats
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from tqdm import tqdm

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

#sys.path.append('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/')
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
    
def get_cols_diag_proc_drug(train_ehr_dataset):
    keywords = ['diagnosis', 'procedures', 'drugs']
    columnas_test_ehr_dataset = [col for col in train_ehr_dataset.columns if any(palabra in col for palabra in keywords)]
    return columnas_test_ehr_dataset
    
def obtain_most_freuent(train_ehr_dataset,columnas_test_ehr_dataset,num):
    code_sums = train_ehr_dataset[columnas_test_ehr_dataset].sum(axis=0).sort_values(ascending=False)
    top_300_codes = code_sums.head(num).index.tolist()
    return top_300_codes

def change_tosyn_stickers_temporal(synthetic_ehr_dataset,columnas_test_ehr_dataset,id_patient):
    scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
    scaler.fit(synthetic_ehr_dataset[columnas_test_ehr_dataset])
    df_normalized = scaler.transform( synthetic_ehr_dataset[columnas_test_ehr_dataset])
    df_inversed = scaler.inverse_transform(df_normalized)
    synthetic_ehr = pd.DataFrame(df_inversed, columns=synthetic_ehr_dataset[columnas_test_ehr_dataset].columns)

        
    percentil_10 = synthetic_ehr.quantile(0.5)

    # Establecer los valores en cada columna a 1 o 0 dependiendo de si son mayores que el promedio
    synthetic_ehr = (synthetic_ehr > percentil_10).astype(int)
    if id_patient:
        # Copiar la columna 'id_patient' a synthetic_ehr
        synthetic_ehr['id_patient'] = synthetic_ehr_dataset['id_patient'].values


    return synthetic_ehr

##################################################ESTAS FUNCIONES TIENE QUE VER CON OBTENER ATTRIBUTE ATTACK#############################
##################################################ESTAS FUNCIONES TIENE QUE VER CON OBTENER ATTRIBUTE ATTACK#############################
##################################################ESTAS FUNCIONES TIENE QUE VER CON OBTENER ATTRIBUTE ATTACK#############################
##################################################ESTAS FUNCIONES TIENE QUE VER CON OBTENER ATTRIBUTE ATTACK#############################
##################################################ESTAS FUNCIONES TIENE QUE VER CON OBTENER ATTRIBUTE ATTACK#############################
def obtetenr_dataset(train_ehr_dataset,top_300_codes,columnas_test_ehr_dataset):
    '''Funcion que itera por cada visita u se agrega, labels si estan en los ma comunes y codes si no estan en los mas comunes'''
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

def attribute_attack(synthetic_ehr_dataset,test_ehr_dataset,train_ehr_dataset,K,synthetic_ehr,top_300_codes,columnas_test_ehr_dataset):
    ''''funcion que realiza el attack attribu HALO_Synthetic'''
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



    # Filtrar los elementos donde tanto 'codes' como 'labels' son conjuntos vacíos

    train = [item for item in train_ehr_dataset if item['codes']]
    test = [item for item in test_ehr_dataset if item['codes']]
    syn = [item for item in synthetic_ehr_dataset if item['codes']]

    train_ehr_dataset = [element for element in train_ehr_dataset if element['codes'] or element['labels']]
    test_ehr_dataset = [element for element in test_ehr_dataset if element['codes'] or element['labels']]
    synthetic_ehr_dataset = [element for element in synthetic_ehr_dataset if element['codes'] or element['labels']]

    att_risk = calc_attribute_risk(train_ehr_dataset, synthetic_ehr_dataset, K)
    att_risk_u = calc_attribute_risk(train, syn, K)
    baseline_risk = calc_attribute_risk(train_ehr_dataset, test_ehr_dataset, K)
    baseline_risk_u = calc_attribute_risk(train, test, 1)
    results_unitu = {
        "Attribute Attack F1 Score": att_risk_u,
        "Baseline Attack F1 Score": baseline_risk_u
    }
    results = {
        "Attribute Attack F1 Score": att_risk,
        "Baseline Attack F1 Score": baseline_risk
    }

    return results
# funciones para membershio_inference
def find_hamming_visits_only(ehr, dataset):
    min_d = 1e10  # Initialize minimum distance with a large number
    visits = ehr['visits']
    
    for p in dataset:
        d = 0  # Start with a distance of 0 for each record in the dataset
        # Compare the number of visits; if not equal, increment distance
        d += abs(len(visits) - len(p['visits']))
        
        # Calculate the set difference for each corresponding visit
        max_index = min(len(visits), len(p['visits']))  # Find the minimum index to avoid out-of-range errors
        for i in range(max_index):
            v = visits[i]  # Visit from the ehr
            v2 = p['visits'][i]  # Corresponding visit from the current record in the dataset
            d += len(v.union(v2)) - len(v.intersection(v2))  # Symmetric difference size
        
        # If ehr has more visits than p, add the number of codes in the extra visits
        if len(visits) > len(p['visits']):
            for i in range(max_index, len(visits)):
                d += len(visits[i])
        
        # If p has more visits than ehr, add the number of codes in the extra visits
        if len(p['visits']) > len(visits):
            for i in range(max_index, len(p['visits'])):
                d += len(p['visits'][i])
        
        # Update the minimum distance if the new calculated distance is smaller
        min_d = min(min_d, d)

    return min_d


def obtain_dataset(train_ehr_dataset,columnas_test_ehr_dataset):
    df = train_ehr_dataset[columnas_test_ehr_dataset+["id_patient"]]


      # Agrupar las filas por 'patient_id' y 'visit_id', y convertir cada grupo en un conjunto de códigos
    df_grouped = df.groupby(['id_patient']).apply(lambda x: set(x.columns[(x == 1).any()]))

    # Agrupar las filas por 'patient_id' y convertir cada grupo en una lista de visitas
 
    df_grouped = df_grouped.groupby('id_patient').apply(list)

    # Convertir el resultado en un diccionario
    dataset = [{'visits': visits} for visits in df_grouped]
    dataset = [item for item in dataset if item['visits']]
    #new_dataset_train = {'visits' : value for key, value in dataset_train.items()}
    return dataset

def membership_attack(train_ehr_dataset, test_ehr_dataset,synthetic_ehr,columnas_test_ehr_dataset,NUM_TEST_EXAMPLES,NUM_TOT_EXAMPLES,NUM_VAL_EXAMPLES):    
    import random
    from sklearn import metrics
    dataset_train = obtain_dataset(train_ehr_dataset,columnas_test_ehr_dataset)
    dataset_test = obtain_dataset(test_ehr_dataset,columnas_test_ehr_dataset)

    dataset_syn = obtain_dataset(synthetic_ehr,columnas_test_ehr_dataset)


    train_ehr_dataset = [(p, 1) for p in dataset_train]
    test_ehr_dataset = [(p,0) for p in dataset_test]
    synthetic_ehr_dataset = [p for p in dataset_syn if len(p['visits']) > 0]



    attack_dataset_pos = list(random.sample(train_ehr_dataset, NUM_TOT_EXAMPLES))
    attack_dataset_neg = list(random.sample(test_ehr_dataset, NUM_TOT_EXAMPLES))
    np.random.shuffle(attack_dataset_pos)
    np.random.shuffle(attack_dataset_neg)
    test_attack_dataset = attack_dataset_pos[:NUM_TEST_EXAMPLES] + attack_dataset_neg[:NUM_TEST_EXAMPLES]
    val_attack_dataset = attack_dataset_pos[NUM_TEST_EXAMPLES:NUM_TEST_EXAMPLES+NUM_VAL_EXAMPLES] + attack_dataset_neg[NUM_TEST_EXAMPLES:NUM_TEST_EXAMPLES+NUM_VAL_EXAMPLES]
    np.random.shuffle(test_attack_dataset)
    np.random.shuffle(val_attack_dataset)
    attack_dataset_pos = attack_dataset_pos[NUM_TEST_EXAMPLES+NUM_VAL_EXAMPLES:]
    attack_dataset_neg = attack_dataset_neg[NUM_TEST_EXAMPLES+NUM_VAL_EXAMPLES:]



    ds = [(find_hamming_visits_only(ehr, synthetic_ehr_dataset), l) for (ehr, l) in tqdm(test_attack_dataset)]
    median_dist = np.median([d for (d,l) in ds])
    preds = [1 if d < median_dist else 0 for (d,l) in ds]
    labels = [l for (d,l) in ds]
    results = {
        "Accuracy membership attack": metrics.accuracy_score(labels, preds),
        "Precision membership attack": metrics.precision_score(labels, preds),
        "Recall membership attack": metrics.recall_score(labels, preds),
        "F1 membership attack": metrics.f1_score(labels, preds)
    }
    return results

def find_hamming_visits_nn(ehr, dataset):
    min_d = 1e10  # Initialize minimum distance with a very large number
    visits = ehr['visits']
    
    for p in dataset:
        d = 0  # Start with a distance of 0 for each record in the dataset
        # Increment distance if the number of visits does not match
        if len(visits) != len(p['visits']):
            d += abs(len(visits) - len(p['visits']))
        
        # Calculate the set difference for each corresponding visit
        max_index = min(len(visits), len(p['visits']))  # Find the minimum index to avoid out-of-range errors
        for i in range(max_index):
            v = visits[i]  # Visit from the ehr
            v2 = p['visits'][i]  # Corresponding visit from the current record in the dataset
            # Calculate symmetric difference size
            d += len(v.union(v2)) - len(v.intersection(v2))
        
        # Handle extra visits in ehr that are not in p
        if len(visits) > len(p['visits']):
            for i in range(max_index, len(visits)):
                d += len(visits[i])  # Add the number of codes in the extra visits
        
        # Handle extra visits in p that are not in ehr
        if len(p['visits']) > len(visits):
            for i in range(max_index, len(p['visits'])):
                d += len(p['visits'][i])  # Add the number of codes in the extra visits
        
        # Update the minimum distance if the new calculated distance is smaller and not zero
        if d < min_d and d > 0:
            min_d = d

    return min_d


def calc_nnaar(train, evaluation, synthetic,NUM_SAMPLES):
    val1 = 0
    val2 = 0
    val3 = 0
    val4 = 0
    for p in tqdm(evaluation):
        des = find_hamming_visits_nn(p, synthetic)
        dee = find_hamming_visits_nn(p, evaluation)
        if des > dee:
            val1 += 1
    
    for p in tqdm(train):
        dts = find_hamming_visits_nn(p, synthetic)
        dtt = find_hamming_visits_nn(p, train)
        if dts > dtt:
            val3 += 1

    for p in tqdm(synthetic):
        dse = find_hamming_visits_nn(p, evaluation)
        dst = find_hamming_visits_nn(p, train)
        dss = find_hamming_visits_nn(p, synthetic)
        if dse > dss:
            val2 += 1
        if dst > dss:
            val4 += 1

    val1 = val1 / NUM_SAMPLES
    val2 = val2 / NUM_SAMPLES
    val3 = val3 / NUM_SAMPLES
    val4 = val4 / NUM_SAMPLES

    aaes = (0.5 * val1) + (0.5 * val2)
    aaet = (0.5 * val3) + (0.5 * val4)
    return aaes - aaet

def evaluate_attacks(list_metric, test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset,synthetic_ehr, top_300_codes, columnas_test_ehr_dataset):
    list_res = []
    result_final = {}



    if "attributes_attack" in list_metric:
        K = 1

        result = attribute_attack(synthetic_ehr,test_ehr_dataset,train_ehr_dataset,K,synthetic_ehr,top_300_codes,columnas_test_ehr_dataset)
        list_res.append(result)

    #################################################M
    #################################################M
    #################################################MEMBERShip attributes_
    #################################################M
    #################################################M
    #################################################M




    if "memebership_attack" in list_metric:
        import random
        NUM_TEST_EXAMPLES = 40
        NUM_TOT_EXAMPLES = 40
        NUM_VAL_EXAMPLES = 20
        synthetic_ehr_ = change_tosyn_stickers_temporal(synthetic_ehr_dataset,columnas_test_ehr_dataset,True)
    
        #synthetic_ehr_ = change_tosyn_stickers_temporal(synthetic_ehr,columnas_test_ehr_dataset,True)
      
        results = membership_attack(train_ehr_dataset, test_ehr_dataset,synthetic_ehr_,columnas_test_ehr_dataset,NUM_TEST_EXAMPLES,NUM_TOT_EXAMPLES,NUM_VAL_EXAMPLES)
        list_res.append(results)

    #################################################M
    #################################################M
    #################################################MNEIGHTBOUr DISTANCE
    #################################################M
    #################################################M
    #################################################M
    # Example usage



    if "nn_distance_attack" in list_metric:


        dataset_train = obtain_dataset(train_ehr_dataset,columnas_test_ehr_dataset)
        dataset_test = obtain_dataset(test_ehr_dataset,columnas_test_ehr_dataset)

        synthetic_ehr = change_tosyn_stickers_temporal(synthetic_ehr_dataset,columnas_test_ehr_dataset,True)
        dataset_syn = obtain_dataset(synthetic_ehr,columnas_test_ehr_dataset)


        NUM_SAMPLES = min(len(dataset_train), len(dataset_test), len(synthetic_ehr))
        dataset_train = np.random.choice(dataset_train, NUM_SAMPLES)
        dataset_test = np.random.choice(dataset_test, NUM_SAMPLES)
        synthetic_ehr = np.random.choice([p for p in dataset_syn if len(p['visits']) > 0], NUM_SAMPLES)


        nnaar = calc_nnaar(dataset_train, dataset_test, synthetic_ehr,NUM_SAMPLES)
        results = {
            "nn_distance_attack": nnaar
        }
        list_res.append(results)

    if "delta" in list_metric:
        train_test= " test"
        delta = DeltaPresence()
        delta_s = delta.evaluate(test_ehr_dataset.values,synthetic_ehr_dataset.iloc[:,1:].values)
        print("Delta Presence:", delta_s)
        results["Delta Presence "+train_test] = delta_s
        list_res.append(results)
    
    for i in list_res:
        result_final.update(i)   
    return  result_final 

#codigo para attribute attack    
if __name__ == "__main__":
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

    #tengo eliminacion de una colman se va tener que modificar mas adelante 
    test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset = preprocess_data(total_cols,total_features_synthethic,total_cols1,total_fetura_valid,total_features_train)


    #obtener coluans que contenga diagnosis,procedures,drugs
    columnas_test_ehr_dataset = get_cols_diag_proc_drug(train_ehr_dataset)

    #obtener n mas frequent codes
    top_300_codes = obtain_most_freuent(train_ehr_dataset,columnas_test_ehr_dataset,100)

    #obtener un syntethic datafram que considere el percentil y si es mayor a eso se considera 1 si no 0, si es false no se le agrega la columnas id_patient
    synthetic_ehr = change_tosyn_stickers_temporal(synthetic_ehr_dataset,columnas_test_ehr_dataset,False)
           
    list_metric = ["attributes_attack","memebership_attack","nn_distance_attack","delta"]
    results = evaluate_attacks(list_metric, test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset,synthetic_ehr, top_300_codes, columnas_test_ehr_dataset) 
    print (results   )