
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