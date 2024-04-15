from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import numpy as np
from typing import Dict
import numpy as np
from scipy.special import rel_entr
from typing import Any, Dict
import numpy as np
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from typing import Any

from scipy.stats import ks_2samp
import numpy as np
from scipy.stats import chisquare, ks_2samp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class JensenShannonDistance:
    """Evaluate the average Jensen-Shannon distance (metric) between two probability arrays."""
    
    def __init__(self, normalize: bool = False) -> None:
        self.normalize = normalize
    
    @staticmethod
    def name() -> str:
        return "jensenshannon_dist"
    
    @staticmethod
    def direction() -> str:
        return "minimize"
    
    def _evaluate_stats(self, X_gt: np.ndarray, X_syn: np.ndarray) -> float:
        if self.normalize:
            X_gt = X_gt / X_gt.sum(axis=1, keepdims=True)
            X_syn = X_syn / X_syn.sum(axis=1, keepdims=True)
        
        distances = []
        for gt, syn in zip(X_gt, X_syn):
            distances.append(jensenshannon(gt, syn))
        distances = np.array(distances)
        distances = distances[np.isfinite(distances)]
  
        return np.nanmean(distances)  # Returns the mean Jensen-Shannon distance
    
    def _evaluate(self, X_gt: np.ndarray, X_syn: np.ndarray) -> Dict[str, float]:
        score = self._evaluate_stats(X_gt, X_syn)
        return {"marginal": score}


class KolmogorovSmirnovTest:
    """
    Performs the Kolmogorov-Smirnov test for goodness of fit.

    Score:
        0: the distributions are totally different.
        1: the distributions are identical.
    """
    def __init__(self) -> None:
        pass  # No additional arguments needed
    @staticmethod
    def name() -> str:
        return "ks_test"
    @staticmethod
    def direction() -> str:
        return "maximize"
    def _evaluate(self, X_gt: np.ndarray, X_syn: np.ndarray) -> Dict:
        # Validate that inputs are 1D arrays
        if X_gt.ndim != 1 or X_syn.ndim != 1:
            raise ValueError("Input arrays must be one-dimensional")
        # Perform the Kolmogorov-Smirnov test
        statistic, _ = ks_2samp(X_gt, X_syn)
        score = 1 - statistic  # The score is the complement of the KS statistic
        return {"marginal": score}

class MaximumMeanDiscrepancy():
    """
    Empirical maximum mean discrepancy. The lower the result the more evidence that distributions are the same.
    Args:
        kernel: "rbf", "linear" or "polynomial"
    Score:
        0: The distributions are the same.
        1: The distributions are totally different.
    """
    def __init__(self, kernel: str = "rbf", **kwargs: Any) -> None:
  
        self.kernel = kernel
    @staticmethod
    def name() -> str:
        return "max_mean_discrepancy"
    @staticmethod
    def direction() -> str:
        return "minimize"
    def _evaluate(
        self,
        X_gt: np.ndarray,
        X_syn: np.ndarray,
    ) -> Dict:
        if self.kernel == "linear":
            """
            MMD using linear kernel (i.e., k(x,y) = <x,y>)
            """
            delta = X_gt.mean(axis=0) - X_syn.mean(axis=0)
            score = delta.dot(delta.T)
        elif self.kernel == "rbf":
            """
            MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
            """
            gamma = 1.0
            XX = metrics.pairwise.rbf_kernel(X_gt, X_gt, gamma)
            YY = metrics.pairwise.rbf_kernel(X_syn, X_syn, gamma)
            XY = metrics.pairwise.rbf_kernel(X_gt, X_syn, gamma)
            score = XX.mean() + YY.mean() - 2 * XY.mean()
        elif self.kernel == "polynomial":
            """
            MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
            """
            degree = 2
            gamma = 1
            coef0 = 0
            XX = metrics.pairwise.polynomial_kernel(X_gt, X_gt, degree, gamma, coef0)
            YY = metrics.pairwise.polynomial_kernel(X_syn, X_syn, degree, gamma, coef0)
            XY = metrics.pairwise.polynomial_kernel(X_gt, X_syn, degree, gamma, coef0)
            score = XX.mean() + YY.mean() - 2 * XY.mean()
        else:
            raise ValueError(f"Unsupported kernel {self.kernel}")
        return {"joint": float(score)}

def plot_marginal_comparison(
    plt: Any, X_gt: np.ndarray, X_syn: np.ndarray, n_histogram_bins: int = 10, normalize: bool = True
) -> None:
    # Assuming X_gt and X_syn are 2D arrays with the same number of columns (features)
    plots_cnt = X_gt.shape[1]
    row_len = 2
    fig, ax = plt.subplots(
        int(np.ceil(plots_cnt / row_len)), row_len, figsize=(14, plots_cnt * 3)
    )
    fig.subplots_adjust(hspace=1)
    if plots_cnt % row_len != 0:
        fig.delaxes(ax.flatten()[-1])
    for idx in range(plots_cnt):
        row_idx = idx // row_len
        col_idx = idx % row_len
        local_ax = ax[row_idx, col_idx] if plots_cnt > row_len else ax[idx]
        # Compute histograms
        gt_hist, bin_edges = np.histogram(X_gt[:, idx], bins=n_histogram_bins, density=normalize)
        syn_hist, _ = np.histogram(X_syn[:, idx], bins=bin_edges, density=normalize)
        # Compute Jensen-Shannon Distance
        m = 0.5 * (gt_hist + syn_hist)
        js_distance = 0.5 * (entropy(gt_hist, m) + entropy(syn_hist, m))
        bar_position = np.arange(n_histogram_bins)
        bar_width = 0.4
        # real distribution
        local_ax.bar(
            x=bar_position,
            height=gt_hist,
            color='blue',
            label='Real',
            width=bar_width,
        )
        # synthetic distribution
        local_ax.bar(
            x=bar_position + bar_width,
            height=syn_hist,
            color='orange',
            label='Synthetic',
            width=bar_width,
        )
        local_ax.set_xticks(bar_position + bar_width / 2)
        local_ax.set_xticklabels([f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)], rotation=90)
        title = "Feature {}\nJensen-Shannon Distance: {:.2f}".format(idx, js_distance)
        local_ax.set_title(title)
        if normalize:
            local_ax.set_ylabel("Probability")
        else:
            local_ax.set_ylabel("Count")
        local_ax.legend()
    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    plt.show()
    
def plot_tsne(
    plt: Any,
    X_gt: np.ndarray,
    X_syn: np.ndarray,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    # Configurar t-SNE para los datos reales
    tsne_gt = TSNE(n_components=2, random_state=0, learning_rate="auto", init="pca")
    proj_gt = tsne_gt.fit_transform(X_gt)
    # Configurar t-SNE para los datos sintéticos
    tsne_syn = TSNE(n_components=2, random_state=0, learning_rate="auto", init="pca")
    proj_syn = tsne_syn.fit_transform(X_syn)
    # Dibujar los scatter plots para los datos transformados por t-SNE
    ax.scatter(x=proj_gt[:, 0], y=proj_gt[:, 1], s=10, label="Datos Reales")
    ax.scatter(x=proj_syn[:, 0], y=proj_syn[:, 1], s=10, label="Datos Sintéticos")
    # Configuraciones adicionales para el plot
    ax.legend(loc="upper left")
    ax.set_title("Gráfico t-SNE")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    plt.show()    
    