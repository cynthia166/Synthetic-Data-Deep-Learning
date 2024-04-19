from scipy.spatial.distance import jensenshannon
import os
os.chdir("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning")
current_directory = os.getcwd()

print(current_directory)
import sys
sys.path.append('preprocessing')
sys.path.append('evaluation')
sys.path.append('privacy')
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
import seaborn as sns
import pickle
from tqdm import tqdm
import numpy as np
from evaluation.privacy.metric_privacy import *

#from evaluation.resemb.resemblance.metric_stat import *
from evaluation.functions import *
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
        return {"JensenShannonDistance": score}


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
        return {"MaximumMeanDiscrepancy": float(score)}

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
    
def corr_plot(total_features_train,name ):
    correlation_matrix = total_features_train.corr()
# Calcular la matriz de correlación
    from scipy.sparse import csr_matrix
    corr_matrix = total_features_train.corr()

    # Crear una figura y un eje para el gráfico
    fig, ax = plt.subplots()

    # Crear un mapa de calor de la matriz de correlación
    sns.heatmap(corr_matrix, annot=True, ax=ax)

    # Mostrar el gráfico
    plt.savefig('results_SD/img/'+name+'_kernelplot.svg')
    plt.show()




# Define a function to calculate statistics on the visits without considering labels
def generate_statistics(ehr_datasets):
    stats = {}

    for label, dataset in ehr_datasets:
        aggregate_stats = {}
        record_lens = [len(p['visits'][0]) for p in dataset]  # List of record lengths
        visit_lens = [len(v) for p in dataset for v in p['visits']]  # List of visit lengths

        # Calculating aggregates
        avg_record_len = np.mean(record_lens)
        std_record_len = np.std(record_lens)
        avg_visit_len = np.mean(visit_lens)
        std_visit_len = np.std(visit_lens)

        # Storing results
        aggregate_stats["Record Length Mean"] = avg_record_len
        aggregate_stats["Record Length Standard Deviation"] = std_record_len
        aggregate_stats["Visit Length Mean"] = avg_visit_len
        aggregate_stats["Visit Length Standard Deviation"] = std_visit_len

        stats[label] = aggregate_stats

    return stats



def plot_age(df,col,name):
    # Set the background style
    from scipy.stats import gaussian_kde
    plt.style.use("seaborn-white")

    # Define two shades of blue
    dark_blue = "#00008B"   # A dark blue
    light_blue = "#ADD8E6"  # A light blue

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(7, 4))

    # Calculate the kernel density estimate for 'Age_max' for each gender
   
    age_max_female = df[col]
   
    kde_age_max_female = gaussian_kde(age_max_female)

    # Generate x-values for the kernel density estimate
    x_age_max = np.linspace(age_max_female.min(), age_max_female.max(), 100)

    ax.plot(x_age_max, kde_age_max_female(x_age_max), color=light_blue, label='Gaussian KDE')

    # Set the labels and title for the 'Age_max' plot
    ax.set_xlabel(str(col))
    ax.set_ylabel('Density')
    ax.set_title('Kernel Density Plot - ' + str(col))
    ax.set_ylim(bottom=0)  # Set the y-axis limit to start at zero
    plt.savefig('results_SD/img/'+col+'_'+name+'_kernelplot.svg')
    # Show the plot
    plt.show()


def plot_procedures_diag_drugs(col_prod,train_ehr_dataset,type_procedur,name):
    df = train_ehr_dataset[col_prod+["id_patient","visit_rank"]]
    if type_procedur =="procedures":
        result_subject = df.groupby("id_patient").size().reset_index(name='Count')
        result_admission = df.groupby("visit_rank").size().reset_index(name='Count')
        label_xl = 'Count of ICD-9 codes'
    elif type_procedur=="diagnosis":
        result_subject = df.groupby("id_patient").size().reset_index(name='Count')
        result_admission = df.groupby("visit_rank").size().reset_index(name='Count')
        #result_subject_1 = result_subject[result_subject["Count"]<100]
        #result_admission_1 = result_admission[result_admission["Count"]<100]
        label_xl = 'Count of ICD-9 codes'
    elif type_procedur=="drugs":          
        result_subject = df.groupby("id_patient").size().reset_index(name='Count')
        result_admission = df.groupby("visit_rank").size().reset_index(name='Count')
        #result_subject_1 = result_subject[result_subject["Count"]<300]
        #result_admission_1 = result_admission[result_admission["Count"]<150]
        label_xl = 'Count of drugs'
    else:
        print( "Type of procedure not found"   ) 
    # Create a figure with two matplotlib.Axes objects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # Assigning a graph to each ax
    sns.histplot(data=result_subject, x="Count", ax=ax1, color='darkblue',bins = 50)
    sns.histplot(data=result_admission, x="Count", ax=ax2, color='lightblue',bins = 30)
    # Set x-axis and y-axis labels for each subplot
    ax1.set(xlabel='Count of ICD-9 codes per patient' + name, ylabel='Frequency')
    ax2.set(xlabel=label_xl+' per admission', ylabel='Frequency')
    ax1.text(-0.1, -0.2, '(a)', transform=ax1.transAxes, size=10, )
    ax2.text(-0.1, -0.2, '(b)', transform=ax2.transAxes, size=10, )
    fig.suptitle('Count of ICD-9 codes '+type_procedur, fontsize=14)  # Increase the font size of the title
    # Show the plot
    plt.savefig('results_SD/hist/'+type_procedur+'_'+name+'_kernelplot.svg')
    plt.show()


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

def plot_tsne(name,X_gt_df: pd.DataFrame, X_syn_df: pd.DataFrame) -> None:
    # Create the figure and axis for plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Compute t-SNE for ground truth data
    tsne_gt = TSNE(n_components=2, random_state=0, learning_rate="auto", init="pca")
    proj_gt = pd.DataFrame(tsne_gt.fit_transform(X_gt_df.values), columns=['x', 'y'])

    # Compute t-SNE for synthetic data
    tsne_syn = TSNE(n_components=2, random_state=0, learning_rate="auto", init="pca")
    proj_syn = pd.DataFrame(tsne_syn.fit_transform(X_syn_df.values), columns=['x', 'y'])

    # Scatter plot for ground truth and synthetic data
    ax.scatter(x=proj_gt['x'], y=proj_gt['y'], s=10, label="Real data")
    ax.scatter(x=proj_syn['x'], y=proj_syn['y'], s=10, label="Synthetic data")

    # Set legend and labels
    ax.legend(loc="upper left")
    ax.set_title("t-SNE plot")
    ax.set_xlabel("t-SNE axis 1")
    ax.set_ylabel("t-SNE axis 2")
    plt.savefig('results_SD/hist/tsne'+'_'+name+'_kernelplot.svg')
    # Display the plot
    plt.show()
 
def histograms_codes(train_ehr_dataset,test_ehr_dataset,synthetic_ehr_dataset):
    keywords = ['diagnosis', 'procedures', 'drugs']
    for i in keywords:
        col_prod = [col for col in train_ehr_dataset.columns if any(palabra in col for palabra in [i])]
        name = "Train"
        type_procedur = i

        plot_procedures_diag_drugs(col_prod,train_ehr_dataset,type_procedur,name)

        col_prod = [col for col in test_ehr_dataset.columns if any(palabra in col for palabra in [i])]
        name = "Test"


        plot_procedures_diag_drugs(col_prod,test_ehr_dataset,type_procedur,name)
        
        col_prod = [col for col in synthetic_ehr.columns if any(palabra in col for palabra in [i])]
        name = "Synthetic"


        plot_procedures_diag_drugs(col_prod,synthetic_ehr_dataset,type_procedur,name)
        
def get_statistics(train_ehr_dataset,columnas_test_ehr_dataset,test_ehr_dataset,synthetic_ehr):


        dataset_train = obtain_dataset(train_ehr_dataset,columnas_test_ehr_dataset)
        dataset_test = obtain_dataset(test_ehr_dataset,columnas_test_ehr_dataset)

        dataset_syn = obtain_dataset(synthetic_ehr,columnas_test_ehr_dataset)

        ##obtener static lenght o stay
        ehr_datasets = [
            ('Train', dataset_train),
            ('Test', dataset_test),
            ('Synthetic', dataset_syn),
            # ... other datasets
        ]

        # continous_plot

        statistics = generate_statistics(ehr_datasets)
        return statistics
# Example datasets

def calcular_remblencemetric(test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset ,columnas_test_ehr_dataset,top_300_codes,synthetic_ehr,list_metric_resemblance):
    result_resemblence = []
    results_final={}

    if "mmd" in list_metric_resemblance:
        mmd_evaluator = MaximumMeanDiscrepancy(kernel="rbf")
        train_test = "test"
        result =   mmd_evaluator._evaluate(test_ehr_dataset.iloc[:,:synthetic_ehr.shape[1]], synthetic_ehr)
        print("MaximumMeanDiscrepancy (flattened):", result)
        result_resemblence.append(result)

    # Example usage:
    # X_gt and X_syn are two numpy arrays representing empirical distributions
    if "ks_test" in list_metric_resemblance:
        features_1d = test_ehr_dataset.iloc[:,:synthetic_ehr.shape[1]].values.flatten()
        synthetic_features_1d = synthetic_ehr.values.flatten()
        ks_test = KolmogorovSmirnovTest()
        result = ks_test._evaluate(features_1d, synthetic_features_1d)
        print("Kolmog orov-Smirnov Test:", result)
        result_resemblence.append(result)

    if "jensenshannon_dist" in list_metric_resemblance:
        score = JensenShannonDistance()._evaluate(test_ehr_dataset.iloc[:,:synthetic_ehr.shape[1]].values, synthetic_ehr.values)
        #score = JensenShannonDistance()._evaluate(features_2d, synthetic_features_2d)synthetic_ehr
        print("Jensen-Shannon Distance:", score)
        result_resemblence.append(score)


    if "statistics" in list_metric_resemblance:
        statistics = get_statistics(train_ehr_dataset,columnas_test_ehr_dataset,test_ehr_dataset,synthetic_ehr_dataset)
        result_resemblence.append(statistics)
        
    if "kernel_density" in list_metric_resemblance:
        
        plot_age(test_ehr_dataset,"Age_max","test")
        plot_age(synthetic_ehr,"Age_max","synthetic")
        plot_age(train_ehr_dataset,"Age_max","synthetic")
    #plot histograms
        
    if "histogramas" in list_metric_resemblance:
        histograms_codes(train_ehr_dataset,test_ehr_dataset,synthetic_ehr_dataset)    


    # Example usage with two dataframes, `X_gt_df` and `X_syn_df`
    if "tsne" in list_metric_resemblance:
        plot_tsne("Doop",test_ehr_dataset, synthetic_ehr_dataset)
        
    if "corr" in list_metric_resemblance:
        corr_plot(synthetic_ehr_dataset,"Syn" )
        corr_plot(test_ehr_dataset,"Test" )  

    for i in result_resemblence:
        results_final.update(i)  

    return results_final
# Save the statistics to a file



if __name__=="main":
        
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





    #obtener coluans que contenga diagnosis,procedures,drugs
    columnas_test_ehr_dataset = get_cols_diag_proc_drug(train_ehr_dataset)

    #obtener n mas frequent codes
    top_300_codes = obtain_most_freuent(train_ehr_dataset,columnas_test_ehr_dataset,100)

    #obtener un syntethic datafram que considere el percentil y si es mayor a eso se considera 1 si no 0, si es false no se le agrega la columnas id_patient
    synthetic_ehr = change_tosyn_stickers_temporal(synthetic_ehr_dataset,columnas_test_ehr_dataset,True)

    #list_metric_resemblance = ["histogramas","tsne","statistics","kernel_density","mmd","ks_test","jensenshannon_dist"]
    list_metric_resemblance = ["statistics","mmd","ks_test","jensenshannon_dist"]

    results  =    calcular_remblencemetric(test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset ,columnas_test_ehr_dataset,top_300_codes,synthetic_ehr,list_metric_resemblance)
    print(results)