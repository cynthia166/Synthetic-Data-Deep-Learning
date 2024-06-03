from scipy.spatial.distance import jensenshannon
import os
#os.chdir("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning")
os.chdir("/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/")
current_directory = os.getcwd()


print(current_directory)
import sys
sys.path.append('preprocessing')
sys.path.append('evaluation/privacy/metric_privacy')
sys.path.append('privacy')
sys.path.append('evaluation')
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
import pandas as pd
import glob
import numpy as np
import pandas as pd
from scipy import stats

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def obtain_most_freuent(train_ehr_dataset,columnas_test_ehr_dataset,num):
    code_sums = train_ehr_dataset[columnas_test_ehr_dataset].sum(axis=0).sort_values(ascending=False)
    top_300_codes = code_sums.head(num).index.tolist()
    return top_300_codes

def obtain_least_frequent(train_ehr_dataset, columns, num):
    """
    This function calculates the least frequent categories for specified columns in a DataFrame.

    Parameters:
    - train_ehr_dataset: pandas DataFrame from which to calculate frequencies.
    - columns: list of columns to sum frequencies across.
    - num: number of least frequent categories to return.

    Returns:
    - List of the indices of the least frequent categories.
    """
    # Calculate the sum of occurrences across specified columns and sort them in ascending order
    code_sums = train_ehr_dataset[columns].sum(axis=0).sort_values(ascending=True)
    
    # Get the indices of the least frequent categories up to the number specified
    least_frequent_codes = code_sums.head(num).index.tolist()
    
    return least_frequent_codes


#from evaluation.resemb.resemblance.metric_stat import *
#from evaluation.functions import *
def get_cols_diag_proc_drug(train_ehr_dataset):
    keywords = ['diagnosis', 'procedures', 'drugs']
    columnas_test_ehr_dataset = [col for col in train_ehr_dataset.columns if any(palabra in col for palabra in keywords)]
    return columnas_test_ehr_dataset

def obtain_readmission_realdata(total_fetura_valid):
    # Ordenando el DataFrame por 'id_' y 'visit_rank'
    total_fetura_valid = total_fetura_valid.sort_values(by=['id_patient', 'visit_rank'])

    # Crear una nueva columna 'readmission'
    # Comparamos si el siguiente 'visit_rank' es mayor que el actual para el mismo 'id_'
    total_fetura_valid['readmission'] = total_fetura_valid.groupby('id_patient')['visit_rank'].shift(-1).notna().astype(int)  
    return  total_fetura_valid

import numpy as np
import pandas as pd
from scipy.spatial import distance

import numpy as np
import pandas as pd
from scipy.spatial import distance

class JensenShannonDistance2:
    
    def __init__(self, origdst, synthdst, num):
        self.origdst = origdst
        self.synthdst = synthdst
        self.num = num
    
    @staticmethod
    def to_cat(df_real, df_synth):
        df_real_cat = df_real.copy()
        df_synth_cat = df_synth.copy()
        
        for col in df_real_cat.columns:
            df_real_cat[col] = df_real_cat[col].astype('category')
            df_synth_cat[col] = df_synth_cat[col].astype('category')
        
        return df_real_cat, df_synth_cat
    
    def jensen_shannon(self):
        real_cat, synth_cat = self.to_cat(self.origdst, self.synthdst)
        
        target_columns = self.origdst.columns
        
        js_dict = {}
        
        for col in target_columns:
            try:
                categories = set(real_cat[col].unique()).union(set(synth_cat[col].unique()))
                
                col_counts_orig = real_cat[col].value_counts(normalize=True).reindex(categories, fill_value=0).sort_index()
                col_counts_synth = synth_cat[col].value_counts(normalize=True).reindex(categories, fill_value=0).sort_index()
                
                js = distance.jensenshannon(col_counts_orig.values, col_counts_synth.values, base=2)
                
                js_dict[col] = js
            
            except Exception as e:
                print(f'Error processing column {col}: {e}')
                print(f'For the column {col}, you must generate the same unique values as the real dataset.')
                print(f'The number of unique values you should generate for column {col} is {len(self.origdst[col].unique())}.')
        
        # Calculate the average Jensen-Shannon Distance
        avg_js = np.mean(list(js_dict.values()))
        
        # Sort the columns by their Jensen-Shannon Distance
        sorted_js = sorted(js_dict.items(), key=lambda item: item[1])
        
        # Get the 10 least and most different columns
        least_different = sorted_js[:self.num]
        most_different = sorted_js[-self.num:]
        
        # Create DataFrames
        least_different_df = pd.DataFrame(least_different, columns=["Column", "JSD"]).reset_index(drop=True)
        most_different_df = pd.DataFrame(most_different, columns=["Column", "JSD"]).reset_index(drop=True)
        
        # Combine DataFrames
        combined_df = pd.concat([least_different_df, most_different_df], axis=1)
        combined_df.columns = ["Least Different Column", "Least Different JSD", "Most Different Column", "Most Different JSD"]
        
        return combined_df, avg_js



# Example usage
# Assuming `orig_data` and `synth_data` are your real and synthetic datasets as pandas DataFrames


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
        return {"KolmogorovSmirnovTest_marginal": score}

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
    plt: Any, X_gt: np.ndarray, X_syn: np.ndarray, n_histogram_bins: int = 10, normalize: bool = True) -> None:
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


    
def corr_plot(total_features_train,name ):
    #correlation_matrix = total_features_train.corr()
# Calcular la matriz de correlación
    from scipy.sparse import csr_matrix
    '''corr_matrix = total_features_train.corr()
    # Crear una figura y un eje para el gráfico
    fig, ax = plt.subplots()
    # Crear un mapa de calor de la matriz de correlación
    sns.heatmap(corr_matrix, annot=False, ax=ax)
    # Mostrar el gráfico
    plt.savefig('results_SD/img/'+name+'_kernelplot.svg')
    plt.show()'''
    # Suponiendo que total_features_train es tu DataFrame y que ya tienes la matriz de correlación
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Suponiendo que total_features_train es tu DataFrame y que ya tienes la matriz de correlación
    corr_matrix = total_features_train.corr()

    # Crear una figura y un eje para el gráfico
    fig, ax = plt.subplots(figsize=(10, 8))  # Ajusta el tamaño según tus necesidades

    # Definir un mapa de colores personalizado
    # Utilizando colores divergentes: azul para valores cercanos a -1 y rojo para valores cercanos a 1
    cmap = sns.diverging_palette(240, 10, as_cmap=True)  # 240 es azul, 10 es rojo en HUSL

    # Crear un mapa de calor de la matriz de correlación
    sns.heatmap(corr_matrix, annot=False, cmap=cmap, center=0, ax=ax)

    # Configurando las etiquetas de los ejes con condiciones para visualización
    num_columns = corr_matrix.shape[1]
    if num_columns <= 30:
        ax.set_xticklabels([])
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        #ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Mostrar el gráfico
    plt.savefig('results_SD/img/'+name+'_kernelplot.svg')
    plt.show()
        
    return corr_matrix



# Define a function to calculate statistics on the visits without considering labels
def generate_statistics(ehr_datasets,type_s):
    stats_ = {}
    for label, dataset in ehr_datasets:
        aggregate_stats = {}
        record_lens = [len(p['visits'][0]) for p in dataset]  # List of record lengths
        #visit_lens = [len(v) for p in dataset for v in p['visits']]  # List of visit lengths
        # Calculating aggregates
        avg_record_len = np.mean(record_lens)
        std_record_len = np.std(record_lens)
        #avg_visit_len = np.mean(visit_lens)
        #std_visit_len = np.std(visit_lens)
        # Storing results
        aggregate_stats["Record Length Mean" +type_s] = avg_record_len
        aggregate_stats["Record Length Standard Deviation" +type_s] = std_record_len
        #aggregate_stats["Visit Length Mean"] = avg_visit_len
        #aggregate_stats["Visit Length Standard Deviation"] = std_visit_len
        #stats[label] = aggregate_stats
    return aggregate_stats



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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def filtered(real_result_admission):
    total_counts = real_result_admission['Count'].sum()
    # Paso 2: Calcular el porcentaje de cada grupo respecto al total
    real_result_admission['Percentage'] = (real_result_admission['Count'] / total_counts) * 100
    # Paso 3: Filtrar las filas donde el porcentaje sea mayor al 30%
    filtered_result_admission = real_result_admission[real_result_admission['Percentage'] > 2]
    print("filtered_result_admission",filtered_result_admission.shape)
    print("real_result_admission",real_result_admission.shape)      
    return filtered_result_admission    

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_visits_per_patient_histogram(train_ehr_dataset, synthetic_ehr):
    # Group by id_patient and count the number of visits for real data
    real_visits_per_patient = train_ehr_dataset.groupby("id_patient")["visit_rank"].count().reset_index(name='visit_count')

    # Group by id_patient and count the number of visits for synthetic data
    synthetic_visits_per_patient = synthetic_ehr.groupby("id_patient")["visit_rank"].count().reset_index(name='visit_count')

    # Plot the histograms separately but in the same figure
    plt.figure(figsize=(12, 6))
    sns.histplot(real_visits_per_patient['visit_count'], color='blue', label='Real', kde=False, bins=50, stat='frequency',)
    sns.histplot(synthetic_visits_per_patient['visit_count'], color='orange', label='Synthetic', kde=False, bins=50, stat='frequency',)
    
    plt.title('Histogram of Visit Counts per Patient - Real vs Synthetic')
    plt.xlabel('Number of Visits per Patient')
    plt.ylabel('Frequency')
    plt.legend(title='Source')
    plt.tight_layout()
    plt.show()
    
def plot_patients_per_visit_rank_histogram(train_ehr_dataset, synthetic_ehr_dataset):
    # Group by visit_rank and count the number of unique patients for real data
    real_patients_per_visit = train_ehr_dataset.groupby("visit_rank")["id_patient"].nunique().reset_index(name='patient_count')

    # Group by visit_rank and count the number of unique patients for synthetic data
    synthetic_patients_per_visit = synthetic_ehr_dataset.groupby("visit_rank")["id_patient"].nunique().reset_index(name='patient_count')

    # Plot the histograms separately but in the same figure
    plt.figure(figsize=(12, 6))
    sns.histplot(real_patients_per_visit['patient_count'], color='blue', label='Real',  bins=100, stat='count', )
    sns.histplot(synthetic_patients_per_visit['patient_count'], color='orange', label='Synthetic',  bins=100, stat='count', )
    
    plt.title('Histogram of Patient Counts per Visit Rank - Real vs Synthetic')
    plt.xlabel('Number of Patients per Visit Rank')
    plt.ylabel('Frequency')
    plt.legend(title='Source')
    plt.tight_layout()
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data for demonstration

def plot_histograms22_(real_data, synthetic_data, title, xlabel, ylabel):
    # Eliminar duplicados en los índices
    real_data = real_data.reset_index(drop=True)
    synthetic_data = synthetic_data.reset_index(drop=True)
    
    # Definir bins comunes
    max_value = max(real_data.max(), synthetic_data.max())
    bins = np.linspace(0, max_value, 31)  # 31 bins para crear 30 intervalos
    
    # Crear un DataFrame combinando ambos conjuntos de datos para facilitar la comparación
    data_combined = pd.DataFrame({
        'Number of Drugs': pd.concat([real_data, synthetic_data], axis=0),
        'Type': ['Real'] * len(real_data) + ['Synthetic'] * len(synthetic_data)
    })
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data_combined, x='Number of Drugs', hue='Type', element='step', stat='count', bins=bins, common_norm=False)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title='Data Type')
    plt.show()


def calculate_counts(train_ehr_dataset,word):
    # Number of drugs each patient has (sum of drugs for each patient)
    drugs_per_patient = train_ehr_dataset.groupby("id_patient").sum().reset_index()
    drugs_per_patient[word+'_count'] = drugs_per_patient.filter(like=word).sum(axis=1)

    # Number of drugs per admission
    drugs_per_admission = train_ehr_dataset.groupby("visit_rank").sum().reset_index()
    drugs_per_admission[word+'_count'] = drugs_per_admission.filter(like=word).sum(axis=1)

    # Admissions per drug (sum of admissions where each drug is present)
    admissions_per_drug = train_ehr_dataset.filter(like=word).sum().reset_index()
    admissions_per_drug.columns = [word, 'admission_count']
    
    # Patients per drug (sum of patients where each drug is present)
    patients_per_drug = train_ehr_dataset.groupby('id_patient').max().filter(like=word).sum().reset_index()
    patients_per_drug.columns = [word, 'patient_count']


    return drugs_per_patient, drugs_per_admission, admissions_per_drug,patients_per_drug

# Calculate counts for real and synthetic data

# Plot histograms
def plot_histograms(real_data, synthetic_data, title, xlabel, ylabel_real, ylabel_synthetic):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    sns.histplot(real_data, color='blue', label='Real',  bins=30, ax=ax1, stat='count',alpha=0.1)
    sns.histplot(synthetic_data, color='orange', label='Synthetic', bins=30, ax=ax2, stat='count',alpha=0.3)
    
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel_real, color='blue')
    ax2.set_ylabel(ylabel_synthetic, color='orange')
    
    ax1.tick_params(axis='y', colors='blue', pad=10)
    ax2.tick_params(axis='y', colors='orange', pad=10)
    
    #fig.tight_layout()
    fig.legend(loc='upper right', )
    plt.show()
    
def plot_histograms22(real_data, synthetic_data, title, xlabel, ylabel_real, ylabel_synthetic):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Define common bins
    max_value = max(real_data.max(), synthetic_data.max())
    bins = np.linspace(0, max_value, 31)  # 31 bins to create 30 intervals
    
    # Plot histograms with the same bins
    sns.histplot(real_data, color='blue', label='Real', bins=bins, ax=ax1, stat='count', alpha=0.5)
    sns.histplot(synthetic_data, color='orange', label='Synthetic', bins=bins, ax=ax2, stat='count', alpha=0.3)
    
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel_real, color='blue')
    ax2.set_ylabel(ylabel_synthetic, color='orange')
    
    ax1.tick_params(axis='y', colors='blue', pad=10)
    ax2.tick_params(axis='y', colors='orange', pad=10)
    
    fig.legend(loc='upper right')
    plt.show()
    
# Plot histograms
# The code snippet is iterating over a list containing strings 'diagnosis', 'procedures', 'drugs'
# starting from the index 2 (which is 'drugs'). For each element in the list starting from 'drugs', it
# calls the function `calculate_counts` with the respective dataset (train_ehr_dataset and
# synthetic_ehr_dataset) and the current element from the list as an argument. The function returns
# counts related to drugs such as real_drugs_per_patient, real_drugs_per_admission,
# real_admissions_per_drug, real_patients_per_drug for the train_e

# Supongamos que tienes dos DataFrames: df_real y df_synthetic

# Función para detectar outliers usando el IQR
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers

# Función para calcular la relación mínima de outliers sintéticos a reales
def calculate_outlier_ratio(df_real, df_synthetic):

        real_outliers = detect_outliers_iqr(df_real)
        synthetic_outliers = detect_outliers_iqr(df_synthetic)
        
        # Contar outliers
        real_outlier_count = real_outliers.sum()
        synthetic_outlier_count = synthetic_outliers.sum()
        
        # Evitar división por cero
        if real_outlier_count == 0:
            ratio = np.inf
        else:
            ratio = synthetic_outlier_count / real_outlier_count
        
        # Encontrar la relación mínima para cada columna
        mean_ratio = min(ratio,1)
        return mean_ratio

# Calcular la relación mínima de outliers para todas las columnas
def calculate_outlier_ratios_tout__(df_real, df_synthetic,cols_sel):
    min_ratios = []
    for column in df_real[cols_sel]:
        min_ratio = calculate_outlier_ratio(df_real[column], df_synthetic[column])
        min_ratios.append(min_ratio)
    return  np.mean(min_ratios)   


#continuos_var
        
def plot_hist_emp_codes(col_prod, train_ehr_dataset, synthetic_ehr, type_procedur, name):
    # Process and filter data
    real_df = train_ehr_dataset[col_prod + ["id_patient", "visit_rank"]]
    synthetic_df = synthetic_ehr[col_prod + ["id_patient", "visit_rank"]]

    real_result_subject = real_df[[i for i in real_df.columns if i != "visit_rank"]].groupby("id_patient").sum().sum(axis=1).reset_index(name='Count')
    filtered_result_admission = real_df[[i for i in real_df.columns if i != "id_patient"]].groupby("visit_rank").sum().sum(axis=1).reset_index(name='Count')

    synthetic_result_subject = synthetic_df[[i for i in synthetic_df.columns if i != "visit_rank"]].groupby("id_patient").sum().sum(axis=1).reset_index(name='Count')
    filtered_synthetic_result_admission = synthetic_df[[i for i in synthetic_df.columns if i != "id_patient"]].groupby("visit_rank").sum().sum(axis=1).reset_index(name='Count')

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    sns.histplot(data=real_result_subject, x="Count", ax=ax1, color='darkblue', bins=30)
    sns.histplot(data=synthetic_result_subject, x="Count", ax=ax2, color='lightblue', bins=20)
    ax1.set_title('Real Data - Patient Count '+type_procedur)
    ax2.set_title('Synthetic Data - Patient Count '+type_procedur)
    ax1.set_xlabel('Count ' + type_procedur+ ' per Patient')
    ax2.set_xlabel('Count ' +  type_procedur+ ' per Patient')
    ax1.set_ylabel('Frequency')
    ax2.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'results_SD/hist/{type_procedur}__comparison_plot_patient.svg')

    # Figure for Admission Data
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    sns.histplot(data=filtered_result_admission, x="Count", ax=ax3, color='darkblue', bins=20)
    sns.histplot(data=filtered_synthetic_result_admission, x="Count", ax=ax4, color='lightblue', bins=20)
    ax3.set_title('Real Data - Admission Count ' +type_procedur)
    ax4.set_title('Synthetic Data - Admission Count '+type_procedur)
    ax3.set_xlabel('Count '+ type_procedur+' per Admission')
    ax4.set_xlabel('Count '+ type_procedur +' per Admission')
    ax3.set_ylabel('Frequency')
    ax4.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'results_SD/hist/{type_procedur}__comparison_plot_admission.svg')

# Example usage with mock data and parameters
# plot_hist_emp_codes(col_prod, train_ehr_dataset, synthetic_ehr, type_procedur, name)


def plot_hist_emp_codes_auz(col_prod, train_ehr_dataset, synthetic_ehr, type_procedur, name):
    # Process real and synthetic data
    real_df = train_ehr_dataset[col_prod + ["id_patient", "visit_rank"]]
    synthetic_df = synthetic_ehr[col_prod + ["id_patient", "visit_rank"]]
    
    
    

    # Prepare results for real and synthetic data
    real_result_subject = real_df.groupby("id_patient").sum().sum(axis=1).reset_index(name='Count')
    real_result_admission = real_df.groupby("visit_rank").sum().sum(axis=1).reset_index(name='Count')
    synthetic_result_subject = synthetic_df.groupby("id_patient").sum().sum(axis=1).reset_index(name='Count')
    synthetic_result_admission = synthetic_df.groupby("visit_rank").sum().sum(axis=1).reset_index(name='Count')
    filtered_result_admission = filtered(real_result_admission)
    filtered_synthetic_result_admission = filtered(synthetic_result_admission)
    # Determine label based on type of procedure
    if type_procedur in ["procedures", "diagnosis", "drugs"]:
        label_xl = f'Count of {type_procedur}'
    else:
        print("Type of procedure not found")
        return

    # Create a figure with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8), sharey='row')

    # Plotting for real data
    sns.histplot(data=real_result_subject, x="Count", ax=ax1, color='darkblue', bins=40)
    sns.histplot(data=filtered_result_admission, x="Count", ax=ax2, color='lightblue', bins=10)
    ax1.set_ylim(1, max(real_result_subject["Count"]))
    ax2.set_ylim(1, max(filtered_result_admission["Count"])) # Set y-axis limits for 'per admission' plot for real data

    # Plotting for synthetic data
    sns.histplot(data=synthetic_result_subject, x="Count", ax=ax3, color='darkgreen', bins=40)
    sns.histplot(data=filtered_synthetic_result_admission, x="Count", ax=ax4, color='lightgreen', bins=10)
    ax3.set_ylim(1, max(synthetic_result_subject["Count"]))
    ax4.set_ylim(1, max(filtered_synthetic_result_admission["Count"])) # Set y-axis limits for 'per admission' plot for real data

    # Set x-axis and y-axis labels for each subplot
    ax1.set(xlabel=f'Count per patient (Real - {name})', ylabel='Frequency')
    ax2.set(xlabel=f'{label_xl} per admission (Real)', ylabel='Frequency')
    ax3.set(xlabel=f'Count per patient (Synthetic - {name})', ylabel='Frequency')
    ax4.set(xlabel=f'{label_xl} per admission (Synthetic)', ylabel='Frequency')

    # Set subplot titles
    ax1.set_title('Real Data')
    ax2.set_title('Real Data')
    ax3.set_title('Synthetic Data')
    ax4.set_title('Synthetic Data')

    # Add subplot annotations
    ax1.text(-0.1, 1.1, '(a)', transform=ax1.transAxes, size=12)
    ax2.text(-0.1, 1.1, '(b)', transform=ax2.transAxes, size=12)
    ax3.text(-0.1, 1.1, '(c)', transform=ax3.transAxes, size=12)
    ax4.text(-0.1, 1.1, '(d)', transform=ax4.transAxes, size=12)

    # Set a super title for the figure
    fig.suptitle('Comparison of Real and Synthetic Data Counts', fontsize=16)

    # Save and show the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the super title
    plt.savefig(f'results_SD/hist/{type_procedur}__comparison_plot.svg')
    plt.show()



import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

def plot_tsne_r(name,X_gt_df: pd.DataFrame, X_syn_df: pd.DataFrame) -> None:
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
def get_statistics(train_ehr_dataset,columnas_test_ehr_dataset,test_ehr_dataset,synthetic_ehr,synthetic_ehr_dataset):
        dataset_train = obtain_dataset(train_ehr_dataset,columnas_test_ehr_dataset)
        dataset_test = obtain_dataset(test_ehr_dataset,columnas_test_ehr_dataset)
        dataset_syn = obtain_dataset(synthetic_ehr_dataset,columnas_test_ehr_dataset)
        ##obtener static lenght o stay
        ehr_datasets = [
            ('Test', dataset_train),
            # ... other datasets
        ]
        # continous_plot
        statistics = generate_statistics(ehr_datasets,'Train')
        ehr_datasets = [
            ('Synthetic', dataset_syn),
            # ... other datasets
        ]
        statistics2 = generate_statistics(ehr_datasets,'Synthetic')
        statistics.update(statistics2)
        df = pd.DataFrame([statistics])

#  Adjust DataFrame for better representation
        df = df.T.reset_index()  # Transpose and reset index to make it vertical
        df.columns = ['Metric', 'Value']  # Rename columns for clarity
        
        # Convert DataFrame to LaTeX
        latex_table = df.to_latex(index=False, caption='Record Length Statistics', label='tab:record_length', column_format='ll')

        # Print LaTeX code
        print(latex_table)

        
        return statistics
# Example datasets

import numpy as np
from typing import Any, Dict

class CommonRowsProportion:
    """
    Returns the proportion of rows in the real dataset leaked in the synthetic dataset.

    Score:
        0: there are no common rows between the real and synthetic datasets.
        1: all the rows in the real dataset are leaked in the synthetic dataset.
    """

    def __init__(self, **kwargs) -> None:
        self.default_metric = kwargs.get("default_metric", "score")


    def evaluate(self, X_gt: np.ndarray, X_syn: np.ndarray) -> Dict[str, float]:
        if X_gt.shape[1] != X_syn.shape[1]:
            raise ValueError(f"Incompatible array shapes {X_gt.shape} and {X_syn.shape}")

        # Convert numpy arrays to pandas DataFrames
        df_gt = pd.DataFrame(X_gt)
        df_syn = pd.DataFrame(X_syn)

        # Find intersection of rows
        intersection = pd.merge(df_gt, df_syn, how='inner').drop_duplicates()

        # Calculate score
        score = len(intersection) / (len(df_gt) + 1e-8)
        return {"Common Rows Proportion score": score}

import numpy as np
from scipy import stats

import pandas as pd
import numpy as np
from scipy import stats

def descriptive_statistics_matrix(data):
    # Check if the DataFrame is empty
    if data.empty:
        return "Empty DataFrame, no statistics available."

    # Initialize a dictionary to hold statistics for each feature
    stats_dict = {}
    
    # Calculate statistics for each feature (column)
    for column in data.columns:
        column_data = data[column]
        
        # Compute each statistic and concatenate it with the column name
        stats_dict['mean_' + column] = np.mean(column_data)
        stats_dict['median_' + column] = np.median(column_data)
        stats_dict['mode_' + column] = stats.mode(column_data)[0][0] if len(column_data) > 0 else None
        stats_dict['minimum_' + column] = np.min(column_data)
        stats_dict['maximum_' + column] = np.max(column_data)
        stats_dict['range_' + column] = np.ptp(column_data)
        stats_dict['variance_' + column] = np.var(column_data, ddof=1)
        stats_dict['standard_deviation_' + column] = np.std(column_data, ddof=1)
        stats_dict['skewness_' + column] = stats.skew(column_data)
        stats_dict['kurtosis_' + column] = stats.kurtosis(column_data)

    return stats_dict

import numpy as np
import pandas as pd

def multivariate_statistics(data):
    # Check if the data is empty
    if data.size == 0:
        return "Empty array, no statistics available."

    # Compute the covariance matrix
    # Note: numpy.cov assumes rows are variables and columns are observations,
    # so we transpose data for correct calculation as our rows are observations.
    covariance_matrix = np.cov(data.T)

    # Convert the numpy array to a pandas DataFrame to compute the correlation matrix
    df = pd.DataFrame(data)
    correlation_matrix = df.corr()

    # Prepare the results in a dictionary
    stats_dict = {
        'covariance_matrix': covariance_matrix,
        'correlation_matrix': correlation_matrix
    }

    return stats_dict

import numpy as np

def compare_matrices(matrix1, matrix2, method="frobenius"):
    """
    Compare two matrices using the specified method.

    Parameters:
        matrix1 (numpy.ndarray): The first matrix.
        matrix2 (numpy.ndarray): The second matrix.
        method (str): Method of comparison. Options are "frobenius", "spectral", "max_diff".

    Returns:
        float: The comparison result.
    """
    # Check if the matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same dimensions")

    # Compute the difference matrix
    diff_matrix = matrix1 - matrix2

    if method == "frobenius":
        # Frobenius norm of the difference matrix
        return np.linalg.norm(diff_matrix, 'fro')
    elif method == "spectral":
        # Spectral norm (maximum singular value) of the difference matrix
        return np.linalg.norm(diff_matrix, 2)
    elif method == "max_diff":
        # Maximum absolute difference
        return np.max(np.abs(diff_matrix))
    else:
        raise ValueError("Unknown comparison method")

import numpy as np

import pandas as pd

import pandas as pd

def compare_descriptive_statistics(data1, data2):
    """
    Compare descriptive statistics between two dataframes.

    Parameters:
        data1 (pd.DataFrame): First dataset.
        data2 (pd.DataFrame): Second dataset.

    Returns:
        dict: A dictionary containing differences in descriptive statistics between the two datasets.
    """
    # Calculating descriptive statistics for both datasets
    stats1 = descriptive_statistics_matrix(data1)
    stats2 = descriptive_statistics_matrix(data2)

    # Dictionary to store the differences
    stats_differences = {}

    # Compute differences only for keys that exist in both dictionaries
    common_keys = set(stats1.keys()).intersection(set(stats2.keys()))
    for key in common_keys:
        if isinstance(stats1[key], (int, float)) and isinstance(stats2[key], (int, float)):
            # Calculate the difference and store it
            stats_differences[f'diff_{key}'] = abs(stats1[key] - stats2[key])

    return stats_differences
import pandas as pd
def get_proportions(df,type_st):

    # Suponiendo que df es tu DataFrame y que contiene columnas categóricas


    # Lista para almacenar los DataFrames de cada categoría
    tablas_proporciones = []

    # Iterar sobre las columnas categóricas
    for column in df.columns:
        # Calcular recuento y proporciones para la columna actual
        recuento = df[column].value_counts()
        proporciones = recuento / len(df)

        # Crear DataFrame de proporciones para la columna actual
        tabla_actual = pd.DataFrame({
            'Category ': recuento.index,
            'Count '+ type_st: recuento.values,
            'Proportion '+ type_st: proporciones.values
        })
        
        # Agregar el nombre de la columna como una nueva columna en la tabla de proporciones
        tabla_actual['Variable'] = column
        
        # Agregar la tabla de proporciones a la lista
        tablas_proporciones.append(tabla_actual)

    # Concatenar todas las tablas de proporciones en un solo DataFrame
    tabla_proporciones_final = pd.concat(tablas_proporciones, ignore_index=True)

    # Imprimir la tabla de proporciones final
    print(tabla_proporciones_final)

    return tabla_proporciones_final

    #res_propr = pd.concat(prop_datafram)   
    #latex_code = res_propr.to_latex( )

     # Print the LaTeX code
  
    
    
     
        

def compare_proportions_and_return_dictionary(real_data, synthetic_data):
    """
        dict: A dictionary with column names as keys and proportion differences as values.
    """
    # Initialize a dictionary to hold the differences in proportions
    proportion_differences = {}

    # Ensure that both DataFrames have the same columns
    if not real_data.columns.equals(synthetic_data.columns):
        raise ValueError("Both datasets must have the same columns")

    # Calculate proportions for each column
    for column in real_data.columns:
        real_proportion = real_data[column].mean()
        synthetic_proportion = synthetic_data[column].mean()
        
        # Calculate the absolute difference in proportions and store it
        proportion_differences[column] = abs(real_proportion - synthetic_proportion)

    return proportion_differences


def compare_data_ranges(real_data, synthetic_data, columns):
    """
    Compare the maximum values of specified features in real and synthetic datasets.

    Parameters:
        real_data (pd.DataFrame): The real dataset.
        synthetic_data (pd.DataFrame): The synthetic dataset.
        columns (list): List of column names to compare.

    Returns:
        dict: A dictionary with the "real_max_{column}" and "synthetic_max_{column}" for each specified feature.
    """
    if not all(col in real_data.columns and col in synthetic_data.columns for col in columns):
        raise ValueError("All specified columns must exist in both datasets.")
    
    range_comparison = {}

    for col in columns:
        range_comparison[f"real_max_{col}"] = real_data[col].max()
        range_comparison[f"synthetic_max_{col}"] = synthetic_data[col].max()

    return range_comparison

import pandas as pd

def descriptive_statistics_one_hot(data):
    """
    Calculate descriptive statistics for one-hot encoded categorical data.

    Parameters:
        data (pd.DataFrame): The dataset containing one-hot encoded columns.

    Returns:
        dict: A dictionary containing statistics for each one-hot encoded column.
    """
    stats_dict = {}
    for column in data.columns:
        column_data = data[column]
        
        # Calculate and store the frequency of the '1' category (presence of the category)
        frequency_1 = column_data.sum()

        # Store the proportion of '1' values (category presence)
        proportion_1 = frequency_1 / len(column_data)

        
        stats_dict['proportion_' + column] = proportion_1

    return stats_dict

def value_couts_visit(df,type_d):
    mean_count_codes_visit_rank_1 = df.groupby(['visit_rank']).count().iloc[:,0]
    #mean_count_codes_visit_rank_1 = df.groupby(['visit_rank']).count().iloc[:,0].mean()
    top_5_visits_dict = mean_count_codes_visit_rank_1[0:6].to_dict()
    modified_dict = {f"{type_d}_dataset_{key}_visit": value for key, value in top_5_visits_dict.items()}
    return modified_dict
# Compare ranges

    # Calculate the total codes per patient across all visits
 

'''
if "common_proportion" in list_metric_resemblance:
cols = [ 'Age_max', 'LOSRD_sum','LOSRD_avg',
    'L_1s_last_p1','visit_rank',
    'days_between_visits']
columns_take_account = [i for i in cols if i not in  cols]
evaluator = CommonRowsProportion()
print(evaluator.evaluate(test_ehr_dataset[columns_take_account].values, synthetic_ehr_dataset[columns_take_account].values))

X_gt = np.array([1, 2, 3])
X_syn = np.array([1.0, 2.0, 3.0])
evaluator = DataMismatchScore()
print(evaluator.evaluate(X_gt, X_syn))'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_kernel_syn(real_df, synthetic_df, col, name):
    # Set the background style
    plt.style.use("seaborn-white")

    # Define colors for real and synthetic data
    dark_blue = "#00008B"   # A dark blue for real data
    dark_green = "#006400"  # A dark green for synthetic data

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(7, 4))

    # Calculate the kernel density estimate for real and synthetic data
    real_data = real_df[col].dropna()  # Ensure no NaN values
    synthetic_data = synthetic_df[col].dropna()  # Ensure no NaN values

    # Gaussian KDE for real data
    kde_real = gaussian_kde(real_data)
    # Gaussian KDE for synthetic data
    kde_synthetic = gaussian_kde(synthetic_data)

    # Generate x-values for the kernel density plots, covering both datasets
    x_values = np.linspace(min(real_data.min(), synthetic_data.min()), max(real_data.max(), synthetic_data.max()), 100)

    # Plotting the KDE for real data
    ax.plot(x_values, kde_real(x_values), color=dark_blue, label='Real Data KDE')

    # Plotting the KDE for synthetic data
    ax.plot(x_values, kde_synthetic(x_values), color=dark_green, label='Synthetic Data KDE')

    # Set the labels and title for the plot
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
    ax.set_title(f'Kernel Density Plot - {col}')
    ax.legend()  # Add a legend to distinguish between real and synthetic data

    ax.set_ylim(bottom=0)  # Ensure the y-axis starts at zero

    # Save the plot to a file
    plt.savefig(f'results_SD/img/{col}__kernelplot.svg')
    
    # Show the plot
    plt.show()

# Example usage:
# Assuming `real_df` and `synthetic_df` are your dataframes and 'Age' is the column of interest
# plot_age(real_df, synthetic_df, 'Age', 'comparison')


def calculate_means(data, columns):
    # Check if the DataFrame is empty
    if data.empty:
        return "Empty DataFrame, no statistics available."

    # Initialize a dictionary to hold mean statistics for specified columns
    mean_stats = {}

    # Calculate mean for each specified column
    for column in columns:
        if column in data.columns:
            column_data = data[column]
            mean_stats[column] = np.mean(column_data)

    return mean_stats

import matplotlib.pyplot as plt

# Example usage:
import matplotlib.pyplot as plt
import numpy as np

def plot_means(train_ehr_dataset, synthetic_ehr_dataset, columns, column_names=None):
    # Calculate means for both datasets
    real_means = calculate_means(train_ehr_dataset, columns)
    synthetic_means = calculate_means(synthetic_ehr_dataset, columns)

    # Data for plotting
    labels = columns if column_names is None else [column_names.get(col, col) for col in columns]
    real_vals = [real_means[col] for col in columns]
    synthetic_vals = [synthetic_means[col] for col in columns]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, real_vals, width, label='Real', color='blue')
    rects2 = ax.bar(x + width/2, synthetic_vals, width, label='Synthetic', color='lightblue')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Variables')
    ax.set_ylabel('Means')
    ax.set_title('Mean Comparison between Real and Synthetic Data')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)  # Rotate x-axis labels 90 degrees
    ax.legend()

    # Function to attach a text label above each bar in *rects*, displaying its height.
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()
    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_admission_date_bar_charts(features, synthetic_ehr_dataset, col):
    # Filter the dataset for first visits
    first_visits = features[features['visit_rank'] == 1]
    first_visits_syn = synthetic_ehr_dataset[synthetic_ehr_dataset['visit_rank'] == 1]
    
    if col == 'ADMITTIME':
        # Ensure the date column is in datetime format and extract the year
        first_visits['YEAR'] = pd.to_datetime(first_visits['ADMITTIME']).dt.year
        first_visits_syn['YEAR'] = pd.to_datetime(first_visits_syn['ADMITTIME']).dt.year
        col = 'YEAR'

    # Aggregate data to get counts per year
    count_per_year_real = first_visits[col].value_counts().sort_index()
    count_per_year_synthetic = first_visits_syn[col].value_counts().sort_index()

    # Set the style
    sns.set(style="whitegrid")

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # Plot bar chart for real first visits
    sns.barplot(x=count_per_year_real.index, y=count_per_year_real.values, color="skyblue", ax=axes[0])
    axes[0].set_title('Bar Chart of ' + col.lower() + ' for Real Data First Visits')
    axes[0].set_xlabel(col.lower())
    axes[0].set_ylabel('Frequency')

    # Plot bar chart for synthetic first visits
    sns.barplot(x=count_per_year_synthetic.index, y=count_per_year_synthetic.values, color="lightgreen", ax=axes[1])
    axes[1].set_title('Bar Chart of ' + col.lower() + ' for Synthetic Data First Visits')
    axes[1].set_xlabel(col.lower())
    #axes[1].set_ylabel('Frequency')  # Optional: You can uncomment this if you want separate y-labels

    # Improve layout and plot the graph
    plt.tight_layout()
    plt.show()



def filter_cols(categorical_cols,train_ehr_dataset):        
    for i in categorical_cols:
        cols_f = train_ehr_dataset.filter(like=i, axis=1).columns
    return cols_f    
            
def plot_admission_date_histograms(features,synthetic_ehr_dataset,col):
    # Filter the dataset for first visits
    features = features[:synthetic_ehr_dataset.shape[0]]
    first_visits = features[features['visit_rank'] == 1]
    
    # Filter the dataset for subsequent visits
    first_visits_syn = synthetic_ehr_dataset[synthetic_ehr_dataset['visit_rank'] == 1]
    if col =='ADMITTIME':
        

# Ensure the date column is in datetime format
        first_visits_syn['ADMITTIME'] = pd.to_datetime(first_visits_syn['ADMITTIME'])

        # Extract the year from the datetime column
        first_visits_syn['YEAR'] = first_visits_syn['ADMITTIME'].dt.year
        first_visits['ADMITTIME'] = pd.to_datetime(first_visits['ADMITTIME'])

        # Extract the year from the datetime column
        first_visits['YEAR'] = first_visits['ADMITTIME'].dt.year
        col = 'YEAR'
    # Set the style
    sns.set(style="whitegrid")
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharey=True)
    
    # Plot histogram for first visits
    sns.histplot(first_visits[col], color="skyblue", ax=axes[0], bins=30)
    axes[0].set_title('Histogram of '+ col.lower()+' for Train Data First Visits')
    axes[0].set_xlabel(col.lower())
    axes[0].set_ylabel('Frequency')
    
    # Plot histogram for subsequent visits
    sns.histplot(first_visits_syn[col], color="skyblue", ax=axes[1], bins=30)
    axes[1].set_title('Histogram of '+ col.lower()+' for Synthetic Data First Visits')
    axes[1].set_xlabel(col.lower())
    #axes[1].set_ylabel('Frequency')
    
    # Improve layout and plot the graph
    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


from scipy.stats import wasserstein_distance

def plot_histograms_separate_axes2(real_data, synthetic_data, title, xlabel, ylabel):
    # Definir bins comunes
    max_value = max(real_data.max(), synthetic_data.max())
    bins = np.linspace(0, max_value, 31)  # 31 bins para crear 30 intervalos
    
    plt.figure(figsize=(12, 6))
    
    # Histograma para datos reales
    sns.histplot(real_data, color='blue', label='Real', bins=bins, stat='count', alpha=0.5)
    real_counts, real_bins = np.histogram(real_data, bins=bins)
    
    # Histograma para datos sintéticos
    sns.histplot(synthetic_data, color='orange', label='Synthetic', bins=bins, stat='count', alpha=0.5)
    synthetic_counts, synthetic_bins = np.histogram(synthetic_data, bins=bins)
    
    # Calcular la distancia de Wasserstein
    wd = wasserstein_distance(real_bins[:-1], synthetic_bins[:-1], u_weights=real_counts, v_weights=synthetic_counts)
    
    # Añadir el título y las etiquetas
    plt.title(f'{title}\nWasserstein Distance: {wd:.4f}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title='Data Type')
    
    # Mostrar el gráfico
    plt.show()


def plot_histograms_separate_axes3(real_data, synthetic_data, title, xlabel, ylabel):
    # Definir bins comunes
    max_value = max(real_data.max(), synthetic_data.max())
    bins = np.linspace(0, max_value, 31)  # 31 bins para crear 30 intervalos
    
    plt.figure(figsize=(12, 6))
    
    # Histograma para datos reales
    sns.histplot(real_data, color='blue', label='Test', bins=bins, stat='count', alpha=0.5)
    real_counts, real_bins = np.histogram(real_data, bins=bins)
    
    # Histograma para datos sintéticos
    sns.histplot(synthetic_data, color='orange', label='Train', bins=bins, stat='count', alpha=0.5)
    synthetic_counts, synthetic_bins = np.histogram(synthetic_data, bins=bins)
    
    # Calcular la distancia de Wasserstein
    wd = wasserstein_distance(real_bins[:-1], synthetic_bins[:-1], u_weights=real_counts, v_weights=synthetic_counts)
    
    # Añadir el título y las etiquetas
    plt.title(f'{title}\nWasserstein Distance: {wd:.4f}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title='Data Type')
    
    # Mostrar el gráfico
    plt.show()




def plot_boxplots(real_data, synthetic_data, title, xlabel, ylabel):
    # Eliminar duplicados en los índices
    real_data = real_data.loc[~real_data.index.duplicated(keep='first')]
    synthetic_data = synthetic_data.loc[~synthetic_data.index.duplicated(keep='first')]
    
    # Crear un DataFrame combinando ambos conjuntos de datos para facilitar la comparación
    data_combined = pd.DataFrame({
        'Number of Drugs': pd.concat([real_data, synthetic_data], axis=0).reset_index(drop=True),
        'Type': ['Real'] * len(real_data) + ['Synthetic'] * len(synthetic_data)
    })
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Type', y='Number of Drugs', data=data_combined)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
    
def plot_outliers(real_data, synthetic_data, column_name, title, xlabel, ylabel,label):
    # Calcular outliers para ambos conjuntos de datos
    real_outliers = calculate_outlier_ratios_tout2(real_data, column_name)
    synthetic_outliers = calculate_outlier_ratios_tout2(synthetic_data, column_name)
    
    plt.figure(figsize=(12, 6))
    
    # Plotear datos reales y sintéticos
    #sns.boxplot(data=[real_data[column_name + '_count'], synthetic_data[column_name + '_count']], palette=["blue", "orange"])
    
    # Plotear outliers reales
    plt.scatter(real_outliers.index, real_outliers[column_name + '_count'], color='red', label='Real Outliers', marker='o', alpha=0.6)
    
    # Plotear outliers sintéticos
    plt.scatter(synthetic_outliers.index, synthetic_outliers[column_name + '_count'], color='purple', label=label+' Outliers', marker='x', alpha=0.6)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# Ejemplo de uso (asegúrate de tener los datos `real_data` y `synthetic_data` preparados):
# plot_histograms_with_same_axis(real_data['number_of_drugs'], synthetic_data['number_of_drugs'], 'Histogram of Number of Drugs per Patient', 'Number of Drugs', 'Patient Count')

def calculate_outlier_ratios_tout2(real_drugs_per_patient,i):
    sorted_df = real_drugs_per_patient.sort_values(by=i+'_count')
    Q1 = sorted_df[i+'_count'].quantile(0.25)
    Q3 = sorted_df[i+'_count'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = sorted_df[(sorted_df[i+'_count'] < (Q1 - 1.5 * IQR)) | (sorted_df[i+'_count'] > (Q3 + 1.5 * IQR))]
    return outliers
def calcular_remblencemetric(test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset ,columnas_test_ehr_dataset,top_300_codes,synthetic_ehr,list_metric_resemblance):
    result_resemblence = []
    results_final={}
    
    if "hist_admi_patient_drug" in ["hist_admi_patient_drug"]:
        def plot_histograms_separate_axes22(real_data, synthetic_data, title, xlabel, ylabel,label_s):
            # Definir bins comunes con más intervalos para hacer los bins más pequeños
            max_value = max(real_data.max(), synthetic_data.max())
            bins = np.linspace(0, max_value, 101)  # 101 bins para crear 100 intervalos más pequeños
            
            plt.figure(figsize=(12, 6))
            
            # Histograma para datos reales
            sns.histplot(real_data, color='blue', label='Real', bins=bins, stat='count', alpha=0.5)
            real_counts, real_bins = np.histogram(real_data, bins=bins)
            
            # Histograma para datos sintéticos
            sns.histplot(synthetic_data, color='orange', label=label_s, bins=bins, stat='count', alpha=0.5)
            synthetic_counts, synthetic_bins = np.histogram(synthetic_data, bins=bins)
            
            # Calcular la distancia de Wasserstein
            wd = wasserstein_distance(real_bins[:-1], synthetic_bins[:-1], u_weights=real_counts, v_weights=synthetic_counts)
            
            # Añadir el título y las etiquetas
            plt.title(f'{title}\nWasserstein Distance: {wd:.4f}')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend(title='Data Type')
            
            # Mostrar el gráfico
            plt.show()

        for i in ['diagnosis', 'procedures', 'drugs'][:1]:
                # Shynthetic/train
                
                test_ehr_dataset_t = train_ehr_dataset[:test_ehr_dataset.shape[0]]
                real_drugs_per_patient_t, real_drugs_per_admission_t, real_admissions_per_drug_t,real_patients_per_drug_t = calculate_counts(train_ehr_dataset,i)
                synthetic_drugs_per_patient_t, synthetic_drugs_per_admission_t, synthetic_admissions_per_drug_t,synthetic_patients_per_drug = calculate_counts(test_ehr_dataset_t,i)
                #, 'Patient Count (Synthetic)'
                real_out_t = calculate_outlier_ratios_tout2(real_drugs_per_patient_t,i)
                syn_out_t = calculate_outlier_ratios_tout2(synthetic_drugs_per_patient_t,i)
                plot_outliers(real_drugs_per_patient_t, synthetic_drugs_per_patient_t, i,
                                'Outliers of ' +i+ ' per Patient', 'Number of ' + i, 'Patient Count (Train)','Test')
                plot_histograms_separate_axes22(real_out_t[i+'_count'], syn_out_t[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of ' + i, 'Patient Count (Train)','Test')
                
                plot_histograms_separate_axes22(real_drugs_per_patient_t[i+'_count'], synthetic_drugs_per_patient_t[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of ' + i, 'Patient Count (Train)','Test')
      
                plot_boxplots(real_drugs_per_patient_t[i+'_count'], synthetic_drugs_per_patient_t[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of ' + i, 'Patient Count (Real)')
                #, 'Patient Count (Synthetic)'
                 #, 'ADmission Count (Synthetic)'
                real_out_a_t = calculate_outlier_ratios_tout2(real_drugs_per_admission_t,i)
                syn_out_a_t = calculate_outlier_ratios_tout2(synthetic_drugs_per_admission_t,i)
                plot_outliers(real_drugs_per_admission_t, synthetic_drugs_per_admission_t, i,
                                'Outliers of ' +i+ ' per Admission', 'Number of ' + i, 'Patient Count (Real)','Test')
                plot_histograms_separate_axes22(real_out_a_t[i+'_count'], syn_out_a_t[i+'_count'], 
                                'Histogram of ' +i+ ' per Admission', 'Number of' + i, 'Patient Count (Real)','Test')
              
          
                plot_boxplots(real_drugs_per_admission_t[i+'_count'], synthetic_drugs_per_admission_t[i+'_count'], 
                                'Histogram of ' +i+ ' per Admission', 'Number of ' + i, 'Patient Count (Real)')

                plot_histograms_separate_axes22(real_drugs_per_admission_t[i+'_count'], synthetic_drugs_per_admission_t[i+ '_count'], 
                                'Histogram of ' +i+ ' per Admission', 'Number of ' + i, 'Admission Count (Real)'
                            ,'Test')
                #, 'ADmission Count per drug(train, test)'
                
                real_out_a_a_t = calculate_outlier_ratios_tout2(real_admissions_per_drug_t,"admission")
                syn_out_a_a_t = calculate_outlier_ratios_tout2(synthetic_admissions_per_drug_t,"admission")
                       
                plot_outliers(real_admissions_per_drug_t, synthetic_admissions_per_drug_t, "admission",
                                'Outliers of   Admission per ' + i, 'Number of ' + i, 'Patient Count (Real)','Test')
                plot_histograms_separate_axes22(real_out_a_a_t["admission"+'_count'], syn_out_a_a_t["admission"+'_count'], 
                                'Histogram of otliers ' + 'Admission per ' + i, 'Number of ' + i, 'Patient Count (Real)','Test')
                   

                plot_histograms_separate_axes22(real_admissions_per_drug_t['admission_count'], synthetic_admissions_per_drug_t['admission_count'], 
                                'Histogram of Admissions per ' + i, 'Number of Admissions ',  i+ ' Count (Real)','Test' )
                
                plot_boxplots(real_admissions_per_drug_t['admission_count'], synthetic_admissions_per_drug_t['admission_count'], 
                                 'Histogram of Admissions per ' +i, 'Number of Admissions ',  i+ ' Count (Real)', )

                
                #, 'Patient Count (Synthetic/train)'
                real_drugs_per_patient, real_drugs_per_admission, real_admissions_per_drug,real_patients_per_drug = calculate_counts(train_ehr_dataset,i)
                synthetic_drugs_per_patient, synthetic_drugs_per_admission, synthetic_admissions_per_drug,synthetic_patients_per_drug = calculate_counts(synthetic_ehr_dataset,i)
                #, 'Patient Count (Synthetic)'
                real_out = calculate_outlier_ratios_tout2(real_drugs_per_patient,i)
                syn_out = calculate_outlier_ratios_tout2(synthetic_drugs_per_patient,i)
                plot_outliers(real_drugs_per_patient, synthetic_drugs_per_patient, i,
                                'Outliers of ' +i+ ' per Patient', 'Number of '+ i, 'Patient Count (Real)','Synthetic')
                plot_histograms_separate_axes22(real_out[i+'_count'], syn_out[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of '+ i, 'Patient Count (Real)','Synthetic')
              
               
                plot_histograms_separate_axes22(real_drugs_per_patient[i+'_count'], synthetic_drugs_per_patient[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of '+ i, 'Patient Count (Real)','Synthetic')
                
                plot_histograms_separate_axes22(real_drugs_per_patient[i+'_count'], synthetic_drugs_per_patient[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of '+ i, 'Patient Count (Real)','Synthetic')
                
                
                plot_boxplots(real_drugs_per_patient[i+'_count'], synthetic_drugs_per_patient[i+'_count'], 
                                'Histogram of ' +i+ ' per Patient', 'Number of '+ i, 'Patient Count (Real)')

                #, 'ADmission Count (Synthetic)'
                real_out_a = calculate_outlier_ratios_tout2(real_drugs_per_admission,i)
                syn_out_a = calculate_outlier_ratios_tout2(synthetic_drugs_per_admission,i)
                plot_outliers(real_drugs_per_admission, synthetic_drugs_per_admission, i,
                                'Outliers of ' +i+ ' per Admission', 'Number of ' + i, 'Patient Count (Real)','Synthetic')
                plot_histograms_separate_axes22(real_out_a[i+'_count'], syn_out_a[i+'_count'], 
                                'Histogram of ' +i+ ' per Admission', 'Number of ' + i, 'Patient Count (Real)','Synthetic')
              
          
                plot_boxplots(real_drugs_per_admission[i+'_count'], synthetic_drugs_per_admission[i+'_count'], 
                                'Histogram of ' +i+ ' per Admission', 'Number of ' + i, 'Patient Count (Real)')

                plot_histograms_separate_axes22(real_drugs_per_admission[i+'_count'], synthetic_drugs_per_admission[i+ '_count'], 
                                'Histogram of ' +i+ ' per Admission', 'Number of ' + i, 'Admission Count (Real)','Synthetic'
                            )
                
                        #ADmission per drug #, 
                real_out_a_a = calculate_outlier_ratios_tout2(real_admissions_per_drug,"admission")
                syn_out_a_a = calculate_outlier_ratios_tout2(synthetic_admissions_per_drug,"admission")
                       
                plot_outliers(real_admissions_per_drug, synthetic_admissions_per_drug, "admission",
                                'Outliers of   Admission per ' + i, 'Number of ' + i, 'Patient Count (Real)','Synthetic')
                plot_histograms_separate_axes22(real_out_a_a["admission"+'_count'], syn_out_a_a["admission"+'_count'], 
                                'Histogram of otliers ' + 'Admission per ' + i, 'Number of ' + i, 'Patient Count (Real)','Synthetic')
                   

                plot_histograms_separate_axes22(real_admissions_per_drug['admission_count'], synthetic_admissions_per_drug['admission_count'], 
                                'Histogram of Admissions per ' + i, 'Number of Admissions ',  i+ ' Count (Real)','Synthetic' )
                
                plot_boxplots(real_admissions_per_drug['admission_count'], synthetic_admissions_per_drug['admission_count'], 
                                'Histogram of Admissions per ' +i, 'Number of Admissions ',  i+ ' Count (Real)', )
                
              
                #plot_histograms(real_patients_per_drug['patient_count'], synthetic_patients_per_drug['patient_count'], 
                #                'Histogram of Patients per '+i, 'Number of Patients', i +' Count (Real)', i +' Count (Synthetic)')
    if "outliers" in list_metric_resemblance:
        res_ratio = {}
        #diagnosis
        for word in ['diagnosis', 'procedures', 'drugs']:
            cols_sel = train_ehr_dataset.filter(like=word).columns
            res_ratio["Ratio_outlierrs_"+word] =calculate_outlier_ratios_tout(train_ehr_dataset, synthetic_ehr_dataset,cols_sel)
            
        #categorical var
        categorical_cols = ['ADMITTIME',  'RELIGION',
                                'MARITAL_STATUS',  'ETHNICITY','GENDER'] 
        cols_accounts=[]
        for i in categorical_cols:
            cols_f = train_ehr_dataset.filter(like=i, axis=1).columns
            cols_accounts.extend(list(cols_f))
            
        res_ratio["Ratio_outlierrs_categorical"] =calculate_outlier_ratios_tout(train_ehr_dataset, synthetic_ehr_dataset,cols_accounts)
        
        cols = [ 'Age_max', 'LOSRD_avg','visit_rank','days_between_visits']
        res_ratio["Ratio_outlierrs_numerical"] =calculate_outlier_ratios_tout(train_ehr_dataset, synthetic_ehr_dataset,cols)


    
    if "Record_lengh" in list_metric_resemblance:
        statistics = get_statistics(train_ehr_dataset,columnas_test_ehr_dataset,test_ehr_dataset,synthetic_ehr,synthetic_ehr_dataset)
        result_resemblence.append(statistics)
    if "plt_hist_first_visits" in list_metric_resemblance:
        #plots the firs visit histograms
        cols_accounts = []
        categorical_cols = ['ADMITTIME',  'RELIGION',
                        'MARITAL_STATUS',  'ETHNICITY','GENDER'] 
        for i in categorical_cols:
            cols_f = train_ehr_dataset.filter(like=i, axis=1).columns
            cols_accounts.extend(list(cols_f))
        for i in   cols_accounts:  
             plot_admission_date_histograms(train_ehr_dataset,synthetic_ehr_dataset,i)
             plot_admission_date_bar_charts(train_ehr_dataset,synthetic_ehr_dataset,i)
             
    if "cols_plot_mean" in list_metric_resemblance:
        cols = [ 'Age_max', 'LOSRD_sum','visit_rank','days_between_visits']
        # Assuming real_df and synthetic_df are your dataframes    
        plot_means(train_ehr_dataset, synthetic_ehr_dataset, cols)

    if "plot_kernel_vssyn" in list_metric_resemblance:
        cols = [ 'Age_max', 'LOSRD_sum','LOSRD_avg','L_1s_last_p1','visit_rank','days_between_visits']
        for i in cols: 
            plot_kernel_syn(train_ehr_dataset, synthetic_ehr_dataset, i, "Marginal_distribution")
        
    if "distance_max" in list_metric_resemblance:
        # Compare using Frobenius norm
        
        train_test = compare_matrices(test_ehr_dataset.values, train_ehr_dataset.values, method="max_diff")
        train_synthethic = compare_matrices(synthetic_ehr_dataset.values, train_ehr_dataset.values, method="max_diff")
        test_synthethic = compare_matrices(synthetic_ehr_dataset.values, test_ehr_dataset.values, method="max_diff")    
        result = {}
        result["MaxAbsDif_train_test"] = train_test
        result["MaxAbsDif_train_synthethic"] = train_synthethic
        result["MaxAbsDif_test_synthethic"] = test_synthethic
        result_resemblence.append(result)
    if "coorr_matrix_abs_dif" in list_metric_resemblance:
        result_sy = multivariate_statistics(synthetic_ehr_dataset.values)
        result_train = multivariate_statistics(train_ehr_dataset.values)
        result_test = multivariate_statistics(test_ehr_dataset.values) 
        # Printing results
        train_test = compare_matrices(result_test['correlation_matrix'], result_train['correlation_matrix'], method="max_diff")
        train_synthethic = compare_matrices(result_train['correlation_matrix'], result_sy['correlation_matrix'], method="max_diff")
        test_synthethic = compare_matrices(result_test['correlation_matrix'], result_sy['correlation_matrix'], method="max_diff")    
        mean_diff_train_test = np.mean(train_test)
        max_diff_train_test = max(train_test)
        min_diff_train_test = min(train_test)
        std_diff_train_test = np.std(train_test)
        mean_diff_syn_test = np.mean(test_synthethic)
        max_diff_syn_test = max(test_synthethic)
        min_diff_syn_test = min(test_synthethic)
        std_diff_syn_test = np.std(test_synthethic)
        result = {}
        result["MaxAbsDif_test_train_corr_mean"] = mean_diff_train_test
        result["MaxAbsDif_test_train_synthethic_corr_max"] = max_diff_train_test
        result["MaxAbsDif_test_train_corr_min"] = min_diff_train_test
        result["MaxAbsDif_test_synthethic_corr_mean"] = mean_diff_syn_test
        result["MaxAbsDif_test_synthethic_corr_max"] = max_diff_syn_test
        result["MaxAbsDif_test_synthethic_corr_min"] = min_diff_syn_test
        result_resemblence.append(result)
    if "descriptive_statistics" in list_metric_resemblance:
        cols = ['Age_max', 'LOSRD_sum','LOSRD_avg','L_1s_last_p1','visit_rank','days_between_visits']
        result = descriptive_statistics_matrix(synthetic_ehr_dataset[cols])
        result1 = descriptive_statistics_matrix(train_ehr_dataset[cols])
        
        #result = descriptive_statistics_matrix(test_ehr_dataset[cols])
        result_resemblence.append(result)
        data = pd.concat([pd.DataFrame(result,index = [0]),pd.DataFrame(result1,index = [0])])
        x1 = data[[ 'mean_visit_rank', 
        'mode_visit_rank', 'minimum_visit_rank', 'maximum_visit_rank',
        'range_visit_rank', 'mean_days_between_visits',
        'mode_days_between_visits',
        'minimum_days_between_visits', 'maximum_days_between_visits' ]].to_latex()
        print(x1)


    if "frequency_categorical_10" in list_metric_resemblance:
        cols = ['Age_max', 'LOSRD_sum','LOSRD_avg','id_patient','L_1s_last_p1','days_between_visits']
        cols_ = ['ADMITTIME','HADM_ID']
     
        train_ehr_dataset_auz = cols_todrop(train_ehr_dataset,cols_)
        synthetic_ehr_dataset_auz = cols_todrop(synthetic_ehr_dataset,cols_)
  
        columns_take_account = [i for i in train_ehr_dataset_auz.columns if i not in cols]
        res = compare_proportions_and_return_dictionary(train_ehr_dataset_auz[columns_take_account], synthetic_ehr_dataset_auz[columns_take_account])    
        top_10_diff_proportions = dict(sorted(res.items(), key=lambda item: item[1], reverse=True)[:10])
        result_resemblence.append(top_10_diff_proportions)

    
    if "visit_counts2" in list_metric_resemblance:
        df = train_ehr_dataset[columnas_test_ehr_dataset+["visit_rank"]]
        type_d = "train"
        res = df.groupby(["visit_rank"]).count().iloc[:,0].head().to_dict()
        res = {f"{key} test set": value for key, value in res.items()}

        df = synthetic_ehr_dataset[columnas_test_ehr_dataset+["visit_rank"]]
        type_d = "synthetic"
        res2 = df.groupby(["visit_rank"]).count().iloc[:,0].head().to_dict()
        res2 = {f"{key} synthetic set": value for key, value in res2.items()}
        aux_s = pd.concat([pd.DataFrame(res,index = [0]),pd.DataFrame(res2,index = [0])],axis = 1)
        la = aux_s.to_latex()
        print(la)
        res.update(res2)
         
        result_resemblence.append(res)
    if "corr" in list_metric_resemblance:

        def heatmap_diff_corr(df1, df2):
            # Calculate correlation matrices
            corr_matrix1 = df1.corr()
            corr_matrix2 = df2.corr()

            # Calculate the absolute difference of the correlation matrices
            diff_corr_matrix = corr_matrix1 - corr_matrix2
            abs_diff_corr_matrix = np.abs(diff_corr_matrix)

            # Create a mask to find the top 10 differences
            # Flatten the matrix, sort the absolute values, and get the top 10
            k = 10  # Number of top elements to highlight
            indices = np.unravel_index(np.argsort(-abs_diff_corr_matrix.values, axis=None)[:k], abs_diff_corr_matrix.shape)
            top_diff_mask = np.zeros_like(diff_corr_matrix, dtype=bool)
            top_diff_mask[indices] = True

            # Names of the columns (assuming they are the same for both matrices)
            column_names = list(df1.columns)

            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create a heatmap for the matrix of differences
            sns.heatmap(diff_corr_matrix, cmap='coolwarm', center=0, ax=ax)

            # Configure the axis labels
           
                # Only show labels for the rows with top 10 highest differences
            #y_labels = [column_names[i] if top_diff_mask[i].any() else '' for i in range(len(column_names))]
            
            #ax.set_yticks(np.arange(len(column_names)))  # Set y-tick positions
            #ax.set_yticklabels(y_labels, rotation=45)  # Set y-tick labels
            
            
            fontdict = {'fontsize': 8, 'fontweight': 'bold', 'fontname': 'Arial'}

            # Configure the axis labels
            y_labels = [column_names[i] if top_diff_mask[i].any() else '' for i in range(len(column_names))]
            ax.set_yticks(np.arange(len(column_names)))  # Set y-tick positions
            ax.set_yticklabels(y_labels, rotation=45, fontdict=fontdict)  # Set y-tick labels with font properties
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            #ax.set_xticklabels(column_names, rotation=45, horizontalalignment='right')
        

            # Show and save the graph
            plt.title('Heatmap of Correlation Matrix Differences')
            plt.savefig('difference_heatmap.svg')
            plt.show()
            y_labels = [item for item in y_labels if item != '']

            return y_labels
    # Example usage:
    # heatmap_diff_corr(df1, df2)

            
        def correlacion_total(synthetic_ehr_dataset):
            # Supongamos que 'df' es tu DataFrame y que 'corr_matrix' es tu matriz de correlación
            # Supongamos que 'df' es tu DataFrame
            # Supongamos que 'df' es tu DataFrame y que 'corr_matrix' es tu matriz de correlación
            percentage = .9
            threshold = 0.7
            corr_matrix = synthetic_ehr_dataset.corr()

            # Crea una copia de la matriz de correlación para modificarla
            corr_matrix_mod = corr_matrix.copy()

            # Establece la diagonal a NaN para que no se tenga en cuenta la correlación de una columna consigo misma
            np.fill_diagonal(corr_matrix_mod.values, np.nan)

            # Calcula el número de columnas que representan el 97% de la matriz de correlación
            num_cols = int(percentage * len(corr_matrix_mod.columns))

            # Encuentra las columnas donde al menos el 97% de las correlaciones son mayores a 0.9
            cols_with_high_corr = corr_matrix_mod.columns[(corr_matrix_mod.abs() > threshold).sum() ]
            col =  set(list(cols_with_high_corr))
            # Imprime los nombres de las columnas
            for col in cols_with_high_corr:
                print(col)
                
            cols_with_all_nan = corr_matrix.columns[corr_matrix.isna().all()]

            # Imprime los nombres de las columnas
            for col in cols_with_all_nan:
                print(col)
                print(synthetic_ehr_dataset[col].sum())   
                
            return cols_with_high_corr, cols_with_all_nan   # Ahora 'cols_with_high_corr' contiene las columnas que tienen una correlación mayor a 0.97 con al menos una otra columna
        
        corr_plot(synthetic_ehr_dataset,"Syn" )
        corr_plot(train_ehr_dataset,"Train" )  
        cols = [ 'Age_max' ,'LOSRD_avg',
        'visit_rank',
        'days_between_visits']
        syn_c = corr_plot(synthetic_ehr_dataset[cols],"Syn" )
        real_c = corr_plot(train_ehr_dataset[cols],"Train" )
        heatmap_digbet_corr(syn_c,real_c)
        cols_list = []
        categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                        'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                        'MARITAL_STATUS',  'ETHNICITY','GENDER',"visit_rank","HOSPITAL_EXPIRE_FLAG"  ]
        for i in categorical_cols:
            cols_f = train_ehr_dataset.filter(like=i, axis=1).columns
            cols_list.extend(list(cols_f))
        syn = corr_plot(synthetic_ehr_dataset[cols_list],"Syn" )
        cols_with_high_corr = correlacion_otra_col(synthetic_ehr_dataset[cols_list])
        real = corr_plot(train_ehr_dataset[cols_list],"Train" )
        #diferencia de correlationes
        heatmap_diff_corr(syn,real)
        correlacion_otra_col(synthetic_ehr_dataset[cols_list])
        keywords = ['diagnosis', 'procedures', 'drugs']
        for i in keywords:
            col_prod = [col for col in train_ehr_dataset.columns if any(palabra in col for palabra in [i])]
            syn1 = corr_plot(synthetic_ehr_dataset[col_prod],"Syn" )
            #cols_with_high_corr, cols_with_all_nan =correlacion_total(synthetic_ehr_dataset[col_prod])
            real1 =corr_plot(train_ehr_dataset[col_prod],"Train" )    
            #cols_with_high_corr, cols_with_all_nan = correlacion_total(train_ehr_dataset[col_prod])
            y_labels = heatmap_diff_corr(syn1,real1)
            print(y_labels)
        
        
    if "pacampa" in list_metric_resemblance:
        import numpy as np
        import matplotlib.pyplot as plt
        from pacmap import PaCMAP
        
        def PACMAP_PLOT(col_prod,synthetic_ehr_dataset,train_ehr_dataset,i,save=False):
            # Assuming data_real and data_synthetic are already defined and 'col_prod' is a valid column or set of columns
            data_real = train_ehr_dataset[col_prod]
            data_synthetic = synthetic_ehr_dataset[col_prod]
            data_combined = np.vstack([data_real, data_synthetic])
            # Create labels (0 for real, 1 for synthetic)
            labels = np.array([0] * len(data_real) + [1] * len(data_synthetic))
            # Initialize PaCMAP with desired parameters
            pacmap_instance = PaCMAP(n_components=2, n_neighbors=30, MN_ratio=0.5, FP_ratio=2.0)
            # Fit and transform the data
            embedding = pacmap_instance.fit_transform(data_combined)
            # Plotting
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, c=labels, cmap='viridis')
            plt.title('Plot of 2D Data Points '+i)
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.grid(True)
            # Create a colorbar with a label
            cbar = plt.colorbar(scatter)
            cbar.set_label('Data Type')
            cbar.set_ticks([0.25, 0.75])  # Set tick positions based on your label distribution
            cbar.set_ticklabels(['Real', 'Synthetic'])
            # Define the filename dynamically if needed or use a fixed filename
            filename = 'Pacmap.png'
            if save:
               plt.savefig(f'results_SD/img/{filename}')  # Save the plot before displaying it
            # Display the plot
            plt.show()
            
        keywords = ['diagnosis', 'procedures', 'drugs']
        for i in keywords[:2]:
            col_prod = [col for col in train_ehr_dataset.columns if any(palabra in col for palabra in [i])]
            PACMAP_PLOT(col_prod,synthetic_ehr_dataset,train_ehr_dataset,i,save=False)
        cols_list = []
        categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                        'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                        'MARITAL_STATUS',  'ETHNICITY','GENDER',"visit_rank","HOSPITAL_EXPIRE_FLAG"  ]
        for i in categorical_cols:
            cols_f = train_ehr_dataset.filter(like=i, axis=1).columns
            cols_list.extend(list(cols_f))
        PACMAP_PLOT(cols_list,synthetic_ehr_dataset,train_ehr_dataset,"Demographic and admission",save=False)    

        cols = [ 'Age_max', 'LOSRD_sum','LOSRD_avg',
        'visit_rank',
        'days_between_visits']
        PACMAP_PLOT(cols_list,synthetic_ehr_dataset,train_ehr_dataset,"Continuos variables",save=False)    
    if "temporal_time_line" in list_metric_resemblance:
                
        #synthetic_ehr_dataset['days_between_visits_bins'] = pd.qcut(synthetic_ehr_dataset['days_between_visits'], q=10)
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np

        def plot_heatmap(synthetic_ehr_dataset, name, col, cols_num=1, cols_prod="None", type_c="Synthetic"):
            # Assuming synthetic_ehr_dataset is preloaded and contains 'visit_rank'
            
            synthetic_ehr_dataset.sort_values(by='visit_rank', ascending=True, inplace=True)
            
            # Aggregate data to count occurrences per 'visit_rank'
            if cols_num == 1:
                # Create categories if specified column is 'Age_max' or 'days_between_visits'
                if col in ['Age_max', 'days_between_visits']:
                    age_intervals = synthetic_ehr_dataset[col].unique()
                    synthetic_ehr_dataset[col] = pd.Categorical(synthetic_ehr_dataset[col], categories=age_intervals, ordered=True)

                # Generate heatmap data
                heatmap_data = synthetic_ehr_dataset.groupby(['visit_rank', col]).size().unstack(fill_value=0)

                # Configuring the heatmap
                plt.figure(figsize=(12, 8))
                ax = sns.heatmap(heatmap_data, cmap='viridis', annot=False)
                
                ax.set_title(f'{name} by visit ({type_c})')
                ax.set_xlabel(name)
                ax.set_ylabel('Visit rank')

                # Adjusting the color scale dynamically
                vmin = heatmap_data.min().min()  # find the minimum value in the dataframe
                vmax = heatmap_data.max().max()  # find the maximum value in the dataframe
                ax = sns.heatmap(heatmap_data, cmap='viridis', annot=False, vmin=vmin, vmax=vmax)

                # Set labels
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                plt.xticks(rotation=45)
                plt.show()

            else:
                # For multiple columns, sum data and sort
                heatmap_data = synthetic_ehr_dataset[cols_prod + ["visit_rank"]].groupby('visit_rank').sum()
                top_drugs = heatmap_data.head(10).index.tolist()
                heatmap_data.sort_values(by='visit_rank', ascending=False, inplace=True)
                
                plt.figure(figsize=(12, 8))
                ax = sns.heatmap(heatmap_data, cmap='viridis', annot=False)
                ax.set_title(f'{name} by Year ({type_c})')
                ax.set_xlabel(name)
                ax.set_ylabel('Year')
                
                # Generate labels, showing only the top items
                labels = [label if label in top_drugs else '' for label in heatmap_data.columns]
                ax.set_xticks(np.arange(len(labels)))
                ax.set_xticklabels(labels, rotation=45)
                plt.show()

# Example usage:
# plot_heatmap_(dataframe, 'Example Name', 'Age_max')

        def plot_heatmap(synthetic_ehr_dataset, name,col, cols_num=1,cols_prod="None",type_c="Synthetic"):
            #synthetic_ehr_dataset["visit_rank"] = pd.to_datetime(synthetic_ehr_dataset["visit_rank"])
            #synthetic_ehr_dataset['year'] = synthetic_ehr_dataset['visit_rank'].dt.year
            synthetic_ehr_dataset.sort_values(by = 'visit_rank', ascending=True, inplace=True)  # Sort by year
            # Aggregate data to count ICD-9 codes occurrences per year  
            if cols_num == 1:
                if col == 'Age_max' or col == 'days_between_visits':
                    age_intervals  = synthetic_ehr_dataset[col].unique()
                    synthetic_ehr_dataset[col]= pd.Categorical(synthetic_ehr_dataset[col], categories=age_intervals, ordered=True)
                heatmap_data = synthetic_ehr_dataset.groupby(['visit_rank', col]).size().unstack(fill_value=0)
                heatmap_data.sort_values(by = 'visit_rank', ascending=False, inplace=True)  # Sort by year
                plt.figure(figsize=(12, 8))
                ax = sns.heatmap(heatmap_data, cmap='viridis', annot=False)
                ax.set_title(name +' by visit (' + type_c+')')
                ax.set_xlabel(name)
                ax.set_ylabel('Visit rank')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                plt.xticks(rotation=0)
                plt.show()

            else:
                heatmap_data = synthetic_ehr_dataset[cols_prod+["visit_rank"]].groupby(col).sum()   
                heatmap_data.sort_values('visit_rank',ascending=False, inplace=True) 
                top_drugs = heatmap_data.head(10).index.tolist()
                heatmap_data.sort_values(by ='visit_rank', ascending=False, inplace=True)  # Sort by year
                plt.figure(figsize=(12, 8))
                ax = sns.heatmap(heatmap_data, cmap='viridis', annot=False)
                ax = sns.heatmap(heatmap_data, cmap='viridis', annot=False, vmin=your_data_min, vmax=your_data_max)

                ax.set_title(name +' by Year (' + type_c+')')
                ax.set_xlabel(name)
                ax.set_ylabel('Year')
                labels = [label if label in top_drugs else '' for label in heatmap_data.columns]
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels)
                #ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                plt.xticks(rotation=0)
                plt.show()
        def hist_d(col,synthetic_ehr_dataset):
            plt.figure(figsize=(12, 8))
            print(synthetic_ehr_dataset[col].describe())
            ax = synthetic_ehr_dataset[col].hist(bins=30)
            ax.set_xlabel(col)
            plt.show()
            
   
        def plot_heatmap_(synthetic_ehr_dataset, name, col,name2):
            # Calculate the correlation or any necessary transformation
            # Example: We assume synthetic_ehr_dataset is already loaded and filtered appropriately
            
            # Check the range of the data
            heatmap_data = synthetic_ehr_dataset.groupby(['visit_rank', col]).size().unstack(fill_value=0)
            heatmap_data.sort_values('visit_rank',ascending=False, inplace=True) 
            plt.figure(figsize=(10, 8))
            sns.heatmap(heatmap_data, annot=True, cmap='viridis')
            plt.title('Heatmap of '+name+' by Visit Rank '+  name2+' data')
            plt.xlabel(name)
            plt.ylabel('Visit Rank')
            plt.show()
# Adjust parameters according to your specific dataset structure and requirements
    
            
        def hist_betw_a(original_data,synthetic_data,col):
            #original_data = pd.read_csv('original_data.csv')  # Adjust the file path as necessary
            #synthetic_data = pd.read_csv('synthetic_data.csv')  # Adjust the file path as necessary

            # Assume 'days_between_visits' column exists, otherwise calculate it
            # Plotting the histograms
            plt.figure(figsize=(12, 6))

            # Histogram for original data
            plt.hist(original_data[col], bins=50, alpha=0.3, label='Original', color='blue')

            # Histogram for synthetic data
            plt.hist(synthetic_data[col], bins=50, alpha=0.3, label='Synthetic', color='red')

            # Adding titles and labels
            plt.title('Comparison of Days '+col+' Original vs Synthetic Data')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.legend()

            # Show the plot
            plt.show()
       
            
        def box_pltos(df,df_syn,col):
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            axes[0].boxplot(df[col])
            axes[0].set_title('Real ' +col )
            axes[0].set_ylabel('Days')
            axes[1].boxplot(df_syn[col])
            axes[1].set_title('Synthetic ' +col)
            axes[1].set_ylabel('Days')
            plt.tight_layout()
            plt.show()   
        # Rotar las etiquetas del eje y 90 grados
        name = "Days between visits"
        col =   'days_between_visits_bins'
        synthetic_ehr_dataset['days_between_visits_bins'] = pd.qcut(synthetic_ehr_dataset['days_between_visits'], q=10, duplicates='drop')
        hist_d("days_between_visits",synthetic_ehr_dataset) 
        plot_heatmap_(synthetic_ehr_dataset, name,col,"Synthetic")
        train_ehr_dataset['days_between_visits_bins'] = pd.qcut(train_ehr_dataset['days_between_visits'], q=10, duplicates='drop')
        hist_d("days_between_visits",train_ehr_dataset) 
        hist_betw_a(train_ehr_dataset,synthetic_ehr_dataset,"days_between_visits")
        hist_betw_a(train_ehr_dataset,synthetic_ehr_dataset,"id_patient")
        
        plot_heatmap_(train_ehr_dataset, name,col, "Real")
        # age
        name = "Age interval"
        col =   'Age_max_bins'
        synthetic_ehr_dataset['Age_max_bins'] = pd.qcut(synthetic_ehr_dataset['Age_max'], q=5, duplicates='drop')
        hist_d("Age_max",synthetic_ehr_dataset) 
        plot_heatmap_(synthetic_ehr_dataset, name,col, "Synthetic")
        train_ehr_dataset['Age_max_bins'] = pd.qcut(train_ehr_dataset['Age_max'], q=5, duplicates='drop')
        plot_heatmap_(train_ehr_dataset, name,col,"Real")
        hist_d("days_between_visits",train_ehr_dataset) 
        
        
        # Histograms
        box_pltos(train_ehr_dataset,synthetic_ehr_dataset,'days_between_visits')
        box_pltos(train_ehr_dataset,synthetic_ehr_dataset,'Age_max')
        keywords = ['diagnosis', 'procedures', 'drugs']
        for i in keywords:
            col_prod = [col for col in train_ehr_dataset.columns if any(palabra in col for palabra in [i])]
            plot_heatmap(train_ehr_dataset, i,i, 2,col_prod,type_c="Real")
            plot_heatmap(synthetic_ehr_dataset, i,i, 2,col_prod,type_c="Synthetic")
        

    
    if "compare_ranges" in list_metric_resemblance:
        cols = [ 'Age_max', 'LOSRD_sum','LOSRD_avg',
        'L_1s_last_p1','visit_rank',
        'days_between_visits']
        cols = [ 'Age_max', 'LOSRD_sum','LOSRD_avg','L_1s_last_p1','visit_rank','days_between_visits']
            
        result = compare_data_ranges(test_ehr_dataset, synthetic_ehr_dataset,cols)

        print("Feature Ranges:\n", result)
        result_resemblence.append(result        )
    if "dimension_wise":
        def dimensio_wise(real_samples,gen_samples):
            #gen_samples = np.load(os.path.join(opt.expPATH, "synthetic.npy"), allow_pickle=False)

            # Load real data
            #real_samples = dataset_train_object.return_data()[0:gen_samples.shape[0], :]

            # Dimenstion wise probability
            # It appears that the code `prob_real` is a variable name in Python. The code itself does
            # not provide enough information to determine what it is doing. If you provide more
            # context or code, I can help you understand its purpose.
            prob_real = np.mean(real_samples, axis=0)
            prob_syn = np.mean(gen_samples, axis=0)

            p1 = plt.scatter(prob_real, prob_syn, c="b", alpha=0.5, label="ARF")
            x_max = max(np.max(prob_real), np.max(prob_syn))
            x = np.linspace(0, x_max + 0.1, 1000)
            p2 = plt.plot(x, x, linestyle='-', color='k', label="Ideal")  # solid
            plt.tick_params(labelsize=12)
            plt.legend(loc=2, prop={'size': 15})
            # plt.title('Scatter plot p')
            # plt.xlabel('x')
            # plt.ylabel('y')
            plt.show()
        gen_samples = synthetic_ehr_dataset.select_dtypes(include=['number'])    
        real_samples = train_ehr_dataset.select_dtypes(include=['number']) 
        cols_to_drop = ["id_patient","HADM_ID","days_between_visits","Age_max","LOSRD_avg"]
        gen_samples.drop(columns = cols_to_drop,inplace = True)
        real_samples.drop(columns = cols_to_drop, inplace = True)
        dimensio_wise(real_samples,gen_samples)
        
        import matplotlib.pyplot as plt
        from scipy.stats import wasserstein_distance
    if "prevalence_wise":
        # Función para calcular la prevalencia de características binarias
        def calculate_binary_prevalence(df):
            binary_df = (df > 0).astype(int)
            prevalences = binary_df.mean()
            return prevalences

        # Función para calcular la diferencia media absoluta para variables de conteo
        def calculate_relative_frequencies(df):
            frequencies = df.sum() / df.sum().sum()
            return frequencies

        def calculate_mean_absolute_difference(real_frequencies, synthetic_frequencies):
            mad = np.mean(abs(real_frequencies - synthetic_frequencies))
            return mad

        # Función para normalizar características continuas al rango [0,1]
        def normalize_features(df):
            normalized_df = (df - df.min()) / (df.max() - df.min())
            return normalized_df

        # Función para calcular AWD para características continuas
        def calculate_awd(real_df, synthetic_df):
            awd = np.mean([wasserstein_distance(real_df[col], synthetic_df[col]) for col in real_df.columns])
            return awd

        def remove_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            filter = (df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)
            return df.loc[filter]
        # Calcular APD para características binarias
        binary_features = cols_categorical +['readmission','HOSPITAL_EXPIRE_FLAG']
        real_binary_prevalences = calculate_binary_prevalence(train_ehr_dataset[binary_features])
        synthetic_binary_prevalences = calculate_binary_prevalence(synthetic_ehr_dataset[binary_features])
        apd_binary = calculate_mean_absolute_difference(real_binary_prevalences, synthetic_binary_prevalences)

        # Calcular MAD para variables de conteo
        count_features = cols_diagnosis.to_list() +cols_drugs.to_list()+cols_procedures.to_list() +['visit_rank']
        real_count_frequencies = calculate_relative_frequencies(train_ehr_dataset[count_features])
        synthetic_count_frequencies = calculate_relative_frequencies(synthetic_ehr_dataset[count_features])
        mad_count = calculate_mean_absolute_difference(real_count_frequencies, synthetic_count_frequencies)

        # Calcular AWD para variables continuas
        continuous_features = cols_continuous
        normalized_real_data = normalize_features(train_ehr_dataset[continuous_features])
        normalized_synthetic_data = normalize_features(synthetic_ehr_dataset[continuous_features])
        awd_continuous = calculate_awd(normalized_real_data, normalized_synthetic_data)

        # Crear DataFrame combinado para el gráfico
        plot_data = pd.DataFrame({
            'Real': pd.concat([real_binary_prevalences, real_count_frequencies, normalized_real_data.mean()]),
            'Synthetic': pd.concat([synthetic_binary_prevalences, synthetic_count_frequencies, normalized_synthetic_data.mean()]),
            'Feature': ['Binary'] * len(real_binary_prevalences) + ['Count'] * len(real_count_frequencies) + ['Continuous'] * len(normalized_real_data.columns)
        })

        plot_data = remove_outliers(plot_data, 'Real')
        plot_data = remove_outliers(plot_data, 'Synthetic')
        index_to_drop = ["id_patient","HADM_ID","days_between_visits","Age_max","LOSRD_avg"]
        
        plot_data = plot_data.drop(["days_between_visits", "Age_max", "LOSRD_avg"])

        
        
        # Encontrar los límites máximos para ajustar los ejes
        max_real = plot_data['Real'].max()
        max_synthetic = plot_data['Synthetic'].max()
        axis_limit = max(max_real, max_synthetic) * 1.1  # añadir un 10% de margen

        # Crear el gráfico de dispersión combinado
        plt.figure(figsize=(10, 10))
        sns.scatterplot(data=plot_data, x='Real', y='Synthetic', hue='Feature', style='Feature', s=100)

        # Añadir línea de referencia diagonal
        plt.plot([0, 1], [0, 1], ls="--", c=".3")

        # Añadir los valores de APD, MAD y AWD al gráfico
        plt.text(0.05, axis_limit * 0.95, f'APD (Binary) = {apd_binary:.2f}', horizontalalignment='left', size='medium', color='black', weight='semibold')
        plt.text(0.05, axis_limit * 0.90, f'MAD (Count) = {mad_count:.2f}', horizontalalignment='left', size='medium', color='black', weight='semibold')
        plt.text(0.05, axis_limit * 0.85, f'AWD (Continuous) = {awd_continuous:.2f}', horizontalalignment='left', size='medium', color='black', weight='semibold')

        # Ajustar los límites de los ejes
        plt.xlim(0, axis_limit)
        plt.ylim(0, axis_limit)

        # Añadir títulos y etiquetas
        plt.title('Comparison of Feature Prevalence in Real and Synthetic Data')
        plt.xlabel('Real Prevalence')
        plt.ylabel('Synthetic Prevalence')
        plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        
        
        import numpy as np

    if "diference_decriptives" in list_metric_resemblance:
        cols = [ 'Age_max', 'LOSRD_sum','LOSRD_avg',
        'L_1s_last_p1','visit_rank',
        'days_between_visits']
    
        dif =  compare_descriptive_statistics(test_ehr_dataset[cols], synthetic_ehr_dataset[cols])
        top_10_differences = sorted(dif.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_10_dict = {item[0]: item[1] for item in top_10_differences}
        result_resemblence.append(top_10_dict)
        
        
    if "mmd" in list_metric_resemblance:
        mmd_evaluator = MaximumMeanDiscrepancy(kernel="rbf")
        train_test = "test"
        cols = ['ADMITTIME','HADM_ID']
        #cols = "days_between_visits_cumsum"
        train_ehr_dataset_a = cols_todrop(train_ehr_dataset,cols)
        #train_ehr_dataset_a = cols_todrop(train_ehr_dataset,cols)
        synthetic_ehr_a = cols_todrop(synthetic_ehr_dataset,cols)
        result =   mmd_evaluator._evaluate(train_ehr_dataset_a.iloc[:,:synthetic_ehr_a.shape[1]], synthetic_ehr_a)
        print("MaximumMeanDiscrepancy (flattened):", result)
        result_resemblence.append(result)

    # Example usage:
    # X_gt and X_syn are two numpy arrays representing empirical distributions
    if "ks_test" in list_metric_resemblance:
        
        continuous_cols = ['Age_max', 'LOSRD_avg','days_between_visits']        
        ks_results = {'column': [], 'ks_statistic': [], 'p_value': []}

        for col in continuous_cols:
            ks_stat, p_value = ks_2samp(train_ehr_dataset[col], synthetic_ehr_dataset[col])
            ks_results['column'].append(col)
            ks_results['ks_statistic'].append(ks_stat)
            ks_results['p_value'].append(p_value)

        # Convert results to DataFrame
        ks_df = pd.DataFrame(ks_results)

        # Calculate summary statistics
        summary_stats = {
            'mean_ks_statistic': ks_df['ks_statistic'].mean(),
            'median_ks_statistic': ks_df['ks_statistic'].median(),
            'std_ks_statistic': ks_df['ks_statistic'].std(),
            'mean_p_value': ks_df['p_value'].mean(),
            'median_p_value': ks_df['p_value'].median(),
            'std_p_value': ks_df['p_value'].std(),
            'proportion_significant': (ks_df['p_value'] < 0.05).mean()
        }

        summary_df = pd.DataFrame([summary_stats])

        print(ks_df.to_latex())
        from scipy.stats import chi2_contingency
        cols_not_con = [i for i in train_ehr_dataset.columns if i not in continuous_cols+['id_patient','ADMITTIME']]
        real_data = train_ehr_dataset
        synthetic_data = synthetic_ehr_dataset
        # Generate example one-hot encoded data for illustration purposes

        # Perform Chi-Square test for each column and calculate Total Variation Distance
        chi_results = {'column': [], 'chi2_statistic': [], 'p_value': [], 'tv_distance': []}

        for col in cols_not_con:
            contingency_table = pd.crosstab(real_data[col], synthetic_data[col])
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
            
            # Calculate Total Variation Distance
            real_prob = real_data[col].value_counts(normalize=True)
            synthetic_prob = synthetic_data[col].value_counts(normalize=True)
            tv_distance = np.sum(np.abs(real_prob - synthetic_prob)) / 2
            
            chi_results['column'].append(col)
            chi_results['chi2_statistic'].append(chi2_stat)
            chi_results['p_value'].append(p_value)
            chi_results['tv_distance'].append(tv_distance)

        # Convert results to DataFrame
        chi_df = pd.DataFrame(chi_results)
        # Identify the top 10 most different and least different columns by p-value


        most_different_cols = chi_df.nsmallest(10, 'p_value')
        least_different_cols = chi_df.nlargest(10, 'p_value')
        most_different_cols_tv = chi_df.nlargest(10, 'tv_distance')
        least_different_cols_tv = chi_df.nsmallest(10, 'tv_distance')


        # Create the final DataFrame
        final_df = pd.DataFrame({
            'Most Different by p-value': most_different_cols['column'].tolist() + [''] * (10 - len(most_different_cols)),
            'Least Different by p-value': least_different_cols['column'].tolist() + [''] * (10 - len(least_different_cols)),
            'Most Different by TV Distance': most_different_cols_tv['column'].tolist() + [''] * (10 - len(most_different_cols_tv)),
            'Least Different by TV Distance': least_different_cols_tv['column'].tolist() + [''] * (10 - len(least_different_cols_tv)),
        })

        # Display the DataFrame
        print(final_df.to_latex())
        # Combine the results
            

    if "jensenshannon_dist" in list_metric_resemblance:
        cols = ['ADMITTIME','Age_max', 'LOSRD_avg','days_between_visits']
        #cols = "days_between_visits_cumsum"
        test_ehr_dataset_a = cols_todrop(train_ehr_dataset,cols)
        #train_ehr_dataset_a = cols_todrop(train_ehr_dataset,cols)
        synthetic_ehr_dataset_a = cols_todrop(synthetic_ehr_dataset,cols)
        test_ehr_dataset_a2 = cols_todrop(test_ehr_dataset,cols)
        train_ehr_dataset_a2 = test_ehr_dataset_a[:test_ehr_dataset_a.shape[0]]
        jsd_calculator = JensenShannonDistance2(test_ehr_dataset_a, synthetic_ehr_dataset_a,10)
        jsd_results, avg_js = jsd_calculator.jensen_shannon()
        #print(jsd_results.to_latex())
        #print(avg_js)
        #jsd_calculator_test = JensenShannonDistance2(test_ehr_dataset_a2, train_ehr_dataset_a2,10)
        #jsd_results_test, avg_js_test = jsd_calculator_test.jensen_shannon()
        #data_fram_train = pd.DataFrame()
        #data_fram_train["Jensen Shannon avg train /synthetic"] = avg_js
        #data_fram_train["Jensen Shannon avg train /test"] = avg_js_test
        #print(data_fram_train.to_latex())
       
        #score = JensenShannonDistance()._evaluate(features_2d, synthetic_features_2d)synthetic_ehr
        print("Jensen-Shannon Distance:", score)
        
        
        jsd_calculator = JensenShannonDistance2(test_ehr_dataset_a, synthetic_ehr_dataset_a,10)
        jsd_results, avg_js = jsd_calculator.jensen_shannon()

        



    # Example usage with two dataframes, `X_gt_df` and `X_syn_df`

    if "hist_mixed" in      list_metric_resemblance:
        # The code snippet provided is written in Python and it seems to be part of a script or
        # program. Here is a breakdown of what the code is doing:
        keywords = ['diagnosis', 'procedures', 'drugs']
        for i in keywords:
                col_prod = [col for col in train_ehr_dataset.columns if any(palabra in col for palabra in [i])]
                name = file[10:]
                type_procedur = i
                plot_hist_emp_codes(col_prod, train_ehr_dataset, synthetic_ehr, type_procedur, name)
                
    
    if "proportion_demos" in  list_metric_resemblance:
        prop_datafram = []
        categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                        'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                        'MARITAL_STATUS',  'ETHNICITY','GENDER',"visit_rank","HOSPITAL_EXPIRE_FLAG"  ]
        for i in categorical_cols:
            cols_f = train_ehr_dataset.filter(like=i, axis=1).columns
            tabla_proporciones_final = get_proportions(train_ehr_dataset[cols_f],"train "+ i)
            tabla_proporciones_final_2= get_proportions(synthetic_ehr_dataset[cols_f],"synthetic"+ i)
            
            total = pd.merge(tabla_proporciones_final,tabla_proporciones_final_2,  on=["Variable","Category "], how='inner') 
            #total = total['Variable','Category ]+test_ehr_dataset[cols_f],"test "+ i]
            prop_datafram.append(total)
            latex_code = total.to_latex( index = "Variable")
            print(latex_code)
    
    if "exact_match" in list_metric_resemblance:
        df1_sorted = synthetic_ehr_dataset.sort_values(by=synthetic_ehr_dataset.columns.tolist()).reset_index(drop=True)  
        df2_sorted = train_ehr_dataset.sort_values(by=train_ehr_dataset.columns.tolist()).reset_index(drop=True)
        exact_matches = (df1_sorted == df2_sorted).all(axis=1).sum()
        print("Number of exact matching rows:", exact_matches)
        result_resemblence.append({"exact_matches" : exact_matches})               
    if "duplicates" in list_metric_resemblance:
        
        combined = pd.concat([train_ehr_dataset, synthetic_ehr_dataset])
        synthetic_ehr_dataset['is_duplicate'] = combined.duplicated(keep=False).iloc[len(train_ehr_dataset):]

        print("Test data with duplicate flag:")
        value_counts = synthetic_ehr_dataset['is_duplicate'].value_counts()
        new_dict = {"Duplicates - " + str(key): value for key, value in value_counts.items()}
        result_resemblence.append(new_dict)               
        
    if "compare_least_codes" in list_metric_resemblance:
        
        least_cols =  obtain_least_frequent(train_ehr_dataset, columnas_test_ehr_dataset, 50)
        

# Columns to analyze
        columns_to_analyze =least_cols

        # Function to calculate proportions for a given column in a DataFrame
        def calculate_proportions(df, column):
            frequency = df[column].value_counts(normalize=True)
            return frequency

        # Step 3: Create a DataFrame for plotting
        plot_data = {}

        for column in columns_to_analyze:
            real_props = calculate_proportions(train_ehr_dataset, column)
            synth_props = calculate_proportions(synthetic_ehr_dataset, column)
            
            # Prepare data for plotting
            for category in set(real_props.index).union(set(synth_props.index)):
                real_prop = real_props.get(category, 0)
                synth_prop = synth_props.get(category, 0)
                plot_data[(column, category)] = [real_prop, synth_prop]

            # Convert plot data into a DataFrame
            plot_df = pd.DataFrame.from_dict(plot_data, orient='index', columns=['Real', 'Synthetic']).reset_index()
            plot_df_filtered = plot_df[plot_df['index'].apply(lambda x: x[1] == 1)]

        # Imprime el DataFrame filtrado

            latex_table = plot_df_filtered.to_latex(index=False)
            print(latex_table)
            
    if "gradico_acum" in list_metric_resemblance:   
        def fun_grafico_compa(col,train_ehr_dataset):
            cols =    filter_cols([col],train_ehr_dataset) 
            df = pd.DataFrame(train_ehr_dataset[cols])
            # Obtener el nombre de la columna con valor máximo (one-hot) para cada fila
            df['category'] = df.apply(lambda row: row.idxmax(), axis=1)
            # Convertir nobres de categorías a etiquetas numéricas
            mapping_dict = {value: i for i, value in enumerate(df['category'].unique())}
            # Asigna las etiquetas a una nueva columna 'category_label'
            # Imprime el diccionario de mapeo
            aux_df =train_ehr_dataset
            aux_df['category'] = df['category']
            aux_df['visit_count'] = 1
            # Agrupar por fecha y categoría y sumar visitas
            grouped = aux_df.groupby(['ADMITTIME', 'category']).sum().groupby(level='category').cumsum()
            # Calcular el total acumulado para cada categoría al final del período
            total_per_category = grouped.groupby('category')['visit_count'].transform('max')
            # Calcular la proporción
            grouped['proportion'] = grouped['visit_count'] / total_per_category
            # Resetear el índice para facilitar el plotting
            grouped = grouped.reset_index()
            # Asegurarse de que los colores se asignan correctamente
            colors = plt.cm.viridis(np.linspace(0, 1, grouped['category'].nunique()))
            categories = grouped['category'].unique()
            # Crea un diccionario que mapee cada categoría a un color
            color_dict = dict(zip(categories, colors))
            # Imprime el diccionario de colores
            print(color_dict)
            return color_dict,grouped
        
        import matplotlib.pyplot as plt

        def plot_admission_date_bar_charts(color_dict, color_dict_, grouped, grouped_synthetic, col, save=False):
            # Create the figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 6), sharey=True)
            # Unique categories in the real data
            categories_real = grouped['category'].unique()
            categories_synthetic = grouped_synthetic['category'].unique()
            # Plot cumulative proportions for each category in the first subplot
            for (category), data in grouped.groupby('category'):
                ax1.plot(data['ADMITTIME'], data['visit_count'], label=f'Category {category}', color=color_dict.get(category, 'gray'))
            # Configuration of the first graph
            ax1.set_title('Cumulative Visits by Category - Real Data (' + col + ')')
            ax1.set_xlabel('Admission Date')
            ax1.set_ylabel('Cumulative of Visits')
            if len(categories_real) <= 10:
                ax1.legend(title='Category')
            # Plot cumulative proportions for each category in the second subplot, using the synthetic DataFrame
            for (category), data in grouped_synthetic.groupby('category'):
                ax2.plot(data['ADMITTIME'], data['visit_count'], label=f'Category {category}', color=color_dict_.get(category, 'gray'))
            # Configuration of the second graph
            ax2.set_title('Cumulative Visit Proportions by Category - Synthetic Data (' + col + ')')
            ax2.set_xlabel('Admission Date')
            if len(categories_synthetic) <= 10:
                ax2.legend(title='Category')
            fig.autofmt_xdate()  # Automatically format the dates to improve visualization
            # Show the graph
            plt.show()
            
            if save:
                plt.savefig(f'results_SD/img/{col}__cumulative_counts_each_visit.svg')

               
 
            
        def categorilca_cols_fun(train_ehr_dataset,synthetic_ehr_dataset,categorical_cols,save=False):
            for i in categorical_cols:    
                    color_dict,grouped = fun_grafico_compa(i,train_ehr_dataset)
                    color_dict_,grouped_synthetic = fun_grafico_compa(i,synthetic_ehr_dataset)
                    plot_admission_date_bar_charts(color_dict,color_dict_,grouped,grouped_synthetic,i,save = False)                
        categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                    'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                    'MARITAL_STATUS',  'ETHNICITY','GENDER',"visit_rank","HOSPITAL_EXPIRE_FLAG"  ]
        categorilca_cols_fun(train_ehr_dataset,synthetic_ehr_dataset,categorical_cols,save=False)
        cols = [ 'Age_max', 'LOSRD_sum','LOSRD_avg','id_patient','L_1s_last_p1','days_between_visits']
        categorilca_cols_fun(train_ehr_dataset,synthetic_ehr_dataset,cols,save=False)
        keywords = ['diagnosis', 'procedures', 'drugs']
        categorilca_cols_fun(train_ehr_dataset,synthetic_ehr_dataset,keywords,save=False)
    if "common_proportoins" in list_metric_resemblance:
        cp = CommonRowsProportion()
        dict_s = cp.evaluate(train_ehr_dataset, synthetic_ehr_dataset)
        result_resemblence.append(dict_s)
    for i in result_resemblence:
        results_final.update(i)  

    return results_final
# Save the statistics to a file

def obtain_dataset_admission_visit_rank(path_to_directory,file,valid_perc,features_path):
    features = load_data(features_path)
    total_features_synthethic = pd.read_csv(file) 
    cols_unnamed = total_features_synthethic.filter(like='Unnamed', axis=1).columns
    total_features_synthethic.drop(cols_unnamed, axis=1, inplace=True)
   
    #split_dataset
    N = features.shape[0]
    N_train = int(N * (1 - valid_perc))
    N_valid = N - N_train
    total_features_train = features[:N_train]
    total_fetura_valid = features[N_train:]   
    total_features_synthethic = total_features_synthethic[:N_train]
    total_features_train   = total_features_train.rename(columns={"SUBJECT_ID":"id_patient"})
    total_features_synthethic   = total_features_synthethic.rename(columns={"SUBJECT_ID":"id_patient"})
    total_fetura_valid   = total_fetura_valid.rename(columns={"SUBJECT_ID":"id_patient"})
    train_ehr_dataset = obtain_readmission_realdata(total_features_train)
    test_ehr_dataset = obtain_readmission_realdata(total_fetura_valid) 
    #obtener readmission
    total_features_synthethic['ADMITTIME'] = pd.to_datetime(total_features_synthethic['ADMITTIME'])
    total_features_synthethic = total_features_synthethic.sort_values(by=['id_patient', 'ADMITTIME'])
    # Ordena el DataFrame por 'patient_id' y 'visit_date' para garantizar el ranking correcto
    # Agrupa por 'patient_id' y asigna un rango a cada visita
    total_features_synthethic['visit_rank'] = total_features_synthethic.groupby('id_patient')['ADMITTIME'].rank(method='first').astype(int)
    synthetic_ehr_dataset = obtain_readmission_realdata(total_features_synthethic) 
    return  test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset,features

def cols_todrop(total_features_synthethic,cols):
    cols_to_drop = list(total_features_synthethic.filter(like='Unnamed', axis=1).columns) + cols
    total_features_synthethic.drop(cols_to_drop, axis=1, inplace=True) 
    return total_features_synthethic

if __name__=="__main__":
    import wandb
    attributes = "False"
    if attributes == "True":    
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
        
       

     

    # Definir la ruta al directorio que contiene los archivos CSV
    path_to_directory = 'generated_synthcity_tabular/*'  # Asegúrate de incluir el asterisco al final
    csv_files = glob.glob(path_to_directory + '.pkl')
    #file = csv_files[0]
    #"file = 'generated_synthcity_tabular/marginal_distributionstotal_0.2_epochs.pkl'
    #valid_perc = 0.2
    #'generated_synthcity_tabular/arftotal_0.2_epochs.pkl', 'generated_synthcity_tabular/tvaetotal_100_epochs.pkl', 'generated_synthcity_tabular/ctgantotal_0.2_epochs.pkl',  'generated_synthcity_tabular/tvae_100_epochs.pkl', 'generated_synthcity_tabular/rtvaetotal_0.2_epochs.pkl', 'generated_synthcity_tabular/ctgan_100_epochs_.pkl',
    #csv_files = [ 'generated_synthcity_tabular/marginal_distributionstotal_0.2_epochs.pkl', 'generated_synthcity_tabular/ctgan_100_epochs.pkl', 'generated_synthcity_tabular/tvae_100_epochs_midi.pkl', 'generated_synthcity_tabular/marginal_distributions_100_epochs.pkl', 'generated_synthcity_tabular/nflow_100_epochs.pkl']
    
    csv_files = ['generated_synthcity_tabular/adsgantotal_0.2_epochs.pkl','generated_synthcity_tabular/pategantotal_0.2_epochs.pkl']
    valid_perc = 0.2
    results_df = pd.DataFrame()
    #file = 'generated_synthcity_tabular/arftotal_0.2_epochs.pkl'
    file = 'generated_synthcity_tabular/arftotal_best_parans_0.2_epochs.pkl'
    for file in csv_files:
        print(file)
    
        if file == "generated_synthcity_tabular/ctgan_100_epochs.pkl" or file == "generated_synthcity_tabular/tvae_100_epochs_midi.pkl":    
            trained_w = 0.5
        else:    
            trained = 0.2 
        config_w = {
            "model": file,
            "trained":0.2,
            "valid_perc" :0.5
        
            }
            
        
        #wandb.init(project='SD_generation2',config=config_w)
        features_path = "data/intermedi/SD/inpput/entire_ceros_tabular_data.pkl"

        test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset,features = obtain_dataset_admission_visit_rank(file,file,valid_perc,features_path)
        #list_metric_resemblance = ["histogramas","tsne","statistics","kernel_density","mmd","ks_test","jensenshannon_dist"]
        # contraints
        #from # The code you provided is a comment in Python. Comments in Python start with a hash
        # symbol (#) and are used to provide explanations or notes within the code. In this case,
        # the comment "generative_model" is likely indicating the purpose or description of the
        # code that follows.
        from generative_model.SD.constraints import *
        c = EHRDataConstraints(train_ehr_dataset, test_ehr_dataset, synthetic_ehr_dataset)
        c.print_shapes()
        #cols_accounts = c.handle_categorical_data()
        synthetic_ehr_dataset = c.initiate_processing()
        c.print_shapes()
        
        
        unique_synthetic_patients = synthetic_ehr_dataset['id_patient'].nunique()
        unique_train_patients = train_ehr_dataset['id_patient'].unique()
        selected_train_patients = np.random.choice(unique_train_patients, size=unique_synthetic_patients, replace=False)
        train_ehr_dataset = train_ehr_dataset[train_ehr_dataset['id_patient'].isin(selected_train_patients)]
        
        train_ehr_dataset = train_ehr_dataset[ :synthetic_ehr_dataset.shape[0]]
        cols = "days_between_visits_cumsum"
        synthetic_ehr_dataset = cols_todrop(synthetic_ehr_dataset,[cols])
        
              
        print(test_ehr_dataset.shape)
        print(train_ehr_dataset.shape)
        print(synthetic_ehr_dataset.shape)
        print(train_ehr_dataset['id_patient'].nunique())
        print(synthetic_ehr_dataset['id_patient'].nunique())
        
        
        #train_ehr_dataset= test_ehr_dataset 
        #cols = ['ADMITTIME','HADM_ID']
        #test_ehr_dataset = cols_todrop(test_ehr_dataset,cols)
        #train_ehr_dataset = cols_todrop(train_ehr_dataset,cols)
        #synthetic_ehr_dataset = cols_todrop(synthetic_ehr_dataset,cols)
        
  
        
        
        columnas_test_ehr_dataset = get_cols_diag_proc_drug(train_ehr_dataset)

        #obtener n mas frequent codes
        top_300_codes = obtain_most_freuent(train_ehr_dataset,columnas_test_ehr_dataset,100)

        #obtener un syntethic datafram que considere el percentil y si es mayor a eso se considera 1 si no 0, si es false no se le agrega la columnas id_patient
        
        columns_to_drop = ['LOSRD_sum', 'L_1s_last_p1','HADM_ID']

        if all(column in synthetic_ehr_dataset.columns for column in columns_to_drop):
            synthetic_ehr_dataset.drop(columns_to_drop, axis=1, inplace=True)
            train_ehr_dataset.drop(columns_to_drop, axis=1, inplace=True)
            test_ehr_dataset.drop(columns_to_drop, axis=1, inplace=True) 
            
        print(test_ehr_dataset.shape)
        print(train_ehr_dataset.shape)
        print(synthetic_ehr_dataset.shape)
        
        #obtener cols para demosgraphics, contnious, procedures, diagnosis, drugs
        cols_continuous = ['Age_max', 'LOSRD_avg','days_between_visits'] 

        categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                                'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                                'MARITAL_STATUS',  'ETHNICITY','GENDER',"visit_rank","HOSPITAL_EXPIRE_FLAG"  ]
        for i in categorical_cols:
            cols_categorical = train_ehr_dataset.filter(like=i, axis=1).columns
        keywords = ['diagnosis', 'procedures', 'drugs']
        cols_diagnosis = train_ehr_dataset.filter(like= 'diagnosis', axis=1).columns
        cols_procedures = train_ehr_dataset.filter(like= 'procedures', axis=1).columns
        cols_drugs = train_ehr_dataset.filter(like= 'drugs', axis=1).columns
        
        #train and yn
        list_metric_resemblance =["plt_hist_first_visits","cols_plot_mean","plot_kernel_vssyn",
        "descriptive_statistics","frequency_categorical_10","visit_counts2","corr","pacampa","temporal_time_line","ks_test",
        "jensenshannon_dist","hist_mixed","proportion_demos", "exact_match","duplicates","compare_least_codes" ,"gradico_acum",
        "common_proportoins","Record_lengh"]
        results_  =    calcular_remblencemetric(test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset ,columnas_test_ehr_dataset,top_300_codes,synthetic_ehr,list_metric_resemblance)
       
        #same value_matrix test,train and syn
        synthetic_ehr = synthetic_ehr_dataset
        list_metric_resemblance =["distance_max","coorr_matrix_abs_dif","compare_ranges","mmd"]
        synthetic_ehr = synthetic_ehr[:test_ehr_dataset.shape[0]]
        synthetic_ehr_dataset = synthetic_ehr_dataset[:test_ehr_dataset.shape[0]]
        train_ehr_dataset = train_ehr_dataset[:test_ehr_dataset.shape[0]]
        print(test_ehr_dataset.shape)
        print(train_ehr_dataset.shape)
        print(synthetic_ehr_dataset.shape)
        results  =    calcular_remblencemetric(test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset ,columnas_test_ehr_dataset,top_300_codes,synthetic_ehr,list_metric_resemblance)
       
  
       
        results_.update(results)
        results_["Model"] = file
        result_df = pd.DataFrame([results_]) 
        result_df = pd.DataFrame([results_]) 
        results_df = pd.concat([results_df,result_df], ignore_index=True)
        results_df.to_csv('generated_synthcity_tabular/model_tabular_results.csv', index=False)
        #wandb.log( results_)
        #wandb.finish()
        #print(results_)
    #wandb.finish()    
#import wandb
#wandb.init(project="test_project")
## Log a dummy variable
#wandb.log({"dummy": 1})
#wandb.finish()