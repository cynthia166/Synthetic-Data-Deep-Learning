import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from matplotlib.colors import LogNorm  # Add this import
from matplotlib.colors import LinearSegmentedColormap

#import polars as pl
import sys
sys.path.append('')
sys.path.append('preprocessing')
from preprocessing.config import *
import glob
import numpy as np
import pandas as pd
#import plotly.express as px
import os
import glob
#import psycopg2
import datetime
import sys
from operator import itemgetter, attrgetter, methodcaller
import numpy as np
import itertools
import os.path
import matplotlib.pyplot as plt
import math
from multiprocessing import Pool, cpu_count
import polars as pl

#import plotly.express as px
import glob
import pandas as pd
#from icdmappings import Mapper
import pandas as pd
from pathlib import Path
#import plotly.express as px

import glob



#import psycopg2
import datetime
import sys
from operator import itemgetter, attrgetter, methodcaller
import numpy as np
import itertools
import os.path
import matplotlib.pyplot as plt
import math
from multiprocessing import Pool, cpu_count

#from utils import getConnection
from xml.dom.pulldom import ErrorHandler
import pandas as pd
#import dill
import numpy as np
from collections import defaultdict
#from rdkit import Chem
#from rdkit.Chem import BRICS
from config import *
import pickle

from icdmappings import Mapper

def load_data(file_path):
    try:
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        with open(file_path, 'rb') as f:
             data = pickle.load(f)    
             return data
def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)



import pandas as pd

def analyze_icu_and_drugs(df):
    # Ensure the DataFrame is in pandas format
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    # ICU stays per subject_id and hadm_id (unique)
    icu_per_subject_hadm_unique = df.groupby(['SUBJECT_ID', 'HADM_ID'])['ICUSTAY_ID'].nunique().rename('UNIQUE_ICU_STAYS')
   
    # ICU stays per subject_id and hadm_id

    icu_per_subject_hadm = df.groupby(['SUBJECT_ID', 'HADM_ID'])['ICUSTAY_ID'].count().rename('ICU_STAYS')
    
    # ICU stays per subject_id
    icu_per_subject_repeated = df.groupby('SUBJECT_ID')['ICUSTAY_ID'].nunique().rename('ICU_STAYS_SUBJECT')
    
    # Drugs per subject_id and hadm_id
    drugs_per_subject_hadm = df.groupby(['SUBJECT_ID', 'HADM_ID'])['DRUG'].nunique().rename('DRUGS__ham_COUNT')
    
    # Drugs per subject_id
    drugs_per_subject = df.groupby('SUBJECT_ID')['DRUG'].nunique().rename('DRUG_COUNT')

    # NDC per subject_id
    ndc_per_subject = df.groupby('SUBJECT_ID')['NDC'].nunique().rename('UNIQUE_NDC')

    # NDC per admission (hadm_id)
    ndc_subject_hadm = df.groupby(['SUBJECT_ID', 'HADM_ID'])['NDC'].nunique().rename('NDC_HADM')

    # NDC frequency
    ndc_frequency = df['NDC'].value_counts().rename('NDC_COUNT')
    drugs_count = df['DRUG'].value_counts().rename('DRUGS_COUNT')

    # Calculate statistics
    stats = {
        'ICU_repeated_stays_per_admission': icu_per_subject_hadm.describe(),
        'UNIQUE_ICU_STAYS': icu_per_subject_hadm_unique.describe(),
        'ICU_stays_per_subject': icu_per_subject_repeated.describe(),
        'Drugs_per_admission': drugs_per_subject_hadm.describe(),
        'Drugs_per_subject': drugs_per_subject.describe(),
        'DRUG': drugs_count.describe(),
        'NDC_per_subject': ndc_per_subject.describe(),
        'NDC_admission': ndc_subject_hadm.describe(),
        'NDC_frequency': ndc_frequency.describe()
    }

    # Create DataFrame from stats
    stats_df = pd.DataFrame(stats).transpose()
    print(stats_df)
    return stats_df
from collections import Counter
def analyze_rxcui_repetition(ndc2RXCUI):
    # Count occurrences of each RXCUI
    rxcui_counts = Counter(ndc2RXCUI.values())
    
    # Calculate the average repetition
    total_rxcui = len(rxcui_counts)
    total_ndc = len(ndc2RXCUI)
    average_repetition = total_ndc / total_rxcui if total_rxcui > 0 else 0

    print(f"Total unique RXCUI values: {total_rxcui}")
    print(f"Total NDC codes: {total_ndc}")
    print(f"Average repetition of RXCUI: {average_repetition:.2f}")

    # Additional statistics
    max_repetition = max(rxcui_counts.values()) if rxcui_counts else 0
    min_repetition = min(rxcui_counts.values()) if rxcui_counts else 0
    
    print(f"Maximum repetition of an RXCUI: {max_repetition}")
    print(f"Minimum repetition of an RXCUI: {min_repetition}")

    # Distribution of repetitions
    repetition_distribution = Counter(rxcui_counts.values())
    print("\nDistribution of RXCUI repetitions:")
    for rep, count in sorted(repetition_distribution.items()):
        print(f"RXCUI repeated {rep} time(s): {count} occurrences")

def med_process(med_file):
    med_pd = pd.read_csv(med_file, dtype={"NDC": "category"})
    #Initial shape of the medication file: (4156450, 19)
    print("Initial shape of the medication file:", med_pd.shape)
    stats_df = analyze_icu_and_drugs(med_pd)

    # Print the statistics
    print(stats_df)

    # Save to CSV
    stats_df.to_csv('data/analysis/drugs/icu_drug_and_ndc_statistics.csv')
    
    #                            count        mean          std  min   25%   50%     75%       max
    # ICU_stays_per_admission  50216.0   53.941811    59.090568  0.0  18.0  37.0   70.00    1151.0
    # ICU_stays_per_subject    39363.0   68.814420    84.853971  0.0  20.0  44.0   85.00    1509.0
    # Drugs_per_admission      50216.0   82.771427    75.012547  1.0  35.0  65.0  105.00    1400.0
    # Drugs_per_subject        39363.0  105.592816   120.689700  1.0  37.0  74.0  128.00    2378.0
    # DRUG                      4525.0  918.552486  6994.611320  1.0   1.0   4.0   50.00  192993.0
    # NDC_per_subject          39363.0   45.437772    31.162148  0.0  24.0  41.0   60.00     346.0
    # NDC_admission            50216.0   39.384061    22.489691  0.0  23.0  36.0   53.00     214.0
    # NDC_frequency             4204.0  987.627735  9729.619360  1.0   8.0  45.0  312.25  586586.0
    
    print("Processing medication file:", med_pd.shape)
    # med_pd.drop(columns=['ROW_ID','DRUG_TYPE','DRUG_NAME_POE','DRUG_NAME_GENERIC',
    #                     'FORMULARY_DRUG_CD','PROD_STRENGTH','DOSE_VAL_RX',
    #                     'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP', 'GSN', 'FORM_UNIT_DISP',
    #                     'ROUTE','ENDDATE','DRUG'], axis=1, inplace=True)
    med_pd.drop(
        columns=[
            "ROW_ID",
            "DRUG_TYPE",
            "DRUG_NAME_POE",
            "DRUG_NAME_GENERIC",
            "FORMULARY_DRUG_CD",
            "PROD_STRENGTH",
            "DOSE_VAL_RX",
            "DOSE_UNIT_RX",
            "FORM_VAL_DISP",
            "FORM_UNIT_DISP",
            "GSN",
            "FORM_UNIT_DISP",
            "ROUTE",
            "ENDDATE",
        ],
        axis=1,
        inplace=True,
    )
    print("Medication file after dropping columns:", med_pd.shape)
    print("Unique values in NDC:", med_pd["NDC"].nunique()
          )
    print(''' Cols eliminated:
            "ROW_ID",
            "DRUG_TYPE",
            "DRUG_NAME_POE",
            "DRUG_NAME_GENERIC",
            "FORMULARY_DRUG_CD",
            "PROD_STRENGTH",
            "DOSE_VAL_RX",
            "DOSE_UNIT_RX",
            "FORM_VAL_DISP",
            "FORM_UNIT_DISP",
            "GSN",
            "FORM_UNIT_DISP",
            "ROUTE",
            "ENDDATE"''')
    #med_pd.drop(index=med_pd[med_pd["NDC"] == "0"].index, axis=0, inplace=True)
    #print("Filling NaN values with previous values")
    print(med_pd.isnull().sum())
    #there are 1447708 nan values in ICUSTAY_ID, there are 3182 nan values in STARTDATE and 0 nan values in SUBJECT_ID and HADM_ID
    #SUBJECT_ID          0
    #HADM_ID             0
    #ICUSTAY_ID    1447708
    #STARTDATE        3182
    #DRUG                0
    #NDC              4463
    #med_pd.fillna(method="pad", inplace=True)
    med_pd.dropna(subset=["STARTDATE"], inplace=True)
    med_pd.drop_duplicates(inplace=True)
    print("After dropping duplicates:", med_pd.shape) #After dropping duplicates: (3414381, 6)
    #med_pd["ICUSTAY_ID"] = med_pd["ICUSTAY_ID"].astype("int64")
    med_pd["STARTDATE"] = pd.to_datetime(
        med_pd["STARTDATE"], format="%Y-%m-%d %H:%M:%S"
    )
    med_pd.sort_values(
        by=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE"], inplace=True
    )
    med_pd = med_pd.reset_index(drop=True)
    print("After sorting and resetting index:", med_pd.shape)
    med_pd = med_pd.drop(columns=["ICUSTAY_ID"])
    med_pd = med_pd.drop_duplicates()
    
    #columnas consideradas ara drop si en icu stay hay una droga con mismo ndc ya solo aparece una ve ya que nc colmn nose toma en cuenta
    #SUBJECT_ID          0
    #HADM_ID             0 
    #STARTDATE        3182
    #DRUG                0
    #NDC              4463
    
    med_pd = med_pd.reset_index(drop=True)
    print("After dropping ICUSTAY_ID:", med_pd.shape)

    return med_pd

def  codeMapping2atc4(med_pd):
    RXCUI2atc4_file =RAW/"suplement/RXCUI2atc4.csv"
    ndc2RXCUI_file=RAW/"suplement/ndc2RXCUI.txt"
    #read ndc2RXCUI hay 4204 ndc unique values
    with open(ndc2RXCUI_file, "r") as f:
        ndc2RXCUI = eval(f.read())
    #med_pd["RXCUI"] = med_pd["NDC"].map(ndc2RXCUI)
    analyze_rxcui_repetition(ndc2RXCUI)
    #     Total unique RXCUI values: 3055
    # Total NDC codes: 4204
    # Average repetition of RXCUI: 1.38
    # Maximum repetition of an RXCUI: 20
    # Minimum repetition of an RXCUI: 1

    # Distribution of RXCUI repetitions:
    # RXCUI repeated 1 time(s): 2212 occurrences
    # RXCUI repeated 2 time(s): 631 occurrences
    # RXCUI repeated 3 time(s): 157 occurrences
    # RXCUI repeated 4 time(s): 43 occurrences
    # RXCUI repeated 5 time(s): 5 occurrences
    # RXCUI repeated 6 time(s): 5 occurrences
    # RXCUI repeated 12 time(s): 1 occurrences
    # RXCUI repeated 20 time(s): 1 occurrences

    print("med_pd shape before mapping RXCUI",med_pd.shape)
    #3055 rxcui unique values
    med_pd["RXCUI"] = med_pd["NDC"].apply(lambda x: ndc2RXCUI.get(x, -1))
    print("Númber of -1 not mappes "+str("RXCUI"),med_pd["RXCUI"],(med_pd["RXCUI" ]==-1).value_counts()[True])
    print("nulos", med_pd["RXCUI"].isnull().sum())
    print("med_pd shape after mapping RXCUI",med_pd.shape)
    #    Númber of -1 not mappes RXCUI 0          1870676/3416228 = 0.5475852 percent
    med_pd.dropna(inplace=True)
    print("med_pd shape after drop na",med_pd.shape)
    #med_pd shape after mapping RXCUI (3416228, 6)
    RXCUI2atc4 = pd.read_csv(RXCUI2atc4_file)
    RXCUI2atc4 = RXCUI2atc4.drop(columns=["YEAR", "MONTH", "NDC"])
    
    RXCUI2atc4.drop_duplicates(subset=["RXCUI"], inplace=True)
    print("med_pd shape after drop duplicates RXCUI",med_pd.shape)
    atc4_counts = RXCUI2atc4.groupby("RXCUI")["ATC4"].nunique()
    multiple_atc4 = atc4_counts[atc4_counts > 1]

    if not multiple_atc4.empty:
        print("The following RXCUI have multiple ATC4 values:")
        print(multiple_atc4)
    else:
        print("No RXCUI have multiple ATC4 values.")
        
    #med_pd shape after drop duplicates RXCUI (3416228, 6)
    print("the percentage of ndc that map to a rxcui is: ", med_pd["RXCUI"].nunique()/med_pd["NDC"].nunique())
    
    med_pd.drop(index=med_pd[med_pd["RXCUI"].isin([""])].index, axis=0, inplace=True)
    #med_pd shape after reset index (3400412, 6) drop(index=med_pd[med_pd["RXCUI"].isin([""])]
    med_pd["RXCUI"] = med_pd["RXCUI"].astype("int64")
    med_pd = med_pd.reset_index(drop=True)
    print("med_pd shape after reset index",med_pd.shape)
    med_pd = med_pd.merge(RXCUI2atc4, on=["RXCUI"])
    #med_pd y RXCUI2atc4 donde los valores de la columna RXCUI coincidan en ambos DataFrames. 
    #0.6639691411616281 of the ndc codes are mapped to a rxcui
    print("med_pd shape after merge RXCUI2atc4",med_pd.shape,"the percentage of ndc that map to a rxcui is: ", med_pd.shape[0]/3410671)
    #
    #med_pd.drop(columns=["NDC", "RXCUI"], inplace=True)
    med_pd["ATC3"] = med_pd["ATC4"].map(lambda x: x[:4])
    #med_pd = med_pd.rename(columns={"ATC4": "ATC3"})
    #med_pd["ATC4"] = med_pd["ATC4"].map(lambda x: x)
    print("med_pd shape after mapping atc4",med_pd.shape)
    med_pd = med_pd.drop_duplicates()
    #med_pd shape after drop duplicates (2261453, 8)
    print("med_pd shape after drop duplicates",med_pd.shape)
    #After dropping duplicates: (3448013, 6)
    #After sorting and resetting index: (3448013, 6)
    #After dropping ICUSTAY_ID: (3416228, 5)
    med_pd = med_pd.reset_index(drop=True)

    # Check for multiple ATC4 values per RXCUI
    atc4_counts = RXCUI2atc4.groupby("RXCUI")["ATC4"].nunique()
    multiple_atc4 = atc4_counts[atc4_counts > 1]

    if not multiple_atc4.empty:
        print("The following RXCUI have multiple ATC4 values:")
        print(multiple_atc4)
    else:
        print("No RXCUI have multiple ATC4 values.")

    return med_pd


def drug2(d1):
    med_pd = med_process(d1)
    #Processing medication file: (4156450, 19)
    print("med_pd initial shape"  ,med_pd.shape)
    RXCUI2atc4_file = "data/RXCUI2atc4.csv"
    ndc2RXCUI_file="data/ndc2RXCUI.txt"
    
    med_pd = codeMapping2atc4(med_pd)
    print("med_pd shape before drop duplicates",med_pd.shape)
    print("med columnes",med_pd.columns)
    # #Dropping duplicates Index(['SUBJECT_ID', 'HADM_ID', 'STARTDATE', 'DRUG', 'NDC', 'RXCUI', 'ATC4',  'ATC3'],
    print("Dropping duplicates",med_pd.columns,"subset=['SUBJECT_ID', 'HADM_ID','DRUG', 'ATC3','ATC4','NDC','STARTDATE']")
    #eliminatin dupliaetr thso there can be the same drug and ieven if they have different star data if they have subjser column it would be the same
    med_pd.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID','DRUG', 'ATC3','ATC4','NDC','STARTDATE'],inplace=True)
    
    print("med_pd shape after drop duplicates",med_pd.shape)
   
    #med_pd shape after drop duplicates (2261453, 8)
    for i in med_pd.columns:    
        print("unique"+str(i),med_pd[i].nunique() )
    #     uniqueSUBJECT_ID 39087
    # uniqueHADM_ID 49929
    # uniqueSTARTDATE 38265
    # uniqueDRUG 2541
    # uniqueNDC 2929
    # uniqueRXCUI 2037
    # uniqueATC4 348
    # uniqueATC3 155
    print("med_pd after atc3 shape : ",med_pd.shape)
    #med_pd after atc3 shape :  (1452205, 8)
    # Supongamos que df es tu DataFrame
    med_pd["SUBJECT_ID"] = med_pd["SUBJECT_ID"].astype(str)
    med_pd["HADM_ID"] = med_pd["HADM_ID"].astype(str)
    # does taht are not able to be maepd
    print("med_p values after mapping ",med_pd.isnull().sum())
    #no null values
    med_pd = med_pd.fillna(-1)
    
    print("med_pd initial shape: ",med_pd.isnull().sum())
    
    for i in ['SUBJECT_ID', 'HADM_ID', 'ATC3','ATC4','NDC']:    
            print("unique "+str(i),med_pd[i].nunique() )
            try:
               print("Númber of -1 not mappes "+str(i),med_pd[i],(med_pd[i ]==-1).value_counts()[True])
            except:
                print("No number of -1")
            print(" Unique codes that do not mapp: " +str(i), med_pd[med_pd[i]==-1]["NDC"].nunique())
            print(" Unique codes that do not mapp: "+str(i), med_pd[med_pd[i]==-1]["NDC"].nunique()/med_pd["NDC"].nunique())

 
        #  unique SUBJECT_ID 39087
        # No number of -1
        # unique HADM_ID 49929
        # No number of -1
        # unique ATC3 155
        # No number of -1
        # unique ATC4 348
        # unique NDC 2929
        # No number of -1



    return med_pd[['SUBJECT_ID', 'HADM_ID', 'ATC3','ATC4','NDC']]

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import polars as pl

def analyze_drug_distributions_sub(df_):
    # Convert Polars DataFrame to Pandas for easier manipulation
    
    df = df_.select(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'DRUG', 'STARTDATE', 'ENDDATE']).to_pandas()
    print("Analyzing patient with hadm_id 194773")
    aux = df[df["HADM_ID"]==194773]
    for i in aux.columns:
        print(i, aux[i].nunique())
        print(i, aux[i].value_counts())
    print(df.shape)
    
    # Convert date columns to datetime
    df['STARTDATE'] = pd.to_datetime(df['STARTDATE'])
    df['ENDDATE'] = pd.to_datetime(df['ENDDATE'])
    
    # 1. All drug administrations per subject (considering unique combinations)
    administrations_per_subject = df.groupby(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'DRUG', 'STARTDATE', 'ENDDATE']).size().groupby(level=0).sum().reset_index(name='administrations')
    
    # 2. Drug presence (0/1) per subject
    drug_presence_subject = df.groupby(['SUBJECT_ID', 'DRUG']).size().unstack(fill_value=0)
    drug_presence_subject = (drug_presence_subject > 0).astype(int)
    drugs_present_per_subject = drug_presence_subject.sum(axis=1)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Drug Distribution Analysis per Subject', fontsize=16)

    # 1. All drug administrations per subject
    sns.histplot(administrations_per_subject['administrations'], kde=True, ax=axs[0])
    axs[0].set_title('All Drug Administrations per Subject')
    axs[0].set_xlabel('Number of Drug Administrations')
    axs[0].set_ylabel('Frequency')

    # 2. Drug presence (0/1) per subject
    sns.histplot(drugs_present_per_subject, kde=True, ax=axs[1])
    axs[1].set_title('Number of Different Drugs Present per Subject')
    axs[1].set_xlabel('Number of Different Drugs')
    axs[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('drug_distribution_analysis_per_subject.png')
    plt.close()

    # Print summary statistics
    print("Summary Statistics:")
    print("\n1. All drug administrations per subject:")
    print(administrations_per_subject['administrations'].describe())

    print("\n2. Number of different drugs present per subject:")
    print(drugs_present_per_subject.describe())

    # Additional statistics
    print("\nTotal number of subjects:", len(administrations_per_subject))
    print("Total number of unique drugs across all subjects:", df['DRUG'].nunique())
    print("Average number of drug administrations per subject:", administrations_per_subject['administrations'].mean())
    print("Average number of different drugs present per subject:", drugs_present_per_subject.mean())

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

def plot_enhanced_sparse_matrix(matrix, title):
    # Convert to dense array for color mapping
    dense_matrix = matrix.toarray()
    
    # Create a color map
    plt.imshow(dense_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Frequency')
    plt.title(title)
    plt.xlabel('Drug ')
    plt.ylabel(' Admission Index')
    
    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.metrics import jaccard_score

# Assuming you have two matrices:
# matrix1: 0/1 matrix indicating whether a drug was given (binary)
# matrix2: Count matrix indicating how many times a drug was given





# 2. Comparison Statistics
def compare_matrices(matrix1, matrix2,real):
    if real == "ICD9_CODE":
        real="ICD9 CODES"
    else:
      
        real=real.lower()
    # Jaccard similarity for binary matrix
    jaccard = jaccard_score(matrix1.toarray().flatten(), (matrix2.toarray() > 0).flatten())
    
    # Percentage of admissions where count > 1 when drug was present
    mask = matrix1.toarray() == 1
    multiple_admin = np.mean(matrix2.toarray()[mask] > 1) * 100
    
    # Average administrations when drug was present
    avg_admin = np.mean(matrix2.toarray()[mask])
    
    # Correlation between presence and count
    correlation = np.corrcoef(matrix1.toarray().flatten(), matrix2.toarray().flatten())[0, 1]
    spearman_corr, _ = stats.spearmanr(matrix1.toarray().flatten(), matrix2.toarray().flatten())

    print(f"Spearman correlation between presence and count: {spearman_corr:.4f}")
    print(f"Jaccard Similarity: {jaccard:.4f}")
    print(f"Percentage of multiple administrations when {real} present: {multiple_admin:.2f}%")
    print(f"Average administrations when {real} present: {avg_admin:.2f}")
    print(f"Correlation between presence and count: {correlation:.4f}")



# 4. Comparison of Drug Prevalence
def compare_drug_prevalence(matrix1, matrix2,real="DRUG",type_input=None):
    if real == "ICD9_CODE":
        real="ICD9 CODES"
 
    else:
        real=real.lower()
        
    presence = matrix1.sum(axis=0).A1
    counts = matrix2.sum(axis=0).A1
    
    plt.figure(figsize=(10, 6))
    plt.scatter(presence, counts)
    plt.title(real +   ' Prevalence: Presence vs. Total Count')
    plt.xlabel('Number of Admissions with ' + real + ' Present')
    plt.ylabel('Total Number of ' + type_input)
    plt.show()
    plt.savefig("data/analysis/drugs/drug_prevalence_scatter_plot" + real + ".png")



import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_drug_matrices_histograms(frequency_matrix, presence_matrix, title,percentile=80,real="DRUG", type_input=None):
    # Convert sparse matrices to dense if they're not already
    if real == "ICD9_CODE":
        real="ICD9 CODES"
    
    else:
      
        
        real=real.lower()
        
    if hasattr(frequency_matrix, 'toarray'):
        frequency_matrix = frequency_matrix.toarray()
    if hasattr(presence_matrix, 'toarray'):
        presence_matrix = presence_matrix.toarray()
    
    # Calculate drug counts for each matrix
    freq_drug_counts = np.array(frequency_matrix.sum(axis=0)).flatten()
    pres_drug_counts = np.array(presence_matrix.sum(axis=0)).flatten()
    
    # Remove zero counts
    freq_non_zero_counts = freq_drug_counts[freq_drug_counts > 0]
    pres_non_zero_counts = pres_drug_counts[pres_drug_counts > 0]
    
    # Calculate 90th percentile for truncation
    freq_90th = np.percentile(freq_non_zero_counts, percentile)
    pres_90th = np.percentile(pres_non_zero_counts, percentile)
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f"Distribution of {real} {type_input} frequency and  presence (Truncated at 80th Percentile)", fontsize=16)
    
    # Histogram for frequency matrix
    ax1.hist(freq_non_zero_counts[freq_non_zero_counts <= freq_90th], bins=50, edgecolor='black')
    ax1.set_title('Histogram of ' +real + ' ' + type_input + ' frequency')
    ax1.set_xlabel('Number of ' + real + ' ' + type_input + ' over all admissions')
    ax1.set_ylabel('Number of ' +real  )
    
    # Histogram for presence matrix
    ax2.hist(pres_non_zero_counts[pres_non_zero_counts <= pres_90th], bins=50, edgecolor='black')
    ax2.set_title('Histogram of ' +real + ' presence')
    ax2.set_xlabel('Number of admissions where ' + real + ' is present over all admissions')
    ax2.set_ylabel('Number of ' +real)
    
    # Calculate stats and non-visible data points
    freq_non_visible = np.sum(freq_non_zero_counts > freq_90th)
    pres_non_visible = np.sum(pres_non_zero_counts > pres_90th)
    
    # Add statistics to the plots
   
    freq_stats = f"Non-visible {real}: {freq_non_visible}\n"
    freq_stats += f"Non-visible range: {freq_90th:.0f} - {np.max(freq_non_zero_counts):.0f}\n"
    freq_stats += f"Mean: {np.mean(freq_non_zero_counts):.2f}\n"
    freq_stats += f"Median: {np.median(freq_non_zero_counts):.2f}\n"
   
    ax1.text(0.95, 0.95, freq_stats, transform=ax1.transAxes, verticalalignment='top',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
   
    pres_stats = f"Non-visible {real}: {pres_non_visible}\n"
    pres_stats += f"Non-visible range: {pres_90th:.0f} - {np.max(pres_non_zero_counts):.0f}\n"
    pres_stats += f"Mean: {np.mean(pres_non_zero_counts):.2f}\n"
    pres_stats += f"Median: {np.median(pres_non_zero_counts):.2f}\n"
   
    ax2.text(0.95, 0.95, pres_stats, transform=ax2.transAxes, verticalalignment='top',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    plt.savefig("data/analysis/drugs/"+real+"_frequency_distribution_histograms" + real + ".png")
    plt.close()
    # Print additional statistics
       # Calculate and print descriptive statistics manually
    print("\nDescriptive Statistics for Frequency Matrix:")
    print(f"Skewness: {stats.skew(freq_non_zero_counts):.2f}")
    print(f"Kurtosis: {stats.kurtosis(freq_non_zero_counts):.2f}")
    print(f"90th percentile: {freq_90th:.2f}")
   
    print(f"Count: {len(freq_non_zero_counts)}")
    print(f"Mean: {np.mean(freq_non_zero_counts):.2f}")
    print(f"Std Dev: {np.std(freq_non_zero_counts):.2f}")
    print(f"Min: {np.min(freq_non_zero_counts):.2f}")
    print(f"25%: {np.percentile(freq_non_zero_counts, 25):.2f}")
    print(f"50% (Median): {np.median(freq_non_zero_counts):.2f}")
    print(f"75%: {np.percentile(freq_non_zero_counts, 75):.2f}")
    print(f"Max: {np.max(freq_non_zero_counts):.2f}")
    
    print("\nDescriptive Statistics for Presence Matrix:")
    print(f"Skewness: {stats.skew(pres_non_zero_counts):.2f}")
    print(f"Kurtosis: {stats.kurtosis(pres_non_zero_counts):.2f}")

    print(f"Count: {len(pres_non_zero_counts)}")
    print(f"Mean: {np.mean(pres_non_zero_counts):.2f}")
    print(f"Std Dev: {np.std(pres_non_zero_counts):.2f}")
    print(f"Min: {np.min(pres_non_zero_counts):.2f}")
    print(f"25%: {np.percentile(pres_non_zero_counts, 25):.2f}")
    print(f"50% (Median): {np.median(pres_non_zero_counts):.2f}")
    print(f"75%: {np.percentile(pres_non_zero_counts, 75):.2f}")
    print(f"Max: {np.max(pres_non_zero_counts):.2f}")

    # Print additional statistics



def plot_admission_histograms_administrations(frequency_matrix, presence_matrix, title, percentile=100, real="DRUG",type_input=None):
    if real == "ICD9_CODE":
        real = "ICD9 CODES"
        
    else:
       
        real = real.lower()
    
    if hasattr(frequency_matrix, 'toarray'):
        frequency_matrix = frequency_matrix.toarray()
    if hasattr(presence_matrix, 'toarray'):
        presence_matrix = presence_matrix.toarray()
    
    # Calculate counts for each admission
    freq_admission_counts = np.array(frequency_matrix.sum(axis=1)).flatten()
    pres_admission_counts = np.array(presence_matrix.sum(axis=1)).flatten()
    
    # Remove zero counts
    freq_non_zero_counts = freq_admission_counts[freq_admission_counts > 0]
    pres_non_zero_counts = pres_admission_counts[pres_admission_counts > 0]
    
    # Calculate percentile for truncation
    freq_nth = np.percentile(freq_non_zero_counts, percentile)
    pres_nth = np.percentile(pres_non_zero_counts, percentile)
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f"Distribution of {real} " + type_input + " and presence per admission", fontsize=16)
    
    # Histogram for frequency matrix
    ax1.hist(freq_admission_counts, bins=30, edgecolor='black')
    ax1.set_title('Histogram of ' + real + ' ' + type_input + ' frequency')
    ax1.set_xlabel('Number of ' + real + ' ' + type_input + ' per admission')
    ax1.set_ylabel('Number of admissions')
    
    # Histogram for presence matrix
    ax2.hist(pres_admission_counts, bins=30, edgecolor='black')
    ax2.set_title(f'Histogram of {real} presence')
    ax2.set_xlabel(f'Number of different {real} present per admission')
    ax2.set_ylabel('Number of admissions')
    
    # Calculate stats and non-visible data points
    freq_non_visible = np.sum(freq_non_zero_counts > freq_nth)
    pres_non_visible = np.sum(pres_non_zero_counts > pres_nth)
    
    # Add statistics to the plots

    freq_stats = f"Mean: {np.mean(freq_non_zero_counts):.2f}\n"
    freq_stats += f"Median: {np.median(freq_non_zero_counts):.2f}\n"
    ax1.text(0.95, 0.95, freq_stats, transform=ax1.transAxes, verticalalignment='top',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    

    pres_stats = f"Mean: {np.mean(pres_non_zero_counts):.2f}\n"
    pres_stats += f"Median: {np.median(pres_non_zero_counts):.2f}\n"
    ax2.text(0.95, 0.95, pres_stats, transform=ax2.transAxes, verticalalignment='top',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f"data/analysis/drugs/{real}_frequency_distribution_histograms_administrations.png")
    plt.close()
    
    # Print additional statistics

    
    print("\nFrequency Distribution Description:")
    # Calculate and print descriptive statistics manually
    print("\nDescriptive Statistics for Frequency Matrix:")
    print(f"Skewness: {stats.skew(freq_non_zero_counts):.2f}")
    print(f"Kurtosis: {stats.kurtosis(freq_non_zero_counts):.2f}")
    print(f"90th percentile: {freq_nth:.2f}")
    print(f"Count: {len(freq_non_zero_counts)}")
    print(f"Mean: {np.mean(freq_non_zero_counts):.2f}")
    print(f"Std Dev: {np.std(freq_non_zero_counts):.2f}")
    print(f"Min: {np.min(freq_non_zero_counts):.2f}")
    print(f"25%: {np.percentile(freq_non_zero_counts, 25):.2f}")
    print(f"50% (Median): {np.median(freq_non_zero_counts):.2f}")
    print(f"75%: {np.percentile(freq_non_zero_counts, 75):.2f}")
    print(f"Max: {np.max(freq_non_zero_counts):.2f}")
    
    print("\nPresence Matrix Statistics:")
    print(f"Skewness: {stats.skew(pres_non_zero_counts):.2f}")
    print(f"Kurtosis: {stats.kurtosis(pres_non_zero_counts):.2f}")
    print(f"90th percentile: {pres_nth:.2f}")
    print(f"Count: {len(pres_non_zero_counts)}")
    print(f"Mean: {np.mean(pres_non_zero_counts):.2f}")
    print(f"Std Dev: {np.std(pres_non_zero_counts):.2f}")
    print(f"Min: {np.min(pres_non_zero_counts):.2f}")
    print(f"25%: {np.percentile(pres_non_zero_counts, 25):.2f}")
    print(f"50% (Median): {np.median(pres_non_zero_counts):.2f}")
    print(f"75%: {np.percentile(pres_non_zero_counts, 75):.2f}")
    print(f"Max: {np.max(pres_non_zero_counts):.2f}")

    # Convert sparse matrices to dense if they're not already
    if hasattr(frequency_matrix, 'toarray'):
        frequency_matrix = frequency_matrix.toarray()
    if hasattr(presence_matrix, 'toarray'):
        presence_matrix = presence_matrix.toarray()
    
    # Calculate drug counts for each matrix
    freq_drug_counts = np.array(frequency_matrix.sum(axis=0)).flatten()
    pres_drug_counts = np.array(presence_matrix.sum(axis=0)).flatten()
    
    # Remove zero counts for better visualization
    freq_non_zero_counts = freq_drug_counts[freq_drug_counts > 0]
    pres_non_zero_counts = pres_drug_counts[pres_drug_counts > 0]
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f"Distribution of {title}", fontsize=16)
    
    # Histogram for frequency matrix
    ax1.hist(freq_non_zero_counts, bins=50, edgecolor='black')
    ax1.set_title('Histogram of Drug Administration Frequency')
    ax1.set_xlabel('Number of Administrations')
    ax1.set_ylabel('Number of Drugs')
    
    # Histogram for presence matrix
    ax2.hist(pres_non_zero_counts, bins=50, edgecolor='black')
    ax2.set_title('Histogram of Drug Presence')
    ax2.set_xlabel('Number of Admissions Where Drug is Present')
    ax2.set_ylabel('Number of Drugs')
    
    # Add statistics to the plots
    freq_stats = f"Total Drugs: {len(freq_drug_counts)}\n"
    freq_stats += f"Non-zero Drugs: {len(freq_non_zero_counts)}\n"
    freq_stats += f"Mean: {np.mean(freq_non_zero_counts):.2f}\n"
    freq_stats += f"Median: {np.median(freq_non_zero_counts):.2f}\n"
    freq_stats += f"Max: {np.max(freq_non_zero_counts)}"
    ax1.text(0.95, 0.95, freq_stats, transform=ax1.transAxes, verticalalignment='top',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    pres_stats = f"Total Drugs: {len(pres_drug_counts)}\n"
    pres_stats += f"Non-zero Drugs: {len(pres_non_zero_counts)}\n"
    pres_stats += f"Mean: {np.mean(pres_non_zero_counts):.2f}\n"
    pres_stats += f"Median: {np.median(pres_non_zero_counts):.2f}\n"
    pres_stats += f"Max: {np.max(pres_non_zero_counts)}"
    ax2.text(0.95, 0.95, pres_stats, transform=ax2.transAxes, verticalalignment='top',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    plt.savefig("data/analysis/drugs/drug_frequency_distribution_histograms.png")



    # Assuming total_matrix is already defined as in your code snippet
    # total_matrix = csr_matrix(...)

def combined_plot_function(matrix, name_graph="DRUG",diagnosis=False):
    # Prepare data
    dense_matrix = matrix.toarray()
    
    if name_graph == "DRUG":
        truncate = 95
        name = "Drugs"
        real = "drugs"
        type_input_title = "administered"
    else:
        truncate = 100
        name = "ICD-9 Codes"
        real = "ICD-9 CODES"
        type_input_title = "registered"
    
    # Calculate percentages
    total_count = dense_matrix.size
    count_one = (dense_matrix == 1).sum()
    count_greater_than_one = (dense_matrix > 1).sum()
    count_greater_equal_one = (dense_matrix >= 1).sum()
    
    count_greater_equal_one_percentage= count_greater_equal_one/ total_count * 100
    percent_greater_than_one_of_nonzero = count_greater_than_one / (count_one + count_greater_than_one) * 100
    percent_greater_than_one_of_total = count_greater_than_one / total_count * 100
    percenta_one = count_one / total_count* 100
    # Get values for histogram
    values_gt_1 = dense_matrix[dense_matrix > 1]
    truncation_point = np.percentile(values_gt_1, truncate)
    plot_values = values_gt_1[values_gt_1 <= truncation_point]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Enhanced Sparse Matrix
    pale_color = '#E6E6FA'  # Lavender color (a pale tone)
    colors = ['#120052', 'red', pale_color]  # Purple, Red, Pale tone
    cmap = ListedColormap(colors)
    bounds = [0, 0.5, 1.5, dense_matrix.max()]
    norm = BoundaryNorm(bounds, cmap.N)
    
    im = ax1.imshow(dense_matrix, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    if  name_graph=="DRUG":   
        ax1.set_title(f"Number of times {real} are administered\nduring the admission")
    else:    
        ax1.set_title(f"Number of times {real} are registered\nduring the admission")

   
    ax1.set_xlabel(f"Unique {real}")
  
    ax1.set_ylabel('Admission Index',ha='right', va='center')

            
    cbar = plt.colorbar(im, ax=ax1, extend='max')
    cbar.set_label('Frequency')
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['0', '1', '>1'])
        
        # Plot 2: Histogram
    if diagnosis:   
       # For diagnosis, we want only two bins centered at 2 and 3
       bins = [1.5, 2.5, 3.5]  # This creates two bins: [1.5, 2.5) and [2.5, 3.5)
       ax2.hist(plot_values, bins=bins, edgecolor='black', color=pale_color, align='mid')
       ax2.set_xticks([2, 3])  # Set x-axis ticks to 2 and 3
       ax2.set_xticklabels(['2', '3'])  # Label the ticks
    else:    
       ax2.hist(plot_values, bins=8, edgecolor='black', color=pale_color) 
         
    
    if name_graph=="DRUG":
        ax2.set_title(f'Histogram of {real} {type_input_title} more than once\n(Truncated at 95th Percentile)')
        ax2.set_xlabel('Number of administrations')
    else:
        ax2.set_title(f'Histogram of {real} {type_input_title} more than once\n')
        ax2.set_xlabel('Number of registrations')     
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Add stats box (as before)
    # Calculate statistics for the text box
    freq_non_visible = np.sum(values_gt_1 > truncation_point)
    freq_90th = np.percentile(values_gt_1, truncate)
    freq_non_zero_counts = values_gt_1[values_gt_1 > 0]
    
    # Create the text for the stats box
    if name_graph == "DRUG":
        freq_stats = f"Non-visible points: {freq_non_visible}\n"
        freq_stats += f"Non-visible range: {freq_90th:.0f} - {np.max(freq_non_zero_counts):.0f}\n"
        freq_stats += f"Mean: {np.mean(freq_non_zero_counts):.2f}\n"
        freq_stats += f"Median: {np.median(freq_non_zero_counts):.2f}\n"
        ax2.text(0.95, 0.95, freq_stats, transform=ax2.transAxes, verticalalignment='top',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        # Add overall title
        fig.suptitle(f"Analysis of administrations of {real}")
    else:
        freq_stats = f"Mean: {np.mean(freq_non_zero_counts):.2f}\n"
        freq_stats += f"Median: {np.median(freq_non_zero_counts):.2f}\n"
        ax2.text(0.95, 0.95, freq_stats, transform=ax2.transAxes, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        # Add overall title
        fig.suptitle(f"Analysis of registrations of {real}",)

    # Add the text box to the plot

    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Save the figure
    plt.savefig(f"data/analysis/drugs/combined_plot_{name_graph.lower()}_updated.png")
     
    # Print percentages and additional statistics
    
    print(f"Percentage of {name}  greater equal 1 over all {name} (including 0): {count_greater_equal_one_percentage}%")
    print(f"Percentage of {name}  equal 1 over all {name} (including 0): {percenta_one}%")
    print(f"Percentage of {name} greater than 1 with respect to {name} that have 1: {percent_greater_than_one_of_nonzero:.2f}%")
    print(f"Percentage of {name} greater than 1 over all {name} (including 0): {percent_greater_than_one_of_total:.2f}%")
    print(f"Number of values not visible: {len(values_gt_1)}")
    print(f"Number of values in truncated plot: {len(plot_values)}")
    print(f"Truncation point ({truncate}th percentile): {truncation_point:.2f}")
 
    # Convert the sparse matrix to a dense array

def plot_one_values(total_matrix,name_graph="DRUG"):   
    if name_graph == "DRUG":
        truncate = 95
        name = "Drugs"
    else:
        truncate = 100    
        name = "ICD-9 - Codes"
    dense_matrix = total_matrix.toarray()

    # Get all values greater than 1
    values_gt_1 = dense_matrix[dense_matrix > 1]

    # Calculate the 70th percentile for truncation
    truncation_point = np.percentile(values_gt_1, truncate)

    # Filter values for plotting (up to 70th percentile)
    plot_values = values_gt_1[values_gt_1 <= truncation_point]

    # Create the histogram
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(plot_values, bins=8, edgecolor='black')
    if name_graph != "ICD9_CODE":
        ax1.set_title(f'Histogram of {name} Count Matrix - Values Greater Than 1 (Truncated at 95th Percentile)')
    else:
        ax1.set_title(f'Histogram of  {name} Count Matrix - Values Greater Than 1 ')


    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

   

    # Calculate statistics for the text box
    freq_non_visible = np.sum(values_gt_1 > truncation_point)
    freq_90th = np.percentile(values_gt_1, truncate)
    freq_non_zero_counts = values_gt_1[values_gt_1 > 0]

    # Create the text for the stats box
      # Replace with the actual column name
    if name_graph == "DRUG":
        freq_stats = f"Non-visible points: {freq_non_visible}\n"
        freq_stats += f"Non-visible range: {freq_90th:.0f} - {np.max(freq_non_zero_counts):.0f}\n"
        freq_stats += f"Mean: {np.mean(freq_non_zero_counts):.2f}\n"
        freq_stats += f"Median: {np.median(freq_non_zero_counts):.2f}\n"
    else:    
        freq_stats = f"Mean: {np.mean(freq_non_zero_counts):.2f}\n"
        freq_stats += f"Median: {np.median(freq_non_zero_counts):.2f}\n"

        
    # Add the text box to the plot
    ax1.text(0.95, 0.95, freq_stats, transform=ax1.transAxes, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Print some additional statistics
    print(f"Number of values not vissible: {len(values_gt_1)}")
    print(f"Number of values in truncated plot: {len(plot_values)}")
    print(f"Truncation point (70th percentile): {truncation_point:.2f}")
    plt.savefig("/Users/cynthiagarcia/Desktop/Synthetic-Data-Deep-Learning/data/analysis/drugs")

def plot_sparse_matrix_functionv2(total_number_of_admissions, unique_values,real,percentile=80):
    
    
    # Create pivot tables and convert to sparse matrices
    total_matrix = csr_matrix(total_number_of_admissions[[real, "SUBJECT_ID", "HADM_ID"]].pivot_table(
        index=['SUBJECT_ID', "HADM_ID"],
        columns=real,
        aggfunc='size',
        fill_value=0
    ))
    
    unique_matrix = csr_matrix(unique_values[[real, "SUBJECT_ID", "HADM_ID"]].pivot_table(
        index=['SUBJECT_ID', "HADM_ID"],
        columns=real,
        aggfunc='size',
        fill_value=0
    ))
    print('combined fig')
    combined_plot_function(total_matrix,real,diagnosis=False)
    print(" matrix greater than one ")
    plot_one_values(total_matrix,real)
    print("total matrix dividida")
    if real == "ICD9_CODE":
        real1="ICD-9 CODES"
        type_input="registrations"
    else:  
        type_input="administrations"
        real1=real.lower()
      
    plot_enhanced_sparse_matrix_single_graph(total_matrix, "Number of times a "+real1+" was given during the admission",
                                            real,type_input)
    
    print("histogram number of " + type_input + " per " + real1)
    plot_drug_matrices_histograms(total_matrix, unique_matrix, "Distribution of Number of " + type_input + " per " + real1,
                                  percentile,real,type_input)
    print("histogram number of " + type_input + " per admission")
    plot_admission_histograms_administrations(total_matrix, unique_matrix, "Distribution of Number of " + type_input + " per " + real1, percentile=80, real=real,type_input=type_input)
    print("compare_drug_prevalence")
    compare_drug_prevalence(unique_matrix, total_matrix,real,type_input)    

    print("compare_matrices")
    compare_matrices(unique_matrix, total_matrix,real)
    # Calculate and print some statistics
    total_density = total_matrix.nnz / np.prod(total_matrix.shape)
    unique_density = unique_matrix.nnz / np.prod(unique_matrix.shape)
    
    print(f"Number of times a {real} was given during the admission Matrix Density: {total_density:.4f}")
    print(f"Unique Values Matrix Density: {unique_density:.4f}")
    
    # Calculate the difference matrix
    difference_matrix = total_matrix - unique_matrix
    
    # Plot the difference matrix
    plt.figure(figsize=(8, 6))
    plot_enhanced_sparse_matrix(difference_matrix, "Difference (Total - Unique)")
    plt.tight_layout()
    plt.show()
    
    
   

    return total_matrix, unique_matrix, difference_matrix
# Usage
# analyze_drug_distributions(df_)
from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_enhanced_sparse_matrix_single_graph(matrix, title, real="DRUG", type_input=None):
    if real == "ICD9_CODE":
        real = "ICD-9 CODES"
        real_c = "ICD-9 CODES"
        type_input_title = "registered"
    else:
        type_input_title = "administered"
        real_c = "drugs"
        real = real.lower()
    
    dense_matrix = matrix.toarray()
     #Calculate percentages
    total_count = dense_matrix.size
    count_one = (dense_matrix == 1).sum()
    count_greater_than_one = (dense_matrix > 1).sum()
    
    percent_greater_than_one_of_nonzero = count_greater_than_one / (count_one + count_greater_than_one) * 100
    percent_greater_than_one_of_total = count_greater_than_one / total_count * 100
    
    print(f"Percentage of {real_c} greater than 1 with respect to {real_c} that have 1: {percent_greater_than_one_of_nonzero:.2f}%")
    print(f"Percentage of {real_c} greater than 1 over all {real_c} (including 0): {percent_greater_than_one_of_total:.2f}%")

    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Custom colormap
    colors = ['#120052', 'red', 'white']  # Purple, Red, White
    cmap = ListedColormap(colors)
    bounds = [0, 0.5, 1.5, dense_matrix.max()]
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Plot: Number of times a drug was given
    im = ax.imshow(dense_matrix, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    ax.set_title(f"Number of times {real_c} are {type_input_title} during the admission", fontsize=14)
    ax.set_xlabel(real_c, fontsize=12)
    ax.set_ylabel('Admission Index', fontsize=12)
    cbar = plt.colorbar(im, ax=ax, extend='max')
    cbar.set_label('Frequency')
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['0', '1', '>1'])
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f"data/analysis/drugs/drug_frequency_distribution_single_graph_{real}.png")


def plot_enhanced_sparse_matrix_dual_graphs(matrix,matrix2 ,title,real="DRUG",type_input=None):
    if real == "ICD9_CODE":
        real="ICD9 CODES"
        real_c="ICD9 CODES"
        type_input_title="registered"
    else:
        type_input_title = "administered"
        real_c=real.lower().title()
        real=real.lower()
        
    dense_matrix = matrix.toarray()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Custom colormap for the left plot
    colors_left = ['#120052', 'red', 'white']  # Purple, Red, White
    cmap_left = ListedColormap(colors_left)
    bounds_left = [0, 0.5, 1.5, dense_matrix.max()]
    norm_left = BoundaryNorm(bounds_left, cmap_left.N)
    
    # Left plot: Number of times a drug was given
    im1 = ax1.imshow(dense_matrix, cmap=cmap_left, norm=norm_left, aspect='auto', interpolation='nearest')
    ax1.set_title("Number of times a " + real_c  + " " + type_input_title + " during the admission", fontsize=14)
    ax1.set_xlabel(real_c, fontsize=12)
    ax1.set_ylabel('Admission Index', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1, extend='max')
    cbar1.set_label('Frequency')
    cbar1.set_ticks([0, 1, 2])
    cbar1.set_ticklabels(['0', '1', '>1'])
    
    # Custom colormap for the right plot
    colors_right = ['#120052', 'red']  # Purple, Yellow
    cmap_right = ListedColormap(colors_right)
    
    # Right plot: Presence or Absence of Drugs
    #binary_matrix = (dense_matrix > 0).astype(int)
    dense_matrix2 = matrix2.toarray()
    im2 = ax2.imshow(dense_matrix2, cmap=cmap_right, aspect='auto', interpolation='nearest',
                     vmin=0, vmax=1)
    ax2.set_title("Presence (1) or Absence (0) of " + real_c + " during the Admission", fontsize=14)
    ax2.set_xlabel(real_c, fontsize=12)
    ax2.set_ylabel('Admission Index', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1])
    #cbar2.set_label('Presence (0 or 1)')
    cbar2.set_ticklabels(['0', '1'])
    
    plt.tight_layout()
    plt.show()
    plt.savefig("data/analysis/drugs/drug_frequency_distribution_dual_graphs"+real+".png")
    
def plot_enhanced_sparse_matrix_grid(matrix, title):

    # Convert to dense array for color mapping
    dense_matrix = matrix.toarray()
    
    # Calculate the maximum frequency
    max_freq = dense_matrix.max()
    
    # Create a single figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
   
    
    im = ax.imshow(dense_matrix, cmap='viridis', aspect='auto', interpolation='nearest',
                       norm=LogNorm(vmin=1, vmax=max_freq))  # Use LogNorm directly
        
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Drug', fontsize=12)
    ax.set_ylabel('Admission Index', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Frequency')
    cbar.ax.set_ylabel('Frequency', fontsize=12)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    plt.savefig("data/analysis/drugs/drug_frequency_distribution_grid.png") 
# Example usage
# Assuming 'matrix' is your sparse matrix
# plot_enhanced_sparse_matrix_grid(matrix, "Drug Frequency Distribution")
from scipy.sparse import csr_matrix

def plot_sparse_matrix(matrix, title):
        plt.spy(matrix, markersize=1,aspect='auto')
        plt.title(title)
# Create two example sparse matrices

def plot_sparse_matrix_funcion(total_number_of_admissions,unique_values,size=200):
    real = "DRUG"
    total_number_of_admissions = total_number_of_admissions[[real, "SUBJECT_ID", "HADM_ID"]].pivot_table(
    index=['SUBJECT_ID', "HADM_ID"],
    columns=real,
    aggfunc='size',
    fill_value=0
 )  
    unique_values = unique_values[[real, "SUBJECT_ID", "HADM_ID"]].pivot_table(
    index=['SUBJECT_ID', "HADM_ID"],
    columns=real,
    aggfunc='size',
    fill_value=0
 )
    # Function to plot a sparse matrix


    # Plot both matrices side by side
    
    plt.figure(figsize=(30,20))

    plt.subplot(121)
    plot_sparse_matrix(total_number_of_admissions, "Number of times a drug was given during the admission")

    plt.subplot(122)
    plot_sparse_matrix(unique_values, "Number of different drugs given during the admission")

    plt.tight_layout()
    plt.show()

import regex as re
def analyze_drug_distributions(df):
    # Convert Polars DataFrame to Pandas for easier manipulation
    #df = df_.select(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'DRUG', 'STARTDATE', 'ENDDATE']).to_pandas()
    print(df.shape)
    aux = df[df["HADM_ID"]==134622]
    for i in aux.columns:
        print(i, aux[i].nunique())
        print(i, aux[i].value_counts())
    print(aux.DRUG.value_counts())
    # Convert date columns to datetime
    df['STARTDATE'] = pd.to_datetime(df['STARTDATE'])
    df['ENDDATE'] = pd.to_datetime(df['ENDDATE'])
    
    # 1. All drug administrations per admission 
    total_drugs_per_admission = df.groupby(['SUBJECT_ID', 'HADM_ID']).size().reset_index(name='total_drugs')
    
    # 2. Drug presence (0/1) per admission
    drug_presence_admission = df.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID', 'DRUG'])
    drug_presence_admission = df.groupby(['SUBJECT_ID', 'HADM_ID', 'DRUG']).size().unstack(fill_value=0)
    drug_presence_admission = (drug_presence_admission > 0).astype(int)
    drugs_present_per_admission = drug_presence_admission.sum(axis=1)
    
    # Create subplots
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Drug Distribution Analysis per Admission', fontsize=16)

    # 1. All drug administrations per admission
    sns.histplot(total_drugs_per_admission['total_drugs'], kde=True, ax=axs[0])
    axs[0].set_title('Total Drug Administrations per Admission')
    axs[0].set_xlabel('Number of Drug Administrations')
    axs[0].set_ylabel('Frequency')

    # 2. Drug presence (0/1) per admission
    sns.histplot(drugs_present_per_admission, kde=True, ax=axs[1])
    axs[1].set_title('Number of Different Drugs Present per Admission')
    axs[1].set_xlabel('Number of Different Drugs')
    axs[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('data/analysis/drugs/drug_distribution_analysis_per_admission.png')
    plt.close()

    # Print summary statistics
    print("Summary Statistics:")
    print("\n1. All drug administrations per admission:")
    print(total_drugs_per_admission['total_drugs'].describe())

    print("\n2. Number of different drugs present per admission:")
    print(drugs_present_per_admission.describe())

    # Additional statistics
    print("\nTotal number of admissions:", len(total_drugs_per_admission))
    print("Total number of unique drugs across all admissions:", total_drugs_per_admission['total_drugs'].nunique())

import unicodedata

def clean_medicine_names(df, column_name):
    def clean_name(name):
        # Convert to lowercase
        name = name.lower()
        
        # Remove extra spaces
        name = ' '.join(name.split())
        
        # Remove special characters and punctuation, except hyphens
        name = re.sub(r'[^\w\s-]', '', name)
        
        # Remove any non-ASCII characters
        #name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
        
        # Replace multiple hyphens with a single hyphen
        name = re.sub(r'-+', '-', name)
        
        # Remove leading and trailing hyphens
        name = name.strip('-')
        
        return name

    # Apply the cleaning function to the specified column
    df[column_name] = df[column_name].astype(str).apply(clean_name)
    print("Before duplicates", df.shape)
    # Remove duplicates based on the cleaned names
    df.drop_duplicates()
    print("after duplicates", df.shape)
    print("UNIQUE DRUGS AFTER CLEANING", df.DRUG.nunique())
    # Sort the dataframe by the cleaned names
    df.sort_values(by=column_name, inplace=True)
    
    # Reset the index
    df.reset_index(drop=True, inplace=True)
    
    return df


def stats(df):\
     # Patient-level statistics (per SUBJECT_ID)
    patient_level = df.groupby('SUBJECT_ID')['DRUG'].nunique()
    patient_level_stats = patient_level.agg(['mean', 'max', 'std']).reset_index()
    patient_level_stats.columns = ['SUBJECT_ID', 'patient_mean_drugs', 'patient_max_drugs', 'patient_std_drugs']

    # Admission-level statistics (per HADM_ID)
    admission_level = df.groupby(['SUBJECT_ID', 'HADM_ID'])['DRUG'].nunique()
    admission_level_stats = admission_level.agg(['mean', 'max', 'std']).reset_index()
    admission_level_stats.columns = ['SUBJECT_ID', 'HADM_ID', 'admission_mean_drugs', 'admission_max_drugs', 'admission_std_drugs']

    # Calculate overall statistics
    overall_stats = {
        'patient_level': {
            'mean': patient_level_stats['patient_mean_drugs'].mean(),
            'max': patient_level_stats['patient_max_drugs'].max(),
            'std': patient_level_stats['patient_std_drugs'].mean()
        },
        'admission_level': {
            'mean': admission_level_stats['admission_mean_drugs'].mean(),
            'max': admission_level_stats['admission_max_drugs'].max(),
            'std': admission_level_stats['admission_std_drugs'].mean()
        }
    }

    # Generate LaTeX tables
    patient_latex = patient_level_stats.head().to_latex(index=False, float_format="{:0.2f}".format)
    admission_latex = admission_level_stats.head().to_latex(index=False, float_format="{:0.2f}".format)
    print("patient_latex",patient_latex)
    print("admission_latex",admission_latex)
    
    import pandas as pd
import numpy as np

def calculate_medication_stats(clean_df_pd):
    # Patient-level statistics (per SUBJECT_ID)
    patient_level = clean_df_pd.groupby('SUBJECT_ID')['DRUG'].nunique()
    patient_level_stats = patient_level.agg(['mean', 'max', 'std']).reset_index()
    print("patient level stat")
    print(patient_level_stats)
       # Admission-level statistics (per HADM_ID)
    admission_level = clean_df_pd.groupby(['SUBJECT_ID', 'HADM_ID'])['DRUG'].nunique()
    admission_level_stats = admission_level.agg(['mean', 'max', 'std']).reset_index()
    print("admission level stat")
    print(admission_level_stats)
       # Admission-level st

    # Generate LaT


# Example usage:
# patient_stats, admission_stats, overall_stats, patient_latex, admission_latex, overall_latex = calculate_medication_stats(df)
# print("Patient-level stats (LaTeX):\n", patient_latex)
# print("\nAdmission-level stats (LaTeX):\n", admission_latex)
# print("\nOverall stats (LaTeX):\n", overall_latex)
    
    

   
    
# Usage
def drugst(d1, n, name, save_path_prefix="data/analysis/drugs/drugs2",administrations=False,plot_matrix=True,save_prproces=True,analyze_distributions=False):
    # Read the CSV file using Polars
    df_ = pl.read_csv(d1, infer_schema_length=10000, ignore_errors=True)
    df_pd = df_.to_pandas()
    print("df_ shape initial",df_pd.shape)
    df_pd=df_pd.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE', 'ENDDATE', 'DRUG_TYPE', 'DRUG', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC', 'FORMULARY_DRUG_CD', 'GSN', 'NDC', 'PROD_STRENGTH', 'DOSE_VAL_RX', 'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'ROUTE'])
    print("df_ shape after drop duplicates",df_pd.shape)
    #duplicate considering the following columns ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE', 'ENDDATE',
    # Convert to pandas for easier manipulation
    df_pd=df_pd.drop_duplicates(subset=[ 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE', 'ENDDATE',
       'DRUG',  'PROD_STRENGTH', 'DOSE_VAL_RX',
        'FORM_UNIT_DISP', 'ROUTE'])
    print("df_ shape after drop duplicates",df_pd.shape)
    
    if analyze_distributions:
        analyze_icu_and_drugs(df_pd)
        
        analyze_drug_distributions(df_pd)
    
    df_pd.drop(
        columns=[
            "ROW_ID",
            "DRUG_TYPE",
            "DRUG_NAME_POE",
            "DRUG_NAME_GENERIC",
            "FORMULARY_DRUG_CD",
            "PROD_STRENGTH",
            "DOSE_VAL_RX",
            "DOSE_UNIT_RX",
            "FORM_VAL_DISP",
            "FORM_UNIT_DISP",
            "GSN",
            "FORM_UNIT_DISP",
            "ROUTE",
          
            
        ],
        axis=1,
        inplace=True,
    )
    print("Medication file after dropping columns:", df_pd.shape)
    print("Unique values in NDC:", df_pd["NDC"].nunique()
          )
    print(''' Cols eliminated:
            "ROW_ID",
            "DRUG_TYPE",
            "DRUG_NAME_POE",
            "DRUG_NAME_GENERIC",
            "FORMULARY_DRUG_CD",
            "PROD_STRENGTH",
            "DOSE_VAL_RX",
            "DOSE_UNIT_RX",
            "FORM_VAL_DISP",
            "FORM_UNIT_DISP",
            "GSN",
            "FORM_UNIT_DISP",
            "ROUTE",
          ''')
    #med_pd.drop(index=med_pd[med_pd["NDC"] == "0"].index, axis=0, inplace=True)
    #print("Filling NaN values with previous values")
    print(df_pd.isnull().sum())
    print("Datashape before cleaning",df_pd.shape)
    clean_df_pd = clean_medicine_names(df_pd, 'DRUG')
    stats(clean_df_pd)
    save_pickle(clean_df_pd,"/Users/cynthiagarcia/Desktop/Synthetic-Data-Deep-Learning/data/analysis/"+"drugs_raw.pkl")
    #coun matrix drugs
    administrations_per_admission =clean_df_pd[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'DRUG', 'STARTDATE', 'ENDDATE']]                                              
    
    # 2. Drug presence (0/1) per admission
    drug_presence_admission =clean_df_pd.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID', 'DRUG'])
    print("drug_presence_admission.shape",drug_presence_admission.shape)

    if plot_matrix:
         plot_sparse_matrix_functionv2(administrations_per_admission, drug_presence_admission,"DRUG")
   
    result_df = pd.DataFrame()
    
    for threshold in n:
        # For administrations
        admin_codes = list(administrations_per_admission['DRUG'])
        num_bins_admin = len(set(admin_codes))
        save_path_admin = f"{save_path_prefix}_admin_{threshold}.svg" if save_path_prefix else None
        bins_before_threshold_admin, bins_before_threshold_i_admin = cumulative_plotvv2(
            admin_codes, num_bins_admin, threshold, f"Drug Administrations (threshold {threshold})", save_path_admin)
        
        # For drug presence
        presence_codes = list(drug_presence_admission['DRUG'])
        num_bins_presence = len(set(presence_codes))
        save_path_presence = f"{save_path_prefix}_presence_{threshold}.svg" if save_path_prefix else None
        bins_before_threshold_presence, bins_before_threshold_i_presence = cumulative_plotvv2(
            presence_codes, num_bins_presence, threshold, f"Drug Presence (threshold {threshold})", save_path_presence)
        
        # Apply thresholds
        administrations_per_admission[f'threshold_admin_{threshold}'] = administrations_per_admission['DRUG'].apply(
            lambda x: x if x in bins_before_threshold_i_admin else -1)
        
        print("administrations_per_admission")
        print(administrations_per_admission.columns)
        print(administrations_per_admission.isnull().sum())
        for i in administrations_per_admission.columns:
            print("unique"+str(i),administrations_per_admission[i].nunique() )
        
        drug_presence_admission[f'threshold_presence_{threshold}'] = drug_presence_admission['DRUG'].apply(
            lambda x: x if x in bins_before_threshold_i_presence else -1)
        print("drug_presence_threshold")
        print(drug_presence_admission.columns)
        print(drug_presence_admission.isnull().sum())
        for i in drug_presence_admission.columns:
            print("unique"+str(i),drug_presence_admission[i].nunique() )
     
        # Merge results
    if administrations:
        result_df = administrations_per_admission
    else:
        result_df = drug_presence_admission
    print(result_df.columns)
    print(result_df.isnull().sum())
    for i in result_df.columns:
        print("unique"+str(i),result_df[i].nunique() )
        # Convert back to Polars DataFrame
    #result_df = pl.from_pandas(result_df)
    if save_prproces:
       save_pickle(result_df,DARTA_INTERM_intput_redoo + "with_ascii_codes_mapping_drugs.pkl")
    return result_df

    # Usage of the function

def drugs1(d1,n,name):
    df_ = pl.read_csv(d1, infer_schema_length=10000, ignore_errors=True, )
    #analyze_drug_distributions(df_)
    #analyze_drug_distributions_sub(df_)
    #nuevo_df = funcion_acum(nuevo_df, n, name)
    df_filtered = df_.with_columns(pl.col("SUBJECT_ID").cast(pl.Utf8))
    df_filtered = df_filtered.with_columns(pl.col("HADM_ID").cast(pl.Utf8))
    aux_nuevo_df =df_filtered[["HADM_ID","SUBJECT_ID", "DRUG","DRUG_NAME_GENERIC", "FORMULARY_DRUG_CD","NDC",'STARTDATE']].to_pandas()
    for i in aux_nuevo_df.columns:    
        print("unique"+str(i),aux_nuevo_df[i].nunique() )
        
    nuevo_df =df_filtered[["HADM_ID","SUBJECT_ID", "DRUG",'STARTDATE']].to_pandas()
    for i in nuevo_df.columns:    
        print("unique"+str(i),nuevo_df[i].nunique() )
    print(nuevo_df.shape)
    #     niqueHADM_ID 50216
    # uniqueSUBJECT_ID 39363
    # uniqueDRUG 4525
    # uniqueDRUG_NAME_GENERIC 2863
    # uniqueFORMULARY_DRUG_CD 3267
     # uniqueNDC 4204
    # uniqueSTARTDATE 38497
    # uniqueHADM_ID 50216
    # uniqueSUBJECT_ID 39363
    # uniqueDRUG 4525
    # uniqueSTARTDATE 38497
    print("nuevo_df before drop duplicates",nuevo_df.isnull().sum())
    #      nuevo_df before drop duplicates HADM_ID          0
    # SUBJECT_ID       0
    # DRUG             0
  #STARTDATE     3182

    nuevo_df.drop_duplicates( inplace=True)
    #nuevo_df shape after drop duplicates (3071980, 4) they are lees than when consider NDC because here we dont consider NDC
    print("considering the following columns['HADM_ID','SUBJECT_ID', 'DRUG','STARTDATE']")
    print("nuevo_df shape after drop duplicates",nuevo_df.shape)
    #     considering the following columns['HADM_ID','SUBJECT_ID', 'DRUG','STARTDATE']
    # nuevo_df shape after drop duplicates (3071980, 4)

    nuevo_df = funcion_acum(nuevo_df,n,name)
    
    for col in [col for col in nuevo_df.columns if "threshold" in col]:
        conteo_negativos = (nuevo_df[col] == -1).sum()  # Count occurrences of -1
        print(col,conteo_negativos)

        print(f"Percentage of drugs coniderign all admission and duplica for {col}: {( conteo_negativos)/nuevo_df.shape[0]}")
        print(f"Number of admissions for {col}: {nuevo_df.shape[0] - conteo_negativos}")
        #        threshold_0.88 369475
        # Percentage of drugs coniderign all admission and duplica for threshold_0.88: 0.12027259292052683
        # Number of admissions for threshold_0.88: 2702505
        # Total unique drugs: 4525
        # Drugs mapped as -1 in threshold_0.88: 4288
        # Percentage of drugs mapped as -1 in threshold_0.88: 94.76%
        # Drugs without -1 in threshold_0.88: 237
        # Percentage of drugs without -1 in threshold_0.88: 5.24%
        # threshold_0.95 153928
        # Percentage of drugs coniderign all admission and duplica for threshold_0.95: 0.05010709705141309


        # Calcular y mostrar el porcentaje de drogas mapeadas como -1 para la columna actual
        total_drogas = nuevo_df['DRUG'].nunique()  # Total de drogas únicas
        drogas_mapeadas_neg1 = nuevo_df[nuevo_df[col] == -1]['DRUG'].nunique()  # Drogas con -1 en la columna actual
        porcentaje_neg1 = (drogas_mapeadas_neg1 / total_drogas) * 100 if total_drogas > 0 else 0

        print(f"Total unique drugs: {total_drogas}")
        print(f"Drugs mapped as -1 in {col}: {drogas_mapeadas_neg1}")
        print(f"Percentage of drugs mapped as -1 in {col}: {porcentaje_neg1:.2f}%")

        # Calcular y mostrar el porcentaje de drogas que no tienen -1 para la columna actual
        drogas_sin_neg1 = nuevo_df[nuevo_df[col] != -1]['DRUG'].nunique()  # Drogas sin -1 en la columna actual
        porcentaje_sin_neg1 = (drogas_sin_neg1 / total_drogas) * 100 if total_drogas > 0 else 0

        print(f"Drugs without -1 in {col}: {drogas_sin_neg1}")
        print(f"Percentage of drugs without -1 in {col}: {porcentaje_sin_neg1:.2f}%")
        #         Number of admissions for threshold_0.95: 2918052
        # Total unique drugs: 4525
        # Drugs mapped as -1 in threshold_0.95: 4089
        # Percentage of drugs mapped as -1 in threshold_0.95: 90.36%
        # Drugs without -1 in threshold_0.95: 436
        # Percentage of drugs without -1 in threshold_0.95: 9.64%
        # threshold_0.98 61498
        # Percentage of drugs coniderign all admission and duplica for threshold_0.98: 0.020019010540433205
        # Number of admissions for threshold_0.98: 3010482
        # Total unique drugs: 4525
        # Drugs mapped as -1 in threshold_0.98: 3825
        # Percentage of drugs mapped as -1 in threshold_0.98: 84.53%
        # Drugs without -1 in threshold_0.98: 700
        # Percentage of drugs without -1 in threshold_0.98: 15.47%
        # threshold_0.999 3072
        # Percentage of drugs coniderign all admission and duplica for threshold_0.999: 0.0010000065104590525
        # Number of admissions for threshold_0.999: 3068908
        # Total unique drugs: 4525
        # Drugs mapped as -1 in threshold_0.999: 2212
        # Percentage of drugs mapped as -1 in threshold_0.999: 48.88%
        # Drugs without -1 in threshold_0.999: 2313
        # Percentage of drugs without -1 in threshold_0.999: 51.12%


    print("nuevo_df after acum",nuevo_df.isnull().sum())
    #     nuevo_df after acum HADM_ID               0
    # SUBJECT_ID            0
    # DRUG                  0
    # STARTDATE          1199    
    for i in nuevo_df.columns:    
        print("unique after drop duplicates"+str(i),nuevo_df[i].nunique() )
        #         unique after drop duplicatesHADM_ID 50216
        # unique after drop duplicatesSUBJECT_ID 39363
        # unique after drop duplicatesDRUG 4525
        # unique after drop duplicatesSTARTDATE 38497
        # unique after drop duplicatesthreshold_0.88 238
        # unique after drop duplicatesthreshold_0.95 437
        # unique after drop duplicatesthreshold_0.98 701
        # unique after drop duplicatesthreshold_0.999 2314
    print("null shape:", nuevo_df.isnull().sum())
    nuevo_df = nuevo_df.fillna(-1)
    conteo_negativos = nuevo_df.apply(lambda x: (x == -1).sum())
    print("number of other admissions:", conteo_negativos)
    return nuevo_df


#################################################################################################################################################################################################333
#funcion para concatenar archivo en el folde s_data con poalrs
def encoding2(res,categorical_cols):
    # Identificar y reemplazar las categorías que representan el 80% inferior
    for col in categorical_cols:
        counts = res[col].value_counts(normalize=True)
        lower_80 = counts[counts.cumsum() > 0.8].index
        res[col] = res[col].replace(lower_80, 'Otra')

    # Aplicar One Hot Encoding
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_cols = encoder.fit_transform(res[categorical_cols])
    encoded_cols_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names(categorical_cols))

    # Concatenar el DataFrame original con el DataFrame codificado
    res_final = pd.concat([res[[i for i in res.columns if i not in categorical_cols]], encoded_cols_df], axis=1)



    for col in categorical_cols:
        print(res[col].unique())
        #res[col] = res[col].replace('Not specified', 'Otra')

    res_final.to_csv("generative_input/"+name_encodeing)      
    return res_final

# Lee todos los archivos con la extensión especificada
def concat_archivo_primeto(procedures,admi,ruta_archivos,save,nom_archivo):
    df = pl.read_csv(admi)
    df_filtered = df
 
    df_filtered = df_filtered.with_columns(pl.col("SUBJECT_ID").cast(pl.Utf8))
    df_filtered = df_filtered.with_columns(pl.col("HADM_ID").cast(pl.Utf8))
    archivos = glob.glob(ruta_archivos)
    archivos
    df_filtered.shape
    unique_subid = []

    for i in archivos:
        aux =  pl.read_csv(i,infer_schema_length=0
                        )
    
        try:
            aux = aux.with_columns(pl.col("SUBJECT_ID").cast(pl.Utf8))
            aux = aux.with_columns(pl.col("HADM_ID").cast(pl.Utf8))
            aux = aux.groupby(['SUBJECT_ID','HADM_ID'], maintain_order=True).all()
            #NOTA  SE ELIMIA ICUSTARY_ID del archivo 'dataset/ICUSTAYS.csv.gz' ya que esta duplicado"ICUSTAY_ID"
            aux = aux.select(pl.exclude("ROW_ID"))
            if i == 'dataset/ICUSTAYS.csv.gz':
                aux = aux.select(pl.exclude("ICUSTAY_ID"))
                
            elif i ==procedures:    
                aux = aux.select(pl.exclude("SEQ_NUM"))
            df_filtered=df_filtered.join(aux, on=['SUBJECT_ID','HADM_ID'], how="left")

            #df_filtered = pd.merge(df_filtered, aux, on=['SUBJECT_ID','HADM_ID'], how='left')
            print("concat, "+i)
        except:
            
                
            aux = aux.with_columns(pl.col("SUBJECT_ID").cast(pl.Utf8))
            aux = aux.filter(pl.col('SUBJECT_ID').is_in( ids["0"]))
            #aux = aux.groupby(['SUBJECT_ID'], maintain_order=True).all()
            
            aux = aux.select(pl.exclude("ROW_ID"))
            df_filtered=df_filtered.join(aux, on=['SUBJECT_ID'], how="left")
            unique_subid.append(i)
            print(i)
    if save == True:


# Ahora intenta escribir el archivo
        df_filtered.write_parquet(nom_archivo)
        print(df_filtered)     

# Assuming df1, df2, df3 are your dataframes
#df_drugs =pd.read_csv('./input_model_pred_drugs_u/ATC3_outs_visit_non_filtered.csv')
#df_diagnosis = pd.read_csv('./input_model_pred_diagnosis_u/CCS_CODES_diagnosis_outs_visit_non_filtered.csv')
#df_procedures = pd.read_csv('./input_model_visit_procedures/CCS CODES_proc_outs_visit_non_filtered.csv')
# Drop the columns categotical

#funcion para poder concatenar los 3 inputs, manteniendo las columasn del mayot



#n = nuevo_df["icd9_category"].unique()



#######parte del preprocesamiento###################


##########nulos##########

# Reemplazar 'Not specified' con 'Otra'



#from utils import getConnection



# Supongamos que ya tienes un DataFrame 'df' con las columnas 'subject_id', 'ham_id' y 'lista_codigos'


def convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        if 'V' in value:
            return 18
        elif 'E' in value:
            return 19
        else:
            return None  # Replace "none" with "None"
def diagnosis2(d1,n,name):
   
    df_ = pl.read_csv(d1)
    print("Initial df shape with out desconcatenate: ", df_.shape)
    df_filtered = df_.with_columns(pl.col("SUBJECT_ID").cast(pl.Utf8))
    df_filtered = df_filtered.with_columns(pl.col("HADM_ID").cast(pl.Utf8))
    txt =df_filtered[["HADM_ID","SUBJECT_ID","ICD9_CODE"]].to_pandas()
    for i in txt.columns:    
        print("unique"+str(i),txt[i].nunique() )
        
    real="ICD9_CODE"

    df_filtered =df_filtered.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'])
    #funcion_acum(unique_values,n,name)
    print("df_filtered.shape",df_filtered.shape)
   
    plot_sparse_matrix_functionv2(total_number_of_admissions, unique_values,real,percentile=80)
    

    nuevos_datos = descocatenar_codes(txt,name)
    print("Initial df shape  desconcatenate: ", nuevos_datos.shape)
    #map to ixd codes
    nuevo_df = codes_diag(nuevos_datos)
    print("Númber of null values after obtaining CCS codes and Level 3", nuevo_df.isnull().sum())
    
    
    nuevo_df = funcion_acum2(nuevo_df,n,name)
        
    print("Númber of null values after obtaining threshhold", nuevo_df.isnull().sum())
    print("df shape : ", nuevo_df.shape)
   
    nuevo_df = nuevo_df.fillna(-1)
        
    for i in nuevo_df.columns:    
            print("unique"+str(i),nuevo_df[i].nunique() )
            try:
               print("Númber of -1 not mappes"+str(i),nuevo_df[i],(nuevo_df[i ]==-1).value_counts()[True])
            except:  
                print("No existen -1") 
            print(" Unique codes that do not mapp: ", nuevo_df[nuevo_df["CCS CODES"]==-1]["ICD9_CODE"].nunique())
            print(" Unique codes that do not mapp: ", nuevo_df[nuevo_df["CCS CODES"]==-1]["ICD9_CODE"].nunique()/nuevo_df["ICD9_CODE"].nunique())
    
    return nuevo_df

import polars as pl
import pandas as pd

def diagnosis(d1, n, name,plot_matrix):
    
    #columns ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE']\\
    #Unique values in SEQ_NUM: 40   
    #Initial df shape with out desconcatenate:  (651047, 5)   
    # Read the CSV file using Polars
    df_ = pl.read_csv(d1)
    print("Shape of input DataFrame:", df_.shape)
   
    # Print unique values for each column
    for col in df_.columns:    
        unique_count = df_.select(pl.col(col).n_unique()).item()
        print(f"Unique values in {col}: {unique_count}")
   
    print("Initial df shape without desconcatenate: ", df_.shape)
    
    # Cast columns to strings
    df_filtered = df_.with_columns([
        pl.col("SUBJECT_ID").cast(pl.Utf8),
        pl.col("HADM_ID").cast(pl.Utf8)
    ])
    
    # Convert to pandas for further processing
    txt = df_filtered.select(["HADM_ID", "SUBJECT_ID", "ICD9_CODE"]).to_pandas()
    
    total_number_of_admissions = txt[["HADM_ID","SUBJECT_ID","ICD9_CODE"]]
    
    print("total_number_of_admissions.shape",total_number_of_admissions.shape)

    real="ICD9_CODE"
    unique_values =txt.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'])
    print("number of duplicates ", total_number_of_admissions.shape[0]-unique_values.shape[0])
    print("percenta of ",unique_values.shape[0]/total_number_of_admissions.shape[0])
 
    #funcion_acum(unique_values,n,name)
    print("drug_presence_admission.shape",unique_values.shape)
   
    #plots 
    if plot_matrix:
        plot_sparse_matrix_functionv2(total_number_of_admissions, unique_values,real,percentile=80)
    
    for i in txt.columns:    
        print(f"unique {i}", txt[i].nunique())
           # uniqueHADM_ID 58976
    # uniqueSUBJECT_ID 46520
    # uniqueICD9_CODE 6984 

    # unique after eliminating naHADM_ID 58929
    # unique after eliminating naSUBJECT_ID 46517
    # unique after eliminating naICD9_CODE 6984
    nuevos_datos = descocatenar_codes(txt, name)
    print("Initial df shape after desconcatenate: ", nuevos_datos.shape)
    
    # Continue with the rest of your function...
    # Make sure to use pandas methods from this point on

    nuevo_df = codes_diag(nuevos_datos)
    print("Number of null values after obtaining CCS codes and Level 3", nuevo_df.isnull().sum())
    print("number of ICDE.codes not mapped", nuevo_df[nuevo_df["CCS CODES"].isnull()]["ICD9_CODE"].nunique())
    print("number of LEVE3 CODES not mapped", nuevo_df[nuevo_df["LEVE3 CODES"].isnull()]["ICD9_CODE"].nunique())
    nuevo_df = funcion_acum(nuevo_df, n, name)
    #111Númber of null values after obtaining CCS codes and Level 3 HADM_ID        0
    # SUBJECT_ID     0
    # ICD9_CODE      0
    # CCS CODES      0
    # LEVE3 CODES    0
    # dtype: int64
  
        
    print("Number of null values after obtaining threshold", nuevo_df.isnull().sum())
    print("df shape : ", nuevo_df.shape)
    
    #Númber of null values after obtaining threshhold HADM_ID            0
    # SUBJECT_ID         0
    # ICD9_CODE          0
    # CCS CODES          0
    # LEVE3 CODES        0
    # threshold_0.88     0
    # threshold_0.95     0
    # threshold_0.98     0
    # threshold_0.999    0
    # dtype: int64
    #     # df shape :  (651000, 9)
   
    nuevo_df = nuevo_df.fillna(-1)
        
    for i in nuevo_df.columns:    
        print(f"unique {i}", nuevo_df[i].nunique())
        try:
            print(f"Number of -1 not mapped {i}", (nuevo_df[i] == -1).sum())
        except:  
            print("No -1 values exist") 
    
    print("Unique codes that do not map: ", nuevo_df[nuevo_df["CCS CODES"] == -1]["ICD9_CODE"].nunique())
    print("Proportion of unique codes that do not map: ", 
          nuevo_df[nuevo_df["CCS CODES"] == -1]["ICD9_CODE"].nunique() / nuevo_df["ICD9_CODE"].nunique())
    #     Count-1: HADM_ID                0
    # SUBJECT_ID             0
    # ICD9_CODE              0
    # CCS CODES              0
    # LEVE3 CODES            0
    # threshold_0.88     78142
    # threshold_0.95     32565
    # threshold_0.98     13029
    # threshold_0.999      652    
    save_pickle(nuevo_df,"/Users/cynthiagarcia/Desktop/Synthetic-Data-Deep-Learning/data/intermedi/SD/inpput/second/diagnosis/diagnosis_total.pkl")
    return nuevo_df
def ccsc_codes(nuevo_df):
    
    
    ccs = pd.read_csv("data/raw/suplement/$prref 2015.csv")
    ccs["CCS CODES"] = ccs["'CCS CATEGORY'"].replace(r'\s+', '', regex=True)
    ccs["ICD9_CODE"]= ccs["'ICD-9-CM CODE'"].str.replace(r'[^0-9]', '', regex=True)
    ccs["ICD9_CODE"]  = ccs["ICD9_CODE"].astype(str)
    nuevo_df["ICD9_CODE"]  = nuevo_df["ICD9_CODE"].astype(str)
    nuevo_df = nuevo_df.drop(columns="CCS CODES")
    resultado_inner_join = pd.merge(nuevo_df, ccs[["ICD9_CODE","CCS CODES"]], on="ICD9_CODE", how='left')

    ccs.head()
    return resultado_inner_join


def procedures(d2,n,name,plot_matrix=True):
    real = name
    # read original file
    df_ = pl.read_csv(d2)
    #'ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE' 
    #shape 240 095
    df_filtered = df_.with_columns(pl.col("SUBJECT_ID").cast(pl.Utf8))
    df_filtered = df_filtered.with_columns(pl.col("HADM_ID").cast(pl.Utf8))
    print("df_filtered.shape",df_filtered.shape)
    
    #subset of procedures only  care about the columns that have admision subject
    nuevos_datos =df_filtered[["HADM_ID","SUBJECT_ID","ICD9_CODE"]].to_pandas()
    print("nuevos_datos.ICD9_CODE.nunique()",nuevos_datos.ICD9_CODE.nunique())
    for i in nuevos_datos.columns:    
        print("unique"+str(i),nuevos_datos[i].nunique() )
    #valores unicos: uniqueHADM_ID 52243
    # uniqueSUBJECT_ID 42214
    # uniqueICD9_CODE 2009
    # Númber of null values after obtaining individual values of ICD9-CODES :  HADM_ID       0
    # SUBJECT_ID    0
    # ICD9_CODE     0    
    print("Númber of null values after obtaining individual values of ICD9-CODES : ", nuevos_datos.isnull().sum())
    nuevo_df = nuevos_datos.dropna()
    print("nuevo_df.shape",nuevo_df.shape)
    for i in nuevo_df.columns:    
        print("unique"+str(i),nuevo_df[i].nunique() )
    print("Númber of null values dropna : ", nuevos_datos.isnull().sum())
    
    total_number_of_admissions = nuevos_datos[["HADM_ID","SUBJECT_ID","ICD9_CODE"]]
    print("total_number_of_admissions.shape",total_number_of_admissions.shape)

    real="ICD9_CODE"

    unique_values =nuevos_datos.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'])
    #funcion_acum(unique_values,n,name)
    print("drug_presence_admission.shape",unique_values.shape)
    if plot_matrix:
       plot_sparse_matrix_functionv2(total_number_of_admissions, unique_values,real,percentile=80)
    
    print("Shape df : ", nuevos_datos.shape)
    #maps to icd9 codes
    nuevo_df = codes_diag(nuevos_datos)
    #nukk valuesCCS CODES   145,494  shape df (240095= usando la apu
    print("number of ICDE.codes not mapped", nuevo_df[nuevo_df["CCS CODES"].isnull()]["ICD9_CODE"].nunique())
    print("Númber of null values dropna : ", nuevo_df.isnull().sum())
    nuevo_df = ccsc_codes(nuevo_df)
    #usando la apiCCS CODES      12422 total 240095\
        
    print("Númber of null values after obtaining CCS codes and Level 3", nuevo_df.isnull().sum())
    print("nuevo_df : ", nuevo_df.shape)
    #threshold 
    print("number of ICDE.codes not mapped", nuevo_df[nuevo_df["CCS CODES"].isnull()]["ICD9_CODE"].nunique())
    nuevo_df = funcion_acum(nuevo_df,n,name)

   
    print("Númber of null values after obtaining threshhold", nuevo_df.isnull().sum())
    print("df shape : ", nuevo_df.shape)
    # -1 for does ccs codes and level codes that had no direct mapping
    #nuevo_df = nuevo_df.fillna(-1)
    print("Númber of null values after eliminating null:", nuevo_df.isnull().sum())
   
    
    print("df shape : ", nuevo_df.shape)
    
    print(" Unique codes that do not mapp: ", nuevo_df[nuevo_df["CCS CODES"]==-1]["ICD9_CODE"].nunique())
    print(" Unique codes that do not mapp: ", nuevo_df[nuevo_df["CCS CODES"]==-1]["ICD9_CODE"].nunique()/nuevo_df["ICD9_CODE"].nunique())
    print("unique ICD9_CODE",nuevo_df["ICD9_CODE"].nunique() )
    print("unique ICD9_CODE",nuevo_df["CCS CODES"].nunique() )
    save_pickle(nuevo_df,"/Users/cynthiagarcia/Desktop/Synthetic-Data-Deep-Learning/data/intermedi/SD/inpput/second/procedures/ procedures_total.pkl")
    # #unique
    #     unique ICD9_CODE 2009
    # unique ICD9_CODE 211
    # uniqueHADM_ID 52243
    # uniqueSUBJECT_ID 42214
    # uniqueICD9_CODE 2009
    # uniqueLEVE3 CODES 711
    # uniqueCCS CODES 211
    # uniquethreshold_0.88 216
    # uniquethreshold_0.95 443
    # uniquethreshold_0.98 749
    # uniquethreshold_0.999 1769
    
    
    for i in nuevo_df.columns:    
            print("unique"+str(i),nuevo_df[i].nunique() )
  
    
    nuevo_df[real] = nuevo_df[real].astype(str)
    nuevo_df[real]=[item.replace("'", '') for item in nuevo_df[real]]
    nuevo_df[real ] = nuevo_df[real].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    print(nuevo_df.shape)
    nuevo_df[real ] = nuevo_df[real ].fillna(-1)
    print(nuevo_df.shape)
    try:
        print("Númber of -1 not mappes",(nuevo_df[real ]==-1).value_counts()[True])    
    except:
        pass        
    #ccs nt mapped 12422
    print(nuevo_df.shape)
    
    return nuevo_df

def procedures2(d2,n,name):

    df_ = pl.read_csv(d2)
    df_filtered = df_.with_columns(pl.col("SUBJECT_ID").cast(pl.Utf8))
    df_filtered = df_filtered.with_columns(pl.col("HADM_ID").cast(pl.Utf8))
    nuevos_datos =df_filtered[["HADM_ID","SUBJECT_ID","ICD9_CODE"]].to_pandas()
    for i in nuevos_datos.columns:    
        print("unique"+str(i),nuevos_datos[i].nunique() )
    print("Númber of null values after obtaining individual values of ICD9-CODES : ", nuevos_datos.isnull().sum())
    nuevo_df = nuevos_datos.dropna()
    for i in nuevo_df.columns:    
        print("unique"+str(i),nuevo_df[i].nunique() )
    print("Númber of null values dropna : ", nuevos_datos.isnull().sum())
    
    print("Shape df : ", nuevos_datos.shape)
    #maps to icd9 codes
    nuevo_df = codes_diag(nuevos_datos)
    nuevo_df = ccsc_codes(nuevo_df)
    print("Númber of null values after obtaining CCS codes and Level 3", nuevo_df.isnull().sum())
    print("nuevo_df : ", nuevo_df.shape)
    #threshold 
    nuevo_df = funcion_acum2(nuevo_df,n,name)

   
    print("Númber of null values after obtaining threshhold", nuevo_df.isnull().sum())
    print("df shape : ", nuevo_df.shape)
    # -1 for does ccs codes and level codes that had no direct mapping
    #nuevo_df = nuevo_df.fillna(-1)
    print("Númber of null values after eliminating null:", nuevo_df.isnull().sum())
   
    
    print("df shape : ", nuevo_df.shape)
    
    print(" Unique codes that do not mapp: ", nuevo_df[nuevo_df["CCS CODES"]==-1]["ICD9_CODE"].nunique())
    print(" Unique codes that do not mapp: ", nuevo_df[nuevo_df["CCS CODES"]==-1]["ICD9_CODE"].nunique()/nuevo_df["ICD9_CODE"].nunique())
    print("unique ICD9_CODE",nuevo_df["ICD9_CODE"].nunique() )
    print("unique ICD9_CODE",nuevo_df["CCS CODES"].nunique() )
    for i in nuevo_df.columns:    
            print("unique"+str(i),nuevo_df[i].nunique() )
  
    real = "CCS CODES"
    nuevo_df[real] = nuevo_df[real].astype(str)
    nuevo_df[real]=[item.replace("'", '') for item in nuevo_df[real]]
    nuevo_df[real ] = nuevo_df[real].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    nuevo_df[real ] = nuevo_df[real ].fillna(-1)
    print("Númber of -1 not mappes",(nuevo_df[real ]==-1).value_counts()[True])        
    
    return nuevo_df


def descocatenar_codes(txt,name):
    # nuevos_datos = []
    # print(txt)
    # for index, row in txt.iterrows():
    #     subject_id = row['SUBJECT_ID']
    #     ham_id = row['HADM_ID']
    #     lista_codigos = row[name]
    #     if lista_codigos is not None:
    #         for codigo in lista_codigos:
    #             nuevo_registro = {
    #                 'SUBJECT_ID': subject_id,
    #                 'HADM_ID': ham_id,
    #                name: codigo
    #             }
    #             nuevos_datos.append(nuevo_registro)
    # nuevo_df = pd.DataFrame(nuevos_datos)
    print(txt)
    for i in txt.columns:    
         print("unique before eliminating na"+str(i),txt[i].nunique() )

    nuevo_df = txt.dropna()
    for i in nuevo_df.columns:    
        print("unique after eliminating na"+str(i),nuevo_df[i].nunique() )

    
    
    return nuevo_df

def convert_to_int(value):
    try:
        return str(value)
    except ValueError:
        if 'V' in value:
            return 18
        elif 'E' in value:
            return 19
        else:
            return None

def codes_diag(nuevo_df):
    nuevo_df["ICD9_CODE"] = nuevo_df["ICD9_CODE"].apply(convert_to_int)
    nuevo_df["ICD9_CODE"] = nuevo_df["ICD9_CODE"].astype(str) 
    icd9codes = list(nuevo_df["ICD9_CODE"])
    mapper = Mapper()
    nuevo_df["CCS CODES"]  =  mapper.map(icd9codes, mapper='icd9toccs')
    nuevo_df["LEVE3 CODES"]  =  mapper.map(icd9codes, mapper='icd9tolevel3')
    return nuevo_df



#n = nuevo_df["icd9_category"].unique()
def cumulative_plotvv2(icd9_codes, num_bins,threshold_value,cat,save_path):
    # Create a DataFrame with ICD-9 codes and their frequencies
    icd9_df = pd.DataFrame(icd9_codes, columns=['ICD-9 Code'])
    icd9_df['Frequency'] = icd9_df['ICD-9 Code'].map(icd9_df['ICD-9 Code'].value_counts())
    icd9_df= icd9_df.sort_values(by='Frequency', ascending=False)

    # Drop duplicate rows to get unique ICD-9 codes and their frequencies
    unique_icd9_df = icd9_df.drop_duplicates().sort_values(by='Frequency', ascending=False)

    # Calculate cumulative frequency percentage
    unique_icd9_df['Cumulative Frequency'] = unique_icd9_df['Frequency'].cumsum()
    total_frequency = unique_icd9_df['Cumulative Frequency'].iloc[-1]
    unique_icd9_df['Cumulative F percentage'] = unique_icd9_df['Cumulative Frequency'] / total_frequency

    # Create the plot using Matplotlib
    fig, ax1 = plt.subplots(figsize=(12, 8))  # Increased figure size for better readability

    # Histogram with fewer bins
      # Adjust the number of bins
    n, bins, patches = ax1.hist(icd9_df['ICD-9 Code'], bins=num_bins, color='blue', alpha=0.7)
    ax1.set_xlabel('ICD-9 Codes', fontsize=12)
    ax1.set_ylabel('Frequency', color='blue', fontsize=12)

    # Add a vertical dashed line at the threshold value

    threshold_x_value = unique_icd9_df[unique_icd9_df["Cumulative F percentage"] >= threshold_value].sort_values(by="Cumulative F percentage", ascending=False).iloc[-1]['ICD-9 Code']
    
    bins_before_threshold = unique_icd9_df[unique_icd9_df["Cumulative F percentage"] < threshold_value].sort_values(by="Cumulative F percentage", ascending=False)['ICD-9 Code'].nunique()
    bins_before_threshold_i = unique_icd9_df[unique_icd9_df["Cumulative F percentage"] < threshold_value].sort_values(by="Cumulative F percentage", ascending=False)['ICD-9 Code'].unique()

    ax1.axvline(x=threshold_x_value, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold_value:.2f}')
    ax1.annotate(f'{bins_before_threshold} Bins Before Threshold', xy=(threshold_x_value, 0), xytext=(10, 20), 
                 textcoords='offset points', fontsize=10, color='red')

    # Customize the plot
    ax1.set_title(f'Histogram of: {cat}', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # Create a secondary y-axis for cumulative frequency
    ax2 = ax1.twinx()
    ax2.plot(unique_icd9_df['ICD-9 Code'], unique_icd9_df['Cumulative F percentage'], color='green', label='Cumulative Frequency')
    ax2.set_ylabel('Cumulative Frequency', color='green', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='green', labelsize=10)

    # Adjust legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(1.3, 0.5), fontsize=10)

    # Hide x-axis tick labels
    ax1.set_xticks([])

    # Adjust layout and show plot
    plt.tight_layout()
    
    # Save the plot if a save path is provided
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Show the plot
    plt.show()
    plt.close()

    return bins_before_threshold, bins_before_threshold_i


def cumulative_plot(icd9_codes, num_bins,threshold_value,cat):
    # Create a DataFrame with ICD-9 codes and their frequencies
    icd9_df = pd.DataFrame(icd9_codes, columns=['ICD-9 Code'])
    icd9_df['Frequency'] = icd9_df['ICD-9 Code'].map(icd9_df['ICD-9 Code'].value_counts())
    icd9_df= icd9_df.sort_values(by='Frequency', ascending=False)

    # Drop duplicate rows to get unique ICD-9 codes and their frequencies
    unique_icd9_df = icd9_df.drop_duplicates().sort_values(by='Frequency', ascending=False)

    # Calculate cumulative frequency percentage
    unique_icd9_df['Cumulative Frequency'] = unique_icd9_df['Frequency'].cumsum()
    total_frequency = unique_icd9_df['Cumulative Frequency'].iloc[-1]
    unique_icd9_df['Cumulative F percentage'] = unique_icd9_df['Cumulative Frequency'] / total_frequency

    # Create the plot using Matplotlib
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Adjust the figsize as needed

    # Histogram with fewer bins
      # Adjust the number of bins
    n, bins, patches = ax1.hist(icd9_df['ICD-9 Code'], bins=num_bins, color='blue', alpha=0.7)
    ax1.set_xlabel('ICD-9 Codes')
    ax1.set_ylabel('Frequency', color='blue')

    # Add a vertical dashed line at the threshold value

    threshold_x_value = unique_icd9_df[unique_icd9_df["Cumulative F percentage"] >= threshold_value].sort_values(by="Cumulative F percentage", ascending=False).iloc[-1]['ICD-9 Code']
    
    bins_before_threshold = unique_icd9_df[unique_icd9_df["Cumulative F percentage"] < threshold_value].sort_values(by="Cumulative F percentage", ascending=False)['ICD-9 Code'].nunique()
    bins_before_threshold_i = unique_icd9_df[unique_icd9_df["Cumulative F percentage"] < threshold_value].sort_values(by="Cumulative F percentage", ascending=False)['ICD-9 Code'].unique()

        
    ax1.axvline(x=threshold_x_value, color='red', linestyle='--', linewidth=2, label='Threshold: ' + str(threshold_value) )
    ax1.annotate(f'{bins_before_threshold} Bins Before Threshold', xy=(threshold_x_value, 0), xytext=(10, 20), textcoords='offset points', fontsize=10, color='red')

        
    
    #percentage = count / total_frequency * 100
    #ax1.annotate(f'\n{i}', xy=(bins[i] + (bins[i+1] - bins[i])/2, count), ha='center', va='bottom', fontsize=10)
        #ax1.annotate(f'{int(count)}', xy=(bins[i] + (bins[i+1] - bins[i])/2, count), ha='center', va='bottom', fontsize=10, )

    # Customize the plot
    ax1.set_title('Histogram of:' +str(cat))
    ax1.legend(loc='upper right')



    # Create a secondary y-axis for cumulative frequency
    ax2 = ax1.twinx()
    ax2.plot(unique_icd9_df['ICD-9 Code'], unique_icd9_df['Cumulative F percentage'], color='green', label='Cumulative Frequency')
    ax2.set_ylabel('Cumulative Frequency', color='green')
    ax2.legend(loc='upper right')

    legend_position = (1, .2)  # Adjust the position as needed
    ax1.legend(loc='upper right', bbox_to_anchor=legend_position)
    ax2.legend(loc='lower right', bbox_to_anchor=legend_position)

    # Hide x-axis tick labels
    ax1.set_xticks([])

    # Show the plot
    plt.tight_layout()  # Adjust layout for labels
    #plt.show()
    #try:
    #   plt.savefig(IMAGES_Demo/'thresholds_'/str(threshold_value)/'.svg')
    #except:
    #   plt.savefig(Path('..')/IMAGES_Demo/'thresholds_'str(threshold_value)'.svg')     
    return bins_before_threshold,bins_before_threshold_i


def asignar_valor(series, lista_especifica):
    # Usamos una comprensión de lista para asignar "Otro" a los valores que no están en la lista
    nueva_serie = series.apply(lambda x: x if x in lista_especifica else -1)
    return nueva_serie


    bins_before_threshold = []
    bins_before_threshold_index = []
    nuevo_df = nuevo_df.to_pandas()
    # 1. All drug administrations per admission (considering unique combinations)
    administrations_per_admission = nuevo_df.groupby(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'DRUG', 'STARTDATE', 'ENDDATE']).size().groupby(level=[0,1]).sum().reset_index(name='administrations')
    
    # 2. Drug presence (0/1) per admission
    drug_presence_admission = nuevo_df.groupby(['SUBJECT_ID', 'HADM_ID', 'DRUG']).size().unstack(fill_value=0)
    drug_presence_admission = (drug_presence_admission > 0).astype(int)
    drugs_present_per_admission = drug_presence_admission.sum(axis=1).reset_index(name='unique_drugs')
    
    for i in n:
        # For all drug administrations
        num_bins_admin = len(administrations_per_admission['administrations'].unique())
        admin_codes = list(administrations_per_admission['administrations'])
        a_admin, b_admin = cumulative_plotv2(admin_codes, num_bins_admin, i, f"Administrations (threshold {i})")
        
        # For drug presence
        num_bins_presence = len(drugs_present_per_admission['unique_drugs'].unique())
        presence_codes = list(drugs_present_per_admission['unique_drugs'])
        a_presence, b_presence = cumulative_plotv2(presence_codes, num_bins_presence, i, f"Drug Presence (threshold {i})")
        
        bins_before_threshold.extend([a_admin, a_presence])
        bins_before_threshold_index.extend(list(b_admin) + list(b_presence))
        
        # Assign values for administrations
        serie_original_admin = administrations_per_admission['administrations']
        serie_modificada_admin = asignar_valorv2(serie_original_admin, b_admin)
        administrations_per_admission[f"threshold_admin_{i}"] = serie_modificada_admin
        
        # Assign values for drug presence
        serie_original_presence = drugs_present_per_admission['unique_drugs']
        serie_modificada_presence = asignar_valor(serie_original_presence, b_presence)
        drugs_present_per_admission[f"threshold_presence_{i}"] = serie_modificada_presence
    
    # Merge the results back to the original dataframe
    nuevo_df = nuevo_df.merge(administrations_per_admission[['SUBJECT_ID', 'HADM_ID'] + [col for col in administrations_per_admission.columns if 'threshold' in col]], 
                              on=['SUBJECT_ID', 'HADM_ID'], how='left')
    nuevo_df = nuevo_df.merge(drugs_present_per_admission[['SUBJECT_ID', 'HADM_ID'] + [col for col in drugs_present_per_admission.columns if 'threshold' in col]], 
                              on=['SUBJECT_ID', 'HADM_ID'], how='left')
    
    return nuevo_df



def cumulative_plotv2(codes, num_bins, threshold_value, cat):
    # Create a DataFrame with codes and their frequencies
    df = pd.DataFrame(codes, columns=['Code'])
    df['Frequency'] = df['Code'].map(df['Code'].value_counts())
    df = df.sort_values(by='Frequency', ascending=False)

    # Drop duplicate rows to get unique codes and their frequencies
    unique_df = df.drop_duplicates().sort_values(by='Frequency', ascending=False)

    # Calculate cumulative frequency percentage
    unique_df['Cumulative Frequency'] = unique_df['Frequency'].cumsum()
    total_frequency = unique_df['Cumulative Frequency'].iloc[-1]
    unique_df['Cumulative F percentage'] = unique_df['Cumulative Frequency'] / total_frequency

    # Find the threshold
    threshold_value = unique_df[unique_df["Cumulative F percentage"] >= threshold_value].sort_values(by="Cumulative F percentage", ascending=False).iloc[-1]['Code']
    
    bins_before_threshold = unique_df[unique_df["Cumulative F percentage"] < threshold_value].sort_values(by="Cumulative F percentage", ascending=False)['Code'].nunique()
    bins_before_threshold_i = unique_df[unique_df["Cumulative F percentage"] < threshold_value].sort_values(by="Cumulative F percentage", ascending=False)['Code'].unique()

    return bins_before_threshold, bins_before_threshold_i

def asignar_valorv2(series, lista_especifica):
    nueva_serie = series.apply(lambda x: x if x in lista_especifica else -1)
    return nueva_serie

def funcion_acum(nuevo_df,n,name):
    
    bins_before_threshold = []
    bins_before_threshold_index = []
    for i in n:
        aux_thresh = nuevo_df.copy()
        #aux_thresh = nuevo_df[nuevo_df["icd9_category"]==i]
        num_bins = len(aux_thresh[name].unique())
        icd9_codes = list(aux_thresh[name])

        a,b = cumulative_plotvv2(icd9_codes, num_bins,i,i,"data/analysis/procedures_threshold"+str(name)+"_"+str(i)+".svg")
        bins_before_threshold.append(a)
        bins_before_threshold_index.extend(list(b))
        serie_original = nuevo_df[name]  
        lista_especifica = bins_before_threshold_index
        #lista_especifica = b

        # Llama a la función para asignar valores
        serie_modificada = asignar_valor(serie_original, lista_especifica)
        nuevo_df["threshold_"+str(i)] = serie_modificada
    return nuevo_df

def funcion_acum2(nuevo_df,n,name):
    
    bins_before_threshold = []
    bins_before_threshold_index = []
    for i in n:
        aux_thresh = nuevo_df.copy()
        #aux_thresh = nuevo_df[nuevo_df["icd9_category"]==i]
        num_bins = len(aux_thresh[name].unique())
        icd9_codes = list(aux_thresh[name])
        a,b = cumulative_plotv2(icd9_codes, num_bins,i,i)
        bins_before_threshold.append(a)
        bins_before_threshold_index.extend(list(b))
        serie_original = nuevo_df[name]  
        lista_especifica = bins_before_threshold_index
        #lista_especifica = b

        # Llama a la función para asignar valores
        serie_modificada = asignar_valorv2(serie_original, lista_especifica)
        nuevo_df["threshold_"+str(i)] = serie_modificada
    return nuevo_df
d1 = '..\s_data\PRESCRIPTIONS.csv.gz'
name1 = "DRUG"
def drugs(d1,name1):
    df_ = pl.read_csv(d1, infer_schema_length=10000, ignore_errors=True, )
    df_filtered = df_.with_columns(pl.col("SUBJECT_ID").cast(pl.Utf8))
    df_filtered = df_filtered.with_columns(pl.col("HADM_ID").cast(pl.Utf8))
    nuevo_df =df_filtered[["HADM_ID","SUBJECT_ID", "DRUG"]].to_pandas()
    for i in nuevo_df.columns:    
        print("unique"+str(i),nuevo_df[i].nunique() )
    print(nuevo_df.shape)
    nuevo_df.drop_duplicates( inplace=True)
    nuevo_df = funcion_acum(nuevo_df,n,name1)
    for i in nuevo_df.columns:    
        print("unique after droping duplicates"+str(i),nuevo_df[i].nunique() )
    nuevo_df = nuevo_df.fillna(-1)
    for i in nuevo_df.columns:    
        print("unique -1"+str(i),nuevo_df[i].nunique() )

    # Iterar sobre las columnas de umbral
    for col in [col for col in nuevo_df.columns if "threshold" in col]:
        conteo_negativos = (nuevo_df[col] == -1).sum()  # Contar ocurrencias de -1
        print(col, conteo_negativos)
        print(f"Percentage of categories for {col}: {(conteo_negativos) / nuevo_df.shape[0]}")
        print(f"Number of admissions for {col}: {nuevo_df.shape[0] - conteo_negativos}")

        # Calcular y mostrar el porcentaje de drogas mapeadas como -1 para la columna actual
        total_drogas = nuevo_df['DRUG'].nunique()  # Total de drogas únicas
        drogas_mapeadas_neg1 = nuevo_df[nuevo_df[col] == -1]['DRUG'].nunique()  # Drogas con -1 en la columna actual
        porcentaje_neg1 = (drogas_mapeadas_neg1 / total_drogas) * 100 if total_drogas > 0 else 0

        print(f"Total unique drugs: {total_drogas}")
        print(f"Drugs mapped as -1 in {col}: {drogas_mapeadas_neg1}")
        print(f"Percentage of drugs mapped as -1 in {col}: {porcentaje_neg1:.2f}%")

        # Calcular y mostrar el porcentaje de drogas que no tienen -1 para la columna actual
        drogas_sin_neg1 = nuevo_df[nuevo_df[col] != -1]['DRUG'].nunique()  # Drogas sin -1 en la columna actual
        porcentaje_sin_neg1 = (drogas_sin_neg1 / total_drogas) * 100 if total_drogas > 0 else 0

        print(f"Drugs without -1 in {col}: {drogas_sin_neg1}")
        print(f"Percentage of drugs without -1 in {col}: {porcentaje_sin_neg1:.2f}%")

    print("nuevo_df after acum", nuevo_df.isnull().sum())
    return nuevo_df

import pandas as pd
import numpy as np
from scipy.stats import mode



def max_patient_add(x):
    return x.max() - x.min()

def max_visit_add(x):
    return x - x.min()

'''def last_firs(ADMISSIONS):
    
    try: 
        ADMISSIONS[['ADMITTIME','DISCHTIME']] = ADMISSIONS[['ADMITTIME','DISCHTIME']].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
    except:
        ADMISSIONS[['ADMITTIME','DISCHTIME']] = ADMISSIONS[['ADMITTIME','DISCHTIME']].apply(lambda x: pd.to_datetime(x, infer_datetime_format=True))    
        
    ADMISSIONS["days from last visit"] = ADMISSIONS.groupby('SUBJECT_ID')['ADMITTIME'].transform(lambda x: x - x.min())
    return ADMISSIONS[["SUBJECT_ID","HADM_ID","days from last visit"]] 
'''

import pandas as pd

def last_firs(ADMISSIONS,level):
    try: 
            ADMISSIONS[['ADMITTIME','DISCHTIME']] = ADMISSIONS[['ADMITTIME','DISCHTIME']].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
    except:
            ADMISSIONS[['ADMITTIME','DISCHTIME']] = ADMISSIONS[['ADMITTIME','DISCHTIME']].apply(lambda x: pd.to_datetime(x, infer_datetime_format=True))    

    if level== "Patient":
                   
        ADMISSIONS["days from last visit"] = ADMISSIONS.groupby('SUBJECT_ID')['ADMITTIME'].transform(lambda x: x - x.min())
        ADMISSIONS["days from last visit"] =[int(i.days) for i in ADMISSIONS["days from last visit"]]

        
    else:    
    # Asegurar que ADMITTIME es tipo datetime
           
        # Ordenar por SUBJECT_ID y ADMITTIME para asegurar el orden cronológico de las visitas
        ADMISSIONS.sort_values(by=['SUBJECT_ID', 'ADMITTIME'], inplace=True)
        
        # Calcular la diferencia en tiempo hasta la visita anterior
        ADMISSIONS['days from last visit'] = ADMISSIONS.groupby('SUBJECT_ID')['ADMITTIME'].diff().dt.days
        
        # Llenar los valores NaN con cero
        ADMISSIONS['days from last visit'] = ADMISSIONS['days from last visit'].fillna(0)
        
    return ADMISSIONS[['SUBJECT_ID', 'HADM_ID', 'days from last visit']]




def calculate_pivot_df(duplicados, real, level,type_g):
    """Calculate the pivot table.
    
    Args:
        duplicados (DataFrame): Pre-cleaned DataFrame used for the pivot table.
        real (str): Name of the columns.
        level (str): Indicates whether it's at patient level or visit level.
        
    Returns:
        pivot_df (DataFrame): The pivoted DataFrame.
    """
    print("na before fillin -1", duplicados.isnull().sum())

    if type_g == "drug2":
         duplicados["SUBJECT_ID"] = duplicados["SUBJECT_ID"].astype(str)
    if type_g == "diagnosis":
       
# Assuming `duplicados` is your DataFrame and `real` is the column you're trying to convert
       duplicados[real] = pd.to_numeric(duplicados[real], errors='coerce')
       duplicados[real] = duplicados[real].fillna(-1).astype(int)
       conteo_negativos = duplicados.apply(lambda x: (x == -1).sum())
       print("Count-1:",conteo_negativos)
       
    if type_g != "drug2" and type_g != "drug1" and type_g != "drugst":   

        if real != "CCS CODES":    
            duplicados[real ] = duplicados[real ].astype(int)
        else:
            def clean_and_convert(value):
                if isinstance(value, str):
                    # Remover comillas extras y espacios
                    cleaned = value.strip("'").strip()
                    # Intentar convertir a float (para manejar decimales si los hay)
                    try:
                        return float(cleaned)
                    except ValueError:
                        return 0  # o cualquier otro valor por defecto
                return value

# Aplicar la función de limpieza a todas las columnas excepto 'subject_id' y 'hamd_id'
            columns_to_clean = [col for col in duplicados.columns if col not in ['SUBJECT_ID', 'HADM_ID']]
            for col in columns_to_clean:
                duplicados[col] = duplicados[col].apply(clean_and_convert)

    if type_g == 'drugst':
        duplicados[real] = duplicados[real].replace(-1, "Other")
           
    duplicados["SUBJECT_ID"] = duplicados["SUBJECT_ID"].astype(str)
    duplicados["HADM_ID"] = duplicados["HADM_ID"].astype(str)

    if level == "Patient":
        pivot_df = duplicados[[real, "SUBJECT_ID"]].pivot_table(
            index='SUBJECT_ID',
            columns=real,
            aggfunc='size',
            fill_value=0
        )
        columns_to_convert = [col for col in pivot_df.columns if col not in [ "SUBJECT_ID"]]

        pivot_df[columns_to_convert] = (pivot_df[columns_to_convert] >= 1).astype(int)
    


    else:
        pivot_df = duplicados[[real, "SUBJECT_ID", "HADM_ID"]].pivot_table(
            index=['SUBJECT_ID', "HADM_ID"],
            columns=real ,
            aggfunc='size',
            fill_value=0
        )
        columns_to_convert = [col for col in pivot_df.columns if col not in [ "SUBJECT_ID", "HADM_ID"]]

        pivot_df[columns_to_convert] = (pivot_df[columns_to_convert] >= 1).astype(int)

    
    
    pivot_df.reset_index(inplace=True)
    print(pivot_df.nunique().value_counts())
    unique_counts = pivot_df.nunique()

    # Filtrar columnas con valores distintos a 2
    columnas_no_binarias = unique_counts[unique_counts != 2]

    # Imprimir los resultados
    print(columnas_no_binarias)
    if type_g == "drugs":
        pivot_df['Other'] = np.where(pivot_df['Other'].notna() & (pivot_df['Other'] > 0), 1, 0)
    unique_counts = pivot_df.nunique()
    columnas_no_binarias = unique_counts[unique_counts != 2]
    print("columnas_no_binarias",columnas_no_binarias)
    return pivot_df

# X_preprocessed = preprocess(X, 'std', ['col1', 'col2', 'col3', 'col4'])

def calculate_demographics(adm,pa, categorical_cols, level,cat_considered,prod_ipvot=None):
    """Calculate the aggregated demographics DataFrame.
    
    Args:
        duplicados (DataFrame): Pre-cleaned DataFrame used for the pivot table.
        categorical_cols (list): List of categorical columns to include in aggregation.
        archivo: File containing additional data.
        level (str): Indicates whether it's at patient level or visit level.
        
    Returns:
        agregacion_cl (DataFrame): DataFrame with aggregated demographics.
    """
    adm = pd.read_csv(adm)
    aux_ad = last_firs(adm,level)
    
    pa = pd.read_csv(pa)
    adm["SUBJECT_ID"] = adm["SUBJECT_ID"].astype(str)
    adm["HADM_ID"] = adm["HADM_ID"].astype(str)
    aux_ad["SUBJECT_ID"] = aux_ad["SUBJECT_ID"].astype(str)
    aux_ad["HADM_ID"] = aux_ad["HADM_ID"].astype(str)

    pa["SUBJECT_ID"] = pa["SUBJECT_ID"].astype(str)

    for i in ["SUBJECT_ID","HADM_ID"]:    
            print("unique"+str(i),adm[i].nunique() )

    if prod_ipvot is None:
        duplicados = adm[cat_considered + ['SUBJECT_ID', 'HADM_ID']]
    else:    
        duplicados = prod_ipvot.merge(adm[cat_considered], on=['SUBJECT_ID', 'HADM_ID'], how='left')
    duplicados=duplicados.merge(pa[['SUBJECT_ID', 'GENDER', 'DOB']], on=['SUBJECT_ID'], how='left')
    duplicados=duplicados.merge(aux_ad, on=['SUBJECT_ID','HADM_ID'], how='left')
    print(duplicados.shape)
    duplicados["DISCHTIME"] = pd.to_datetime(duplicados["DISCHTIME"])
    
    
    

    #duplicados['DOB'] = pd.to_datetime(duplicados['DOB'], format='%Y-%m-%d %H:%M:%S')
    duplicados['DEATHTIME'] = pd.to_datetime(duplicados['DEATHTIME'])
    duplicados['DEATHTIME'] = pd.to_datetime(duplicados['DEATHTIME'])
    duplicados['DOB'] = pd.to_datetime(duplicados['DOB'], format='%Y-%m-%d %H:%M:%S')
    duplicados['DOB'] = pd.to_datetime(duplicados['DOB'])
    duplicados['ADMITTIME'] = pd.to_datetime(duplicados['ADMITTIME'])
    duplicados["LOSRD"] = duplicados["DISCHTIME"] - duplicados["ADMITTIME"]
    duplicados['DOB'] = [timestamp.to_pydatetime() for timestamp in duplicados['DOB']]
    duplicados['ADMITTIME'] = [timestamp.date() for timestamp in duplicados['ADMITTIME']]
    duplicados['DOB'] = [timestamp.date() for timestamp in duplicados['DOB']]
    duplicados["age"] = (duplicados['ADMITTIME'].to_numpy() - duplicados['DOB'].to_numpy())
    duplicados["year_age"] = [i.days/365 for i in duplicados["age"]]

    print("People with more than 100 years: ",duplicados.loc[duplicados['year_age'] > 100].shape)
    duplicados.loc[duplicados['year_age'] > 100, 'year_age'] = 89  
        
    
    
    
    duplicados['LOSRD']  = [i.days for i in duplicados['LOSRD']]
    print("Before eliminate LOSRD<0: ",duplicados.shape)
    duplicados = duplicados[duplicados["LOSRD"]>0]   
    for i in["SUBJECT_ID","HADM_ID"]:    
            print("unique"+str(i),duplicados[i].nunique() )

    print("After eliminate LOSRD<0: ",duplicados.shape)
    if level == "Patient":
        agregacion_cl = duplicados.groupby(['SUBJECT_ID']).agg(
            Age_max=("year_age", 'max'),
            GENDER=("GENDER", lambda x: mode(x)[0][0]),
            ADMISSION_LOCATION=("ADMISSION_LOCATION", lambda x: mode(x)[0][0]),
            ETHNICITY=('ETHNICITY', lambda x: mode(x)[0][0]),
            MARITAL_STATUS=("MARITAL_STATUS", lambda x: mode(x)[0][0]),
            RELIGION=("RELIGION", lambda x: mode(x)[0][0]),
            ADMISSION_TYPE=("ADMISSION_TYPE", lambda x: mode(x)[0][0]),
            INSURANCE=("INSURANCE", lambda x: mode(x)[0][0]),
            DISCHARGE_LOCATION=("DISCHARGE_LOCATION", lambda x: mode(x)[0][0]),
            LOSRD_sum=("LOSRD", 'sum'),
            LOSRD_avg=("LOSRD", np.mean),
            L_1s_last_p1=("days from last visit",  'max'),
            ADMITTIME_max=("ADMITTIME", 'max'),
            ADMITTIME_min=("ADMITTIME", 'min')
        )
        agregacion_cl = agregacion_cl.rename(columns={'L_1s_last_p1': 'days from last visit'})
    else:
        agregacion_cl = duplicados.groupby(['SUBJECT_ID', "HADM_ID"]).agg(
            Age_max=("year_age", 'max'),
            GENDER=("GENDER", lambda x: mode(x)[0][0]),
            ADMISSION_LOCATION=("ADMISSION_LOCATION", lambda x: mode(x)[0][0]),
            ETHNICITY=('ETHNICITY', lambda x: mode(x)[0][0]),
            MARITAL_STATUS=("MARITAL_STATUS", lambda x: mode(x)[0][0]),
            RELIGION=("RELIGION", lambda x: mode(x)[0][0]),
            ADMISSION_TYPE=("ADMISSION_TYPE", lambda x: mode(x)[0][0]),
            INSURANCE=("INSURANCE", lambda x: mode(x)[0][0]),
            DISCHARGE_LOCATION=("DISCHARGE_LOCATION", lambda x: mode(x)[0][0]),
            LOSRD_sum=("LOSRD", 'sum'),
            LOSRD_avg=("LOSRD", np.mean),
            L_1s_last_p1=("days from last visit",  'max'),
            ADMITTIME_max=("ADMITTIME", 'max'),
            ADMITTIME_min=("ADMITTIME", 'min')
        ).reset_index()
        agregacion_cl = agregacion_cl.rename(columns={'L_1s_last_p1': 'days from last visit'})

  
    #agregacion_cl["L_1s_last_p1"] = (agregacion_cl["ADMITTIME_max"] - agregacion_cl["ADMITTIME_min"]).apply(lambda x: x.days)
    #agregacion_cl["L_1s_last_p1"] =[int(i.days) for i in agregacion_cl["L_1s_last_p1"]]


    agregacion_cl = agregacion_cl.rename(columns={'L_1s_last_p1': 'days from last visit'})

    agregacion_cl = agregacion_cl[["Age_max", "LOSRD_sum", "days from last visit", "LOSRD_avg","DISCHTIME"] +categorical_cols+['SUBJECT_ID', 'HADM_ID']] 
    print("Before replacing 0: ",agregacion_cl.isnull().sum())
    #for i in ["Age_max", "LOSRD_sum", "L_1s_last_p1", "LOSRD_avg"]:
    #    agregacion_cl[i]= agregacion_cl[i].replace(np.nan,0)
    #print("After replacing 0: ",agregacion_cl.isnull().sum())
    for col in categorical_cols:
        agregacion_cl[col]= agregacion_cl[col].replace(np.nan, "Unknown")
        agregacion_cl[col] = agregacion_cl[col].replace(0, 'Unknown')
        agregacion_cl[col] = agregacion_cl[col].replace('** INFO NOT AVAILABLE **', 'Unknown')
        agregacion_cl[col] = agregacion_cl[col].replace('UNKNOWN (DEFAULT)', 'Unknown')
        agregacion_cl[col] = agregacion_cl[col].replace('UNOBTAINABLE', 'Unknown')
        agregacion_cl[col] = agregacion_cl[col].replace('OTHER', 'Unknown')
        agregacion_cl[col] = agregacion_cl[col].replace('NOT SPECIFIED', 'Unknown')
        agregacion_cl = agregacion_cl.replace('UNKNOWN/NOT SPECIFIED', 'Unknown')

    return agregacion_cl

def calculate_agregacion_cl(adm,pa, categorical_cols, level,cat_considered,prod_ipvot):
    """Calculate the aggregated demographics DataFrame.
    
    Args:
        duplicados (DataFrame): Pre-cleaned DataFrame used for the pivot table.
        categorical_cols (list): List of categorical columns to include in aggregation.
        archivo: File containing additional data.
        level (str): Indicates whether it's at patient level or visit level.
        
    Returns:
        agregacion_cl (DataFrame): DataFrame with aggregated demographics.
    """
    try:
       adm = pd.read_csv(adm)
    except:
        adm = pd.read_csv(Path('..')/adm) 
    aux_ad = last_firs(adm,level)
    
    pa = pd.read_csv(pa)
    adm["SUBJECT_ID"] = adm["SUBJECT_ID"].astype(str)
    adm["HADM_ID"] = adm["HADM_ID"].astype(str)
    aux_ad["SUBJECT_ID"] = aux_ad["SUBJECT_ID"].astype(str)
    aux_ad["HADM_ID"] = aux_ad["HADM_ID"].astype(str)

    pa["SUBJECT_ID"] = pa["SUBJECT_ID"].astype(str)

    for i in ["SUBJECT_ID","HADM_ID"]:    
            print("unique"+str(i),adm[i].nunique() )

    if level == 'Patient':
        duplicados = prod_ipvot.merge(adm[cat_considered], on=['SUBJECT_ID'], how='left')
    else:    
        duplicados = prod_ipvot.merge(adm[cat_considered], on=['SUBJECT_ID', 'HADM_ID'], how='left')
    duplicados=duplicados.merge(pa[['SUBJECT_ID', 'GENDER', 'DOB']], on=['SUBJECT_ID'], how='left')
    duplicados=duplicados.merge(aux_ad, on=['SUBJECT_ID','HADM_ID'], how='left')
    print(duplicados.shape)
    duplicados["DISCHTIME"] = pd.to_datetime(duplicados["DISCHTIME"])
    
    
    

    #duplicados['DOB'] = pd.to_datetime(duplicados['DOB'], format='%Y-%m-%d %H:%M:%S')
    duplicados['DEATHTIME'] = pd.to_datetime(duplicados['DEATHTIME'])
    duplicados['DEATHTIME'] = pd.to_datetime(duplicados['DEATHTIME'])
    duplicados['DOB'] = pd.to_datetime(duplicados['DOB'], format='%Y-%m-%d %H:%M:%S')
    duplicados['DOB'] = pd.to_datetime(duplicados['DOB'])
    duplicados['ADMITTIME'] = pd.to_datetime(duplicados['ADMITTIME'])
    duplicados["LOSRD"] = duplicados["DISCHTIME"] - duplicados["ADMITTIME"]
    duplicados['DOB'] = [timestamp.to_pydatetime() for timestamp in duplicados['DOB']]
    duplicados['ADMITTIME'] = [timestamp.date() for timestamp in duplicados['ADMITTIME']]
    duplicados['DOB'] = [timestamp.date() for timestamp in duplicados['DOB']]
    duplicados["age"] = (duplicados['ADMITTIME'].to_numpy() - duplicados['DOB'].to_numpy())
    duplicados["year_age"] = [i.days/365 for i in duplicados["age"]]

    print("People with more than 100 years: ",duplicados.loc[duplicados['year_age'] > 100].shape)
    duplicados.loc[duplicados['year_age'] > 100, 'year_age'] = 89  
        
    
    
    
    duplicados['LOSRD']  = [i.days for i in duplicados['LOSRD']]
    print("Before eliminate LOSRD<0: ",duplicados.shape)
    duplicados = duplicados[duplicados["LOSRD"]>0]   
    for i in["SUBJECT_ID","HADM_ID"]:    
            print("unique"+str(i),duplicados[i].nunique() )

    print("After eliminate LOSRD<0: ",duplicados.shape)
    if level == "Patient":
        agregacion_cl = duplicados.groupby(['SUBJECT_ID']).agg(
            Age_max=("year_age", 'max'),
            GENDER=("GENDER", lambda x: mode(x)[0][0]),
            #ADMISSION_LOCATION=("ADMISSION_LOCATION", lambda x: mode(x)[0][0]),
            ETHNICITY=('ETHNICITY', lambda x: mode(x)[0][0]),
            MARITAL_STATUS=("MARITAL_STATUS", lambda x: mode(x)[0][0]),
            RELIGION=("RELIGION", lambda x: mode(x)[0][0]),
            #ADMISSION_TYPE=("ADMISSION_TYPE", lambda x: mode(x)[0][0]),
            #INSURANCE=("INSURANCE", lambda x: mode(x)[0][0]),
            #DISCHARGE_LOCATION=("DISCHARGE_LOCATION", lambda x: mode(x)[0][0]),
            LOSRD_sum=("LOSRD", 'sum'),
            LOSRD_avg=("LOSRD", np.mean),
            L_1s_last_p1=("L_1s_last_p1",  'max'),
            ADMITTIME_max=("ADMITTIME", 'max'),
            DISCHTIME=("DISCHTIME", 'max'),
           
            
        ).reset_index()
        agregacion_cl = agregacion_cl.rename(columns={'L_1s_last_p1': 'days from last visit'})

  
    else:
        agregacion_cl = duplicados.groupby(['SUBJECT_ID', "HADM_ID"]).agg(
            Age_max=("year_age", 'max'),
            GENDER=("GENDER", lambda x: mode(x)[0][0]),
            #ADMISSION_LOCATION=("ADMISSION_LOCATION", lambda x: mode(x)[0][0]),
            ETHNICITY=('ETHNICITY', lambda x: mode(x)[0][0]),
            MARITAL_STATUS=("MARITAL_STATUS", lambda x: mode(x)[0][0]),
            RELIGION=("RELIGION", lambda x: mode(x)[0][0]),
            #ADMISSION_TYPE=("ADMISSION_TYPE", lambda x: mode(x)[0][0]),
            #INSURANCE=("INSURANCE", lambda x: mode(x)[0][0]),
            #DISCHARGE_LOCATION=("DISCHARGE_LOCATION", lambda x: mode(x)[0][0]),
            LOSRD_sum=("LOSRD", 'sum'),
            LOSRD_avg=("LOSRD", np.mean),
            L_1s_last_p1=("days from last visit",  'max'),
            ADMITTIME_max=("ADMITTIME", 'max'),
            ADMITTIME_min=("ADMITTIME", 'min'),
             DISCHTIME=("DISCHTIME", 'max'),
        ).reset_index()
        agregacion_cl = agregacion_cl.rename(columns={'L_1s_last_p1': 'days from last visit'})

  
        
    #agregacion_cl["L_1s_last_p1"] = (agregacion_cl["ADMITTIME_max"] - agregacion_cl["ADMITTIME_min"]).apply(lambda x: x.days)
    #agregacion_cl["days from last visit"] =[int(i.days) for i in agregacion_cl["days from last visit"]]

    if level == 'Patient':
        agregacion_cl = agregacion_cl[["Age_max", "LOSRD_sum", "days from last visit", "LOSRD_avg","DISCHTIME","ADMITTIME_max","ADMITTIME_min"] +categorical_cols+['SUBJECT_ID']] 
    else:
         agregacion_cl = agregacion_cl[["Age_max", "LOSRD_sum", "days from last visit", "LOSRD_avg","DISCHTIME","ADMITTIME_max"] +categorical_cols+['SUBJECT_ID', 'HADM_ID']] 
    print("Before replacing 0: ",agregacion_cl.isnull().sum())
    #for i in ["Age_max", "LOSRD_sum", "days from last visit", "LOSRD_avg"]:
    #    agregacion_cl[i]= agregacion_cl[i].replace(np.nan,0)
    #print("After replacing 0: ",agregacion_cl.isnull().sum())
    for col in categorical_cols:
        agregacion_cl[col]= agregacion_cl[col].replace(np.nan, "Unknown")
        agregacion_cl[col] = agregacion_cl[col].replace(0, 'Unknown')
        agregacion_cl[col] = agregacion_cl[col].replace('** INFO NOT AVAILABLE **', 'Unknown')
        agregacion_cl[col] = agregacion_cl[col].replace('UNKNOWN (DEFAULT)', 'Unknown')
        agregacion_cl[col] = agregacion_cl[col].replace('UNOBTAINABLE', 'Unknown')
        agregacion_cl[col] = agregacion_cl[col].replace('OTHER', 'Unknown')
        agregacion_cl[col] = agregacion_cl[col].replace('NOT SPECIFIED', 'Unknown')
        agregacion_cl = agregacion_cl.replace('UNKNOWN/NOT SPECIFIED', 'Unknown')
    
    #add admitted time
    agregacion_cl['ADMITTIME_max'] = pd.to_datetime(agregacion_cl['ADMITTIME_max'])

    agregacion_cl["year"] = agregacion_cl['ADMITTIME_max'].dt.year
    agregacion_cl['month'] = agregacion_cl['ADMITTIME_max'].dt.month
    agregacion_cl['visit_rank'] = agregacion_cl.groupby('SUBJECT_ID')['ADMITTIME_max'].rank(method='first').astype(int)

    return agregacion_cl

def merge_df(agregacion_cl, prod_ipvot,level):
        if level == "Patient":
            return agregacion_cl.merge(prod_ipvot, on=['SUBJECT_ID'], how='left')
        else:
            return agregacion_cl.merge(prod_ipvot, on=["HADM_ID",'SUBJECT_ID'], how='left')
    
import pandas as pd
import numpy as np

import numpy as np
import pandas as pd

def normalize_count_matrix_aux(pivot_df, level):
    """
    Processes the count matrix by normalizing based on zero entries and then concatenates demographic data.
    
    :param pivot_df: DataFrame with count matrix of icd9-codes
    :param level: 'Patient', 'visit', or 'outs_visit'
    :return: DataFrame with normalized counts
    """
    # Drop identifiers based on the level
    if level == "Patient":
        matrix_df = pivot_df.drop('SUBJECT_ID', axis=1)
    else:
        matrix_df = pivot_df.drop(['HADM_ID', 'SUBJECT_ID'], axis=1)
    
    # Calculate the number of zero entries for each column
    num_zeros = (matrix_df == 0).sum()

    # Avoid division by zero by adding a small constant, if needed
    # This step depends on your specific requirements and data characteristics
    num_zeros += (num_zeros == 0)  # This will add 1 to columns with no zeros to avoid division by zero

    # Normalize each column by the number of zero entries
    normalized_matrix_df = matrix_df.div(num_zeros, axis=1)
    
    # Prepare the result DataFrame
    result_df = pd.concat([pivot_df[['SUBJECT_ID', 'HADM_ID']], normalized_matrix_df], axis=1)

    return result_df


def normalize_count_matrix__aux(pivot_df, level):
    """
    Processes the count matrix by normalizing and then concatenates demographic data.
    
    :param pivot_df: DataFrame with count matrix of icd9-codes
    :param stri: 'visit', 'Patient', or 'outs_visit'
    :param agregacion_cl: DataFrame with demographic data
    :param categorical_cols: List of categorical columns to include
    :return: Concatenated DataFrame with normalized counts and demographics
    """
    # Normalize the count matrix
    if level == "Patient":
        matrix = pivot_df.drop('SUBJECT_ID', axis=1).values
        
    else:
        matrix = pivot_df.drop(['HADM_ID','SUBJECT_ID'], axis=1).values
        
    num_non_zeros = np.count_nonzero(matrix)
    normalized_matrix = matrix / num_non_zeros  # Dividing the matrix by the number of non-zero elements
    
    # Create the result DataFrame
    result_df = pd.DataFrame(normalized_matrix, columns=pivot_df.columns.difference(['SUBJECT_ID', 'HADM_ID'], sort=False))
    if 'SUBJECT_ID' in pivot_df.columns:
        result_df['SUBJECT_ID'] = pivot_df['SUBJECT_ID']
    if 'HADM_ID' in pivot_df.columns:
        result_df['HADM_ID'] = pivot_df['HADM_ID']
    
    # Concatenate with demographic data
       
    return result_df

from sklearn.preprocessing import FunctionTransformer


from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

def encoding(res, categorical_cols, encoding_type='onehot',proportion =False,prop=0.09 ):
    """
    Aplica codificación a las columnas categóricas de un DataFrame.

    Parámetros:
    - res: DataFrame original.
    - categorical_cols: Lista de columnas categóricas para codificar.
    - encoding_type: Tipo de codificación ('onehot' o 'label').
    - output_file_name: Nombre del archivo CSV para guardar el resultado.

    Retorna:
    - res_final: DataFrame con columnas categóricas codificadas.
    """

        
    for col in [
    'RELIGION',
    'ETHNICITY',
    ]:
            if proportion:
                num_categorias = len(res[col].value_counts())
                print("Número de categorías únicas en la columna {}: {}".format(col, num_categorias))
                proporcion = res[col].value_counts(normalize=True)
                print(proporcion)
                lower = proporcion[proporcion < prop].index
               
                res[col] = res[col].replace(lower, 'Other')
                num_categorias = len(res[col].value_counts())
                print("Número de categorías únicas después de la agregacion {}: {}".format(col, res[col].value_counts()))
                print("num of categories",num_categorias)                
            else:
                counts = res[col].value_counts(normalize=True)
                num_categorias = len(res[col].value_counts())
                print("Número de categorías únicas en la columna {}: {}".format(col, num_categorias))
                lower_80 = counts[counts.cumsum() > 0.8].index
                print("Number of cat agregated"+col,len(lower_80))
                print("Number of cat agregated"+col,len(lower_80))
                res[col] = res[col].replace(lower_80, 'Otra')
            
        
        # Aplicar codificación basada en el tipo especificado
    if encoding_type == 'onehot':
        encoder = OneHotEncoder()
        encoded_cols = encoder.fit_transform(res[categorical_cols])
        #encoded_cols_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names(categorical_cols))
        encoded_cols_df = pd.DataFrame(encoded_cols.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
        num_filas = encoded_cols_df.shape
        print("Shape of encoded cosl:: {}".format(num_filas))
        # Concatenar el DataFrame original con el DataFrame codificado
    elif encoding_type == 'label':
        encoded_cols_df = res[categorical_cols].apply(LabelEncoder().fit_transform)
        
    # Concatenar el DataFrame original con el DataFrame codificado
    
    res_final = pd.concat([res[[i for i in res.columns if i not in categorical_cols]], encoded_cols_df], axis=1)

    
    return res_final



from sklearn.preprocessing import FunctionTransformer

def apply_log_transformation(merged_df, column_name):
    """
    Applies log transformation to a column in the DataFrame.

    :param merged_df: DataFrame resulting from process_and_concat function
    :param column_name: The name of the column to which log transformation will be applied
    :return: DataFrame with the log-transformed column
    """
    transformer = FunctionTransformer(np.log1p, validate=False)  # np.log1p handles log(0) by returning 0
    merged_df[f"{column_name}"] = transformer.transform(merged_df[[column_name]].values)
    
    # Replace -inf with 0 if there are any -inf values resulting from log(0)
    merged_df[f"{column_name}"] = merged_df[f"{column_name}"].replace(-np.inf, 0)
    
    return merged_df


categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                'MARITAL_STATUS', 'ETHNICITY','GENDER']


from sklearn.preprocessing import StandardScaler, MaxAbsScaler, PowerTransformer
import numpy as np
import pandas as pd

def preprocess(X, prep, columns_to_normalize):
    '''
    Normalizes specified numerical features in a DataFrame.
    
    Parameters:
    - X: DataFrame to preprocess.
    - prep: String indicating the preprocessing method: 'std', 'max', 'power'.
    - columns_to_normalize: List of column names to normalize.
    
    Returns:
    - X: DataFrame with preprocessed numerical features.
    '''
    
    if prep == "std":
        scaler = StandardScaler()
        X[columns_to_normalize] = scaler.fit_transform(X[columns_to_normalize])
        
    elif prep == "max":
        transformer = MaxAbsScaler()
        X[columns_to_normalize] = transformer.fit_transform(X[columns_to_normalize])
        
    elif prep == "power":
        pt = PowerTransformer()
        X[columns_to_normalize] = pt.fit_transform(X[columns_to_normalize])
        
    # No else case needed, if 'prep' is not one of the above, X is returned unchanged
    
    return X

# Ejemplo de uso
# X es tu DataFrame
# 'prep' es el método de preprocesamiento: 'std', 'max', 'power'
# 'columns_to_normalize' es una lista de las columnas numéricas que deseas normalizar
# X_preprocessed = preprocess(X, 'std', ['col1', 'col2', 'col3', 'col4'])