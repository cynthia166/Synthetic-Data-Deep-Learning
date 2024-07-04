from scipy.spatial.distance import jensenshannon
import os
#os.chdir("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning")
os.chdir("/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/")
import sys
sys.path.append('/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/')
current_directory = os.getcwd()
from scipy.stats import beta, uniform, triang, truncnorm, expon, kstest, gaussian_kde
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits



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
from scipy.stats import gaussian_kde
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from generative_model.SD.constraints import *
import pickle


import random
import numpy as np
import pandas as pd
from scipy.spatial import distance

import matplotlib.pyplot as plt
import math
#import cairosvg
from io import BytesIO
import svgutils.transform as sg
import svgutils.transform as sg


import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
#equalize_length_dataframe

import logging

# Configura el logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


import svgutils.transform as sg

def get_feature_importance(clf,synthetic_raw_data,num_impor):
    feature_importances = clf.feature_importances_
    print("Feature Importances:", feature_importances)
    df = pd.DataFrame({
        'importance': feature_importances,
        'name': synthetic_raw_data.columns
    })

    # Sort by importance
    df_sorted = df.sort_values(by='importance',ascending = False)

    # Get the names of the lowest and highest 20 variables
    print(df_sorted.head(num_impor).to_latex())
    print(df_sorted.tail(num_impor).to_latex())

    return  df_sorted


def convert_to_pixels(value):
    if value.endswith('px'):
        return float(value.replace('px', ''))
    elif value.endswith('pt'):
        return float(value.replace('pt', '')) * 1.333
    else:
        return float(value)

def concatenate_svgs(path_img, list_img, output_path):
    # Load the SVG files
    svgs = [sg.fromfile(path_img + img) for img in list_img]
    # Get dimensions of the SVG files
    widths = [convert_to_pixels(svg.root.get('width')) for svg in svgs]
    heights = [convert_to_pixels(svg.root.get('height')) for svg in svgs]
    # Calculate the total width and maximum height
    total_width = sum(widths)
    max_height = max(heights)
    # Create a new SVGFigure to hold the combined image
    combined_svg = sg.SVGFigure(f"{total_width}px", f"{max_height}px")
    # Set the initial x position
    x_offset = 0
    # Append each SVG to the combined SVG, adjusting the x position
    for svg, width in zip(svgs, widths):
        svg_element = svg.getroot()
        svg_element.moveto(x_offset, 0)
        combined_svg.append(svg_element)
        x_offset += width
    # Save the combined SVG to the output path
    combined_svg.save(output_path)
#

# list_img = ["admission_date_histagram_8851.svg", "admission_date_bar_9228.svg"]
# output_path = "combined_image.svg"
# concatenate_svgs(path_img, list_img, output_path)

def combine_svgs(svg_paths, path_img):
    # Crea una nueva figura de matplotlib
    fig = plt.figure(figsize=(10, 10))  # Ajusta el tamaño de la figura según sea necesario

    # Calcula el número de filas y columnas para la cuadrícula de subtramas
    num_images = len(svg_paths)
    num_cols = math.ceil(math.sqrt(num_images))
    num_rows = math.ceil(num_images / num_cols)

    # Carga las imágenes SVG y las añade a la figura
    for i, svg_path in enumerate(svg_paths):
        svg = sg.fromfile(path_img + svg_path)
        plot = sg.SVGFigure(svg.width, svg.height)
        plot.append(svg.getroot())
        
        # Convierte el SVG a PNG usando cairosvg
        png_data = cairosvg.svg2png(bytestring=plot.to_str())
        
        # Convierte los datos PNG a una imagen que plt.imread puede manejar
        subplot = fig.add_subplot(num_rows, num_cols, i + 1)
        subplot.imshow(plt.imread(BytesIO(png_data), format='png'))
        subplot.axis('off')  # Oculta los ejes

    # Ajusta el espaciado entre subtramas
    plt.tight_layout()

    # Guarda la figura combinada
    if path_img is not None:
        save_plot_as_svg(fig, path_img, "concatenated_images")

# Uso de la función
def  get_hist(dy_df):    
    random_node = random.randint(0, max(dy_df["nodeid"]))  
    random_tree = random.randint(0, max(dy_df["tree"])) 
    df_serti = dy_df[(dy_df["nodeid"]==random_node)&(dy_df["tree"]==random_tree)]
    #plot
    hist_counts_node(df_serti,"value",random_node,random_tree,path_img=None)    
    



def save_plot_as_svg(plt, path_img, filename):
    random_number = random.randint(1000, 9999)  # Genera un número aleatorio entre 1000 y 9999
    plt.savefig(f"{path_img}{filename}_{random_number}.svg", format='svg')

def cols_drop_ehr(columns_to_drop, synthetic_ehr_dataset, train_ehr_dataset, test_ehr_dataset):
        #drop columns not needed 
        print("cols to drop: " ,columns_to_drop) 
        if all(column in synthetic_ehr_dataset.columns for column in columns_to_drop):
            synthetic_ehr_dataset.drop(columns_to_drop, axis=1, inplace=True)
            train_ehr_dataset.drop(columns_to_drop, axis=1, inplace=True)
            test_ehr_dataset.drop(columns_to_drop, axis=1, inplace=True) 
            
        print(test_ehr_dataset.shape)
        print(train_ehr_dataset.shape)
        print(synthetic_ehr_dataset.shape)
        #lista de metricas
        return synthetic_ehr_dataset, train_ehr_dataset, test_ehr_dataset


def get_columns_codes_commun(train_ehr_dataset,keywords,categorical_cols):
        columnas_test_ehr_dataset = get_cols_diag_proc_drug(train_ehr_dataset)
        #obtener cols para demosgraphics, contnious, procedures, diagnosis, drugs
        cols_categorical,cols_diagnosis,cols_procedures,cols_drugs = cols_to_filter(     train_ehr_dataset,keywords,categorical_cols,)
        cols_diagnosis,cols_procedures,cols_drugs=cols_filter_codes_drugs(train_ehr_dataset)
        
        #obtener 300 codes
        top_300_codes = obtain_most_freuent(train_ehr_dataset,columnas_test_ehr_dataset,100)
        return columnas_test_ehr_dataset,cols_categorical,cols_diagnosis,cols_procedures,cols_drugs,top_300_codes 
    
def load_create_ehr(read_ehr,save_ehr,file_path_dataset,sample_patients_path,file,valid_perc,features_path,name_file_ehr,type_file = 'ARFpkl'):
    #obtain dataset admission
    if read_ehr:
        test_ehr_dataset = load_pickle(file_path_dataset + 'test_ehr_dataset'+name_file_ehr+'.pkl')
        train_ehr_dataset = load_pickle(file_path_dataset + 'train_ehr_dataset'+name_file_ehr+'.pkl')
        synthetic_ehr_dataset = load_pickle(file_path_dataset + 'synthetic_ehr_dataset'+name_file_ehr+'.pkl')
        features = load_pickle(file_path_dataset + 'features'+name_file_ehr+'.pkl')
    else:    
        test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset,features = obtain_dataset_admission_visit_rank(sample_patients_path,file,valid_perc,features_path,type_file)
        if save_ehr:
            save_pickle(test_ehr_dataset, file_path_dataset + 'test_ehr_dataset'+name_file_ehr+'.pkl')
            save_pickle(train_ehr_dataset, file_path_dataset + 'train_ehr_dataset'+name_file_ehr+'.pkl')
            save_pickle(synthetic_ehr_dataset, file_path_dataset + 'synthetic_ehr_dataset'+name_file_ehr+'.pkl')
            save_pickle(features, file_path_dataset + 'features'+name_file_ehr+'.pkl')
    return test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset,features    

def make_read_constraints(make_contrains,save_constrains,train_ehr_dataset,test_ehr_dataset,synthetic_ehr_dataset,columns_to_drop,columns_to_drop_syn,type_archivo,invert_normalize,cols_continuous,subject_continous,
                          file_path_dataset = None,make_read_constraints_name='synthetic_ehr_dataset_contrainst.pkl'):
    if make_contrains:
        c = EHRDataConstraints(train_ehr_dataset, test_ehr_dataset, synthetic_ehr_dataset,columns_to_drop,columns_to_drop_syn,cols_continuous,True,type_archivo,invert_normalize,subject_continous)
        c.print_shapes()
        #cols_accounts = c.handle_categorical_data()
        synthetic_ehr_dataset, train_ehr_dataset, test_ehr_dataset = c.initiate_processing()
        c.print_shapes()
        #drop column between_cum sum made from constrains       
        
        #synthetic_ehr_dataset = cols_todrop(synthetic_ehr_dataset,+)
        if save_constrains: 
            save_pickle(synthetic_ehr_dataset, file_path_dataset +make_read_constraints_name )
            save_pickle(train_ehr_dataset, file_path_dataset +str("train_ehr_")+make_read_constraints_name )
            save_pickle(test_ehr_dataset, file_path_dataset +str("test_ehr_")+make_read_constraints_name )
    else: 
        synthetic_ehr_dataset = load_pickle(file_path_dataset + make_read_constraints_name)       
        train_ehr_dataset = load_pickle(file_path_dataset +str("train_ehr_")+make_read_constraints_name )
        test_ehr_dataset = load_pickle(file_path_dataset+str("test_ehr_")+make_read_constraints_name )
    #   get sam num patient in train set as synthetic
    #train_ehr_dataset = get_same_numpatient_as_synthetic(   train_ehr_dataset, synthetic_ehr_dataset)      
    # same shape_ synthetic train
    #train_ehr_dataset,synthetic_ehr_dataset = equalize_length_dataframe(train_ehr_dataset,synthetic_ehr_dataset)
    logging.info(f'test_ehr_dataset shape: {test_ehr_dataset.shape}')
    logging.info(f'train_ehr_dataset shape: {train_ehr_dataset.shape}')
    logging.info(f'synthetic_ehr_dataset shape: {synthetic_ehr_dataset.shape}')
    logging.info(f'itrain_unicos: {train_ehr_dataset["id_patient"].nunique()}')
    logging.info(f'synthetic unique patients: {synthetic_ehr_dataset["id_patient"].nunique()}')  
    return train_ehr_dataset,synthetic_ehr_dataset,test_ehr_dataset
def equalize_length_dataframe(df1, df2):
    # Obtiene la longitud de cada DataFrame
    len_df1 = len(df1)
    len_df2 = len(df2)

    # Encuentra la longitud mínima
    min_len = min(len_df1, len_df2)

    # Recorta los DataFrames a la longitud mínima
    df1 = df1.iloc[:min_len]
    df2 = df2.iloc[:min_len]

    return df1, df2    
def cols_filter_codes_drugs(train_ehr_dataset):
    cols_diagnosis = train_ehr_dataset.filter(like= 'diagnosis', axis=1).columns
    cols_procedures = train_ehr_dataset.filter(like= 'procedures', axis=1).columns
    cols_drugs = train_ehr_dataset.filter(like= 'drugs', axis=1).columns
    return cols_diagnosis,cols_procedures,cols_drugs

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def cols_to_filter(     train_ehr_dataset,keywords,categorical_cols,):
        for i in categorical_cols:
            cols_categorical = train_ehr_dataset.filter(like=i, axis=1).columns
        keywords = keywords
        cols_diagnosis = train_ehr_dataset.filter(like= 'diagnosis', axis=1).columns
        cols_procedures = train_ehr_dataset.filter(like= 'procedures', axis=1).columns
        cols_drugs = train_ehr_dataset.filter(like= 'drugs', axis=1).columns
        return cols_categorical,cols_diagnosis,cols_procedures,cols_drugs
import pandas as pd

def concat_to_dict_tolatex_syn_vs_real(result_syn, result_train):
    result_syn = pd.DataFrame(result_syn, index=[0])  # syntehti
    result_train = pd.DataFrame(result_train, index=[0])
    result_syn['Source'] = 'Synthetic'
    result_train['Source'] = 'Train'
    data = pd.concat([result_syn, result_train], ignore_index=True)
    # Reemplaza los subrayados en los nombres de las columnas con un espacio vacío
    data.columns = data.columns.str.replace('_', ' ')
    data.set_index('Source', inplace=True)
    data = data.transpose()
    print(data.to_latex())
    return data.to_dict()    

    
def same_size_synthetic(    train_ehr_dataset,synthetic_ehr_dataset):  
        if synthetic_ehr_dataset.shape[0] > train_ehr_dataset.shape[0]:
            synthetic_ehr_dataset = synthetic_ehr_dataset[ :train_ehr_dataset.shape[0]]
        else:
            train_ehr_dataset = train_ehr_dataset[ :synthetic_ehr_dataset.shape[0]]
        return train_ehr_dataset,synthetic_ehr_dataset    
 
def get_same_numpatient_as_synthetic(   train_ehr_dataset, synthetic_ehr_dataset):    
        unique_synthetic_patients = synthetic_ehr_dataset['id_patient'].nunique()
        unique_train_patients = train_ehr_dataset['id_patient'].unique()
        selected_train_patients = np.random.choice(unique_train_patients, size=unique_synthetic_patients, replace=False)
        train_ehr_dataset = train_ehr_dataset[train_ehr_dataset['id_patient'].isin(selected_train_patients)]
        return train_ehr_dataset      


def doppleganger_data_synthetic(data, synthetic_data,  attributes,  path_o, attributes_path_train, features_path_train, features_path_valid, attributes_path_valid,synthetic_path_attributes,synthetic_path_features):
            """
            This function will create a synthetic dataset with the same attributes as the original dataset. 
            The synthetic dataset will have the same number of rows as the original dataset. 
            """
            synthetic_data = synthetic_data.copy()
            synthetic_data.columns = data.columns
            synthetic_data.index = data.index
           
            total_features_synthethic, total_fetura_valid,total_features_train,attributes =  get_valid_train_synthetic (path_o, attributes_path_train, features_path_train, features_path_valid, attributes_path_valid,synthetic_path_attributes,synthetic_path_features)
            total_features_synthethic,total_fetura_valid,total_features_train = obtener_dataframe_inicial_denumpyarrray(total_features_synthethic, total_fetura_valid,total_features_train )

            
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
            return synthetic_ehr,train_ehr_dataset,test_ehr_dataset,attributes,features,static,con_cols,cat,cat1,total_cols,total_cols1,top_300_codes

def print_latex(result):
    aux_s =pd.DataFrame(result,index = [0])
    la = aux_s.to_latex()
    print(la)



def calculate_wasserstein_distances(real_data, synthetic_data, columns):
    """
    Calculate Wasserstein distances between real and synthetic data for specified columns.

    Parameters:
        real_data (pd.DataFrame): The real dataset.
        synthetic_data (pd.DataFrame): The synthetic dataset.
        columns (list): List of column names to compare.

    Returns:
        pd.DataFrame: A DataFrame with Wasserstein distances for each specified column.
    """
    distances = []

    for col in columns:
        real_values = real_data[col].dropna()
        synthetic_values = synthetic_data[col].dropna()

        dist = wasserstein_distance(real_values, synthetic_values)
        distances.append({
            'Column': col,
            'Wasserstein Distance': dist
        })

    return pd.DataFrame(distances)

def generate_comparison_table(real_data, synthetic_data, test_data, columns):
    """
    Generate a comparison table with Wasserstein distances and outlier counts.

    Parameters:
        real_data (pd.DataFrame): The real dataset.
        synthetic_data (pd.DataFrame): The synthetic dataset.
        test_data (pd.DataFrame): The test dataset.
        columns (list): List of column names to compare.

    Returns:
        pd.DataFrame: A DataFrame with Wasserstein distances and outlier counts.
    """
    comparison_data = []

    for col in columns:
        real_values = real_data[col].dropna()
        synthetic_values = synthetic_data[col].dropna()
        test_values = test_data[col].dropna()

        ws_dist_synthetic_train = wasserstein_distance(synthetic_values, real_values)
        ws_dist_test_train = wasserstein_distance(test_values, real_values)
        ws_dist_synthetic_test = wasserstein_distance(synthetic_values, test_values)

        comparison_data.append({
            'Feature': col,
            'Synthetic/Train': ws_dist_synthetic_train,
            'Test/Train': ws_dist_test_train,
            'Synthetic/Test': ws_dist_synthetic_test
        })

    return pd.DataFrame(comparison_data)

# # Ejemplo de uso con tus datos
# columns = ['admissions_per_drug', 'drugs_per_admission', 'drugs_per_patient', 'admissions_per_diagnosis', 
#            'diagnosis_per_admission', 'diagnosis_per_patient', 'admissions_per_procedure', 'procedures_per_admission', 
#            'procedures_per_patient']

# real_data = pd.read_csv('path_to_real_data.csv')
# synthetic_data = pd.read_csv('path_to_synthetic_data.csv')
# test_data = pd.read_csv('path_to_test_data.csv')

# # Generar la tabla de comparación
# comparison_table = generate_comparison_table(real_data, synthetic_data, test_data, columns)
# print(comparison_table)

# # Guardar la tabla de comparación en un archivo CSV
# comparison_table.to_csv('comparison_table.csv', index=False)

def plot_admission_date_bar_charts(color_dict, color_dict_, grouped, grouped_synthetic, col, save=False,path_img=None):
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
            
            if path_img != None:
                save_plot_as_svg(plt,path_img,'plot_admission_date_bar_charts')    
               
 
            
def categorilca_cols_fun(train_ehr_dataset,synthetic_ehr_dataset,categorical_cols,save=False,path_img=None):
    for i in categorical_cols:    
            color_dict,grouped = fun_grafico_compa(i,train_ehr_dataset)
            color_dict_,grouped_synthetic = fun_grafico_compa(i,synthetic_ehr_dataset)
            plot_admission_date_bar_charts(color_dict,color_dict_,grouped,grouped_synthetic,i,False,path_img)           
                 
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
         data = pickle.load(f)
    return data

def save_load_numpy(sample_patients=None,save=False,load=False,name='sample_patients.npy'):
    # Save the numpy array to disk
    if save:
        np.save(name, sample_patients)
        return
    if load: 
    # Load the numpy array from disk
       return np.load(name)    


def fit_distributions(data, n_components=3):
    distributions = {
        'beta': beta,
        'uniform': uniform,
        'triang': triang,
        'truncnorm': truncnorm,
        'expon': expon,
        'kde': gaussian_kde,
        'gmm': GaussianMixture
    }
    
    results = []
    
    # If data is a Series, convert it to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
        
        
        column_results = {'feature': "var"}
        feature_data = data.values
        #Reshape for GMM
        
        # Handle constant features separately
          
        for name, distribution in distributions.items():
            print(distribution)
            if name == 'truncnorm':
                min_val, max_val = feature_data.min(), feature_data.max()
                mean, std = feature_data.mean(), feature_data.std()
                a, b = (min_val - mean) / std, (max_val - mean) / std
                params = (a, b, mean, std)
            elif name == 'kde':
                try: 
                   kde = distribution(feature_data.flatten())
                except:
                    continue   
                
                params = kde
            elif name == 'gmm':
                with threadpool_limits(limits=1, user_api='blas'):
                        gmm = distribution(n_components=n_components, random_state=0)
                        gmm.fit(feature_data)
                        params = gmm
            elif name == 'beta':
                # Ensure that data is within (0, 1) interval
                min_val = feature_data.min()
                max_val = feature_data.max()
                if min_val == max_val:  # Avoid fitting Beta to constant data
                    continue
                scaled_data = (feature_data - min_val) / (max_val - min_val)
                # Beta distribution fitting
                try:
                    a, b, loc, scale = distribution.fit(scaled_data.flatten(), floc=0, fscale=1)
                    params = (a, b, loc, scale)
                except Exception:
                    continue
            elif name == 'uniform':
                loc, scale = distribution.fit(feature_data.flatten())
                params = (loc, scale)
            elif name == 'triang':
                c, loc, scale = distribution.fit(feature_data.flatten())
                params = (c, loc, scale)
            elif name == 'expon':
                loc, scale = distribution.fit(feature_data.flatten())
                params = (loc, scale)
            else:
                params = distribution.fit(feature_data.flatten())
            if np.isscalar(feature_data):
                  feature_data = np.array([feature_data])
            # Use Kolmogorov-Smirnov test for goodness of fit
            if name == 'truncnorm':
                ks_stat, ks_p_value = kstest(feature_data.flatten(), 'truncnorm', args=params)
            elif name == 'kde':
                cdf = lambda x: np.mean(params.integrate_box_1d(-np.inf, x))
                cdf_values = np.array([cdf(xi) for xi in feature_data.flatten()])
                ks_stat, ks_p_value = kstest(feature_data.flatten(), lambda x: cdf_values)
            elif name == 'gmm':
                gmm_samples, _ = params.sample(len(feature_data))
                gmm_samples = gmm_samples.flatten()
                cdf = lambda x: np.mean(gmm_samples <= x)
                cdf_values = np.array([cdf(xi) for xi in feature_data.flatten()])
                ks_stat, ks_p_value = kstest(feature_data.flatten(), lambda x: cdf_values)
            elif name == 'beta':
                ks_stat, ks_p_value = kstest(scaled_data.flatten(), 'beta', args=params)
            elif name == 'uniform':
                ks_stat, ks_p_value = kstest(feature_data.flatten(), 'uniform', args=params)
            elif name == 'triang':
                ks_stat, ks_p_value = kstest(feature_data.flatten(), 'triang', args=params)
            elif name == 'expon':
                ks_stat, ks_p_value = kstest(feature_data.flatten(), 'expon', args=params)
            else:
                ks_stat, ks_p_value = kstest(feature_data.flatten(), name, args=params)
            
            column_results[name + '_params'] = params
            column_results[name + '_ks_stat'] = ks_stat
            column_results[name + '_ks_p_value'] = ks_p_value
        results.append(column_results)
    
    return pd.DataFrame(results).to_dict()


def load_pkl(name):
    with open(name+'.pkl', 'rb') as f:
        data = pickle.load(f)
    return data        
       
def obtain_dataset_admission_visit_rank(sample_patients_path,file,valid_perc,features_path,type_archivo='csv'):
    features = load_data(features_path)
    #load csv 
    if type_archivo == 'csv':
      synthetic_ehr_dataset = pd.read_csv(file) 
    else:
       synthetic_ehr_dataset = load_pickle(file)
    # if there is an unnamed column drop it    
    cols_unnamed = synthetic_ehr_dataset.filter(like='Unnamed', axis=1).columns
    synthetic_ehr_dataset.drop(cols_unnamed, axis=1, inplace=True)
    sample_patients_r = load_pkl(sample_patients_path)
    train_ehr_dataset = features[features['SUBJECT_ID'].isin(sample_patients_r)]
    test_ehr_dataset  = features[~features['SUBJECT_ID'].isin(sample_patients_r)]  
    
    # if total_features_train.shape[0] > total_features_synthethic.shape[0]: 
    #    total_features_train = total_features_train[:total_features_synthethic.shape[0]]
    # if total_features_train.shape[0] < total_features_synthethic.shape[0]:
    #      total_features_synthethic = total_features_synthethic[:total_features_train.shape[0]]  
    
    return  test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset,features

def cols_todrop(total_features_synthethic,cols):
    try:
        cols_to_drop = list(total_features_synthethic.filter(like='Unnamed', axis=1).columns) + cols
        total_features_synthethic.drop(cols_to_drop, axis=1, inplace=True) 
    except:
        pass    
    return total_features_synthethic
       
def plot_admission_date_bar_charts(color_dict, color_dict_, grouped, grouped_synthetic, col, save=False,path_img=None):
            # Create the figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 6), sharey=True)
            # Unique categories in the real data
            categories_real = grouped['category'].unique()
            categories_synthetic = grouped_synthetic['category'].unique()
            # Plot cumulative proportions for each category in the first subplot
            for (category), data in grouped.groupby('category'):
                ax1.plot(data['visit_rank'], data['visit_count'], label=f'Category {category}', color=color_dict.get(category, 'gray'))
            # Configuration of the first graph
            ax1.set_title('Cumulative Visits by Category - Real Data (' + col + ')')
            ax1.set_xlabel('Visit rank ')
            ax1.set_ylabel('Cumulative of Visits')
            if len(categories_real) <= 10:
                ax1.legend(title='Category')
            # Plot cumulative proportions for each category in the second subplot, using the synthetic DataFrame
            for (category), data in grouped_synthetic.groupby('category'):
                ax2.plot(data['visit_rank'], data['visit_count'], label=f'Category {category}', color=color_dict_.get(category, 'gray'))
            # Configuration of the second graph
            ax2.set_title('Cumulative Visit Proportions by Category - Synthetic Data (' + col + ')')
            ax2.set_xlabel('Visit rank ')
            if len(categories_synthetic) <= 10:
                ax2.legend(title='Category')
            fig.autofmt_xdate()  # Automatically format the dates to improve visualization
            # Show the graph
            plt.show()
            if path_img != None:
                save_plot_as_svg(plt,path_img,'plot_admission_date_bar_charts')
       
       
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
            grouped = aux_df.groupby(['visit_rank', 'category']).sum().groupby(level='category').cumsum()
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
        

def obtain_most_freuent(train_ehr_dataset,columnas_test_ehr_dataset,num):
    code_sums = train_ehr_dataset[columnas_test_ehr_dataset].sum(axis=0).sort_values(ascending=False)
    top_300_codes = code_sums.head(num).index.tolist()
    return top_300_codes

def obtain_least_frequent_fun(train_ehr_dataset, columns, num):
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

def proportion_non_zeros(df, columns):
    # Cuenta el número total de valores en cada columna
    total_counts = df[columns].apply(lambda x: len(x))
    # Cuenta el número de valores no cero en cada columna
    non_zero_counts = df[columns].apply(lambda x: np.count_nonzero(x))
    # Calcula la proporción de valores no cero
    non_zero_proportions = non_zero_counts / total_counts
    # Convierte la serie resultante en un diccionario y la devuelve
    return non_zero_proportions.to_dict()
    
    return least_frequent_codes
def calculate_proportions(df, column):
            frequency = df[column].value_counts(normalize=True)
            return frequency

def dimensio_wise(real_samples,gen_samples,name,path_img=None):
            prob_real = np.mean(real_samples, axis=0)
            prob_syn = np.mean(gen_samples, axis=0)

            plt.scatter(prob_real, prob_syn, c="b", alpha=0.5, label="ARF")
            x_max = max(np.max(prob_real), np.max(prob_syn))
            x = np.linspace(0, x_max + 0.1, 1000)
            plt.plot(x, x, linestyle='-', color='k', label="Ideal")  # solid
            
            plt.tick_params(labelsize=12)
            plt.legend(loc=2, prop={'size': 15})
            plt.title('Mean of dimensions')
            plt.xlabel('Real data ' + name)
            plt.ylabel('Synthetic data ' + name)
            
            plt.xlim(0, x_max + 0.1)  # Ensure x-axis starts at 0
            plt.ylim(0, x_max + 0.1)  # Ensure y-axis starts at 0
            
            plt.show()
            
            if path_img is not None:
                save_plot_as_svg(plt, path_img, 'dimensio_wise')
            
            plt.close()
                    

#from evaluation.resemb.resemblance.metric_stat import *
#from evaluation.functions import *
def get_cols_diag_proc_drug(train_ehr_dataset):
    keywords = ['diagnosis', 'procedures', 'drugs']
    columnas_test_ehr_dataset = [col for col in train_ehr_dataset.columns if any(palabra in col for palabra in keywords)]
    return columnas_test_ehr_dataset




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
        

def plot_heatmap(synthetic_ehr_dataset, name,col, cols_num=1,cols_prod="None",type_c="Synthetic",path_img=None):
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
                if path_img!= None:
                     save_plot_as_svg(plt,path_img,'plot_kernel_syn')


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
                if path_img!= None:
                     save_plot_as_svg(plt,path_img,'plot_kernel_syn')

def hist_d(col,synthetic_ehr_dataset,path_img = None):
    plt.figure(figsize=(12, 8))
    print(synthetic_ehr_dataset[col].describe())
    ax = synthetic_ehr_dataset[col].hist(bins=30)
    ax.set_xlabel(col)
    plt.show()
    if path_img != None:
        save_plot_as_svg(plt, path_img, "histogram")
    

def plot_heatmap_(synthetic_ehr_dataset, name, col,name2,path_img=None):
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
    if path_img!= None:
        
        save_plot_as_svg(plt,path_img,'plot_kernel_syn')

# Adjust parameters according to your specific dataset structure and requirements

    
def hist_betw_a(original_data,synthetic_data,col,path_img=None):
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
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'plot_kernel_syn')

def hist_counts_node(synthetic_data,col,node_num,tree_node,path_img=None):
    #original_data = pd.read_csv('original_data.csv')  # Adjust the file path as necessary
    #synthetic_data = pd.read_csv('synthetic_data.csv')  # Adjust the file path as necessary

    # Assume 'days_between_visits' column exists, otherwise calculate it
    # Plotting the histograms
              
    plt.figure(figsize=(12, 6))

    # Histogram for original data
    #plt.hist(original_data[col], bins=50, alpha=0.3, label='Original', color='blue')

        # Histogram for synthetic data
    plt.hist(synthetic_data[col], bins=50,alpha=0.3, label='Synthetic', color='red')

    # Adding titles and labels
    plt.title('Comparison of Days '+col+' Synthetic Data node ' +str(node_num) + 'and tree ' + str(tree_node))
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.legend()

    # Show the plot
    plt.show()
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'plot_kernel_syn_node')

    
def box_pltos(df,df_syn,col,path_img=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axes[0].boxplot(df[col])
    axes[0].set_ylabel('Days')
    axes[1].boxplot(df_syn[col])
    
    axes[1].set_ylabel('Days')
    col = limpiar_lista(col)
    try:
        axes[0].set_title('Real ' +col )
        axes[1].set_title('Synthetic ' +col)
    except:
        axes[0].set_title('Real ' )
        axes[1].set_title('Synthetic ' )
        
    plt.tight_layout()
    plt.show()   
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'plot_kernel_syn')

        
def plot_heatmap(synthetic_ehr_dataset, name, col, cols_num=1, cols_prod="None", type_c="Synthetic",path_img=None):
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
                if path_img != None:
                    save_plot_as_svg(plt, path_img, 'plot_heatmap')

            else:
                # For multiple columns, sum data and sort
                heatmap_data = synthetic_ehr_dataset[cols_prod + ["visit_rank"]].groupby('visit_rank').sum()
                top_drugs = heatmap_data.head(10).index.tolist()
                heatmap_data.sort_values(by='visit_rank', ascending=False, inplace=True)
                
                plt.figure(figsize=(12, 8))
                ax = sns.heatmap(heatmap_data, cmap='viridis', annot=False)
                ax.set_title(f'{name} by Visit rank ({type_c})')
                ax.set_xlabel(name)
                ax.set_ylabel('Visit rank')
                
                # Generate labels, showing only the top items
                labels = [label if label in top_drugs else '' for label in heatmap_data.columns]
                ax.set_xticks(np.arange(len(labels)))
                ax.set_xticklabels(labels, rotation=45)
                plt.show()
                if path_img != None:
                    save_plot_as_svg(plt, path_img, 'plot_heatmap')


 
def PACMAP_PLOT(col_prod,synthetic_ehr_dataset,train_ehr_dataset,i,save=False,path_img=None ):
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
               # Display the plot
            plt.show()
            if path_img != None:
                    save_plot_as_svg(plt, path_img, 'PACMAP_PLOT')

 
 
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
        
def dist_node_tree_cat(FORED,col):
    cat = FORED["cat"]
    aux2 = cat[cat["variable"]==col]
    s = aux2.groupby(["tree"])["nodeid"].value_counts().value_counts()
    return  print(s)
        
def heatmap_diff_corr(df1, df2,path_img):
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
            fig, ax = plt.subplots(figsize=(20, 16))

            # Create a heatmap for the matrix of differences
            sns.heatmap(abs_diff_corr_matrix, cmap='coolwarm', center=0, ax=ax)

            # Configure the axis labels
           
                # Only show labels for the rows with top 10 highest differences
            #y_labels = [column_names[i] if top_diff_mask[i].any() else '' for i in range(len(column_names))]
            
            #ax.set_yticks(np.arange(len(column_names)))  # Set y-tick positions
            #ax.set_yticklabels(y_labels, rotation=45)  # Set y-tick labels
            
            
            fontdict = {'fontsize': 6, 'fontweight': 'bold', 'fontname': 'Arial'}
            num_columns = corr_matrix1.shape[1]
            if num_columns <= 50:
                ax.set_xticklabels([])
                y_labels = [limpiar_lista(label.get_text()) for label in ax.get_yticklabels()]
                #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
                ax.set_yticklabels(y_labels, rotation=0)
                #ax.xaxis.set_tick_params(labelsize=10)
                ax.yaxis.set_tick_params(labelsize=12)
                ax.set_xticklabels([])
            else:
            # Configure the axis labels
                y_labels = [column_names[i] if top_diff_mask[i].any() else '' for i in range(len(column_names))]
                y_labels = [limpiar_lista(label) for label in y_labels]
                ax.set_yticks(np.arange(len(column_names)))  # Set y-tick positions
                ax.set_yticklabels(y_labels, rotation=0, fontdict=fontdict)  # Set y-tick labels with font properties
                ax.set_xticklabels([])
                
                #ax.set_xticklabels(column_names, rotation=45, horizontalalignment='right')
        

            # Show and save the graph
            plt.title('Heatmap of Correlation Matrix Differences')
            plt.savefig('difference_heatmap.svg')
            plt.tight_layout()
            plt.show()
            
            if path_img!= None:
               save_plot_as_svg(plt,path_img,'difference_heatmap_correaltion')
            y_labels = [item for item in y_labels if item != '']
            plt.close()
            return y_labels
    
def plot_histograms_separate_axes22(real_data, synthetic_data, title, xlabel, ylabel,label_s,path_img=None):
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
            if path_img!= None:
               save_plot_as_svg(plt,path_img,'plot_histograms_separate_axes22')
            return wd
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



    
def corr_plot(total_features_train,name ,path_img=None):
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
    fig, ax = plt.subplots(figsize=(14, 12))  # Ajusta el tamaño según tus necesidades

    # Definir un mapa de colores personalizado
    # Utilizando colores divergentes: azul para valores cercanos a -1 y rojo para valores cercanos a 1
    cmap = sns.diverging_palette(240, 10, as_cmap=True)  # 240 es azul, 10 es rojo en HUSL

    # Crear un mapa de calor de la matriz de correlación
    sns.heatmap(corr_matrix, annot=False, cmap=cmap, center=0, ax=ax)

    # Configurando las etiquetas de los ejes con condiciones para visualización
    num_columns = corr_matrix.shape[1]
    if num_columns <= 50:
        ax.set_xticklabels([])
        y_labels = [limpiar_lista(label.get_text()) for label in ax.get_yticklabels()]
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.set_yticklabels(y_labels, rotation=0)
        #ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=12)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()
    if path_img!= None:
        save_plot_as_svg(plt,path_img,name+'_kernelplot')       
  
        
    return corr_matrix



# Define a function to calculate statistics on the visits without considering labels
def generate_statistics(ehr_datasets,type_s,):
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



def plot_age(df,col,name,path_img=None):
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
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'kentnel_densitu_plot')



def filter_and_equalize_datasets(df1, df2):
    # Filtra solo las columnas numéricas
    df1_numeric = df1.select_dtypes(include=['number'])
    df2_numeric = df2.select_dtypes(include=['number'])

    # Asegúrate de que los dos conjuntos de datos tengan las mismas columnas
    common_columns = set(df1_numeric.columns) & set(df2_numeric.columns)
    df1_numeric = df1_numeric[common_columns]
    df2_numeric = df2_numeric[common_columns]

    # Asegúrate de que los dos conjuntos de datos tengan el mismo tamaño
    min_size = min(df1_numeric.shape[0], df2_numeric.shape[0])
    df1_numeric = df1_numeric.sample(n=min_size, random_state=1)
    df2_numeric = df2_numeric.sample(n=min_size, random_state=1)

    return df1_numeric, df2_numeric




def plot_procedures_diag_drugs(col_prod,train_ehr_dataset,type_procedur,name,path_img=None):
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
    
    plt.show()
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'count_of_drugs')


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

def plot_visits_per_patient_histogram(train_ehr_dataset, synthetic_ehr,path_img=None):
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
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'Histogram_numbervisit')

    
def plot_patients_per_visit_rank_histogram(train_ehr_dataset, synthetic_ehr_dataset,path_img=None):
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
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'visit_rank_histogram')



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data for demonstration

def plot_histograms22_(real_data, synthetic_data, title, xlabel, ylabel,path_img=None):
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
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'plot_histograms22_')


def obtatain_df(serie1,serie2,serie3,serie4,serie5,serie6,name1,name2,name3,name4,name5,name6):
        # Obtener las descripciones y convertirlas en DataFrames
        desc1 = serie1.describe().reset_index()
        desc1.columns = ['Statistic', name1]

        desc2 = serie2.describe().reset_index()
        desc2.columns = ['Statistic', name2]

        desc3 = serie3.describe().reset_index()
        desc3.columns = ['Statistic', name3]

        desc4 = serie4.describe().reset_index()
        desc4.columns = ['Statistic', name4]

        desc5 = serie5.describe().reset_index()
        desc5.columns = ['Statistic', name5]

        desc6 = serie6.describe().reset_index()
        desc6.columns = ['Statistic', name6]

        # Unir las descripciones en un solo DataFrame
        descriptions = desc1.merge(desc2, on='Statistic').merge(desc3, on='Statistic').merge(desc4, on='Statistic').merge(desc5, on='Statistic').merge(desc6, on='Statistic')

        return descriptions

# Combinar las descripciones en un solo DataFrame


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
def plot_histograms(real_data, synthetic_data, title, xlabel, ylabel_real, ylabel_synthetic,path_img=None):
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
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'plot_histograms')

    
def plot_histograms22(real_data, synthetic_data, title, xlabel, ylabel_real, ylabel_synthetic,path_img=None):
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
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'plot_histomas_')

    
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
        
def plot_hist_emp_codes(col_prod, train_ehr_dataset, synthetic_ehr, type_procedur, name,path_img=None):
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
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'patient_count')


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
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'admission_count')


# Example usage with mock data and parameters
# plot_hist_emp_codes(col_prod, train_ehr_dataset, synthetic_ehr, type_procedur, name)


def plot_hist_emp_codes_auz(col_prod, train_ehr_dataset, synthetic_ehr, type_procedur, name,path_img=None):
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
    
    plt.show()
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'plot_hist_emp_codes_auz')




 
def histograms_codes(train_ehr_dataset,test_ehr_dataset,synthetic_ehr_dataset,path_img=None):
    keywords = ['diagnosis', 'procedures', 'drugs']
    for i in keywords:
        col_prod = [col for col in train_ehr_dataset.columns if any(palabra in col for palabra in [i])]
        name = "Train"
        type_procedur = i

        plot_procedures_diag_drugs(col_prod,train_ehr_dataset,type_procedur,name,path_img=None)

        col_prod = [col for col in test_ehr_dataset.columns if any(palabra in col for palabra in [i])]
        name = "Test"


        plot_procedures_diag_drugs(col_prod,test_ehr_dataset,type_procedur,name, path_img =None  )
        
        col_prod = [col for col in synthetic_ehr.columns if any(palabra in col for palabra in [i])]
        name = "Synthetic"


        plot_procedures_diag_drugs(col_prod,synthetic_ehr_dataset,type_procedur,name,path_img=None)
        
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
def get_statistics(train_ehr_dataset,columnas_test_ehr_dataset,test_ehr_dataset,synthetic_ehr_dataset):
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
def descriptive_statistics_matrix_dif(data, data_type):
    # Check if the DataFrame is empty
    if data.empty:
        return "Empty DataFrame, no statistics available."

    # Initialize a dictionary to hold statistics for each feature
    stats_dict = {}
    data_type = ""
    # Calculate statistics for each feature (column)
    for column in data.columns:
        column_data = data[column]
        
        # Compute each statistic and concatenate it with the column name and data type
        stats_dict[f'mean_{column}_{data_type}'] = np.mean(column_data)
        stats_dict[f'median_{column}_{data_type}'] = np.median(column_data)
        #stats_dict[f'mode_{column}_{data_type}'] = stats.mode(column_data)[0][0] if len(column_data) > 0 else None
        stats_dict[f'minimum_{column}_{data_type}'] = np.min(column_data)
        stats_dict[f'maximum_{column}_{data_type}'] = np.max(column_data)
        stats_dict[f'range_{column}_{data_type}'] = np.ptp(column_data)
        stats_dict[f'variance_{column}_{data_type}'] = np.var(column_data, ddof=1)
        stats_dict[f'standard_deviation_{column}_{data_type}'] = np.std(column_data, ddof=1)
        stats_dict[f'skewness_{column}_{data_type}'] = stats.skew(column_data)
        stats_dict[f'kurtosis_{column}_{data_type}'] = stats.kurtosis(column_data)

    return stats_dict

def descriptive_statistics_matrix(data, data_type):
    # Check if the DataFrame is empty
    if data.empty:
        return "Empty DataFrame, no statistics available."

    # Initialize a dictionary to hold statistics for each feature
    stats_dict = {}
    
    # Calculate statistics for each feature (column)
    for column in data.columns:
        column_data = data[column]
        
        # Compute each statistic and concatenate it with the column name and data type
        stats_dict[f'mean_{column}_{data_type}'] = np.mean(column_data)
        stats_dict[f'median_{column}_{data_type}'] = np.median(column_data)
        #stats_dict[f'mode_{column}_{data_type}'] = stats.mode(column_data)[0][0] if len(column_data) > 0 else None
        stats_dict[f'minimum_{column}_{data_type}'] = np.min(column_data)
        stats_dict[f'maximum_{column}_{data_type}'] = np.max(column_data)
        stats_dict[f'range_{column}_{data_type}'] = np.ptp(column_data)
        stats_dict[f'variance_{column}_{data_type}'] = np.var(column_data, ddof=1)
        stats_dict[f'standard_deviation_{column}_{data_type}'] = np.std(column_data, ddof=1)
        stats_dict[f'skewness_{column}_{data_type}'] = stats.skew(column_data)
        stats_dict[f'kurtosis_{column}_{data_type}'] = stats.kurtosis(column_data)

    return stats_dict
def compare_descriptive_statistics_fun(test_ehr_dataset, syn_ehr):
    """
    Compare descriptive statistics between two dataframes.

    Parameters:
        data1 (pd.DataFrame): First dataset.
        data2 (pd.DataFrame): Second dataset.

    Returns:
        dict: A dictionary containing differences in descriptive statistics between the two datasets.
    """
    # Calculating descriptive statistics for both datasets
    stats1 = descriptive_statistics_matrix_dif(test_ehr_dataset,"Train")
    stats2 = descriptive_statistics_matrix_dif(syn_ehr ,"Synthetic")

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

def extract_relevant_part(subset):
    parts = subset['Variable'].iloc[0].split('_')
    if len(parts) > 2:
        return subset['Variable'].apply(lambda x: '_'.join(x.split('_')[:2])) # Join parts after the second underscore
    else:
        return subset['Variable'].apply(lambda x: '_'.join(x.split('_')[:1]))  # Part after the first underscore

def limpiar_lista(var):
    var_new = ' '.join(var.split('_'))
    
    return var_new.replace('Otra', 'Other')
    
def plot_pie_proportions(variables, filtered_df, type_data):
    # Combined plot
    fig, axes = plt.subplots(len(filtered_df['Variable'].unique()), 1, figsize=(10, 15))
    #fig.patch.set_facecolor('white')
    
    for i, variable in enumerate(filtered_df['Variable'].unique()):
        subset = filtered_df[filtered_df['Variable'] == variable]
        #col = subset['Variable'].apply(lambda x: '_'.join(x.split('_')[:2])).iloc[0]
        col =extract_relevant_part(subset).iloc[0]
        
        if col=='visit':   
            col =  'visit_rank'
    

        if type_data =='Synthetic data': 
            counts = subset['Count synthetic'+col].values
            labels = subset['Category '].values
            proportions = subset['Proportion synthetic'+col].values
            colors = sns.color_palette('Blues', len(labels))
        else: 
            counts = subset['Count train '+col].values
            labels = subset['Category '].values
            proportions = subset['Proportion train '+col].values
            colors = sns.color_palette('Blues', len(labels))
        if col == 'visit_rank':
            counts =counts[:4]
            labels = ["Visit " +str(i) for i in labels[:4] ]
            pie = axes.pie(counts, labels=labels, autopct='%1.0f%%', startangle=90, colors=colors, textprops={'fontsize': 12})
            # Agrega el título que desees
            name_limpit = subset['Variable'].iloc[0]
            limipiar_nom = limpiar_lista(name_limpit)
            plt.title("Number of visits" + " " +type_data)
            plt.axis('off')  # Desactiva los ejes
            plt.legend('')  # Desactiva las leyendas
            plt.show()
            save_plot_as_svg(plt,path_img,'pie_chart_proportion')
        else:       
        #pie = axes.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 12})
            pie = axes[i].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 12})
            axes[i].legend(pie[0], labels, loc='center left', fontsize=12, bbox_to_anchor=(1, 0.5))
            name_limpit = subset['Variable'].iloc[0]
            limipiar_nom = limpiar_lista(name_limpit)
            axes[i].set_title(limipiar_nom + " " +type_data, fontsize=15)
            axes[i].set_aspect('equal')  # Ensure pie is circular
            axes[i].set_xlabel('')  # Remove x-axis label
            axes[i].set_ylabel('')  # Remove y-axis label

        plt.tight_layout()
        
        plt.show()
        
    save_plot_as_svg(plt,path_img,'pie_chart_proportion')
    plt.close()
def plot_vistis(    df ):    
    
    df['Percentage Difference'] = (df['Proportion syntheticvisit_rank'] - df['Proportion train visit_rank']) / df['Proportion train visit_rank'] * 100

    # Plot the data
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Bar plot for proportions
    bar1 = sns.barplot(x='Category ', y='Proportion train visit_rank', data=df, color='red', alpha=0.8, label='Train Proportion', ax=ax1, dodge=False)
    bar2 = sns.barplot(x='Category ', y='Proportion syntheticvisit_rank', data=df, color='blue', alpha=0.4, label='Synthetic Proportion', ax=ax1, dodge=False)

    # Add proportions on top of each bar
    for p in bar1.patches:
        ax1.text(p.get_x() + p.get_width() / 2., p.get_height(), '{0:.2f}'.format(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

    for p in bar2.patches:
        ax1.text(p.get_x() + p.get_width() / 2., p.get_height(), '{0:.2f}'.format(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

    ax1.set_xlabel('Visit Number ')
    ax1.set_ylabel('Proportion')
    ax1.set_title('Proportion Comparison of Train and Synthetic Data by Visit Number')
    ax1.legend(loc='upper left')

    
    # Adding legends and titles
    fig.tight_layout()
    fig.legend(loc='upper right', bbox_to_anchor=(0.5, -0.05))

    plt.show()    
    save_plot_as_svg(fig, path_img, "pie_char_vist")
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


def compare_data_ranges(real_data, synthetic_data, columns,type_col):
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
    
    
    numeric_columns = real_data[columns].select_dtypes(include=[np.number]).columns

    synthetic_ranges = synthetic_data[numeric_columns].max() - synthetic_data[numeric_columns].min()
    real_ranges = real_data[numeric_columns].max() - real_data[numeric_columns].min()

    # Comparer les ranges
    range_diff = abs(real_ranges - synthetic_ranges)

    # Trier les différences et obtenir les 10 premières colonnes
    top_diff = range_diff.sort_values(ascending=False).head(10)
    top_diff = range_diff.sort_values(ascending=False).head(10)

    # Crear un DataFrame con las columnas y sus diferencias
    result ={'Column '+type_col: top_diff.index.tolist(), 'Range Difference '+ type_col: top_diff.values.tolist()}


    return result

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
 


def plot_kernel_syn(real_df, synthetic_df, col, name ,path_img=None):
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
    
    col = limpiar_lista(col)
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
    ax.set_title(f'Kernel Density Plot - {col}')
    ax.legend()  # Add a legend to distinguish between real and synthetic data

    ax.set_ylim(bottom=0)  # Ensure the y-axis starts at zero

    # Save the plot to a file
    
    
    # Show the plot
    plt.show()
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'plot_kernel_syn')

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
def count_non_zeros(df, columns):
    # Aplica la función count_nonzero a cada columna especificada
    non_zero_counts = df[columns].apply(lambda x: np.count_nonzero(x))
    # Convierte la serie resultante en un diccionario y la devuelve
    return non_zero_counts.to_dict()

def calculate_means(dataset, columns):
    return {col: dataset[col].mean() for col in columns}

def plot_means(train_ehr_dataset, synthetic_ehr_dataset, columns,  path_img=None):
    # Calculate means for both datasets
    real_means = calculate_means(train_ehr_dataset, columns)
    synthetic_means = calculate_means(synthetic_ehr_dataset, columns)

    # Ensure column_names is a dictionary
    
    column_names = {col: col for col in columns}

    # Data for plotting
    labels = [column_names.get(col, col).replace('_', ' ') for col in columns]
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

    if path_img != None:
        save_plot_as_svg(plt, path_img, 'mean_comparison_'+str(np.random.randint(0,1000)))
    else:
        plt.show()

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

def plot_admission_date_bar_charts2(features, synthetic_ehr_dataset, col,path_img=None):
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
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'admission_date_bar')




def filter_cols(categorical_cols,train_ehr_dataset):        
    col_total = []
    for i in categorical_cols:
        cols_f = train_ehr_dataset.filter(like=i, axis=1).columns
        col_total.extend(cols_f.tolist())
    return col_total    
            
def plot_admission_date_histograms(features,synthetic_ehr_dataset,col,path_img=None):
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
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'admission_date_histagram')




import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


from scipy.stats import wasserstein_distance

def plot_histograms_separate_axes2(real_data, synthetic_data, title, xlabel, ylabel,path_img=None):
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
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'plot_histogram_seperate')



def plot_histograms_separate_axes3(real_data, synthetic_data, title, xlabel, ylabel,path_img=None):
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
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'plot_histogram_separee2')





def plot_boxplots(real_data, synthetic_data, title, xlabel, ylabel,path_img=None):
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
    if path_img!= None:
               save_plot_as_svg(plt,path_img,'box_plot')

    
    
def plot_outliers( real_data, synthetic_data, column_name, title, xlabel, ylabel_real, ylabel_synthetic,path_img=None):
        plt.figure(figsize=(14, 7))

        # Gráfico de caja para los datos reales
        plt.subplot(1, 2, 1)
        sns.histplot  (y=real_data[column_name+'_count'])
        plt.title(f'{title} - Real Data')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel_real)

        # Gráfico de caja para los datos sintéticos
        plt.subplot(1, 2, 2)
        sns.histplot(y=synthetic_data[column_name+'_count'])
        plt.title(f'{title} - Synthetic Data')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel_synthetic)

        plt.tight_layout()
        plt.show()
        if path_img!= None:
               save_plot_as_svg(plt,path_img,'plot_outier')


# Ejemplo de uso (asegúrate de tener los datos `real_data` y `synthetic_data` preparados):
# plot_histograms_with_same_axis(real_data['number_of_drugs'], synthetic_data['number_of_drugs'], 'Histogram of Number of Drugs per Patient', 'Number of Drugs', 'Patient Count')
def calculate_outlier_ratios_tout(df, columns):
    outliers_dict = {}
    total_values = df[columns].sum().sum()
    total_outliers = 0
    for i in columns:
        sorted_df = df[columns].sort_values(by=i)
        Q1 = sorted_df[i].quantile(0.25)
        Q3 = sorted_df[i].quantile(0.75)
        IQR = Q3 - Q1
        outliers = sorted_df[(sorted_df[i] < (Q1 - 1.5 * IQR)) | (sorted_df[i] > (Q3 + 1.5 * IQR))]
        total_outliers += outliers[i].sum()
        outliers_dict[i] = outliers
    ratio = total_outliers / total_values if total_values != 0 else 0
    return  outliers_dict

def calculate_outlier_ratios_tout2(real_drugs_per_patient,i):
    sorted_df = real_drugs_per_patient.sort_values(by=i+'_count')
    Q1 = sorted_df[i+'_count'].quantile(0.25)
    Q3 = sorted_df[i+'_count'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = sorted_df[(sorted_df[i+'_count'] < (Q1 - 1.5 * IQR)) | (sorted_df[i+'_count'] > (Q3 + 1.5 * IQR))]
    return outliers

def test_distributions_nodes(long,ruta_save_dist):
    '''params
    #input function that  test the distributions
    long: dataframe data de nodos en arboles de variables continuas
    ruta_save_dist: path where result are stored
    '''
    test_list = []
    cont_var_above_threshold = 0
    params = pd.DataFrame()
    for (t, n, var), group in long.groupby(['tree', 'nodeid', 'variable']):
        if len(group)<threshold:
            continue
        else:
            cont_var_above_threshold += 1
            rest = fit_distributions(group['value'], n_components=3)
            
            test_list.append([t, n, var, rest])
            res = pd.DataFrame(test_list, columns=['tree', 'nodeid', 'variable', 'test_results'])
            params = pd.concat([params, res])
            params["cont_var_above_threshold"] =cont_var_above_threshold       
        save_pickle(params,ruta_save_dist)
    return params


def get_proportion_significant_kst(long_):
    list_par = list(long_["test_results"].iloc[0].keys())
    col_ks_p_value = [i for i in list_par if "p_value" in i]        


    uniform_l = []
    triang_l = []
    truncnorm_l =[]
    expon_l = []
    gmm_l = []

    for j in range(len(long_)):
        x =long_["test_results"].iloc[j]
        for i in col_ks_p_value:
            p_value =  int(x[i][0]>0.05)
            print(p_value)
            if i  == 'uniform_ks_p_value':
                uniform_l.append(p_value)
            elif i == 'triang_ks_p_value':
                triang_l.append(p_value)
            elif i == 'truncnorm_ks_p_value':
                truncnorm_l.append(p_value)
            elif i == 'expon_ks_p_value':
                expon_l.append(p_value)
            elif i == 'gmm_ks_p_value':
                gmm_l.append(p_value)
        
    long_["uniform_ks_stat"]=  uniform_l
    long_["triang_ks_stat"]=  triang_l
    long_["truncnorm_ks_stat"]=  truncnorm_l
    long_["expon_ks_stat"]=  expon_l
    long_["gmm_ks_stat"]=  gmm_l
    mena_kst = long_.groupby([ 'variable'])[["uniform_ks_stat","triang_ks_stat","truncnorm_ks_stat","expon_ks_stat","gmm_ks_stat"]].mean()

    return long_,uniform_l,triang_l,truncnorm_l,expon_l,gmm_l

def prom(uniform_l):
    print(uniform_l.shape)
    return sum(uniform_l[-2149701:])/len(uniform_l[-2149701:])        


def res_promedio_ks_stat(uniform_l,triang_l,truncnorm_l,expon_l,gmm_l):
    list_dist_resp = [uniform_l,triang_l,truncnorm_l,expon_l,gmm_l]
    res = [prom(i) for i in list_dist_resp]    
    print(res)
    return res

def hist(df,col,title,x_label,lsup=None):
    plt.figure(figsize=(12, 6))
    vmin = df[col].min()
    vmax = df[col].max()
    if lsup==None:
        plt.hist(df[col], bins=100,alpha=0.3, label='Patient Counts', color='red')
    else:    
        plt.hist(df[col], bins=100,range=(vmin,lsup),alpha=0.3, label='Patient Counts', color='red')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def create_plot_count_matrix_for_specific_subject(train_ehr_dataset,subjects,type_data  = 'Train data',path_img=None):


# Seleccionamos las columnas correspondientes
    diagnosis_columns = train_ehr_dataset.filter(like="diagnosis").columns
    procedure_columns = train_ehr_dataset.filter(like="procedures").columns
    medication_columns = train_ehr_dataset.filter(like="drugs").columns

    # Seleccionamos tres sujetos específicos por sus índices
    # Cambia estos índices según los sujetos que quieras analizar


    fig, axes = plt.subplots(nrows=len(subjects), ncols=3, figsize=(20, 15))
    
    diagnosis_palette = sns.color_palette("Blues", n_colors=len(subjects))
    procedure_palette = sns.color_palette("Greens", n_colors=len(subjects))
    medication_palette = sns.color_palette("Reds", n_colors=len(subjects))



    for i, subject in enumerate(subjects):
        df = train_ehr_dataset[train_ehr_dataset["id_patient"] == subject].sort_values("visit_rank")
        
        # Diagnósticos
        df['diagnosis_sum'] = df[diagnosis_columns].sum(axis=1)
        sns.barplot(x="visit_rank", y="diagnosis_sum", data=df, ax=axes[i, 0], color=diagnosis_palette[i])
        axes[i, 0].set_title(f'Subject {subject} - Diagnosis {type_data}', fontsize=10)
        axes[i, 0].set_xlabel('Visit Number')
        axes[i, 0].set_ylabel('Number of ICD-9 Codes')
        
        # Procedimientos
        df['procedure_sum'] = df[procedure_columns].sum(axis=1)
        sns.barplot(x="visit_rank", y="procedure_sum", data=df, ax=axes[i, 1],color=procedure_palette[i])
        axes[i, 1].set_title(f'Subject {subject} - Procedures {type_data}'  , fontsize=10)
        axes[i, 1].set_xlabel('Visit Number')
        axes[i, 1].set_ylabel('Number of ICD-9 Codes')
        
        # Medicamentos
        df['medication_sum'] = df[medication_columns].sum(axis=1)
        sns.barplot(x="visit_rank", y="medication_sum", data=df, ax=axes[i, 2],color=medication_palette[i])
        axes[i, 2].set_title(f'Subject {subject} - Drugs {type_data}', fontsize=10)
        axes[i, 2].set_xlabel('Visit Number')
        axes[i, 2].set_ylabel('Number of ICD-9 Codes')

    plt.tight_layout()
    plt.show()

    if path_img is not None:
        save_plot_as_svg(plt, path_img, "count_matrix_progression")

    plt.tight_layout()
    plt.show()

    if path_img is not None:
        save_plot_as_svg(plt, path_img, "count_matrix_progression")
        
         
def plot_visit_trejetory(   train_ehr_dataset,diagnosis_columns,procedure_columns,medication_columns,num_visit_count = 10, patient_visit = 3,path_img = None, type_data = "Train"):
            visit_counts = train_ehr_dataset['id_patient'].value_counts()
            patients_with_10_visits = visit_counts[visit_counts == num_visit_count].index
            # Seleccionar 3 pacientes al azar
            selected_patients = np.random.choice(patients_with_10_visits, patient_visit, replace=False)
            create_plot_count_matrix_for_specific_subject(train_ehr_dataset,selected_patients,type_data,path_img=path_img)
            #plot trajectories
            for patient_id in selected_patients:
                plot_patient_trajectory(train_ehr_dataset, patient_id=patient_id, diagnosis_columns=diagnosis_columns, visit_column='visit_rank',name= 'ICD-9 Codes',type_anal='Diagnoses',type_data="Train")        
                plot_patient_trajectory(train_ehr_dataset, patient_id=patient_id, diagnosis_columns=procedure_columns, visit_column='visit_rank',name= 'ICD-9 Codes',type_anal='Procedure',type_data="Train")        
                plot_patient_trajectory(train_ehr_dataset, patient_id=patient_id, diagnosis_columns=medication_columns, visit_column='visit_rank',name= 'Drugs',type_anal='Drugs',path_img = path_img,type_data="Train")        


def plot_patient_trajectory(df, patient_id, diagnosis_columns, visit_column='visit_rank', id_column='id_patient',name= 'ICD-9 Codes',type_anal='Diagnoses',path_img = None,type_data="Train"):
    # Filter data for the specific patient
    patient_data = df[df[id_column] == patient_id].sort_values(visit_column)    
    # Select only the diagnosis columns and the visit column
    plot_data = patient_data[diagnosis_columns + [visit_column]]
    # Set the visit column as the index
    plot_data = plot_data.set_index(visit_column)
    # Get the top 10 most frequent diagnoses
    diagnosis_sums = plot_data.sum().nlargest(10)
    top_diagnoses = diagnosis_sums.index.tolist()
    # Select only the top 10 diagnoses
    plot_data = plot_data[top_diagnoses]
    # Create the plot
    plt.figure(figsize=(15, 8))
    sns.heatmap(plot_data, cmap='YlOrRd', annot=True, fmt='g', cbar_kws={'label': 'Count'})
    plt.title(f'Top 10  Trajectory for Patient {patient_id} {type_data}')
    plt.xlabel(name)
    plt.ylabel('Visit Rank')
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    if path_img is not None:
        save_plot_as_svg(plt, path_img, "trajectories_patient_")



        
