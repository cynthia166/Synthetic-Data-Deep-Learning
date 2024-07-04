# cols_continuous = ['Age_max', 'LOSRD_avg','days_between_visits']     
# fored = load_pkl("/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/FORED")
# trnorm_modified = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/ARF_tnorm/synthetic_data_generative_model_arf_adjs_per_0.7.pkl"
# initial_tnorm = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/synthetic_data_generative_model_arf_per_0.7.pkl"
import os
import sys
os.chdir('/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/')
sys.path.append('/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning')
from generative_model.utils_arf import *
from evaluation.resemb.resemblance.utilsstats import *
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices
import scipy
from arfpy import utils
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, kstest
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from collections import Counter
from evaluation.resemb.resemblance.config_fored  import *


class AnalysisFORED(
    original_data_path ,
    sample_patients_path,
    columns_to_drop,
    cols_continuous,
    save_path_features,
    create_features,
    fored_fixed,
    ruta_continous_observations,
    get_count_variables_per_node_tree_cont,
    continous_variable_ks_test,
    obtain_promedio_por_nodo_ks_dist,
    perform_ks_test,
    threshold_continous,
    col_continous,
    col_cat,
    columns_to_drop,
    save_path_count_features
    
    
):
    
    def __init__(self):
        self.original_data_path = original_data_path
        self.sample_patients_path = sample_patients_path
        self.columns_to_drop = columns_to_drop
        self.cols_continuous = cols_continuous
        self.save_path_features = save_path_features
        self.create_features = create_features
        self.FORED = load_pkl(fored_fixed)
        self.params   = load_pkl(ruta_continous_observations)        
        self.aux_real =  load_pkl(save_path_features)
        self.get_count_variables_per_node_tree_cont = get_count_variables_per_node_tree_cont
        self.continous_variable_ks_test = continous_variable_ks_test
        self.obtain_promedio_por_nodo_ks_dist = obtain_promedio_por_nodo_ks_dist
        self.perform_ks_test = perform_ks_test
        self.threshold_continous = threshold_continous
        self.col_continous = col_continous
        self.save_path_count_features = save_path_count_features
        self.col_cat = col_cat
        
        
    def initialize(self):
    

        if  self.create_features:
            self.x_real = self.get_featuresfun()
        else:
            self.x_real =load_pkl(self.save_path_features)
        if  get_count_variables_per_node_tree_cont:
            self.get_count_pervariables_fun()
        if  continous_variable_ks_test:
            self.get_continous_variable_ks_test
        if  self.get_analysis_categotical:    
            self.get_analysis_per_node()
    def  get_featuresfun(self):
       
            train_data_features = load_data(self.original_data_path)
            train_data_features["year"] = train_data_features['ADMITTIME'].dt.year
            train_data_features['month'] = train_data_features['ADMITTIME'].dt.month    
            # se quitan columnas que no se utilizan y se convierte en categoricas, la matrix de conteo, subject id, admission data
            train_data_features = train_data_features.drop(columns=self.columns_to_drop)  
            categorical_cols =[col for col in train_data_features.columns if col not in self.cols_continuous]
            train_data_features = convertir_categoricas(train_data_features,categorical_cols)
            print(train_data_features.dtypes)
            sample_patients_r = load_pkl(self.sample_patients_path)
            x_real = train_data_features[train_data_features['SUBJECT_ID'].isin(sample_patients_r)]
            save_pickle(x_real,self.save_path_features)
    
                 
    def   get_count_pervariables_fun(self):
        
            
            #feature importance
            clf = self.FORED["forest"]

            #inputs
            factor_cols = self.x_real.dtypes == "category"
                
            dist = "truncnorm"
            oob = False
            alpha = 0
            num_trees = 30
            orig_colnames = list(x_real)
            p = self.x_real.shape[1]
            # Get terminal nodes for all observations
            
            pred = clf.apply(self.x_real)
            # If OOB, use only OOB trees
            if oob:
                for tree in range(num_trees):
                    idx_oob = np.isin(range(self.x_real.shape[0]), _generate_unsampled_indices(clf.estimators_[tree].random_state, x.shape[0], x.shape[0]))
                    pred[np.invert(idx_oob), tree] = -1
                
            # Get probabilities of terminal nodes for each tree 
            # node_probs dims: [nodeid, tree]
            #node_probs = np.apply_along_axis(func1d= utils.bincount, axis = 0, arr =pred, nbins = np.max(pred))
            
            # compute leaf bounds and coverage
            bnds = pd.concat([utils.bnd_fun(tree=j, p = p, forest = clf, feature_names = orig_colnames) for j in range(num_trees)])
            bnds['f_idx']= bnds.groupby(['tree', 'leaf']).ngroup()

            bnds_2 = pd.DataFrame()
            for t in range(num_trees):
                unique, freq = np.unique(pred[:,t], return_counts=True)
                vv = pd.concat([pd.Series(unique, name = 'leaf'), pd.Series(freq/pred.shape[0], name = 'cvg')], axis = 1)
                zz = bnds[bnds['tree'] == t]
                bnds_2 =pd.concat([bnds_2,pd.merge(left=vv, right=zz, on=['leaf'])])
            bnds = bnds_2
            del(bnds_2)

            # set coverage for nodes with single observations to zero
            if np.invert(factor_cols).any():
               bnds.loc[bnds['cvg'] == 1/pred.shape[0],'cvg'] = 0
            
            # no parameters to learn for zero coverage leaves - drop zero coverage nodes
            bnds = bnds[bnds['cvg'] > 0]

            # rename leafs to nodeids
            bnds.rename(columns={'leaf': 'nodeid'}, inplace=True)

            # save bounds to later use coverage for drawing new samples
            bnds= bnds
            # Fit continuous distribution in all terminal nodes
            params = pd.DataFrame()
            if np.invert(factor_cols).any():
                for tree in range(num_trees):
                    dt = self.x_real.loc[:, np.invert(factor_cols)].copy()
                    dt["tree"] = tree
                    dt["nodeid"] = pred[:,tree]
                    # merge bounds and make it long format
                    long = pd.merge(right = bnds[['tree', 'nodeid','variable', 'min', 'max', 'f_idx']], left = pd.melt(dt[dt["nodeid"] >= 0], id_vars = ["tree", "nodeid"]), on = ['tree', 'nodeid', 'variable'], how = 'left')
                    long["count"] = 1
                    # get distribution parameters
                    if dist == "truncnorm":
                        
                        res = long.groupby([ 'tree',"nodeid", "variable"], as_index = False).agg(mean=("value", "mean"), sd=("value", "std"), min = ("min", "min"), max = ("max", "max"), count = ("count","count"))
                    else:
                        raise ValueError('unknown distribution, make sure to enter a vaild value for dist')
                        exit()
                    params = pd.concat([params, res])
            save_pickle(params,self.save_path_count_features)
            print("Process finished!")    
            # Get class probabilities in all terminal nodes
            
        
        
    def get_continous_variable_ks_test(self):
            
        
        # perfom ks test 5 distribution
        if perform_ks_test:
            long = load_pkl(save_path_dist)
            test_distributions_nodes(long,save_path_dist)

        if obtain_promedio_por_nodo_ks_dist:
            long_ =  load_pkl(save_path_dist)
            #count signifiant prom
            long_,mena_kst,long_,uniform_l,triang_l,truncnorm_l,expon_l,gmm_l = get_proportion_significant_kst(long_)
            res = res_promedio_ks_stat(uniform_l,triang_l,truncnorm_l,expon_l,gmm_l)

    def get_analysis_categotical(self):
        
        fored_fixed =   FORED
        # agrupar por nodos en lista conta valore s repeditos
        df = self.FORED["cat"]
        aux  = df[df["variable"]== self.col_cat]
        #ciontar el promedio de aquellos con menso de 13 valores
        value_counts_prob = aux.groupby(["tree","nodeid"])["value"].nunique().value_counts()
        print("Count of unique values_per node and tree",value_counts_prob)
        df_count_subject_id = aux.groupby(["tree","nodeid"] ,as_index = False).agg( mean = ("value","mean"),count = ("value","count"))
        print("Count of values per node and tree" ,df_count_subject_id)
        print("Stats of Count of values per node and tree" ,df_count_subject_id.describe())
        
        print("Analyzing variables with less than " + s+ "per node")
        # Those values that have less thatn 30 values per node
        less_thirteen = df_count_subject_id[df_count_subject_id["count"]<30]
        tree_l = less_thirteen["tree"].unique()
        node_l = less_thirteen["nodeid"].unique()
        subset_thirten_tree = aux[aux["tree"].isin(list(tree_l))]
        subset_thirten_nodeandtree = subset_thirten_tree[subset_thirten_tree["nodeid"].isin(list(node_l))]
        print("Shape of subset with less than 30 values per node",subset_thirten_nodeandtree.shape)
        print("Unique values of the subset",subset_thirten_nodeandtree["value"].nunique())
        print("Percentage of all the observation",df_count_subject_id.shape[0]/df_count_subject_id.shape[0])
        unique_values_=subset_thirten_nodeandtree.groupby(["tree","nodeid"])["prob"].nunique().value_counts()
        print("Count of unique values per node and tree",unique_values_)
        
        
        
        groupe_nodes = aux.groupby(["tree","nodeid"])["value"].apply(list).reset_index()
        grouped_len= aux.groupby(["tree","nodeid"])["prob"].apply(len).reset_index()
        grouped_len["list_prob"] =[ len(set(groupe_nodes["value"][i]))/len(groupe_nodes["value"][i]) for i in range(len(grouped_len))]
        grouped_len["list_prob"] = [grouped_len["list_prob"][i] / grouped_len["prob"][i] for i in  range(len(grouped_len))]
        print("stat of count per values per node",grouped_len["prob"].describe())
        x_label = "Count of values patient per node"
        title = "Count of total patient per node in each tree"
        hist(grouped_len,"prob",title,x_label,100)





        #revuisar la probas cuantas son unicas por nodo y grupo
        groupe_nodes_2 = aux.groupby(["tree","nodeid"])["prob"].apply(list).reset_index()

        grouped_len["prob_unique"] =[ len(set(groupe_nodes_2["prob"][i])) for i in range(len(grouped_len))]
        #revisar el promedio por cada lista y nodo 
        grouped_len["prob_mean"] =[ sum(groupe_nodes_2["prob"][i])/len(groupe_nodes_2["prob"][i]) for i in range(len(grouped_len))]
        grouped_len["prob_mean"].describe()
        print("Mean of probability per node, for all the trees in de ARF",grouped_len["prob_mean"].describe())
        title = "Mean of probability per node, for all the trees in de ARF"
        x_label  = "Mean of probability per node"
        hist(grouped_len,"prob_mean",title,x_label,)
        


        #plot of all 
        title = "Probability per observation, for all node of all the trees in de ARF"
        x_label  = "Mean of probability per node"
        hist(aux,"prob",title,x_label,)
        #repetition of values
        count_all_patinets_repetions_forest = aux.groupby(["value"])["prob"].apply("count").reset_index()
        print("Count of repetition of values in the whole Forest considering all the node",count_all_patinets_repetions_forest["prob"   ].describe())
        title = "Count of repetition of values in the whole Forest considering all the node"
        xlab = "Count of patients"
        count_all_patinets_repetions_forest.describe()
        hist(count_all_patinets_repetions_forest,"prob",title,xlab)
        
        count_all_patinets_repetions_forest_mean = aux.groupby(["value"])["prob"].apply("mean").reset_index()
        print("Count of repetition of values in the whole Forest considering all the node",count_all_patinets_repetions_forest_mean["prob"   ].describe())
        title = "Mean of prob of values in the whole Forest considering all the node"
        xlab = "Count of patients"
        count_all_patinets_repetions_forest_mean.describe()
        hist(count_all_patinets_repetions_forest_mean,"prob",title,xlab)
        
        #distribucion de numero de datos por nodo y arbol
        print("Distribution Count of obervation per node ")
        dist_node_tree_cat(FORED,self.col_cat)


    def get_analysis_continous(self):

        cnt = self.FORED[self]
        aux2 = cnt[cnt["variable"]==self.col_continous]
        col  ="mean"
        title = "Mean of days parameter mean tcnorm of  visit considering all the nodes and trees"
        xlab = "Mean Days between visit considering"
        print("Mean of all the nodes considetin all the treas",aux2["mean"].describe())
        hist(aux2,"mean",title,xlab)
        # contar por nodo
        
        
        
        print("Mean of all the node and trees",count_all_patinets_repetions_forest_mean = aux2["mean"].mean())
        #title = "Mean of days parameter standard deviation tcnorm of  visit considering all the nodes and trees"
        #xlab = "Mean of days between visits"
        # count_all_patinets_repetions_forest_mean.describe()
        
        # hist(count_all_patinets_repetions_forest_mean,"sd",title,xlab)

        #description real data
        
        print("Train data distribution of "+ self.col_continous,self.aux_real["days_between_visits"].describe())
        hist(self.aux_real,"days_between_visits",title,xlab)


    
        aux = self.params[self.params["variable"]==self.col_continous]
        print("Distribution Count of obervation per node ",aux.describe())

    #countof observation in continous variables.        
        pmenor_30_aux =aux[aux["count"]>self.threshold_continous]
        print("Distribution of count of values  less thatn",pmenor_30_aux)

        print("Mean of the persons with less than 30 observations",pmenor_30_aux["mean"].mean())


        col  ="mean"
        title = "Count of observation per node and trees"
        xlab = "Count Observations"
        hist(aux,"count",title,xlab)
        print("Truncated Graph")
        hist(aux,"count",title,xlab,100)
if __name__ == "__main__":
    AnalysisFORED(
    original_data_path ,
    sample_patients_path,
    columns_to_drop,
    cols_continuous,
    save_path_features,
    create_features,
    fored_fixed,
    ruta_continous_observations,
    get_count_variables_per_node_tree_cont,
    continous_variable_ks_test,
    obtain_promedio_por_nodo_ks_dist,
    perform_ks_test,
    threshold_continous,
    col_continous,
    col_cat,
    columns_to_drop,
    save_path_count_features).initialize()
    
'''
# fit in every node a distribution:
import numpy as np
import pandas as pd
from scipy.stats import beta, uniform, triang, truncnorm, expon, kstest, gaussian_kde
from sklearn.mixture import GaussianMixture

print(rest.to_latex())
rest.to_csv("results/results_arf/results_trunxated.csv")


def evaluate_kde_methods(data, kernels=['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear']):
    """
    Estimate the density parameters of a feature using various KDE methods and evaluate with KS test.
    
    Parameters:
    data (pd.Series): Input feature data.
    kernels (list): List of kernels to evaluate.
    
    Returns:
    dict: A dictionary containing the best kernel and its KS statistic.
    """
    # Ensure data is in the correct format
    if isinstance(data, pd.Series):
        data = data.values
    data = data[:, np.newaxis]

    # Initialize results dictionary
    results = {}

    # Cross-validation to find the best bandwidth for each kernel
    bandwidths = np.logspace(-1, 1, 20)
    for kernel in kernels:
        grid = GridSearchCV(KernelDensity(kernel=kernel), {'bandwidth': bandwidths}, cv=5)
        grid.fit(data)
        
        # Get the best bandwidth
        best_bandwidth = grid.best_params_['bandwidth']
        
        # Fit KDE with the best bandwidth
        kde = KernelDensity(kernel=kernel, bandwidth=best_bandwidth)
        kde.fit(data)
        
        # Generate samples from the fitted KDE
        log_density = kde.score_samples(data)
        kde_samples = np.exp(log_density)
        
        # Perform KS test
        ks_stat, p_value = kstest(data.flatten(), kde_samples)
        
        # Store the results
        results[kernel] = {'bandwidth': best_bandwidth, 'ks_stat': ks_stat, 'p_value': p_value}
    
    # Find the best kernel with the lowest KS statistic
    best_kernel = min(results, key=lambda k: results[k]['ks_stat'])
    best_result = results[best_kernel]
    
    return {'best_kernel': best_kernel, 'best_result': best_result, 'all_results': results}

# Example usage
results = evaluate_kde_methods(real_data)


print("Best Kernel:", results['best_kernel'])
print("Best Result:", results['best_result'])
print("All Results:", results['all_results'])

# def select_best_distribution(fit_results):
#     best_distributions = {}
#     for index, row in fit_results.iterrows():
#         best_dist = None
#         best_stat = float('inf')
#         for dist in ['beta', 'gamma', 'norm', 'lognorm', 'truncnorm']:
#             if row[dist + '_ks_stat'] < best_stat:
#                 best_stat = row[dist + '_ks_stat']
#                 best_dist = dist
#         best_distributions[row['feature']] = best_dist
#     return best_distributions


# def validate_best_distributions(data, best_distributions, fit_results):
#     for column in data.columns:
#         best_dist = best_distributions[column]
#         params = fit_results.loc[fit_results['feature'] == column, best_dist + '_params'].values[0]
#         plt.figure(figsize=(10, 5))
#         sns.histplot(data[column], kde=True, label='Real Data', color='blue')
#         if best_dist == 'truncnorm':
#             a, b, mean, std = params
#             synthetic_data = truncnorm.rvs(a, b, loc=mean, scale=std, size=1000)
#         else:
#             synthetic_data = eval(f"{best_dist}.rvs(*params, size=1000)")
#         sns.histplot(synthetic_data, kde=True, label='Synthetic Data', color='red', alpha=0.5)
#         plt.title(f'Validation for {column} using {best_dist} distribution')
#         plt.xlabel(column)
#         plt.ylabel('Frequency')
#         plt.legend()
#         plt.show()

# # Validar visualmente las mejores distribuciones seleccionadas

# fit_results = fit_distributions(real_training_data[cols_continuous])
# print(fit_results)
# fit_results.to_csv("fit_results.csv")
# best_distributions = select_best_distribution(fit_results)
# #{'Age_max': 'beta', 'LOSRD_avg': 'lognorm', 'days_between_visits': 'norm'}
# print("Mejores distribuciones por característica:")
# print(best_distributions)

# validate_best_distributions(real_training_data[cols_continuous], best_distributions, fit_results)



df_sorted = get_feature_importance(clf,synthetic_raw_data,10)

# synthetic_ehr_dataset_constrains = load_pickle("generated_synthcity_tabular/ARF/" + 'synthetic_ehr_dataset_contrainst_tnorm.pkl')
real_training_data =load_pickle("generated_synthcity_tabular/ARF/" + 'train_ehr_dataset.pkl')

aux = FORED['cat']
aux2 = aux[aux['variable'] == 'SUBJECT_ID']
aggregated_stats = aux.groupby('variable').agg({
    'prob': ['mean'],
   
}).reset_index()
aggregated_stats.sort_values(ascending=False, by=(    'prob', 'mean'))
# Aplanar el MultiIndex de las columnas
# aggregated_stats = df.groupby('age').agg({
#     'variable1': ['mean', 'std', 'min', 'max'],
#     'variable2': ['mean', 'std', 'min', 'max']
# }).reset_index()


#aggregated_stats.columns = ['_'.join(col).strip() for col in aggregated_stats.columns.values]
#print(aggregated_stats.to_latex())
print(aggregated_stats.sort_values(ascending=True, by=(    'prob', 'mean'))[:20].to_latex())
print(aggregated_stats.sort_values(ascending=False, by=(    'prob', 'mean'))[:20].to_latex())
# fored_aux = fored["cnt"][fored["cnt"]["variable"]=="days_between_visits"]

# import matplotlib.pyplot as plt

# min_vals = real_training_data["days_between_visits"].min(axis=0)
# max_vals = real_training_data["days_between_visits"].max(axis=0)



# ['readmission','HADM_ID',"ADMITTIME",'GENDER_0']
real_data = real_training_data["days_between_visits"]
# synthetic_data = synthetic_ehr_dataset["days_between_visits"]
# synthetic_data = np.clip(synthetic_data, min_vals, max_vals)
# plot_hist(real_data,synthetic_ehr_dataset_constrains["days_between_visits"])

# import numpy as np
# import seaborn as sns
# # Supongamos que estas son tus dos series de datos

# def visualize_data(data):
#     for column in data.columns:
#         plt.figure(figsize=(10, 5))
#         sns.histplot(data[column], kde=True)
#         plt.title(f'Distribution of {column}')
#         plt.xlabel(column)
#         plt.ylabel('Frequency')
#         plt.show()

# # Supongamos que `data_real` es tu conjunto de datos real
# visualize_data(real_training_data[cols_continuous])


# from scipy.stats import beta, gamma, norm, lognorm, truncnorm, kstest
# import pandas as pd

from scipy.stats import beta, gamma, norm, lognorm, truncnorm, expon, kstest
import pandas as pd
from threadpoolctl import threadpool_limits




# def plot_hist(real_data,synthetic_data):
#     # Calcular los límites combinados de los datos reales y sintéticos
#     data_min = min(real_data.min(), synthetic_data.min())
#     data_max = max(real_data.max(), synthetic_data.max())
#     # Generar los bins en función del rango combinado
#     bins = np.linspace(data_min, data_max, 80)  # 30 bins
#     # Crear los histogramas usando los mismos bins
#     plt.figure(figsize=(10, 6))
#     sns.histplot(real_data, bins=bins, color='blue', alpha=0.5, label='Real data', kde=False)
#     sns.histplot(synthetic_data, bins=bins, color='orange', alpha=0.5, label='Synthetic data', kde=False)
#     plt.legend(loc='upper right')
#     plt.title('Days between visits')
#     plt.xlabel('Days')
#     plt.ylabel('Frequency')
#     plt.show()
'''