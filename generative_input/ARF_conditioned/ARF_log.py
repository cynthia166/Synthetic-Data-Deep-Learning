#from predict_prob_codes import DemographicPredictor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices
import scipy
from predict_prob_codes import MultiModelDemographicPredictor
#from arfpy import utils
from config_conditonalarf import *
import os
import joblib
import xgboost as xgb

def convertir_categoricas(df,categorical_cols):
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    return df  
def safe_truncnorm_params(myclip_a, myclip_b, myloc, myscale):
    """
    Calcula parámetros seguros para scipy.stats.truncnorm, manejando NaN.
    
    Args:
    myclip_a: Límite inferior de truncamiento.
    myclip_b: Límite superior de truncamiento.
    myloc: Media de la distribución no truncada.
    myscale: Desviación estándar de la distribución no truncada.
    
    Returns:
    tuple: (a, b, loc, scale) parámetros seguros para truncnorm.
    """
    # Convertir todos los inputs a Series de Pandas si no lo son ya
    myclip_a = pd.Series(myclip_a)
    myclip_b = pd.Series(myclip_b)
    myloc = pd.Series(myloc)
    myscale = pd.Series(myscale)

    # Reemplazar NaN con valores seguros
    myclip_a = myclip_a.fillna(myloc - 10*myscale)
    myclip_b = myclip_b.fillna(myloc + 10*myscale)
    myloc = myloc.fillna(myclip_a.mean())
    myscale = myscale.fillna((myclip_b - myclip_a) / 4)

    # Ajustar myscale si es menor o igual a cero
    myscale = myscale.where(myscale > 0, (myclip_b - myclip_a).abs() / 1000)
    myscale = myscale.where(myscale > 1e-8, 1e-8)

    # Calcular a y b
    a = ((myclip_a - myloc) / myscale).clip(lower=-10)
    b = ((myclip_b - myloc) / myscale).clip(upper=10)

    # Ajustar a y b si a >= b
    mask = a >= b
    mid = (a + b) / 2
    a = a.where(~mask, mid - 0.5)
    b = b.where(~mask, mid + 0.5)

    return a, b, myloc, myscale



def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)    

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def bnd_fun(tree, p, forest, feature_names ):
    my_tree = forest.estimators_[tree].tree_
    num_nodes = my_tree.node_count
    lb = np.full(shape=(num_nodes, p), fill_value=float('-inf'))
    ub = np.full(shape=(num_nodes, p), fill_value=float('inf'))
    for i in range(num_nodes):
        left_child = my_tree.children_left[i]
        right_child = my_tree.children_right[i]
        if left_child > -1: # leaf nodes are indicated by -1
            ub[left_child,: ] = ub[right_child, :] = ub[i,: ]
            lb[left_child,: ] = lb[right_child,: ] = lb[i,: ]
            if left_child != right_child:
                # If no pruned node, split changes bounds
                ub[left_child, my_tree.feature[i]] = lb[right_child,  my_tree.feature[i]] = my_tree.threshold[i]
    leaves = np.where(my_tree.children_left < 0)[0]
    # lower and upper bounds, to long format, to single data frame for return
    l = pd.concat([pd.Series(np.full(shape=(leaves.shape[0]), fill_value=tree), name = 'tree'), 
                             pd.Series(leaves, name = 'leaf'), 
                             pd.DataFrame(lb[leaves,], columns = feature_names)], axis = 1)
    u = pd.concat([pd.Series(np.full(shape=(leaves.shape[0]), fill_value=tree), name = 'tree'), 
                             pd.Series(leaves, name = 'leaf'), 
                             pd.DataFrame(ub[leaves,], columns = feature_names)], axis = 1)
    ret = pd.merge(left=pd.melt(l, id_vars=['tree', 'leaf'], value_name='min'),
              right = pd.melt(u, id_vars=['tree', 'leaf'], value_name='max'),
              on=['tree','leaf', 'variable'])
    del(l,u)
    return ret
    


import pandas as pd
class arf2:
    """Implements Adversarial Random Forests (ARF) in python
    Usage:
    1. fit ARF model with arf()
    2. estimate density with arf.forde()
    3. generate data with arf.forge().

    :param x: Input data.
    :type x: pandas.Dataframe
    :param num_trees:  Number of trees to grow in each forest, defaults to 30
    :type num_trees: int, optional
    :param delta: Tolerance parameter. Algorithm converges when OOB accuracy is < 0.5 + `delta`, defaults to 0
    :type delta: float, optional
    :param max_iters: Maximum iterations for the adversarial loop, defaults to 10
    :type max_iters: int, optional
    :param early_s(top): Terminate loop if performance fails to improve from one round to the next?, defaults to True
    :type early_stop: bool, optional
    :param verbose: Print discriminator accuracy after each round?, defaults to True
    :type verbose: bool, optional
    :param min_node_size: minimum number of samples in terminal node, defaults to 5 
    :type min_node_size: int
    """   
    def __init__(self, x, num_trees=30, delta=0, max_iters=1, early_stop=True, verbose=True, min_node_size=5, 
                    demographic_predictor=None, demographic_cols=None, medical_code_cols=None, **kwargs):
            # ... (keep the existing initialization code)

            # Add new attributes for demographic prediction
        # assertions
        assert isinstance(x, pd.core.frame.DataFrame), f"expected pandas DataFrame as input, got:{type(x)}"
        assert len(set(list(x))) == x.shape[1], f"every column must have a unique column name"
        assert max_iters >= 0, f"negative number of iterations is not allowed: parameter max_iters must be >= 0"
        assert min_node_size > 0, f"minimum number of samples in terminal nodes (parameter min_node_size) must be greater than zero"
        assert num_trees > 0, f"number of trees in the random forest (parameter num_trees) must be greater than zero"
        assert 0 <= delta <= 0.5, f"parameter delta must be in range 0 <= delta <= 0.5"


        # initialize values 
        x_real = x.copy()
        self.p = x_real.shape[1]
        self.orig_colnames = list(x_real)
        self.num_trees = num_trees
        self.demographic_predictor = demographic_predictor
        self.demographic_cols = demographic_cols
        self.medical_code_cols = medical_code_cols

        # Find object columns and convert to category
        self.object_cols = x_real.dtypes == "object"
        for col in list(x_real):
            if self.object_cols[col]:
                x_real[col] = x_real[col].astype('category')
            
        # Find factor columns
        self.factor_cols = x_real.dtypes == "category"
        
        # Save factor levels
        self.levels = {}
        for col in list(x_real):
            if self.factor_cols[col]:
                self.levels[col] = x_real[col].cat.categories
            
        # Recode factors to integers
        for col in list(x_real):
            if self.factor_cols[col]:
                x_real[col] = x_real[col].cat.codes
            
        # If no synthetic data provided, sample from marginals
        x_synth = x_real.apply(lambda x: x.sample(frac=1).values)
        
        # Merge real and synthetic data
        x = pd.concat([x_real, x_synth])
        y = np.concatenate([np.zeros(x_real.shape[0]), np.ones(x_real.shape[0])])
        # real observations = 0, synthetic observations = 1

        # pass on x_real
        self.x_real = x_real

        # Fit initial RF model
        clf_0 = RandomForestClassifier( oob_score= True, n_estimators=self.num_trees,min_samples_leaf=min_node_size, **kwargs) 
        clf_0.fit(x, y)

        iters = 0

        acc_0 = clf_0.oob_score_ # is accuracy directly
        acc = [acc_0]

        if verbose is True:
           print(f'Initial accuracy is {acc_0}')

        if (acc_0 > 0.5 + delta and iters < max_iters):
            converged = False
            while (not converged): # Start adversarial loop
                # get nodeIDs
                nodeIDs = clf_0.apply(self.x_real) # dimension [terminalnode, tree]

                # add observation ID to x_real
                x_real_obs = x_real.copy()
                x_real_obs['obs'] = range(0,x_real.shape[0])

                # add observation ID to nodeIDs
                nodeIDs_pd = pd.DataFrame(nodeIDs)
                tmp = nodeIDs_pd.copy()
                #tmp.columns = [ "tree" + str(c) for c in tmp.columns ]
                tmp['obs'] = range(0,x_real.shape[0])
                tmp = tmp.melt(id_vars=['obs'], value_name="leaf", var_name="tree")

                # match real data to trees and leafs (node id for tree)
                x_real_obs = pd.merge(left=x_real_obs, right=tmp, on=['obs'], sort=False)
                x_real_obs.drop('obs', axis = 1, inplace= True)

                # sample leafs
                tmp.drop("obs", axis=1, inplace=True)
                tmp = tmp.sample(x_real.shape[0], axis=0, replace=True)
                tmp = pd.Series(tmp.value_counts(sort = False ), name = 'cnt').reset_index()
                draw_from = pd.merge(left = tmp, right = x_real_obs, on=['tree', 'leaf'], sort=False )

                # sample synthetic data from leaf
                grpd =  draw_from.groupby(['tree', 'leaf'])
                x_synth = [grpd.get_group(ind).apply(lambda x: x.sample(n=grpd.get_group(ind)['cnt'].iloc[0], replace = True).values) for ind in grpd.indices]
                x_synth = pd.concat(x_synth).drop(['cnt', 'tree', 'leaf'], axis=1)
                
                # delete unnecessary objects 
                del(nodeIDs, nodeIDs_pd, tmp, x_real_obs, draw_from)

                # merge real and synthetic data
                x = pd.concat([x_real, x_synth])
                y = np.concatenate([np.zeros(x_real.shape[0]), np.ones(x_real.shape[0])])
                
                # discrimintator
                clf_1 = RandomForestClassifier( oob_score= True, n_estimators=self.num_trees, min_samples_leaf=min_node_size,**kwargs) 
                clf_1.fit(x, y)

                # update iters and check for convergence
                acc_1 = clf_1.oob_score_
                
                acc.append(acc_1)
                
                iters = iters + 1
                plateau = True if early_stop is True and acc[iters] > acc[iters - 1] else False
                if verbose is True:
                    print(f"Iteration number {iters} reached accuracy of {acc_1}.")
                if (acc_1 <= 0.5 + delta or iters >= max_iters or plateau):
                    converged = True
                else:
                    clf_0 = clf_1
        self.clf = clf_0
        self.acc = acc 
            
        # Pruning
        pred = self.clf.apply(self.x_real)
        for tree_num in range(0, self.num_trees):
            tree = self.clf.estimators_[tree_num]
            left = tree.tree_.children_left
            right = tree.tree_.children_right
            leaves = np.where(left < 0)[0]

            # get leaves that are too small
            unique, counts = np.unique(pred[:, tree_num], return_counts=True)
            to_prune = unique[counts < min_node_size]

            # also add leaves with 0 obs.
            to_prune = np.concatenate([to_prune, np.setdiff1d(leaves, unique)])

            while len(to_prune) > 0:
                for tp in to_prune:
                # Find parent
                    parent = np.where(left == tp)[0]
                    if len(parent) > 0:
                        # Left child
                        left[parent] = right[parent]
                    else:
                        # Right child
                        parent = np.where(right == tp)[0]
                        right[parent] = left[parent]
                    # Prune again if child was pruned
                to_prune = np.where(np.in1d(left, to_prune))[0]

    import numpy as np
from scipy import stats




def forge(n, bnds, num_trees, factor_cols, p, levels, object_cols, res,
                    demographic_predictor=None, demographic_cols=None, medical_code_cols=None):
    params, class_probs, clf, orig_colnames = res["cnt"], res["cat"], res["forest"], res["meta"]["variable"].tolist()
    dist = "truncnorm"

    # Optimizar selección de nodos
    unique_bnds = bnds[['tree', 'nodeid', 'cvg']].drop_duplicates()
    p_draws = unique_bnds['cvg'].values / num_trees
    draws = np.random.choice(len(unique_bnds), size=n, p=p_draws)
    sampled_trees_nodes = unique_bnds.iloc[draws].reset_index(drop=True)
    sampled_trees_nodes['obs'] = np.arange(n)

    # Inicializar DataFrame para datos nuevos
    data_new = pd.DataFrame(index=range(n), columns=orig_colnames)

    # Generar datos no médicos
    non_medical_cols = [col for col in orig_colnames if col not in medical_code_cols]
    for colname in non_medical_cols:
        if factor_cols[orig_colnames.index(colname)]:
            col_probs = class_probs[class_probs["variable"] == colname]
            merged = pd.merge(sampled_trees_nodes[['obs', 'tree', 'nodeid']], col_probs, on=['tree', 'nodeid'])
            grouped = merged.groupby('obs')
            data_new[colname] = grouped.apply(lambda x: np.random.choice(x['value'], p=x['prob'] / x['prob'].sum()))
        else:
            col_params = params[params["variable"] == colname]
            merged = pd.merge(sampled_trees_nodes[['obs', 'tree', 'nodeid']], col_params, on=['tree', 'nodeid'])
            a, b = (merged['min'] - merged['mean']) / merged['sd'], (merged['max'] - merged['mean']) / merged['sd']
            data_new[colname] = stats.truncnorm.rvs(a, b, loc=merged['mean'], scale=merged['sd'])

    # Generar datos médicos
    if demographic_predictor is not None:
        demo_probs = predict_probabilities_u(demographic_predictor[0], demographic_predictor[1], data_new[demographic_cols])
        
        all_groups_info = []
        for code in medical_code_cols:
            code_probs = class_probs[class_probs["variable"] == code]
            if not code_probs.empty:
                merged = pd.merge(sampled_trees_nodes[['obs', 'tree', 'nodeid']], code_probs, on=['tree', 'nodeid'])
                
                if code in demo_probs.columns:
                    demo_prob = demo_probs[code].values
                    demos_prob_act = np.where(merged['value'] == 0, 1 - demo_prob[merged['obs']], demo_prob[merged['obs']])
                    adjusted_probs = merged['prob'] * demos_prob_act
                    adjusted_probs /= adjusted_probs.groupby(merged['obs']).transform('sum')
                else:
                    adjusted_probs = merged['prob'] / merged.groupby('obs')['prob'].transform('sum')
                
                merged['adjusted_probs'] = adjusted_probs
                
                data_new[code] = merged.groupby('obs').apply(
                    lambda x: np.random.choice(x['value'], p=x['adjusted_probs'])
                ).values
                
                all_groups_info.append(merged)
        
        dataframe_total = pd.concat(all_groups_info, ignore_index=True)
    else:
        dataframe_total = pd.DataFrame()

    return data_new, dataframe_total

def forgev2(n, bnds,
        num_trees,
        factor_cols, 
        p,  levels,
        object_cols,
        res,
        demographic_predictor=None, demographic_cols=None, medical_code_cols=None,):
    """This part is for data generation (FORGE)

    :param n: Number of synthetic samples to generate.
    :type n: int
    :return: Returns generated data.
    :rtype: pandas.DataFrame
    """
    # try:
    #     getattr(self, 'bnds')
    # except AttributeError:
    #     raise AttributeError('need density estimates to generate data -- run .forde() first!')

    # # Sample new observations and get their terminal nodes
    # # Draw random leaves with probability proportional to coverage
    params=res["cnt"] 
    class_probs= res["cat"] 
    clf=res["forest"]
    orig_colnames=res["meta"]["variable"]
    dist="truncnorm"
    unique_bnds = bnds[['tree', 'nodeid', 'cvg']].drop_duplicates()

    draws = np.random.choice(a=range(unique_bnds.shape[0]), p=unique_bnds['cvg'] / num_trees, size=n)
    sampled_trees_nodes = unique_bnds[['tree','nodeid']].iloc[draws,].reset_index(drop=True).reset_index().rename(columns={'index': 'obs'})

    # Get distributions parameters for each new obs.
    if np.invert(factor_cols).any():
        obs_params = pd.merge(sampled_trees_nodes, params, on=["tree", "nodeid"]).sort_values(by=['obs'], ignore_index=True)
    
    # Get probabilities for each new obs.
    if factor_cols.any():
        obs_probs = pd.merge(sampled_trees_nodes, class_probs, on=["tree", "nodeid"]).sort_values(by=['obs'], ignore_index=True)
    
    # Sample new data from mixture distribution over trees
    data_new = pd.DataFrame(index=range(n), columns=orig_colnames)
    
    # Generate demographic data and other non-medical code data first
    for j, colname in enumerate(orig_colnames):
        if colname not in medical_code_cols:
            if factor_cols[j]:
                # Factor columns: Multinomial distribution
                data_new[colname] = obs_probs[obs_probs["variable"] == colname].groupby("obs").sample(weights="prob")["value"].reset_index(drop=True)
            else:
                # Continuous columns: Match estimated distribution parameters with r...() function
                if dist == "truncnorm":
                    myclip_a = obs_params.loc[obs_params["variable"] == colname, "min"]
                    myclip_b = obs_params.loc[obs_params["variable"] == colname, "max"]
                    myloc = obs_params.loc[obs_params["variable"] == colname, "mean"]
                    myscale = obs_params.loc[obs_params["variable"] == colname, "sd"]
                    data_new[colname] = scipy.stats.truncnorm(
                        a=(myclip_a - myloc) / myscale,
                        b=(myclip_b - myloc) / myscale,
                        loc=myloc,
                        scale=myscale
                    ).rvs(size=n)

    # Convert categories back to category for non-medical code columns
    # for col in orig_colnames:
    #     if factor_cols[col] and col not in medical_code_cols:
    #         data_new[col] = pd.Categorical.from_codes(data_new[col], categories=levels[col])

    # # Convert object columns back to object for non-medical code columns
    # for col in orig_colnames:
    #     if object_cols[col] and col not in medical_code_cols:
    #         data_new[col] = data_new[col].astype("object")

    # Generate medical code data using demographic prediction
    if demographic_predictor is not None:
        assert demographic_cols is not None, "demographic_cols must be provided when using demographic_predictor"
        assert medical_code_cols is not None, "medical_code_cols must be provided when using demographic_predictor"
        demo_probs =predict_probabilities_u(demographic_predictor[0],demographic_predictor[1], data_new[demographic_cols])
            
        all_groups_info = []
        for code in medical_code_cols:
            code_probs = obs_probs[obs_probs["variable"] == code]
            if not code_probs.empty:
                grouped = code_probs.groupby("obs")
                
                def sample_code(group):
                    prob_values = group['prob'].values
                    values = group['value'].values
                    
                    if code in demo_probs.columns:
                        demo_prob = demo_probs.loc[group.name, code]
                        demos_prob_act = np.where(values == 0, 
                                                (1 - demo_prob), 
                                                demo_prob)
        
                        adjusted_probs = prob_values * demos_prob_act
                        adjusted_probs /= adjusted_probs.sum()
                    else:
                        adjusted_probs = prob_values / prob_values.sum()
                    
                    sampled_value = np.random.choice(values, p=adjusted_probs)
                    group["adjusted_probs"]=adjusted_probs
                    group["demos_prob_act"]=demos_prob_act
                    print(group)
                    # Return a tuple with sampled value and a dict of group info
                    return (sampled_value, group)


                #grouped = code_probs.groupby("obs")
                
                results = grouped.apply(sample_code)
                sampled_values, groups_info = zip(*results)
        
        # Assign the sampled values to data_new
                data_new[code] = pd.Series(sampled_values, index=grouped.groups.keys())
            
                data_groups = pd.concat(groups_info)
            all_groups_info.append(data_groups)
        # Collect group info for this code
        dataframe_total= pd.concat(all_groups_info)       
                 
# Create a DataFrame with all group information
        
    # # Convert medical code columns to appropriate data types
    # for col in medical_code_cols:
    #     if col in factor_cols:
    #         data_new[col] = pd.Categorical.from_codes(data_new[col], categories=levels[col])
    #     elif col in factor_cols:
    #         data_new[col] = data_new[col].astype("object")

    return data_new,dataframe_total
    
def sample_patients_list(ruta_patients, x_train):
    sample_patients = load_pickle(ruta_patients)
    sample_df = x_train[x_train['SUBJECT_ID'].isin(sample_patients)]
    return sample_df

def change_dtypes(x_train,cols_continuous_d):
   

    categorical_cols_d =[col for col in x_train.columns if col not in cols_continuous_d]
    #change to categorilcal 
    train_data_features = convertir_categoricas(x_train,categorical_cols_d)
    print(train_data_features.dtypes)
            # samplear 3# fre los pacientes
    return train_data_features

def load_models_and_scaler(model_path):
    """Load the dictionary of models and the scaler."""
    predictors = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return [predictors,scaler]

def predict_probabilities_u(models,scaler, new_data,model_log=False):
    models= models[0]
    X_scaled_continuous = scaler.fit_transform(new_data[continuous_cols])
    # Convert the scaled array back to a DataFrame
    X_scaled_continuous = pd.DataFrame(X_scaled_continuous, columns=continuous_cols, index=new_data.index)
    # Drop the original continuous columns and concatenate with the scaled DataFrame
    new_data_scaled = new_data.drop(columns=continuous_cols).join(X_scaled_continuous)


    
    probabilities = pd.DataFrame()
    for code, model in models.items():
        if model_log:
           probabilities[code] = model.predict_proba(new_data_scaled)[:, 1]
        else:
            dtest = xgb.DMatrix(new_data_scaled,enable_categorical=True)
        
            # Predict probabilities
            # XGBoost's predict method returns probabilities for binary classification
            probs = model.predict(dtest)
            
            # For binary classification, we only need the probability of the positive class
            probabilities[code] = probs
    
    
    return probabilities





def list_model_file_paths(directory):
    """
    Busca todos los archivos .pkl en el directorio especificado que contengan 'model' en su nombre
    y devuelve una lista de sus rutas completas.
    
    :param directory: Ruta al directorio que contiene los archivos de modelo
    :return: Una lista de strings con las rutas completas de los archivos de modelo
    """
    model_files = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".pkl") and "model" in filename.lower():
            file_path = os.path.join(directory, filename)
            model_files.append(file_path)
    
    return model_files


def predict_probabilities_multi(self, new_data, model_name):
        probabilities = pd.DataFrame()
        for code, model in models[model_name].items():
            scaler = scalers[code]
            X_scaled_continuous = scaler.transform(new_data[continuous_cols])
            X_scaled_continuous = pd.DataFrame(X_scaled_continuous, columns=continuous_cols, index=new_data.index)
            new_data_scaled = new_data.drop(columns=continuous_cols).join(X_scaled_continuous)
            probabilities[code] = model.predict_proba(new_data_scaled)[:, 1]
        return probabilities

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def group_and_encode_demographics(df, demographic_categories):
    """
    Group columns by demographic categories and apply label encoding.
    
    :param df: pandas DataFrame containing the data
    :param demographic_categories: list of demographic category names (e.g., ['gender', 'age_group', 'ethnicity'])
    :return: tuple containing the modified DataFrame and a dictionary of LabelEncoders
    """
    encoders = {}

    for category in demographic_categories:
        # Find all columns that contain the category name
        category_cols = [col for col in df.columns if category in col]
        
        if category_cols:
            # Group the columns and create a new column with the category name
            df[category] = df[category_cols].idxmax(axis=1).str.replace(category + '_', '')
            
            # Drop the original columns
            df = df.drop(columns=category_cols)
            
            # Apply label encoding
            encoder = LabelEncoder()
            df[category + '_encoded'] = encoder.fit_transform(df[category])
            
            # Store the encoder for later use
            encoders[category] = encoder
            df = df.drop(columns=category)
    return df, encoders

def get_input_forde(x_real):
    p = x_real.shape[1]
    orig_colnames = list(x_real)
    

    # Find object columns and convert to category
    object_cols = x_real.dtypes == "object"
    for col in list(x_real):
      if object_cols[col]:
        x_real[col] = x_real[col].astype('category')
    
    # Find factor columns
    factor_cols = x_real.dtypes == "category"
    
    # Save factor levels
    levels = {}
    for col in list(x_real):
      if factor_cols[col]:
        levels[col] = x_real[col].cat.categories
    
    # Recode factors to integers
    for col in list(x_real):
      if factor_cols[col]:
        x_real[col] = x_real[col].cat.codes
    return p, orig_colnames, num_trees,object_cols, factor_cols,levels, x_real    

    # Usage example
if __name__ == "__main__":
    # Assume we have training data in x_train
    archivo ="_cuarto"
    arf_path_car = path_result_arf+"ARF_demosca_r"+archivo+".pkl"
    fored_path_car = path_result+"density_params_5_v2_r"+archivo+".pkl"
    bonds_path =path_result+"bnds2.pkl"
    full_path = os.path.join(path_result, "synthetic_d")
    synthetic_data_path ="synthetic_d/"+archivo+".pkl"
    synthetic_probs_path =path_result+"synthetic_d/"+"demos_prob_2"+archivo+".pkl"


    if os.path.exists(full_path):
        print(f"The path {full_path} exists.")
    else:
        print(f"The path {full_path} does not exist.")
        

    x_train = load_data(path_dataset_demos_whole)[:1000]
    x_train= sample_patients_list(ruta_patients, x_train)    

    
    print(f'lista para excluir {len(list_col_exclute)}')    
        # se quitan columnas que no se utilizan y se convierte en categoricas, la matrix de conteo, subject id, admission dat
    x_train= x_train[[i for i in x_train.columns if i not in list_col_exclute['codes_no'].to_list()]]
    #columns_to_drop = ['LOSRD_sum', 'L_1s_last_p1','HADM_ID',"ADMITTIME",'GENDER_0']
    print("Cols to eliminate",columns_to_drop)
    x_train = x_train.drop(columns=columns_to_drop_arf)  
    x_train =  change_dtypes(x_train,cols_continuous_d)   

    #predictor_models = MultiModelDemographicPredictor(data, demographic_cols, medical_code_cols)
    #predictor_models.load_models()   
    #file_list =  list_model_file_paths(directory_predictors)  
    file_list = ["/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/ARF_conditioned/results/model_XGBoost.pkl"]
    print(file_list[-5:-3])
    
    for file_path in   file_list:   
        print(x_train.shape)
        predictor = load_models_and_scaler(file_path)
        if train_arf:
            arf_model = arf2(x_train,demographic_predictor=predictor, demographic_cols=base_demographic_columns, medical_code_cols=medical_code_cols)
            joblib.dump(arf_model,arf_path_car)
        else: 
            
            arf_model = joblib.load(arf_path_car) 
            arf_model.demographic_predictor = predictor
        # Estimate density with demographic prediction
        
        
        if train_path_forde:
                
            res,bnds, num_trees, factor_cols, p,  levels, object_cols= arf_model.forde()
            joblib.dump(res,fored_path_car)
            bnds = joblib.dump(bnds, bonds_path)

        else:
            res = joblib.load(fored_path_car)
            bnds = joblib.load(bonds_path)
            p, orig_colnames, num_trees,object_cols, factor_cols,levels, x_real =get_input_forde(x_train)      
        # Generate synthetic data
       
        synthetic_data,data_groups  =forge(39645, bnds,
           num_trees,
           factor_cols, 
           p,  levels,
           object_cols,
           res,demographic_predictor=predictor,
            demographic_cols=base_demographic_columns, medical_code_cols=medical_code_cols)
        
        #synthetic_data = arf_model.forge(n=len(x_train))
        save_pickle(data_groups, path_result+synthetic_data_path)
        save_pickle(synthetic_data, synthetic_probs_path)
        print(synthetic_data.head())



#load_df = load_pickle("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/ARF_conditioned/results/synthetic_d/synnthetic_datalog_arf_sin_var2.pkl")