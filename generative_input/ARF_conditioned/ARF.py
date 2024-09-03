#from predict_prob_codes import DemographicPredictor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices
import scipy
#from arfpy import utils
from config_conditonalarf import *
import os   
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
logging.basicConfig(level=logging.INFO)
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



def arf2( x, num_trees=30, delta=0, max_iters=1, early_stop=True, verbose=True, min_node_size=5, 
                    demographic_predictor=None, demographic_cols=None, medical_code_cols=None, **kwargs):
            # ... (keep the existing initialization code)

            # Add new attributes for demographic prediction
        # assertions
       
        assert len(set(list(x))) == x.shape[1], f"every column must have a unique column name"
        assert max_iters >= 0, f"negative number of iterations is not allowed: parameter max_iters must be >= 0"
        assert min_node_size > 0, f"minimum number of samples in terminal nodes (parameter min_node_size) must be greater than zero"
        assert num_trees > 0, f"number of trees in the random forest (parameter num_trees) must be greater than zero"
        assert 0 <= delta <= 0.5, f"parameter delta must be in range 0 <= delta <= 0.5"


        # initialize values 
        x_real = x.copy()
        p = x_real.shape[1]
        orig_colnames = list(x_real)
        num_trees = num_trees
        demographic_predictor = demographic_predictor
        demographic_cols = demographic_cols
        medical_code_cols = medical_code_cols

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
            
        # If no synthetic data provided, sample from marginals
        x_synth = x_real.apply(lambda x: x.sample(frac=1).values)
        
        # Merge real and synthetic data
        x = pd.concat([x_real, x_synth])
        y = np.concatenate([np.zeros(x_real.shape[0]), np.ones(x_real.shape[0])])
        # real observations = 0, synthetic observations = 1

        # pass on x_real
        x_real = x_real

        # Fit initial RF model
        clf_0 = RandomForestClassifier( oob_score= True, n_estimators=num_trees,min_samples_leaf=min_node_size, **kwargs) 
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
                nodeIDs = clf_0.apply(x_real) # dimension [terminalnode, tree]

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
                clf_1 = RandomForestClassifier( oob_score= True, n_estimators=num_trees, min_samples_leaf=min_node_size,**kwargs) 
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
        clf = clf_0
        acc = acc 
            
        # Pruning
        pred = clf.apply(x_real)
        for tree_num in range(0, num_trees):
            tree = clf.estimators_[tree_num]
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
        return clf


def forde2(clf, 
            x_real, 
            p, orig_colnames, 
            object_cols, 
            factor_cols, 
            levels, num_trees,
            dist="truncnorm", oob=False, alpha=0):
    



    # Get terminal nodes for all observations
    pred = clf.apply(x_real)
    
    # If OOB, use only OOB trees
    if oob:
      for tree in range(num_trees):
        idx_oob = np.isin(range(x_real.shape[0]), _generate_unsampled_indices(clf.estimators_[tree].random_state, x.shape[0], x.shape[0]))
        pred[np.invert(idx_oob), tree] = -1
    
    # Get probabilities of terminal nodes for each tree 
    # node_probs dims: [nodeid, tree]
    #node_probs = np.apply_along_axis(func1d= utils.bincount, axis = 0, arr =pred, nbins = np.max(pred))
    
    # compute leaf bounds and coverage
    bnds = pd.concat([bnd_fun(tree=j, p = p, forest = clf, feature_names = orig_colnames) for j in range(num_trees)])
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
        dt = x_real.loc[:, np.invert(factor_cols)].copy()
        dt["tree"] = tree
        dt["nodeid"] = pred[:,tree]
        # merge bounds and make it long format
        long = pd.merge(right = bnds[['tree', 'nodeid','variable', 'min', 'max', 'f_idx']], left = pd.melt(dt[dt["nodeid"] >= 0], id_vars = ["tree", "nodeid"]), on = ['tree', 'nodeid', 'variable'], how = 'left')
        # get distribution parameters
        if dist == "truncnorm":
          res = long.groupby([ 'tree',"nodeid", "variable"], as_index = False).agg(mean=("value", "mean"), sd=("value", "std"), min = ("min", "min"), max = ("max", "max"))
        else:
          raise ValueError('unknown distribution, make sure to enter a vaild value for dist')
          exit()
        params = pd.concat([params, res])
    
    # Get class probabilities in all terminal nodes
    class_probs = pd.DataFrame()
    if factor_cols.any():
      for tree in range(num_trees):
        dt = x_real.loc[:, factor_cols].copy()
        dt["tree"] = tree
        dt["nodeid"] = pred[:,tree]
        dt = pd.melt(dt[dt["nodeid"] >= 0], id_vars = ["tree", "nodeid"])
        long = pd.merge(left = dt, right = bnds, on = ['tree','nodeid', 'variable'])
        long['count_var'] = long.groupby(['tree', 'nodeid', 'variable'])['variable'].transform('count')
        long['count_var_val'] = long.groupby(['tree', 'nodeid', 'variable', 'value'])['variable'].transform('count')
        long.drop_duplicates(inplace=True)
        if alpha == 0:
          long['prob'] = long['count_var_val'] / long['count_var'] 
        else:
          # Define the range of each variable in each leaf
          long['k'] = long.groupby(['variable'])['value'].transform('nunique')  
          long.loc[long['min'] == float('-inf') , 'min'] = 0.5 - 1
          long.loc[long['max'] == float('inf') , 'max'] = long['k'] + 0.5 - 1
          long.loc[round(long['min'] % 1,2) != 0.5 , 'min'] = long['min'] - 0.5
          long.loc[round(long['max'] % 1,2) != 0.5 , 'min'] = long['max'] + 0.5
          long['k'] = long['max'] - long['min']  
          # Enumerate each possible leaf-variable-value combo
          tmp = long[['f_idx','tree', "nodeid", 'variable', 'min','max']].copy()
          tmp['rep_min'] = tmp['min'] + 0.5 
          tmp['rep_max'] = tmp['max'] - 0.5 
          tmp['levels'] = tmp.apply(lambda row:  list(range(int(row['rep_min']), int(row['rep_max'] + 1))), axis=1)
          tmp = tmp.explode('levels')
          cat_val = pd.DataFrame(levels).melt()
          cat_val['levels'] = cat_val['value'] 
          tmp =  pd.merge(left = tmp, right = cat_val, on = ['variable', 'levels'])[['variable', 'f_idx','tree', "nodeid",'value']]
          # populate count, k
          tmp = pd.merge(left = tmp, right = long[['f_idx', 'variable', 'tree', "nodeid",'count_var', 'k']], on = ['f_idx', "nodeid", 'variable', 'tree'])
          # Merge with long, set val_count = 0 for possible but unobserved levels
          long = pd.merge(left = tmp, right = long, on = ['f_idx','tree',"nodeid",  'variable','value','count_var','k'], how = 'left')
          long.loc[long['count_var_val'].isna(), 'count_var_val'] = 0
          long = long[['f_idx','tree',"nodeid",  'variable', 'value', 'count_var_val', 'count_var', 'k']].drop_duplicates()
          # Compute posterior probabilities
          long['prob'] = (long['count_var_val'] + alpha) / (long['count_var'] + alpha*long['k'])
          long['value'] = long['value'].astype('int8')
        
        long = long[['f_idx','tree', "nodeid", 'variable', 'value','prob']]
        class_probs = pd.concat([class_probs, long])
    return {"cnt": params, "cat": class_probs, 
            "forest": clf, "meta" : pd.DataFrame(data={"variable": orig_colnames, "family": dist})},bnds, num_trees, factor_cols, p,  levels, object_cols



def forge2(n, bnds,
           num_trees,
           factor_cols, 
           p,  levels,
           object_cols,
           res):
          
        params=res["cnt"] 
        class_probs= res["cat"] 
        clf=res["forest"]
        orig_colnames=res["meta"]["variable"]
        dist="truncnorm"
        """This part is for data generation (FORGE)

        :param n: Number of synthetic samples to generate.
        :type n: int
        :return: Returns generated data.
        :rtype: pandas.DataFrame
        """
  

        # Sample new observations and get their terminal nodes
        # Draw random leaves with probability proportional to coverage
        unique_bnds = bnds[['tree', 'nodeid', 'cvg']].drop_duplicates()
        draws = np.random.choice(a=range(unique_bnds.shape[0]), p = unique_bnds['cvg'] / num_trees, size=n)
        sampled_trees_nodes = unique_bnds[['tree','nodeid']].iloc[draws,].reset_index(drop =True).reset_index().rename(columns={'index': 'obs'})

        # Get distributions parameters for each new obs.
        if np.invert(factor_cols).any():
            obs_params = pd.merge(sampled_trees_nodes, params, on = ["tree", "nodeid"]).sort_values(by=['obs'], ignore_index = True)
        
        # Get probabilities for each new obs.
        if factor_cols.any():
            obs_probs = pd.merge(sampled_trees_nodes, class_probs, on = ["tree", "nodeid"]).sort_values(by=['obs'], ignore_index = True)
        
        # Sample new data from mixture distribution over trees
        data_new = pd.DataFrame(index=range(n), columns=range(p))
        for j in range(p): 
            colname = orig_colnames[j]
            
            if factor_cols[j]:
                # Factor columns: Multinomial distribution
                data_new.isetitem(j, obs_probs[obs_probs["variable"] == colname].groupby("obs").sample(weights = "prob")["value"].reset_index(drop = True))

            else:
                # Continuous columns: Match estimated distribution parameters with r...() function
                if dist == "truncnorm":
                    # sample from normal distribution, only here for debugging
                    # data_new.loc[:, j] = np.random.normal(obs_params.loc[obs_params["variable"] == colname, "mean"], obs_params.loc[obs_params["variable"] == colname, "sd"], size = n) 
                    
                    # sample from truncated normal distribution
                    # note: if sd == 0, truncnorm will return location parameter -> this is desired; if we have 
                    # all obs. in that leave having the same value, we sample a new obs. with exactly that value as well
                    myclip_a = obs_params.loc[obs_params["variable"] == colname, "min"]
                    myclip_b = obs_params.loc[obs_params["variable"] == colname, "max"]
                    myloc = obs_params.loc[obs_params["variable"] == colname, "mean"]
                    myscale = obs_params.loc[obs_params["variable"] == colname, "sd"]
                    data_new.isetitem(j, scipy.stats.truncnorm(a =(myclip_a - myloc) / myscale,b = (myclip_b - myloc) / myscale, loc = myloc , scale = myscale ).rvs(size = n))
                    del(myclip_a,myclip_b,myloc,myscale)
                else:
                    raise ValueError('Other distributions not yet implemented')
            
            # Use original column names
        data_new = data_new.set_axis(orig_colnames, axis = 1, copy = False)
        
        # Convert categories back to category   
        for col in orig_colnames:
            if factor_cols[col]:
                data_new[col] = pd.Categorical.from_codes(data_new[col], categories = levels[col])

        # Convert object columns back to object
        for col in orig_colnames:
            if object_cols[col]:
                data_new[col] = data_new[col].astype("object")

        # Return newly sampled data
        return data_new

    
    

   
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
    
    full_path = os.path.join(path_result, "synthetic_d")

    if os.path.exists(full_path):
        print(f"The path {full_path} exists.")
    else:
        print(f"The path {full_path} does not exist.")

    if create_demos_var_dataset:     
        x_train = data# Your training data
        x_train["year"] = x_train['ADMITTIME'].dt.year
        x_train['month'] = x_train['ADMITTIME'].dt.month
        x_train = x_train.drop(columns=columns_to_drop) 
        x_train, encoders =group_and_encode_demographics(    x_train, demographiccols)
        print(f'lista para excluir {len(list_col_exclute)}')   
        save_pickle(encoders,encoders_demos )
        save_pickle(x_train,path_dataset_demos_whole) 
        x_train= sample_patients_list(ruta_patients, x_train)    

        
        #print(f'lista para excluir {len(list_col_exclute)}')    
            # se quitan columnas que no se utilizan y se convierte en categoricas, la matrix de conteo, subject id, admission dat
        #x_train= x_train[[i for i in x_train.columns if i not in list_col_exclute['codes_no'].to_list()]]
        #columns_to_drop = ['LOSRD_sum', 'L_1s_last_p1','HADM_ID',"ADMITTIME",'GENDER_0']
        #save_pickle(x_train,path_dataset_demos) 
        print("Cols to eliminate",columns_to_drop)
        x_train = x_train.drop(columns=columns_to_drop_arf)  
        x_train =  change_dtypes(x_train,cols_continuous_d)   
        save_pickle(x_train,path_dataset_demos) 
        logging.info(x_train.shape)
        logging.info(x_train.head())
    else:
        x_train = load_pickle(path_dataset_demos)
        x_train =  change_dtypes(x_train,cols_continuous_d) 
        encoders = load_pickle(encoders_demos)    
        logging.info(x_train.shape)
        logging.info(x_train.head())

        
 
   
        

    if train_arf_demos:
        logging.info("training ARF started")
        #arf_model = arf2(x_train,demographic_predictor=predictor, demographic_cols=demographic_cols, medical_code_cols=medical_code_cols)
        arf_model = arf2( x_train, num_trees=30, delta=0, max_iters=1, early_stop=True, verbose=True, min_node_size=5, 
                demographic_predictor=None, demographic_cols=None, medical_code_cols=None)
        clf = arf_model
        joblib.dump(arf_model, path_result_arf+arf_demos)
        
        logging.info("training ARF finished")
       
    else: 
        logging.info("training ARF started")
        arf_model = joblib.load(path_result_arf+arf_demos) 
        logging.info("training ARF finished")
    clf = arf_model
    num_trees = num_trees
    p, orig_colnames, num_trees,object_cols, factor_cols,levels, x_real =get_input_forde(x_train)
    

    #arf_model.demographic_predictor = predictor
    # Estimate density with demographic prediction
    logging.info("density estimation fored")
    res,bnds, num_trees, factor_cols, p,  levels, object_cols = forde2(clf, 
        x_real, 
        p, orig_colnames, 
        object_cols, 
        factor_cols, 
        levels, num_trees,
        dist="truncnorm", oob=False, alpha=0)
    logging.info("training fored finished")
    if save_path_forde_demos:
            joblib.dump(res, path_result+name_fored_output)
    logging.info("forge ARF")
    n=len(x_train )   
    synthetic_data=forge2(n, bnds,
        num_trees,
        factor_cols, 
        p,  levels,
        object_cols,
        res)
    # Generate synthetic data
    save_pickle(synthetic_data, path_result+synthetic_data_demos)
    print(synthetic_data.head())


