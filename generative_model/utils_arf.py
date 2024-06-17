import gzip 
import pickle
import numpy as np
import pandas as pd
import shap

def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)
    
    
def save_load_numpy(sample_patients,save=False,load=False,name='sample_patients.npy'):
    # Save the numpy array to disk
    if save:
        np.save(name, sample_patients)
        return
    if load: 
    # Load the numpy array from disk
       return np.load(name)    
   
   
def sample_patients(res,percent_patient):
    unique_patients = res['SUBJECT_ID'].unique()

    # Calcular el 20% del total de pacientes únicos
    sample_size = int(percent_patient * len(unique_patients))
    # Obtener una muestra aleatoria del 20% de los pacientes únicos
    sample_patients = np.random.choice(unique_patients, size=sample_size, replace=False)
    # Filtrar el DataFrame para incluir solo los registros de los pacientes en la muestra
    sample_df = res[res['SUBJECT_ID'].isin(sample_patients)]
    
    return sample_df, sample_patients   


def filter_keywords(train_ehr_dataset,keywords):
    for i in keywords:
        col_prod = [col for col in train_ehr_dataset.columns if any(palabra in col for palabra in [i])]
    return col_prod
    
def convertir_categoricas(df,categorical_cols):
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    return df                   


def shap_values(clf, X_test_v):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_v)

    # Plot the SHAP summary plot
    #shap.summary_plot(shap_values, X_test_v, plot_type="bar")

    # If your classifier is binary, shap_values will be a list with two elements.
    # Use the first element if you're interested in the positive class.

    # Calculate mean absolute SHAP values
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)

    # Create a DataFrame with feature names and their SHAP values
    shap_df = pd.DataFrame({
        'Feature': X_test_v.columns,
        'SHAP Value': mean_abs_shap_values
    })

    # Sort the DataFrame by SHAP Value in descending order
    shap_df = shap_df.sort_values(by='SHAP Value', ascending=False).reset_index(drop=True)
    return shap_df



#fun
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

def identify_zero_coverage_variables(clf, x_real, orig_colnames, factor_cols, num_trees):
    pred = clf.apply(x_real)

    bnds = pd.concat([bnd_fun(tree=j, p=x_real.shape[1], forest=clf, feature_names=orig_colnames) for j in range(num_trees)])
    bnds['f_idx'] = bnds.groupby(['tree', 'leaf']).ngroup()

    bnds_2 = pd.DataFrame()
    for t in range(num_trees):
        unique, freq = np.unique(pred[:, t], return_counts=True)
        vv = pd.concat([pd.Series(unique, name='leaf'), pd.Series(freq/pred.shape[0], name='cvg')], axis=1)
        zz = bnds[bnds['tree'] == t]
        bnds_2 = pd.concat([bnds_2, pd.merge(left=vv, right=zz, on=['leaf'])])
    bnds = bnds_2
    del(bnds_2)

    zero_coverage_nodes = bnds[bnds['cvg'] == 0]
    zero_coverage_vars = zero_coverage_nodes['variable'].unique().tolist()

    return zero_coverage_vars

# E
#obtener coversge_9 del modelo






def save_pkl(data,name):
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(data, f)
        
        
import pickle

def load_pkl(name):
    with open(name+'.pkl', 'rb') as f:
        data = pickle.load(f)
    return data        


def train_arf2(x, num_trees=30, delta=0, max_iters=10, early_stop=True, verbose=True, min_node_size=5, **kwargs):
    """
    Implements Adversarial Random Forests (ARF) in python.

    :param x: Input data.
    :type x: pd.DataFrame
    :param num_trees: Number of trees to grow in each forest, defaults to 30
    :type num_trees: int, optional
    :param delta: Tolerance parameter. Algorithm converges when OOB accuracy is < 0.5 + `delta`, defaults to 0
    :type delta: float, optional
    :param max_iters: Maximum iterations for the adversarial loop, defaults to 10
    :type max_iters: int, optional
    :param early_stop: Terminate loop if performance fails to improve from one round to the next?, defaults to True
    :type early_stop: bool, optional
    :param verbose: Print discriminator accuracy after each round?, defaults to True
    :type verbose: bool, optional
    :param min_node_size: Minimum number of samples in terminal node, defaults to 5 
    :type min_node_size: int
    :return: Returns necessary objects for the `forde` function.
    :rtype: dict
    """

    # assertions
    assert isinstance(x, pd.DataFrame), f"expected pandas DataFrame as input, got:{type(x)}"
    assert len(set(list(x))) == x.shape[1], f"every column must have a unique column name"
    assert max_iters >= 0, f"negative number of iterations is not allowed: parameter max_iters must be >= 0"
    assert min_node_size > 0, f"minimum number of samples in terminal nodes (parameter min_node_size) must be greater than zero"
    assert num_trees > 0, f"number of trees in the random forest (parameter num_trees) must be greater than zero"
    assert 0 <= delta <= 0.5, f"parameter delta must be in range 0 <= delta <= 0.5"

    # initialize values 
    x_real = x.copy()
    orig_colnames = list(x_real)
    p = x_real.shape[1]

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

    # Fit initial RF model
    clf_0 = RandomForestClassifier(oob_score=True, n_estimators=num_trees, min_samples_leaf=min_node_size, **kwargs)
    clf_0.fit(x, y)

    iters = 0
    acc_0 = clf_0.oob_score_ # is accuracy directly
    acc = [acc_0]

    if verbose:
        print(f'Initial accuracy is {acc_0}')

    if (acc_0 > 0.5 + delta and iters < max_iters):
        converged = False
        while not converged:  # Start adversarial loop
            # get nodeIDs
            nodeIDs = clf_0.apply(x_real)  # dimension [terminalnode, tree]

            # add observation ID to x_real
            x_real_obs = x_real.copy()
            x_real_obs['obs'] = range(0, x_real.shape[0])

            # add observation ID to nodeIDs
            nodeIDs_pd = pd.DataFrame(nodeIDs)
            tmp = nodeIDs_pd.copy()
            tmp['obs'] = range(0, x_real.shape[0])
            tmp = tmp.melt(id_vars=['obs'], value_name="leaf", var_name="tree")

            # match real data to trees and leafs (node id for tree)
            x_real_obs = pd.merge(left=x_real_obs, right=tmp, on=['obs'], sort=False)
            x_real_obs.drop('obs', axis=1, inplace=True)

            # sample leafs
            tmp.drop("obs", axis=1, inplace=True)
            tmp = tmp.sample(x_real.shape[0], axis=0, replace=True)
            tmp = pd.Series(tmp.value_counts(sort=False), name='cnt').reset_index()
            draw_from = pd.merge(left=tmp, right=x_real_obs, on=['tree', 'leaf'], sort=False)

            # sample synthetic data from leaf
            grpd = draw_from.groupby(['tree', 'leaf'])
            x_synth = [grpd.get_group(ind).apply(lambda x: x.sample(n=grpd.get_group(ind)['cnt'].iloc[0], replace=True).values) for ind in grpd.indices]
            x_synth = pd.concat(x_synth).drop(['cnt', 'tree', 'leaf'], axis=1)
            
            # delete unnecessary objects 
            del(nodeIDs, nodeIDs_pd, tmp, x_real_obs, draw_from)

            # merge real and synthetic data
            x = pd.concat([x_real, x_synth])
            y = np.concatenate([np.zeros(x_real.shape[0]), np.ones(x_real.shape[0])])
            
            # discriminator
            clf_1 = RandomForestClassifier(oob_score=True, n_estimators=num_trees, min_samples_leaf=min_node_size, **kwargs)
            clf_1.fit(x, y)

            # update iters and check for convergence
            acc_1 = clf_1.oob_score_
            acc.append(acc_1)
            iters += 1
            plateau = True if early_stop and acc[iters] > acc[iters - 1] else False
            if verbose:
                print(f"Iteration number {iters} reached accuracy of {acc_1}.")
            if (acc_1 <= 0.5 + delta or iters >= max_iters or plateau):
                converged = True
            else:
                clf_0 = clf_1

    clf = clf_0
    acc = acc 
        
    # Pruning
    pred = clf.apply(x_real)
    for tree_num in range(num_trees):
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

    return {
        "x_real": x_real,
        "clf": clf,
        "num_trees": num_trees,
        "orig_colnames": orig_colnames,
        "factor_cols": factor_cols,
        "object_cols": object_cols,
        "levels": levels
    }

def forde2(x_real, clf, num_trees, orig_colnames, factor_cols, object_cols, levels, dist="truncnorm", oob=False, alpha=0):
    """
    This part is for density estimation (FORDE)

    :param x_real: Real data
    :type x_real: pd.DataFrame
    :param clf: Trained random forest classifier
    :type clf: RandomForestClassifier
    :param num_trees: Number of trees in the forest
    :type num_trees: int
    :param orig_colnames: Original column names
    :type orig_colnames: list
    :param factor_cols: Boolean array indicating which columns are categorical
    :type factor_cols: pd.Series
    :param object_cols: Boolean array indicating which columns are objects
    :type object_cols: pd.Series
    :param levels: Dictionary with category levels for categorical columns
    :type levels: dict
    :param dist: Distribution to use for continuous features, defaults to "truncnorm"
    :type dist: str, optional
    :param oob: Only use out-of-bag samples for parameter estimation?, defaults to False
    :type oob: bool, optional
    :param alpha: Optional pseudocount for Laplace smoothing of categorical features, defaults to 0
    :type alpha: float, optional
    :return: Parameters for the estimated density
    :rtype: dict
    """

    # Get terminal nodes for all observations
    pred = clf.apply(x_real)
    
    # If OOB, use only OOB trees
    if oob:
        for tree in range(num_trees):
            idx_oob = np.isin(range(x_real.shape[0]), _generate_unsampled_indices(clf.estimators_[tree].random_state, x_real.shape[0], x_real.shape[0]))
            pred[np.invert(idx_oob), tree] = -1
    
    # Compute leaf bounds and coverage
    bnds = pd.concat([bnd_fun2(tree=j, p=x_real.shape[1], forest=clf, feature_names=orig_colnames) for j in range(num_trees)])
    bnds['f_idx'] = bnds.groupby(['tree', 'leaf']).ngroup()

    bnds_2 = pd.DataFrame()
    for t in range(num_trees):
        unique, freq = np.unique(pred[:, t], return_counts=True)
        vv = pd.concat([pd.Series(unique, name='leaf'), pd.Series(freq / pred.shape[0], name='cvg')], axis=1)
        zz = bnds[bnds['tree'] == t]
        bnds_2 = pd.concat([bnds_2, pd.merge(left=vv, right=zz, on=['leaf'])])
    bnds = bnds_2
    del(bnds_2)

    # Set coverage for nodes with single observations to zero
    if np.invert(factor_cols).any():
        bnds.loc[bnds['cvg'] == 1 / pred.shape[0], 'cvg'] = 0
    
    # No parameters to learn for zero coverage leaves - drop zero coverage nodes
    bnds = bnds[bnds['cvg'] > 0]

    # Rename leaves to nodeids
    bnds.rename(columns={'leaf': 'nodeid'}, inplace=True)
    #esto se agrego
    # Save bounds to later use coverage for drawing new samples
    global_min = x_real.loc[:, np.invert(factor_cols)].min()
    global_max = x_real.loc[:, np.invert(factor_cols)].max()

    # Fit continuous distribution in all terminal nodes
    params = pd.DataFrame()
    if np.invert(factor_cols).any():
        for tree in range(num_trees):
            dt = x_real.loc[:, np.invert(factor_cols)].copy()
            dt["tree"] = tree
            dt["nodeid"] = pred[:, tree]
            # Merge bounds and make it long format
            long = pd.merge(right=bnds[['tree', 'nodeid', 'variable', 'min', 'max', 'f_idx']],
                            left=pd.melt(dt[dt["nodeid"] >= 0], id_vars=["tree", "nodeid"]),
                            on=['tree', 'nodeid', 'variable'], how='left')
            # Get distribution parameters
            if dist == "truncnorm":
                res = long.groupby(['tree', "nodeid", "variable"], as_index=False).agg(
                    mean=("value", "mean"), sd=("value", "std"),
                    min=("variable", lambda x: global_min.loc[x.iloc[0]]),
                    max=("variable", lambda x: global_max.loc[x.iloc[0]]))
            else:
                raise ValueError('Unknown distribution, make sure to enter a valid value for dist')
            params = pd.concat([params, res])
    
    # Get class probabilities in all terminal nodes
    class_probs = pd.DataFrame()
    if factor_cols.any():
        for tree in range(num_trees):
            dt = x_real.loc[:, factor_cols].copy()
            dt["tree"] = tree
            dt["nodeid"] = pred[:, tree]
            dt = pd.melt(dt[dt["nodeid"] >= 0], id_vars=["tree", "nodeid"])
            long = pd.merge(left=dt, right=bnds, on=['tree', 'nodeid', 'variable'])
            long['count_var'] = long.groupby(['tree', 'nodeid', 'variable'])['variable'].transform('count')
            long['count_var_val'] = long.groupby(['tree', 'nodeid', 'variable', 'value'])['variable'].transform('count')
            long.drop_duplicates(inplace=True)
            if alpha == 0:
                long['prob'] = long['count_var_val'] / long['count_var']
            else:
                # Define the range of each variable in each leaf
                long['k'] = long.groupby(['variable'])['value'].transform('nunique')
                long.loc[long['min'] == float('-inf'), 'min'] = 0.5 - 1
                long.loc[long['max'] == float('inf'), 'max'] = long['k'] + 0.5 - 1
                long.loc[round(long['min'] % 1, 2) != 0.5, 'min'] = long['min'] - 0.5
                long.loc[round(long['max'] % 1, 2) != 0.5, 'min'] = long['max'] + 0.5
                long['k'] = long['max'] - long['min']
                # Enumerate each possible leaf-variable-value combo
                tmp = long[['f_idx', 'tree', "nodeid", 'variable', 'min', 'max']].copy()
                tmp['rep_min'] = tmp['min'] + 0.5
                tmp['rep_max'] = tmp['max'] - 0.5
                tmp['levels'] = tmp.apply(lambda row: list(range(int(row['rep_min']), int(row['rep_max'] + 1))), axis=1)
                tmp = tmp.explode('levels')
                cat_val = pd.DataFrame(levels).melt()
                cat_val['levels'] = cat_val['value']
                tmp = pd.merge(left=tmp, right=cat_val, on=['variable', 'levels'])[['variable', 'f_idx', 'tree', "nodeid", 'value']]
                # Populate count, k
                tmp = pd.merge(left=tmp, right=long[['f_idx', 'variable', 'tree', "nodeid", 'count_var', 'k']], on=['f_idx', "nodeid", 'variable', 'tree'])
                # Merge with long, set val_count = 0 for possible but unobserved levels
                long = pd.merge(left=tmp, right=long, on=['f_idx', 'tree', "nodeid", 'variable', 'value', 'count_var', 'k'], how='left')
                long.loc[long['count_var_val'].isna(), 'count_var_val'] = 0
                long = long[['f_idx', 'tree', "nodeid", 'variable', 'value', 'count_var_val', 'count_var', 'k']].drop_duplicates()
                # Compute posterior probabilities
                long['prob'] = (long['count_var_val'] + alpha) / (long['count_var'] + alpha * long['k'])
                long['value'] = long['value'].astype('int8')

            long = long[['f_idx', 'tree', "nodeid", 'variable', 'value', 'prob']]
            class_probs = pd.concat([class_probs, long])
    
    return {
        "bnds": bnds,
        "params": params,
        "class_probs": class_probs,
        "factor_cols": factor_cols,
        "orig_colnames": orig_colnames,
        "levels": levels,
        "object_cols": object_cols
    }


def bnd_fun2(tree, p, forest, feature_names ):
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

def forge2(n, bnds, params, class_probs, factor_cols, orig_colnames, levels, object_cols, dist="truncnorm"):
    """
    This part is for data generation (FORGE).

    :param n: Number of synthetic samples to generate.
    :type n: int
    :param bnds: DataFrame with bounds and coverage for nodes.
    :type bnds: pd.DataFrame
    :param params: DataFrame with distribution parameters for continuous features.
    :type params: pd.DataFrame
    :param class_probs: DataFrame with class probabilities for categorical features.
    :type class_probs: pd.DataFrame
    :param factor_cols: Boolean array indicating which columns are categorical.
    :type factor_cols: pd.Series
    :param orig_colnames: List of original column names.
    :type orig_colnames: list
    :param levels: Dictionary with category levels for categorical columns.
    :type levels: dict
    :param object_cols: Boolean array indicating which columns are objects.
    :type object_cols: pd.Series
    :param dist: Distribution to use for continuous features, defaults to "truncnorm".
    :type dist: str, optional
    :return: Returns generated data.
    :rtype: pd.DataFrame
    """
    # orig_colnames = orig_colnames[0]
    # factor_cols = factor_cols[0]
    # class_probs = class_probs[0]
    # bnds = bnds[0]
    # params = params[0]
    
    # # Sample new observations and get their terminal nodes
    # Draw random leaves with probability proportional to coverage
    unique_bnds = bnds[['tree', 'nodeid', 'cvg']].drop_duplicates()
    draws = np.random.choice(a=range(unique_bnds.shape[0]), p=unique_bnds['cvg'] / unique_bnds['cvg'].sum(), size=n)
    sampled_trees_nodes = unique_bnds[['tree', 'nodeid']].iloc[draws].reset_index(drop=True).reset_index().rename(columns={'index': 'obs'})

    # Get distributions parameters for each new observation
    if np.invert(factor_cols).any():
        obs_params = pd.merge(sampled_trees_nodes, params, on=["tree", "nodeid"]).sort_values(by=['obs'], ignore_index=True)

    # Get probabilities for each new observation
    if factor_cols.any():
        obs_probs = pd.merge(sampled_trees_nodes, class_probs, on=["tree", "nodeid"]).sort_values(by=['obs'], ignore_index=True)

    # Sample new data from mixture distribution over trees
    data_new = pd.DataFrame(index=range(n), columns=range(len(orig_colnames)))
    for j in range(len(orig_colnames)):
        colname = orig_colnames[j]
        if factor_cols[j]:
            # Factor columns: Multinomial distribution
            data_new.iloc[:, j] = obs_probs[obs_probs["variable"] == colname].groupby("obs").sample(weights="prob")["value"].reset_index(drop=True)
        else:
            # Continuous columns: Match estimated distribution parameters with r...() function
            if dist == "truncnorm":
                myclip_a = obs_params.loc[obs_params["variable"] == colname, "min"]
                myclip_b = obs_params.loc[obs_params["variable"] == colname, "max"]
                myloc = obs_params.loc[obs_params["variable"] == colname, "mean"]
                myscale = obs_params.loc[obs_params["variable"] == colname, "sd"]
                # Asegurar que los rangos de truncamiento sean válidos
                valid_mask = (myclip_a < myclip_b)
                if not valid_mask.all():
                    print(f"Invalid truncation range for variable {colname}. Adjusting invalid ranges...")
                    myclip_a = myclip_a.where(valid_mask, other=myloc - 2 * myscale)
                    myclip_b = myclip_b.where(valid_mask, other=myloc + 2 * myscale)
                if not myscale.all():
                    print(f"Scale parameter is zero for variable {colname}. Adjusting scale...")
                    myscale = 1e-6  # Un pequeño valor positivo      
                data_new.iloc[:, j] = scipy.stats.truncnorm(a=(myclip_a - myloc) / myscale,
                                                            b=(myclip_b - myloc) / myscale,
                                                            loc=myloc, scale=myscale).rvs(size=n)
            else:
                raise ValueError('Other distributions not yet implemented')
    #convertilo back en category
    # # Use original column names
    # data_new.columns = orig_colnames

    # # Convert categories back to category
    # for col in orig_colnames:
    #     if factor_cols[col]:
    #         data_new[col] = pd.Categorical.from_codes(data_new[col], categories=dict(levels)[col])

    # # Convert object columns back to object
    # for col in orig_colnames:
    #     if object_cols[col]:
    #         data_new[col] = data_new[col].astype("object")

    # # Return newly sampled data
    return data_new
