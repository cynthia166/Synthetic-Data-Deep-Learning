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