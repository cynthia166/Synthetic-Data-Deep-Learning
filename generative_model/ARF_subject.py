import numpy as np
import gzip
import pickle
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from deepecho import PARModel
import torch

import logging
file_principal = os.getcwd()
os.chdir(file_principal )



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class SubjectBootstrapARF:
    def __init__(self, x_real, subject_id_col, num_trees=30, delta=0, max_iters=10, early_stop=True, verbose=True, min_node_size=40, random_state=None,oob = False, **kwargs):
        # ... (previous initialization code remains the same)
        self.par_models = {}
        self.par_kwargs = {
            'epochs': 128,
            'sample_size': 1,
            'cuda': torch.cuda.is_available(),
            'verbose': False
        }
        self.oob = oob
        self.bootstrap_samples = []
        self.x_real = x_real
        self.subject_id_col = subject_id_col
        self.num_trees = num_trees
        self.delta = delta
        self.max_iters = max_iters
        self.early_stop = early_stop
        self.verbose = verbose
        self.min_node_size = min_node_size
        self.random_state = check_random_state(random_state)
        self.kwargs = kwargs

        # Group data by subject ID
        self.subjects = x_real.groupby(subject_id_col)
        self.subject_ids = list(self.subjects.groups.keys())

        # Prepare metadata
        self.p = x_real.shape[1] - 1  # Exclude subject_id column
        self.orig_colnames = list(x_real.columns.drop(subject_id_col))

        # Identify data types
        self.object_cols = x_real.dtypes == "object"
        self.factor_cols = x_real.dtypes == "category"
        
        # Save factor levels
        self.levels = {}
        for col in self.orig_colnames:
            if self.factor_cols[col]:
                self.levels[col] = x_real[col].cat.categories

        self.visit_distribution = self.x_real.groupby('SUBJECT_ID')['visit_rank'].count().value_counts(normalize=True)
        logging.info("setting variables")

    # ... (previous methods remain the same)
    def _preprocess_subject(self, subject):
        subject = subject.copy()
        for col in self.orig_colnames:
            if self.factor_cols[col]:
                subject[col] = subject[col].cat.codes
        return subject

    def _subject_bootstrap(self, n_subjects):
        sampled_ids = self.random_state.choice(self.subject_ids, size=n_subjects, replace=False)
        sampled_subjects = [self._preprocess_subject(self.subjects.get_group(id_)) for id_ in sampled_ids]
        return pd.concat(sampled_subjects, keys=sampled_ids, names=[self.subject_id_col])



    def fit(self):
        x_real = self.x_real.copy()
        x_synth = self._subject_bootstrap(len(self.subject_ids))

        X = pd.concat([x_real, x_synth]).drop(columns=[self.subject_id_col])
        y = np.concatenate([np.zeros(len(x_real)), np.ones(len(x_synth))])

        # Fit initial RF model
        clf_0 = RandomForestClassifier(oob_score=True,n_estimators=self.num_trees, **self.kwargs)
        clf_0.fit(X,y)

        iters = 0
        acc_0 = clf_0.oob_score_
        acc = [acc_0]

        if self.verbose:
            print(f'Initial accuracy is {acc_0}')

        while acc[-1] > 0.5 + self.delta and iters < self.max_iters:
            # Get node IDs for real subjects
            nodeIDs = self._get_subject_node_ids(clf_0, x_real)

            # Sample synthetic subjects from leaves
            x_synth = self._sample_from_leaves(nodeIDs, x_real)
            logging.info("Sample synthetic subjects from leavess")
            # Merge real and synthetic data
            X = pd.concat([x_real, x_synth]).drop(columns=[self.subject_id_col])
            y = np.concatenate([np.zeros(len(x_real)), np.ones(len(x_synth))])

            # Train new classifier
            clf_1 = RandomForestClassifier(oob_score=True,n_estimators=self.num_trees, **self.kwargs)
        
            clf_1.fit(X, y)


            acc_1 = clf_1.oob_score_
            acc.append(acc_1)
            iters += 1

            if self.verbose:
                print(f"Iteration number {iters} reached accuracy of {acc_1}.")

            if self.early_stop and acc[iters] > acc[iters - 1]:
                break

            clf_0 = clf_1

        self.clf = clf_0
        self.acc = acc
        # Pruning
        self.clf = self._prune_forest(self.clf,self.x_real,self.min_node_size)
        return self.clf


    def _prune_forest(self,clf, x_real, min_node_size):
        """
        Prune the random forest by removing or merging small leaf nodes.
        
        :param clf: The trained RandomForestClassifier
        :param x_real: The real data used for training
        :param min_node_size: Minimum number of samples required in a leaf node
        :return: The pruned RandomForestClassifier
        """
        # Get terminal nodes for all observations
        pred = clf.apply(x_real.drop(columns=[self.subject_id_col]))
        
        for tree_num in range(len(clf.estimators_)):
            tree = clf.estimators_[tree_num]
            left = tree.tree_.children_left
            right = tree.tree_.children_right
            leaves = np.where(left < 0)[0]

            # Get leaves that are too small
            unique, counts = np.unique(pred[:, tree_num], return_counts=True)
            to_prune = unique[counts < min_node_size]

            # Also add leaves with 0 obs.
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


    def _get_subject_node_ids(self, clf, x_real):
        nodeIDs = np.zeros((len(self.subject_ids), self.num_trees))
        for i, (subject_id, subject_data) in enumerate(self.subjects):
            subject_nodeIDs = np.array([tree.apply(subject_data.drop(columns=[self.subject_id_col])) for tree in clf.estimators_])
            nodeIDs[i] = subject_nodeIDs.mean(axis=1)  # Take the mean node ID for each subject
        return nodeIDs

    def _sample_from_leaves(self, nodeIDs, x_real):
        # Sample entire subjects from leaves
        leaf_subjects = {}
        for i, subject_id in enumerate(self.subject_ids):
            for tree in range(self.num_trees):
                leaf = int(nodeIDs[i, tree])  # Convert to int for indexing
                if (tree, leaf) not in leaf_subjects:
                    leaf_subjects[(tree, leaf)] = []
                leaf_subjects[(tree, leaf)].append(subject_id)

        sampled_subjects = []
        for (tree, leaf), subject_ids in leaf_subjects.items():
            n_samples = len(subject_ids)
            sampled_ids = self.random_state.choice(subject_ids, size=n_samples, replace=False)
            for sampled_id in sampled_ids:
                sampled_subject = self._preprocess_subject(self.subjects.get_group(sampled_id))
                sampled_subjects.append(sampled_subject)

        return pd.concat(sampled_subjects, keys=range(len(sampled_subjects)), names=[self.subject_id_col])


 
   

    def forde(self):
        # Get terminal nodes for all observations
        pred = self.clf.apply(self.x_real.drop(columns=[self.subject_id_col]))
        
        # If OOB, use only OOB trees
        if self.oob:
            for tree in range(self.num_trees):
                idx_oob = np.isin(range(self.x_real.shape[0]), self._generate_unsampled_indices(self.clf.estimators_[tree].random_state, self.x.shape[0], self.x.shape[0]))
                pred[np.invert(idx_oob), tree] = -1
        
        # Compute leaf bounds and coverage
        bnds = pd.concat([self._bnd_fun(tree=j, p=self.p, forest=self.clf, feature_names=self.orig_colnames) for j in range(self.num_trees)])
        bnds['f_idx'] = bnds.groupby(['tree', 'leaf']).ngroup()
        bnds_2 = pd.DataFrame()
        for t in range(self.num_trees):
            unique, freq = np.unique(pred[:,t], return_counts=True)
            vv = pd.concat([pd.Series(unique, name='leaf'), pd.Series(freq/pred.shape[0], name='cvg')], axis=1)
            zz = bnds[bnds['tree'] == t]
            bnds_2 = pd.concat([bnds_2, pd.merge(left=vv, right=zz, on=['leaf'])])
        bnds = bnds_2
        del(bnds_2)
        
        # Set coverage for nodes with single observations to zero
        if np.invert(self.factor_cols).any():
            bnds.loc[bnds['cvg'] == 1/pred.shape[0], 'cvg'] = 0
        
        # No parameters to learn for zero coverage leaves - drop zero coverage nodes
        bnds = bnds[bnds['cvg'] > 0]
        # Rename leafs to nodeids
        bnds.rename(columns={'leaf': 'nodeid'}, inplace=True)
        # Save bounds to later use coverage for drawing new samples
        self.bnds = bnds
        
        # Fit continuous distribution in all terminal nodes
        self.params = pd.DataFrame()
        if np.invert(self.factor_cols).any():
            for tree in range(self.num_trees):
                dt = self.x_real.loc[:, np.invert(self.factor_cols)].copy()
                dt["tree"] = tree
                dt["nodeid"] = pred[:,tree]
                # Merge bounds and make it long format
                long = pd.merge(right=bnds[['tree', 'nodeid','variable', 'min', 'max', 'f_idx']], 
                                left=pd.melt(dt[dt["nodeid"] >= 0], id_vars=["tree", "nodeid"]), 
                                on=['tree', 'nodeid', 'variable'], how='left')
                # Get distribution parameters
                       
        # Get class probabilities in all terminal nodes
        self.class_probs = pd.DataFrame()
        if self.factor_cols.any():
            for tree in range(self.num_trees):
                dt = self.x_real.loc[:, self.factor_cols].copy()
                dt["tree"] = tree
                dt["nodeid"] = pred[:,tree]
                dt = pd.melt(dt[dt["nodeid"] >= 0], id_vars=["tree", "nodeid"])
                long = pd.merge(left=dt, right=bnds, on=['tree', 'nodeid', 'variable'])
                long['count_var'] = long.groupby(['tree', 'nodeid', 'variable'])['variable'].transform('count')
                long['count_var_val'] = long.groupby(['tree', 'nodeid', 'variable', 'value'])['variable'].transform('count')
                long.drop_duplicates(inplace=True)
                self.class_probs = pd.concat([self.class_probs, long])
        
        # Train PARModels for each tree and leaf
        self.par_models = {}
        count_nodes_data = 0
        for tree in range(self.num_trees):
            logging.info('Training tree number {tree}')
            # Merge real and synthetic data

            self.par_models[tree] = {}
            for nodeid in bnds[bnds['tree'] == tree]['nodeid'].unique():
                logging.info('Training node  {nodeid}')
                leaf_data = self.x_real[pred[:, tree] == nodeid]
                if len(leaf_data) >= self.min_node_size:
                    self.par_models[tree][nodeid] = self._train_leaf_par(leaf_data)
                    count_nodes_data +=count_nodes_data
                    logging.info("numero de nodos con datos {count_nodes_data}")
        logging.info("total de nodos con informacion entre nodos totales {count_nodes_data}")             
        return {"par_models": self.par_models, "forest": self.clf}

    def _train_leaf_par(self, leaf_data,context_columns):
        
        model = PARModel(**self.par_kwargs)
        model.fit_sequences(sequences, context_types=[], data_types=data_types)
        model.fit(
                    data=leaf_data,
                    entity_columns=["SUBJECT_ID"],
                    context_columns=context_columns,
                    data_types=data_types,
                    sequence_index="visit_rank"
                )
        return model

    def _generate_unsampled_indices(self, random_state, n_samples, n_samples_bootstrap):
        random_instance = np.random.RandomState(random_state)
        sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)
        unsampled_mask = ~np.isin(range(n_samples), sample_indices)
        return np.where(unsampled_mask)[0]

    def _bnd_fun(self,tree, p, forest, feature_names ):
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
    
    def forge(self, n_subjects, max_visits=42):
            """Generate new subject-level data using PARModel"""
            generated_subjects = []
            
            # Sample the number of visits for each synthetic subject
            sampled_visits = np.random.choice(
                self.visit_distribution.index,
                size=n_subjects,
                p=self.visit_distribution.values
            )
            
            for num_visits in sampled_visits:
                tree_idx = np.random.randint(0, self.num_trees)
                leaf = np.random.choice(list(self.par_models[tree_idx].keys()))
                
                model = self.par_models[tree_idx][leaf]
                sequence = model.sample_sequence(context=[], sequence_length=num_visits)
                
                sequence_df = pd.DataFrame(dict(zip(self.orig_colnames, sequence)))
                sequence_df['SUBJECT_ID'] = len(generated_subjects)
                sequence_df['visit_rank'] = range(len(sequence[0]))
                
                generated_subjects.append(sequence_df)
            
            return pd.concat(generated_subjects, ignore_index=True)

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
# Usage

def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)
    
# Usage
train_data_features = load_data("data/intermedi/SD/inpput/entire_ceros_tabular_data.pkl")[:10000]
columns_to_drop = ['GENDER_0','days_between_visits','ADMITTIME']
#columns_to_drop = ['LOSRD_sum', 'L_1s_last_p1','HADM_ID',"ADMITTIME",'GENDER_0']
print("Cols to eliminate",columns_to_drop)
train_data_features = train_data_features.drop(columns=columns_to_drop)  

subject_data = train_data_features  # Your subject data
subject_arf = SubjectBootstrapARF(subject_data, subject_id_col='SUBJECT_ID')

subject_arf.fit()
logging.info("Finish the training")
subject_arf.forde()
logging.info("Calculating parameter for sampling")
synthetic_data = subject_arf.forge(n_subjects=3, max_visits=42)