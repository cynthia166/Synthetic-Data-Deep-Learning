import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices
import scipy
from arfpy import utils

class arf:
    def __init__(self, x, num_trees=30, delta=0, max_iters=10, early_stop=True, verbose=True, min_node_size=5, **kwargs):
        assert isinstance(x, pd.core.frame.DataFrame), f"expected pandas DataFrame as input, got:{type(x)}"
        assert len(set(list(x))) == x.shape[1], f"every column must have a unique column name"
        assert max_iters >= 0, f"negative number of iterations is not allowed: parameter max_iters must be >= 0"
        assert min_node_size > 0, f"minimum number of samples in terminal nodes (parameter min_node_size) must be greater than zero"
        assert num_trees > 0, f"number of trees in the random forest (parameter num_trees) must be greater than zero"
        assert 0 <= delta <= 0.5, f"parameter delta must be in range 0 <= delta <= 0.5"

        x_real = x.copy()
        self.p = x_real.shape[1]
        self.orig_colnames = list(x_real)
        self.num_trees = num_trees

        self.object_cols = x_real.dtypes == "object"
        for col in list(x_real):
            if self.object_cols[col]:
                x_real[col] = x_real[col].astype('category')

        self.factor_cols = x_real.dtypes == "category"
        self.levels = {}
        for col in list(x_real):
            if self.factor_cols[col]:
                self.levels[col] = x_real[col].cat.categories

        for col in list(x_real):
            if self.factor_cols[col]:
                x_real[col] = x_real[col].cat.codes

        x_synth = x_real.apply(lambda x: x.sample(frac=1).values)
        x = pd.concat([x_real, x_synth])
        y = np.concatenate([np.zeros(x_real.shape[0]), np.ones(x_real.shape[0])])

        self.x_real = x_real

        clf_0 = RandomForestClassifier(oob_score=True, n_estimators=self.num_trees, min_samples_leaf=min_node_size, **kwargs)
        clf_0.fit(x, y)

        iters = 0
        acc_0 = clf_0.oob_score_
        acc = [acc_0]

        if verbose:
            print(f'Initial accuracy is {acc_0}')

        if acc_0 > 0.5 + delta and iters < max_iters:
            converged = False
            while not converged:
                nodeIDs = clf_0.apply(self.x_real)
                x_real_obs = x_real.copy()
                x_real_obs['obs'] = range(0, x_real.shape[0])
                nodeIDs_pd = pd.DataFrame(nodeIDs)
                tmp = nodeIDs_pd.copy()
                tmp['obs'] = range(0, x_real.shape[0])
                tmp = tmp.melt(id_vars=['obs'], value_name="leaf", var_name="tree")
                x_real_obs = pd.merge(left=x_real_obs, right=tmp, on=['obs'], sort=False)
                x_real_obs.drop('obs', axis=1, inplace=True)
                tmp.drop("obs", axis=1, inplace=True)
                tmp = tmp.sample(x_real.shape[0], axis=0, replace=True)
                tmp = pd.Series(tmp.value_counts(sort=False), name='cnt').reset_index()
                draw_from = pd.merge(left=tmp, right=x_real_obs, on=['tree', 'leaf'], sort=False)
                grpd = draw_from.groupby(['tree', 'leaf'])
                x_synth = [grpd.get_group(ind).apply(lambda x: x.sample(n=grpd.get_group(ind)['cnt'].iloc[0], replace=True).values) for ind in grpd.indices]
                x_synth = pd.concat(x_synth).drop(['cnt', 'tree', 'leaf'], axis=1)
                del nodeIDs, nodeIDs_pd, tmp, x_real_obs, draw_from
                x = pd.concat([x_real, x_synth])
                y = np.concatenate([np.zeros(x_real.shape[0]), np.ones(x_real.shape[0])])
                clf_1 = RandomForestClassifier(oob_score=True, n_estimators=self.num_trees, min_samples_leaf=min_node_size, **kwargs)
                clf_1.fit(x, y)
                acc_1 = clf_1.oob_score_
                acc.append(acc_1)
                iters += 1
                plateau = early_stop and acc[iters] > acc[iters - 1]
                if verbose:
                    print(f"Iteration number {iters} reached accuracy of {acc_1}.")
                if acc_1 <= 0.5 + delta or iters >= max_iters or plateau:
                    converged = True
                else:
                    clf_0 = clf_1

        self.clf = clf_0
        self.acc = acc

        pred = self.clf.apply(self.x_real)
        for tree_num in range(0, self.num_trees):
            tree = self.clf.estimators_[tree_num]
            left = tree.tree_.children_left
            right = tree.tree_.children_right
            leaves = np.where(left < 0)[0]

            unique, counts = np.unique(pred[:, tree_num], return_counts=True)
            to_prune = unique[counts < min_node_size]
            to_prune = np.concatenate([to_prune, np.setdiff1d(leaves, unique)])

            while len(to_prune) > 0:
                for tp in to_prune:
                    parent = np.where(left == tp)[0]
                    if len(parent) > 0:
                        left[parent] = right[parent]
                    else:
                        parent = np.where(right == tp)[0]
                        right[parent] = left[parent]
                to_prune = np.where(np.in1d(left, to_prune))[0]

    def forde(self, dist="truncnorm", oob=False, alpha=0):
        self.dist = dist
        self.oob = oob
        self.alpha = alpha

        pred = self.clf.apply(self.x_real)

        if self.oob:
            for tree in range(self.num_trees):
                idx_oob = np.isin(range(self.x_real.shape[0]), _generate_unsampled_indices(self.clf.estimators_[tree].random_state, self.x_real.shape[0]))
                pred[np.invert(idx_oob), tree] = -1

        bnds = pd.concat([utils.bnd_fun(tree=j, p=self.p, forest=self.clf, feature_names=self.orig_colnames) for j in range(self.num_trees)])
        bnds['f_idx'] = bnds.groupby(['tree', 'leaf']).ngroup()

        bnds_2 = pd.DataFrame()
        for t in range(self.num_trees):
            unique, freq = np.unique(pred[:, t], return_counts=True)
            vv = pd.concat([pd.Series(unique, name='leaf'), pd.Series(freq / pred.shape[0], name='cvg')], axis=1)
            zz = bnds[bnds['tree'] == t]
            bnds_2 = pd.concat([bnds_2, pd.merge(left=vv, right=zz, on=['leaf'])])
        bnds = bnds_2
        del bnds_2

        if np.invert(self.factor_cols).any():
            bnds.loc[bnds['cvg'] == 1 / pred.shape[0], 'cvg'] = 0
        bnds = bnds[bnds['cvg'] > 0]
        bnds.rename(columns={'leaf': 'nodeid'}, inplace=True)

        self.bnds = bnds
        self.params = pd.DataFrame()
        if np.invert(self.factor_cols).any():
            for tree in range(self.num_trees):
                dt = self.x_real.loc[:, np.invert(self.factor_cols)].copy()
                dt["tree"] = tree
                dt["nodeid"] = pred[:, tree]
                long = pd.merge(right=bnds[['tree', 'nodeid', 'variable', 'min', 'max', 'f_idx']], left=pd.melt(dt[dt["nodeid"] >= 0], id_vars=["tree", "nodeid"]), on=['tree', 'nodeid', 'variable'], how='left')
                if self.dist == "truncnorm":
                    res = long.groupby(['tree', "nodeid", "variable"], as_index=False).agg(mean=("value", "mean"), sd=("value", "std"), min=("min", "min"), max=("max", "max"))
                elif self.dist == "norm":
                    res = long.groupby(['tree', "nodeid", "variable"], as_index=False).agg(mean=("value", "mean"), sd=("value", "std"))
                else:
                    raise ValueError('unknown distribution, make sure to enter a valid value for dist')
                    exit()
                self.params = pd.concat([self.params, res])

        self.class_probs = pd.DataFrame()
        if self.factor_cols.any():
            for tree in range(self.num_trees):
                dt = self.x_real.loc[:, self.factor_cols].copy()
                dt["tree"] = tree
                dt["nodeid"] = pred[:, tree]
                dt = pd.melt(dt[dt["nodeid"] >= 0], id_vars=["tree", "nodeid"])
                long = pd.merge(left=dt, right=bnds, on=['tree', 'nodeid', 'variable'])
                long['count_var'] = long.groupby(['tree', 'nodeid', 'variable'])['variable'].transform('count')
                long['count_var_val'] = long.groupby(['tree', 'nodeid', 'variable', 'value'])['variable'].transform('count')
                long.drop_duplicates(inplace=True)
                if self.alpha == 0:
                    long['prob'] = long['count_var_val'] / long['count_var']
                else:
                    long['k'] = long.groupby(['variable'])['value'].transform('nunique')
                    long['prob'] = (long['count_var_val'] + self.alpha) / (long['count_var'] + self.alpha * long['k'])
                    long['value'] = long['value'].astype('int8')

                long = long[['f_idx', 'tree', "nodeid", 'variable', 'value', 'prob']]
                self.class_probs = pd.concat([self.class_probs, long])
        return {"cnt": self.params, "cat": self.class_probs, "forest": self.clf, "meta": pd.DataFrame(data={"variable": self.orig_colnames, "family": self.dist})}

    def forge(self, n):
        try:
            getattr(self, 'bnds')
        except AttributeError:
            raise AttributeError('need density estimates to generate data -- run .forde() first!')

        unique_bnds = self.bnds[['tree', 'nodeid', 'cvg']].drop_duplicates()
        draws = np.random.choice(a=range(unique_bnds.shape[0]), p=unique_bnds['cvg'] / unique_bnds.shape[0], size=n)
        sampled_trees_nodes = unique_bnds[['tree', 'nodeid']].iloc[draws].reset_index(drop=True).reset_index().rename(columns={'index': 'obs'})

        if np.invert(self.factor_cols).any():
            obs_params = pd.merge(sampled_trees_nodes, self.params, on=["tree", "nodeid"]).sort_values(by=['obs'], ignore_index=True)

        if self.factor_cols.any():
            obs_probs = pd.merge(sampled_trees_nodes, self.class_probs, on=["tree", "nodeid"]).sort_values(by=['obs'], ignore_index=True)

        data_new = pd.DataFrame(index=range(n), columns=range(self.p))
        for j in range(self.p):
            colname = self.orig_colnames[j]

            if self.factor_cols[j]:
                data_new.iloc[:, j] = obs_probs[obs_probs["variable"] == colname].groupby("obs").sample(weights="prob")["value"].reset_index(drop=True)
            else:
                if self.dist == "truncnorm":
                    myclip_a = obs_params.loc[obs_params["variable"] == colname, "min"]
                    myclip_b = obs_params.loc[obs_params["variable"] == colname, "max"]
                    myloc = obs_params.loc[obs_params["variable"] == colname, "mean"]
                    myscale = obs_params.loc[obs_params["variable"] == colname, "sd"]
                    data_new.iloc[:, j] = scipy.stats.truncnorm(a=(myclip_a - myloc) / myscale, b=(myclip_b - myloc) / myscale, loc=myloc, scale=myscale).rvs(size=n)
                elif self.dist == "norm":
                    mean = obs_params.loc[obs_params["variable"] == colname, "mean"]
                    sd = obs_params.loc[obs_params["variable"] == colname, "sd"]
                    data_new.iloc[:, j] = np.random.normal(loc=mean, scale=sd, size=n)
                else:
                    raise ValueError('Other distributions not yet implemented')

        data_new.columns = self.orig_colnames

        for col in self.orig_colnames:
            if self.factor_cols[col]:
                data_new[col] = pd.Categorical.from_codes(data_new[col], categories=self.levels[col])

        for col in self.orig_colnames:
            if self.object_cols[col]:
                data_new[col] = data_new[col].astype("object")

        return data_new

    if __name__ == "__main__":
        df = pd.DataFrame({
            'feature1': np.random.normal(size=1000),
            'feature2': np.random.normal(size=1000)
        })

        # Fit ARF
        arf_model = arf(df)

        # Estimate densities using normal distribution
        density_params_norm = arf_model.forde(dist="norm")

        # Generate synthetic data using normal distribution
        synthetic_data_norm = arf_model.forge(n=1000)
        print(synthetic_data_norm.head())

        # Estimate densities using truncated normal distribution
        density_params_truncnorm = arf_model.forde(dist="truncnorm")

        # Generate synthetic data using truncated normal distribution
        synthetic_data_truncnorm = arf_model.forge(n=1000)
        print(synthetic_data_truncnorm.head())

                