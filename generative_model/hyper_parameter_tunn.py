# stdlib
# stdlib
import sys
import warnings

from synthcity.benchmark import Benchmarks
from pathlib import Path
# third party
import optuna
from sklearn.datasets import load_diabetes
import joblib
# synthcity absolute
import synthcity.logger as log
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

log.add(sink=sys.stderr, level="INFO")
warnings.filterwarnings("ignore")

import gzip
import pickle
def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

import pandas as pd
num = 1
train_data_features = load_data("generative_input/entire_ceros_tabular_data.pkl")
#train_data_features = train_data_features[:1000]

#reloaded = load_from_file('./adsgan_10_epochs.pkl')

#df = df.iloc[:2000,1:]

# stdlib
# Benchmarking the synthetic data generation process
print(train_data_features.shape)

        #'ctgan','uniform_sampler','ddpm','marginal_distributions', 'tvae','rtvae',
loader = GenericDataLoader(train_data_features, target_column="HOSPITAL_EXPIRE_FLAG", sensitive_columns=['ADMITTIME','SUBJECT_ID', 'GENDER_0', 'GENDER_F', 'GENDER_M'])

train_loader, test_loader = loader.train(), loader.test()


PLUGIN = "arf"
plugin_cls = type(Plugins().get(PLUGIN))
plugin_cls


print(plugin_cls.hyperparameter_space())


from synthcity.utils.optuna_sample import suggest_all
'''
trial = optuna.create_study().ask()
params = suggest_all(trial, plugin_cls.hyperparameter_space())
params['n_iter'] = 100  # speed up
params['delta'] =0.5



plugin = plugin_cls(**params).fit(train_loader)
report = Benchmarks.evaluate(
    [("trial", PLUGIN, params)],
    train_loader,  # Benchmarks.evaluate will split out a validation set
    repeats=1,

      # DELETE THIS LINE FOR ALL METRICS
)

print(report['trial'])

'''
def objective(trial: optuna.Trial):
    hp_space = Plugins().get(PLUGIN).hyperparameter_space()
    hp_space[0].high = 100  # speed up for now
    delta = trial.suggest_float('delta', 0.0, 0.5)  # Ensuring delta is between 0 and 0.5
    num_trees = trial.suggest_int('num_trees', 10, 100)
    max_iters = trial.suggest_int('max_iters', 1, 10)
    min_node_size = trial.suggest_int('min_node_size', 10, 20)
    params = {
        'delta': delta,
        'num_trees': num_trees,
        'max_iters': max_iters,
        'early_stop': False,  # Assuming this is a boolean flag
        'min_node_size': min_node_size,
        'workspace': Path('workspace'),  # Ensure Path is correctly imported
        'random_state': 0  # Fixed for reproducibility
    }
    ID = f"trial_{trial.number}"
    try:
        report = Benchmarks.evaluate(
            [(ID, PLUGIN, params)],
            train_loader,
            repeats=1,
             # DELETE THIS LINE FOR ALL METRICS
        )
    except Exception as e:  # invalid set of params
        print(f"{type(e).__name__}: {e}")
        print(params)
        raise optuna.TrialPruned()
    score = report[ID].query('direction == "minimize"')['mean'].mean()
    # average score across all metrics with direction="minimize"
    return score


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=2)
study.best_params

import json

# Save best parameters to a JSON file
joblib.dump(study, "HO/study"+str(num)+".pkl")

with open('HO/best_parameters.json', 'w') as f:
    json.dump(study.best_params, f)