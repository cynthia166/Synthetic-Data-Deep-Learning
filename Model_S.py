
# stdlib
import sys
import warnings
# synthcity absolute
import os

os.chdir('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning')
# synthcity absolute
import synthcity.logger as log
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import TimeSeriesSurvivalDataLoader
import sys
import warnings

warnings.filterwarnings("ignore")

# third party
from sklearn.datasets import load_iris

# synthcity absolute
import synthcity.logger as log
#from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader


# synthcity absolute
from synthcity.benchmark import Benchmarks


log.add(sink=sys.stderr, level="INFO")
warnings.filterwarnings("ignore")
import pandas as pd

df =  pd.read_csv('generative_input/input_generative_g.csv')
df = df.iloc[:2000,1:]
df
# stdlib

loader = GenericDataLoader(df, target_column="HOSPITAL_EXPIRE_FLAG", sensitive_columns=[])

loader.dataframe()


#"ctgan", "tvae"
score = Benchmarks.evaluate(
    [
        (f"example_{model}", model, {})  # testname, plugin name, plugin args
        for model in ["adsgan", ]
    ],
    loader,
    synthetic_size=10,
    task_type="classification",
    metrics =     ['data_mismatch', 'common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability', 'jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc']
    repeats=3,
)

Benchmarks.print(score)

Benchmarks.highlight(score)
# third party
import numpy as np
import pandas as pd

means = []
for plugin in score:
    data = score[plugin]["mean"]
    directions = score[plugin]["direction"].to_dict()
    means.append(data)

out = pd.concat(means, axis=1)
out.set_axis(score.keys(), axis=1, inplace=True)

bad_highlight = "background-color: lightcoral;"
ok_highlight = "background-color: green;"
default = ""


def highlights(row):
    metric = row.name
    if directions[metric] == "minimize":
        best_val = np.min(row.values)
        worst_val = np.max(row)
    else:
        best_val = np.max(row.values)
        worst_val = np.min(row)

    styles = []
    for val in row.values:
        if val == best_val:
            styles.append(ok_highlight)
        elif val == worst_val:
            styles.append(bad_highlight)
        else:
            styles.append(default)

    return styles


out.style.apply(highlights, axis=1)