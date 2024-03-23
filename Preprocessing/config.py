# config.py

from pathlib import Path  # pathlib is seriously awesome!

# Definir el directorio de datos
from pathlib import Path

# Directorio de datos
data_dir = Path('data/')
# Ruta al archivo de datos específico
data_path = data_dir / 'my_file.csv'

# Directorio principal de resultados
DATA_DIRECTORY_results = Path("results/experimernt_prepro")

_DIRECTORY_results2 = Path("results/experiment_prepo")

# Subdirectorios
PROCEDURES_DIRECTORY = DATA_DIRECTORY_results / "procedures/"
DRUGS_DIRECTORY = DATA_DIRECTORY_results / "drugs/"
DIAGNOSIS_DIRECTORY = DATA_DIRECTORY_results / "diagnosis/"

DIAGNOSIS_DIRECTORY_e = DIAGNOSIS_DIRECTORY / "experiment_prepo"


# Archivos de resultados de predicciones para procedimientos
RESULTS_PREDICTION_FILE_1 = PROCEDURES_DIRECTORY / "results_prediction_30+non_filteredProcedures_1_.csv"
RESULTS_PREDICTION_FILE_2 = PROCEDURES_DIRECTORY / "results_prediction_30+non_filteredProcedures_prods_prediction.csv"
RESULTS_PREDICTION_FILE_3 = PROCEDURES_DIRECTORY / "results_prediction_30+non_filteredProcedures.csv"
LG_PROD_FILE = PROCEDURES_DIRECTORY / "lg_prod.csv"
HP_PROD_XGBOS_FILE = PROCEDURES_DIRECTORY / "hp_prod_xgbos.csv"
RESULTS_FINAL_MERGED_PROCEDURES_FILE = PROCEDURES_DIRECTORY / "results_final_merged_procedures_prepo.csv"

# Archivos de resultados de predicciones para medicamentos
XGBOOST_RP_DRUGA_FILE = DRUGS_DIRECTORY / "xgbost_rp_druga.csv"
DRUGS_LR_HP_FILE = DRUGS_DIRECTORY / "drugs_lr_hp.csv"
RESULTS_PREPRO_NONFILTERED_DRUGS_FINAL_FILE = DRUGS_DIRECTORY / "experiment_preporesults_prepro_nonfilteres_DRUGS_final.csv"

# Archivos de resultados de predicciones para diagnósticos
RESULTS_PREDICTION_NONFILTERED_DIAGNOSIS = [
    DIAGNOSIS_DIRECTORY / "results_prediction_30+non_filteredDiagnosis_.csv",
    DIAGNOSIS_DIRECTORY / "results_prediction_30+non_filteredDiagnosis_1.csv",
    DIAGNOSIS_DIRECTORY / "results_prediction_30+non_filteredDiagnosis_2.csv",
    DIAGNOSIS_DIRECTORY / "results_prediction_30+non_filteredDiagnosis.csv",
    DIAGNOSIS_DIRECTORY / "results_prediction_30+non_filteredDiagnosis_threshopl_.csv",
]
DIAGNOSIS_XGBOOST_HP_FILE = DIAGNOSIS_DIRECTORY / "diagnosis_xgbost_hp.csv"
DIAGNOSIS_LR_HP_FILE = DIAGNOSIS_DIRECTORY / "diagnosis_lr_hp.csv"
RESULTS_PREPRO_NONFILTERED_DIAGNOSIS_FINAL_FILE = DIAGNOSIS_DIRECTORY / "experiment_preporesults_prepro_nonfilteres_diagnosis_final.csv"

IMAGES_PRED_DICT = "report/img/pred/"
IMAGES_Cluster_DICT= "report/img/cluster/"
