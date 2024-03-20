# Synthetic-Data-Deep-Learning
Synthetic Data Generation with deep learning models.

**Exploratory Analysis** (doc_path):


```
Preprocessing/1_eda_2.ipynb
```

**Concatenated selected input (subset(MIMIC-III))** (doc_path):


```
Preprocessing/2_input_1.ipynb

```


**Exploration of Simplification Thresholds Diagnosis** 

```
Preprocessing/3_diagnosis_icd9codes_1.ipynb

```


**Exploration of Simplification Thresholds Procedures** 

```
Preprocessing/3_procedures_icd9codes_1.ipynb

```

**Exploration of Simplification Thresholds Drugs** 

```
Preprocessing/3_drugs_count_1.ipynb

```

**Exploration of Simplification Thresholds Drugs** 

```
Preprocessing/3_diagnosis_icd9codes_1.ipynb

```

**Visualizations** 

```
6_visualization_results.ipynb

```

**Experiment Clustering Prepocessing (std, max, pwr) configuration (4,8,12)** 

```
Preprocessing/cluster_prepo_script.py

```

**Experiment Mutual information** 

```
Preprocessing/mutual_information_script.py

```



**Input for prediction and readmission** 

```
Preprocessing/input_pred_dinamic.py
```


**Experiment Prediction of 30 days Readmission** 

```
Preprocessing/readmission_pred_script.py

```

**Hyperparameter tuning** 

```
Preprocessing/readmission_pred_script_wandb2_drugsprocedures.py

```

**Hyperparameter tuning** 

```
Preprocessing/readmission_pred_script_wandb2_drugsprocedures.py

```

**Functions Helpers scripts** 

```
Preprocessing/preprocess.py
Preprocessing/features_eng.py
Preprocessing/function_mapping.py
Preprocessing/function_pred.py
preprocess_input1.py
utils.py
```



**Preprocess for Generative Model** 

```
preprocess_input_SDmodel.py

```


**Generative Models** 

```
Model_S.py
Model_Stimeseries.py
Dopplenganger.py

```



Raw input model (without preprocessing):

In the script: 

```
preprocess_class1.py
```


1.run following for type_g =  "diagnosos", "prescripction", "drugs"

doc_path = PROCEDURES_ICD.csv.gz

admission_path = ADMISSIONS.csv.gz

patients_path = PATIENTS.csv.gz

categorical_cols = demographic and admidsion cols considered

real = "name of preprocessing ej: thershold x, ATC3, CCS CODES"

name = "ICD9_CODES", "DRUGS"

n = thresholds to be applied



```
preprocessor = DataPreprocessor(type_p,doc_path, admissions_path, patients_path, categorical_cols, real, level, numerical_cols, prepomax,name,n, cols_to = None,normalize_matrix=False, log_transformation=False, encode_categorical=False, final_preprocessing=True)

data = preprocessor.run(type_p)
count_matrix = preprocessor.calculate_count_matrix(data)
demographics = preprocessor.calculate_demographics(count_matrix)
merged_data = preprocessor.merge_data(demographics, count_matrix)

```

Preprocessing for prediction analysis and clustering

```
preprocessor = DataPreprocessor(type_p,doc_path, admissions_path, patients_path, categorical_cols, real, level, numerical_cols, prepomax,name,n, cols_to = None,normalize_matrix=True, log_transformation=True, encode_categorical=True, final_preprocessing=True)
df = preprocessor.run(type_p)
```
