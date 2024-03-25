
from Preprocessing.preprocess_input1 import *

class DataPreprocessor:
    def __init__(self, type_p, doc_path, admissions_path, patients_path, categorical_cols, real, level, numerical_cols, prepomax, name, n, cols_to=None, normalize_matrix=False, log_transformation=False, encode_categorical=False, final_preprocessing=False,proportion = False,prop = 0.09 ):
        self.type_p = type_p  # Correctly initializing type_p here
        self.doc_path = doc_path
        self.admissions_path = admissions_path
        self.patients_path = patients_path
        self.categorical_cols = categorical_cols
        self.real = real
        self.level = level
        self.numerical_cols = numerical_cols
        self.prepomax = prepomax
        self.name = name
        self.n = n
        self.cols_to = cols_to
        self.normalize_matrix = normalize_matrix
        self.log_transformation = log_transformation
        self.encode_categorical = encode_categorical
        self.final_preprocessing = final_preprocessing
        self.proportion = proportion
        self.prop = prop
    def load_data_clean_data(self, type_p):
        if type_p == "procedures":
            data = procedures(self.doc_path, self.n, self.name)
        elif type_p == "diagnosis":
            data = diagnosis(self.doc_path, self.n, self.name)
        elif  type_p == "drug2":
            data = drug2(self.doc_path)
            # ATC
        elif  type_p == "drug1":
            # thresholds
              data = drugs1(self.doc_path, self.n, self.name)  
        return data



    def calculate_count_matrix(self, data):
        return calculate_pivot_df(data, self.real, self.level,self.type_p)

    def normalize_count_matrix(self, data):
        if self.normalize_matrix:
            return normalize_count_matrix__aux(data, self.level)
        return data

    def calculate_demographics(self, data):
        cat_considered = ['ADMITTIME', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'DEATHTIME'] + ['DISCHTIME', 'SUBJECT_ID', 'HADM_ID']
        return calculate_agregacion_cl(self.admissions_path, self.patients_path, self.categorical_cols, self.level, cat_considered, data)

    def apply_log_transformation(self, data):
        if self.log_transformation:
            return apply_log_transformation(data, 'L_1s_last_p1')
        return data

    def merge_data(self, demographic_data, count_data):
        return merge_df(demographic_data, count_data,self.level)

    def encode_categorical_data(self, data):
        if self.encode_categorical:
            return encoding(data, self.categorical_cols, 'onehot', self.proportion,self.prop)
        return data

    def final_preprocessing_fun(self, data):
        data.columns = data.columns.astype(str)
        if self.cols_to is None:
            cols_to_normalize = [col for col in data.columns if col not in ['SUBJECT_ID', 'HADM_ID'] ]
        else:
            cols_to_normalize = self.cols_to
        return preprocess(data, self.prepomax, cols_to_normalize)

    def run(self,type_p):
        data = self.load_data_clean_data(type_p)
        count_matrix = self.calculate_count_matrix(data)
        count_matrix = self.normalize_count_matrix(count_matrix)
        demographics = self.calculate_demographics(count_matrix)
        demographics = self.apply_log_transformation(demographics)
        encoded_data_demographics = self.encode_categorical_data(demographics)
        merged_data = self.merge_data(encoded_data_demographics, count_matrix)

        
        if self.final_preprocessing:
            final_data = self.final_preprocessing_fun(merged_data)
        else:
            final_data = merged_data
        return final_data
    
########################### ########################### ########################### ########################### ########################### ###########################    
########################### ########################### Procedures########################### ########################### ########################### ########################### 
########################### ########################### ########################### ########################### ########################### ########################### 
type_p= "diagnosis"

if type_p == "procedures":
    name="ICD9_CODE"
    n = [.88,.95,.98,.999]
    numerical_cols =  ['Age_max', 'LOSRD_sum',
            'LOSRD_avg','L_1s_last_p1']

    prepomax = 'std'
    # Ejemplo de uso
    #concat
    nom_archivo = 'data\df_non_filtered.parquet'

    categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                    'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                    'MARITAL_STATUS',  'ETHNICITY','GENDER']
    type_p = "procedures"

    name="ICD9_CODE"
    n = [.88,.95,.98,.999]
    numerical_cols =  ['Age_max', 'LOSRD_sum',
            'LOSRD_avg','L_1s_last_p1']

    prepomax = 'std'
    # Ejemplo de uso
    #concat
    nom_archivo = 'data\df_non_filtered.parquet'

    categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                    'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                    'MARITAL_STATUS', 'ETHNICITY','GENDER']

    real = "CCS CODES"
    level = "Otro"

    doc_path = '/Users/cgarciay/Desktop/Laval_Master_Computer/research/MIMIC/PROCEDURES_ICD.csv.gz'
    admissions_path = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/MIMIC/ADMISSIONS.csv.gz"
    patients_path = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/MIMIC/PATIENTS.csv.gz"
    # type_p, doc_path, admissions_path, patients_path, categorical_cols, real, level, numerical_cols, prepomax, name, n, cols_to=None, normalize_matrix=False, log_transformation=False, encode_categorical=False, final_preprocessing=False
    preprocessor = DataPreprocessor(type_p,doc_path, admissions_path, patients_path, categorical_cols, real, level, numerical_cols, prepomax,name,n, cols_to = None,normalize_matrix=True, log_transformation=True, encode_categorical=True, final_preprocessing=True,proportion = True)
    #df = preprocessor.load_data(type_p)
    df_final = preprocessor.run(type_p)



    df_final.to_csv("aux/"+ real +type_p+".csv")
    # Crear una lista de las columnas a normalizar (todas las columnas excepto 'CCS CODES', 'HADM_ID' y 'SUBJECT_ID')
    cols_to_itereate = [col for col in df.columns if col not in ['LEVE3 CODES', 'HADM_ID', 'SUBJECT_ID']]
'''for i in cols_to_itereate:
    real = i
    preprocessor = DataPreprocessor(type_p,doc_path, admissions_path, patients_path, categorical_cols, real, level, numerical_cols, prepomax,name,n, cols_to = None,normalize_matrix=False, log_transformation=False, encode_categorical=True, final_preprocessing=None)
    df_final = preprocessor.run(type_p)
    df_final.to_csv("input_pred_p/"+ real +type_p+".csv")

    
aux = pd.read_csv("input_pred_p/"+ real +type_p+".csv")
int_index_columns = []
for col in aux.columns:
    try:
        int(col)  # Intenta convertir el índice de la columna a un entero
          # Si la conversión es exitosa, agrega la columna a la lista
    except ValueError:
        int_index_columns.append(col)
          # Si la conversión falla, ignora la columna

print(int_index_columns)
aux = aux[int_index_columns]
cols_to_drop = aux.filter(like='Unnamed', axis=1).columns
aux.drop(cols_to_drop, axis=1, inplace=True)
aux.to_csv("input_pred_p/sin_codigo.csv")'''  
########################### ########################### ########################### ########################### ########################### ###########################    
########################### ########################### DRUGS########################### ########################### ########################### ########################### 
########################### ########################### ########################### ########################### ########################### ########################### 
##### Para las que no tiene columnas##################################


if type_p == "drug2" or type_p == "drug1":
   
    n = [.88,.95,.98,.999]
    numerical_cols =  ['Age_max', 'LOSRD_sum',
            'LOSRD_avg','L_1s_last_p1']

    prepomax = 'std'
    # Ejemplo de uso
    #concat
    nom_archivo = 'data\df_non_filtered.parquet'

    categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                    'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                    'MARITAL_STATUS', 'ETHNICITY','GENDER']
    name = "DRUG"

    
    n = [.88,.95,.98,.999]



   

    numerical_cols =  ['Age_max', 'LOSRD_sum',
            'LOSRD_avg','L_1s_last_p1']

    prepomax = 'std'
    # Ejemplo de uso
    #concat
    nom_archivo = 'data\df_non_filtered.parquet'

    categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                    'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                    'MARITAL_STATUS', 'ETHNICITY','GENDER']

    real = "ATC3"
    level = "Otro"

    doc_path = '/Users/cgarciay/Desktop/Laval_Master_Computer/research/MIMIC/PRESCRIPTIONS.csv.gz'
    admissions_path = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/MIMIC/ADMISSIONS.csv.gz"
    patients_path = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/MIMIC/PATIENTS.csv.gz"
    # type_p, doc_path, admissions_path, patients_path, categorical_cols, real, level, numerical_cols, prepomax, name, n, cols_to=None, normalize_matrix=False, log_transformation=False, encode_categorical=False, final_preprocessing=False
    preprocessor = DataPreprocessor(type_p,doc_path, admissions_path, patients_path, categorical_cols, real, level, numerical_cols, prepomax,name,n, cols_to = None,normalize_matrix=True, log_transformation=True, encode_categorical=True, final_preprocessing=True,proportion=True)
    aux = preprocessor.load_data_clean_data(type_p)
    df_final = preprocessor.run(type_p)
    df_final.to_csv("aux/"+ real +type_p+".csv")
    
    



########################### ########################### ########################### ########################### ########################### ###########################    
########################### ########################### DIAGNOSIS########################### ########################### ########################### ########################### 
########################### ########################### ########################### ########################### ########################### ########################### 
##### Para las que no tiene columnas##################################
if type_p == "diagnosis":
    type_p = "diagnosis"

    name="ICD9_CODE"
    n = [.88,.95,.98,.999]
    numerical_cols =  ['Age_max', 'LOSRD_sum',
            'LOSRD_avg','L_1s_last_p1']

    prepomax = 'std'
    # Ejemplo de uso
    #concat
    nom_archivo = 'data\df_non_filtered.parquet'

    categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                    'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                    'MARITAL_STATUS', 'ETHNICITY','GENDER']

    #real = "LEVE3 CODES"
    real = "CCS CODES"
    level = "Otro"

    doc_path = '/Users/cgarciay/Desktop/Laval_Master_Computer/research/MIMIC/DIAGNOSES_ICD.csv.gz'
    admissions_path = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/MIMIC/ADMISSIONS.csv.gz"
    patients_path = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/MIMIC/PATIENTS.csv.gz"
    # type_p, doc_path, admissions_path, patients_path, categorical_cols, real, level, numerical_cols, prepomax, name, n, cols_to=None, normalize_matrix=False, log_transformation=False, encode_categorical=False, final_preprocessing=False
   
    preprocessor = DataPreprocessor(type_p,doc_path, admissions_path, patients_path, categorical_cols, real, level, numerical_cols, prepomax,name,n, cols_to = None,normalize_matrix=True, log_transformation=True, encode_categorical=True, final_preprocessing=True,proportion=True)
    #df = preprocessor.load_data(type_p)
    df_final = preprocessor.run(type_p)



    df_final.to_csv("aux/"+ real +type_p+".csv")


####################################Conatenacion primera
#ruta_archivos = 's_data\*.csv.gz'  # Puedes cambiar '*.csv' por la extensión que desees
#save = True #dfale

#mi_objeto = PreprocessInput()
#df_concat_primero = concat_archivo_primeto(procedures,admi,ruta_archivos,save,nom_archivo)



#d1 = '.\s_data\DIAGNOSES_ICD.csv.gz'


#didf_diagnosisa = diagnosis(d1,n,name)

#desconcatenación, y se obtiene threshold y mapping
#d2 = '.\s_data\PROCEDURES_ICD.csv.gz' 
#prod = procedures(d2,n,name)
#prod = limipiar_Codigos(prod)
#caluco de matrix de conteo

#prod_ipvot  = calculate_pivot_df(prod, real, level) #if normalize matrix == True
#prod_ipvot = normalize_count_matrix(prod_ipvot, level, )
#prod_ipvot

#prod_ipvot
#calcul de variable demos
#adm = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/MIMIC/ADMISSIONS.csv.gz"
#pa = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/MIMIC/PATIENTS.csv.gz"


#cat_considered = ['ADMITTIME','ADMISSION_TYPE', 'ADMISSION_LOCATION',
#                'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
#                'MARITAL_STATUS', 'ETHNICITY','DEATHTIME'] + ['DISCHTIME','SUBJECT_ID', 'HADM_ID']
#demos_pivot = calculate_agregacion_cl(adm,pa, categorical_cols, level,cat_considered,prod_ipvot)


#print(demos_pivot.isnull().sum()) #if variable logtransforamtio == true
#demos_pivot = apply_log_transformation(demos_pivot, 'L_1s_last_p1')
#merge demographic df with count matrix
#df_merge  = merge_df(demos_pivot, prod_ipvot)
#df_merge.shape
#if encoding categorical variables=true
#df_merge_e = encoding(df_merge, categorical_cols, 'onehot')
#df_merge_e


# if preprocesamiento final == True
#numerical_cols =  ['Age_max', 'LOSRD_sum',
#    'LOSRD_avg','L_1s_last_p1']
#df_merge_e.columns = df_merge_e.columns.astype(str)
#cols_to_normalize = [col for col in df_merge_e.columns if col not in ['SUBJECT_ID', 'HADM_ID']]



#df_final_prepo = preprocess(df_merge_e, "max", cols_to_normalize)
       
       
#d1 = '..\s_data\PRESCRIPTIONS.csv.gz'
#name1 = "DRUG"
#df_drugs = drugs(d1,name1)

#name_df = "raw_input.csv"
#name_encodeing = "input_onehot_encoding.csv"


#df_final = concat_input(df_drugs, df_diagnosis, df_procedures,numerical_cols,categorical_cols,name_df)
#df_final_encoded = encoding(df_final,categorical_cols)