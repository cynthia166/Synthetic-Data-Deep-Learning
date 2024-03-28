
from preprocess_input1 import *
# The above code is importing the necessary modules, such as `config` and `os`, in a Python script. It
# then changes the current working directory to a new path specified as `'../'`, which typically means
# moving up one directory level from the current working directory.
from config import *

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



def procedures_data(type_p,doc_path,admissions_path,patients_path,numerical_cols,n,categorical_cols,normalize_matrix, log_transformation, encode_categorical):
    name="ICD9_CODE"   
    prepomax = 'std'
    type_p = "procedures"
    name="ICD9_CODE"
    prepomax = 'std'
    real = "CCS CODES"
    level = "Otro"
    # type_p, doc_path, admissions_path, patients_path, categorical_cols, real, level, numerical_cols, prepomax, name, n, cols_to=None, normalize_matrix=False, log_transformation=False, encode_categorical=False, final_preprocessing=False
    preprocessor = DataPreprocessor(type_p,doc_path, admissions_path, patients_path, categorical_cols, real, level, numerical_cols, prepomax,name,n, cols_to = None,normalize_matrix=normalize_matrix, log_transformation=log_transformation, encode_categorical=encode_categorical, final_preprocessing=True,proportion = True)
    #df = preprocessor.load_data(type_p)
    df_final = preprocessor.run(type_p)
    df_final.to_csv(str(DARTA_INTERM_intput) + real +"_"+type_p+".csv")
    print(" Procedures Done")
    
    
def drugs_data(type_p,doc_path,admissions_path,patients_path,numerical_cols,n,categorical_cols,normalize_matrix, log_transformation, encode_categorical):
    prepomax = 'std'
    name = "DRUG"
    real = "ATC3"
    level = "Otro"
  # type_p, doc_path, admissions_path, patients_path, categorical_cols, real, level, numerical_cols, prepomax, name, n, cols_to=None, normalize_matrix=False, log_transformation=False, encode_categorical=False, final_preprocessing=False
    preprocessor = DataPreprocessor(type_p,doc_path, admissions_path, patients_path, categorical_cols, real, level, numerical_cols, prepomax,name,n, cols_to = None,normalize_matrix=normalize_matrix, log_transformation=log_transformation, encode_categorical=encode_categorical, final_preprocessing=True,proportion=True)
    aux = preprocessor.load_data_clean_data(type_p)
    df_final = preprocessor.run(type_p)
    df_final.to_csv(str(DARTA_INTERM_intput)+ real +"_"+type_p+".csv")

    print(" Drugs Done")
def diagnosis_data(type_p,doc_path,admissions_path,patients_path,numerical_cols,n,categorical_cols,normalize_matrix, log_transformation, encode_categorical):

    type_p = "diagnosis"
    name="ICD9_CODE"
    prepomax = 'std'
    real = "CCS CODES"
    level = "Otro"
    preprocessor = DataPreprocessor(type_p,doc_path, admissions_path, patients_path, categorical_cols, real, level, numerical_cols, prepomax,name,n, cols_to = None,normalize_matrix=normalize_matrix, log_transformation=log_transformation, encode_categorical=encode_categorical, final_preprocessing=True,proportion=True)
    #df = preprocessor.load_data(type_p)
    df_final = preprocessor.run(type_p)
    df_final.to_csv(str(DARTA_INTERM_intput)+ real +"_"+type_p+".csv")
    print(" Diagnosis Done")

def main(type_p,normalize_matrix, log_transformation, encode_categorical):
    
    admissions_path = MIMIC/'ADMISSIONS.csv.gz'
    patients_path = MIMIC /'PATIENTS.csv.gz'
    numerical_cols =  ['Age_max', 'LOSRD_sum',
            'LOSRD_avg','L_1s_last_p1']

        
    n = [.88,.95,.98,.999]
    categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                    'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                    'MARITAL_STATUS',  'ETHNICITY','GENDER']
    if type_p == "diagnosis":
        try:
            doc_path = MIMIC/'DIAGNOSES_ICD.csv.gz'
            diagnosis_data(type_p,doc_path,admissions_path,patients_path,numerical_cols,n,categorical_cols,normalize_matrix, log_transformation, encode_categorical)
        except:
            
            doc_path =Path('..')/ MIMIC/'DIAGNOSES_ICD.csv.gz'    
            diagnosis_data(type_p,doc_path,admissions_path,patients_path,numerical_cols,n,categorical_cols,normalize_matrix, log_transformation, encode_categorical)
    
    elif type_p == "procedures":
        try:
            doc_path = MIMIC/'PROCEDURES_ICD.csv.gz'
            procedures_data(type_p,doc_path,admissions_path,patients_path,numerical_cols,n,categorical_cols,normalize_matrix, log_transformation, encode_categorical)
        except:
            doc_path =Path('..')/'PROCEDURES_ICD.csv.gz'
            procedures_data(type_p,doc_path,admissions_path,patients_path,numerical_cols,n,categorical_cols,normalize_matrix, log_transformation, encode_categorical)

    elif type_p in ["drug1", "drug2"]:
        try:
            doc_path = MIMIC/'PRESCRIPTIONS.csv.gz'
            drugs_data(type_p,doc_path,admissions_path,patients_path,numerical_cols,n,categorical_cols,normalize_matrix, log_transformation, encode_categorical)
        except:
            doc_path =Path('..')/'PRESCRIPTIONS.csv.gz'
            drugs_data(type_p,doc_path,admissions_path,patients_path,numerical_cols,n,categorical_cols,normalize_matrix, log_transformation, encode_categorical)
    else:
        print("Tipo de procesamiento no reconocido.")

      
if __name__ == "__main__":
    import argparse
    import os

    ruta = str(RAW/"suplement/RXCUI2atc4.csv")

    if os.path.exists(ruta):
        print("La ruta existe.")
    else:
        print("La ruta no existe.")

    parser = argparse.ArgumentParser(description="Script para procesar datos de salud con opciones adicionales.")

    # Este argumento es obligatorio, así que no tiene un valor por defecto.
    parser.add_argument("type_p", type=str, choices=["diagnosis", "procedures", "drug1", "drug2"],default="diagnosis",
                        help="Tipo de procesamiento a realizar.")

    # Argumentos opcionales con valores por defecto para facilitar la depuración.
    parser.add_argument("--normalize_matrix", action="store_true", 
                        help="Normaliza la matriz durante el procesamiento. Por defecto es True para depuración.")

    parser.add_argument("--log_transformation", action="store_true", 
                        help="Aplica transformación logarítmica durante el procesamiento. Por defecto es True para depuración.")

    parser.add_argument("--encode_categorical", action="store_true", 
                        help="Codifica variables categóricas durante el procesamiento. Por defecto es True para depuración.")

    args = parser.parse_args()

    print(f"Process: {args.type_p}")
    print(f"Standard matriz: {args.normalize_matrix}")
    print(f"Logarithmic transformation: {args.log_transformation}")
    print(f"Codify categorical variables: {args.encode_categorical}")

        
    main(args.type_p, args.normalize_matrix, args.log_transformation, args.encode_categorical)    
    # Crear una lista de las columnas a normalizar (todas las columnas excepto 'CCS CODES', 'HADM_ID' y 'SUBJECT_ID')
    #cols_to_itereate = [col for col in df.columns if col not in ['LEVE3 CODES', 'HADM_ID', 'SUBJECT_ID']]
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
    ########################### ########################### DIAGNOSIS########################### ########################### ########################### ########################### 
    ########################### ########################### ########################### ########################### ########################### ########################### 
    ##### Para las que no tiene columnas##################################


