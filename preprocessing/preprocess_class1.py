
import sys
sys.path.append('')
sys.path.append('preprocessing')
from preprocess_input1 import *
from preprocessing.config import *

import logging
logging.basicConfig(
    filename='app.log',  # Log file name
    level=logging.INFO,  # Set the minimum level of log messages to capture
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DataPreprocessor:
    def __init__(self, type_p,
                  doc_path, 
                  admissions_path,
                    patients_path,
                      categorical_cols,
                        real,
                          level,
                            numerical_cols,
                              prepomax, name,
                                n,
                                columns_to_drop,
                                cols_to_accumulate,
                                feature_accumulative_path,
                                save_accumulate_df,
                                  cols_to=None,
                                    normalize_matrix=False,
                                      log_transformation=False, encode_categorical=False,
                                        final_preprocessing=False,proportion = False,prop = 0.8,
                                         make_initial_preprocess = True,
                                          features_path = "",
                                           create_added_paste_feautres_actual_summed=True,
                                           plot_matrix=True,
                                           save_prproces=True,read_archivo_threshold=True):
        self.type_p = type_p  #drugs procedures diagnosis
        self.doc_path = doc_path # Path to the document above
        self.admissions_path = admissions_path #admissions_path
        self.patients_path = patients_path #patients_path
        self.categorical_cols = categorical_cols #8 categorical cols
        self.real = real #CCS CODES whar simpñification is to be considered
        self.level = level #patient level or admission level
        self.numerical_cols = numerical_cols # 4 numerical cols
        self.prepomax = prepomax # preporcessing method
        self.name = name # ICD9_CODE or DRUGS
        self.n = n # thresholds
        self.cols_to = cols_to # columns to normalize
        self.normalize_matrix = normalize_matrix # normalize matrix or not / true or false
        self.log_transformation = log_transformation # log transformation or not / true or false for lenght of stay variable
        self.encode_categorical = encode_categorical # encode categorical variables or not / true or false
        self.final_preprocessing = final_preprocessing  # final preprocessing or not / true or false
        self.proportion = proportion  # proportion to considere for the agruping of demographical variables or not / true or false         
        self.prop = prop #preprocessing portion
        self.make_initial_preprocess = make_initial_preprocess
        self.features_path = features_path
        self.create_added_paste_feautres_actual_summed = create_added_paste_feautres_actual_summed
        self.columns_to_drop  = columns_to_drop
        self.cols_to_accumulate = cols_to_accumulate
        self.feature_accumulative_path = feature_accumulative_path
        self.save_accumulate_df = save_accumulate_df
        self.plot_matrix =plot_matrix
        self.save_prproces=save_prproces
        self.read_archivo_threshold=read_archivo_threshold
    def initialize(self):
        if self.make_initial_preprocess == True:
           self.df = self.run()
           logging.info("Running features and concatenations")
        else: 
            self.df = self.load_initial_preocess()   
            logging.info("Pre -loaddings features and concatenations")
            if   len(self.columns_to_drop)!=0:
                self.eliminate_columns()
                logging.info(f'eliminating {self.columns_to_drop}')
        if self.create_added_paste_feautres_actual_summed:
           self.calculate_accumulative_los()
           logging.info("Creating accumulative features")
        if self.save_accumulate_df:
            save_pickle(self.df,self.feature_accumulative_path)
            logging.info("Saving pickle")
           
    def calculate_accumulative_los(self):
    # Convert the input data to a pandas DataFrame


        # Sort the DataFrame by patient_id and visit_rank
        self.df = self.df.sort_values(['SUBJECT_ID', 'visit_rank'])
        for col in self.cols_to_accumulate:          
        # Calculate the accumulative LOS for each patient
            self.df['accumulative_'+col] =self.df.groupby('SUBJECT_ID')[col].cumsum()



        return self.df        

    def eliminate_columns(self):
        
        self.df.drop(columns=self.columns_to_drop, inplace=True)

    def load_initial_preocess(self):
        return load_data(self.features_path)
                
    def load_data_clean_data(self, type_p):
        #clean data, and create indiidual counts of drugs or codes
        if type_p == "procedures":
            data = procedures(self.doc_path, self.n, self.name,self.plot_matrix)
        elif type_p == "diagnosis":
            data = diagnosis(self.doc_path, self.n, self.name,self.plot_matrix)
        elif  type_p == "drug2":
            data = drug2(self.doc_path,self.plot_matrix)
            # ATC
        elif  type_p == "drugst":
            # thresholds
              data = drugst(self.doc_path, self.n, self.name, 
                            save_path_prefix="data/analysis/drugs/drugs2",
                            administrations=False,
                            plot_matrix=self.plot_matrix,
                            save_prproces=self.save_prproces,
                            analyze_distributions=False)  
        return data



    def calculate_count_matrix(self, data):
        # calculate pivot matrix
        return calculate_pivot_df(data, self.real, self.level,self.type_p)

    def normalize_count_matrix(self, data):
        #normalize the coun matrix
        if self.normalize_matrix:
            return normalize_count_matrix__aux(data, self.level)
        return data

    def calculate_demographics(self, data):
        #calculates demographical data
        cat_considered = ['ADMITTIME', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'DEATHTIME', 'DISCHTIME'] + [ 'SUBJECT_ID', 'HADM_ID']
        return calculate_agregacion_cl(self.admissions_path, self.patients_path, self.categorical_cols, self.level, cat_considered, data)


    def apply_log_transformation(self, data):
        #transform the lenght of stay variable
        if self.log_transformation:
            return apply_log_transformation(data, 'days from last visit')
        return data

    def merge_data(self, demographic_data, count_data):
        #concatenate demographic and count_data
        return merge_df(demographic_data, count_data,self.level)

    def encode_categorical_data(self, data):
        #encode categorical data
        if self.encode_categorical:
            return encoding(data, self.categorical_cols, 'label', self.proportion,self.prop)
        return data

    def final_preprocessing_fun(self, data):
        #final preprocessing
        data.columns = data.columns.astype(str)
        if self.cols_to is None:
            cols_to_normalize = [col for col in data.columns if col not in ['SUBJECT_ID', 'HADM_ID'] ]
        else:
            cols_to_normalize = self.cols_to
        return preprocess(data, self.prepomax, cols_to_normalize)

    def run(self,type_p):
        # run the whole process sequentially
        if type_p == "drugst":
            if self.read_archivo_threshold:
                archivo2 = "/Users/cynthiagarcia/Desktop/Synthetic-Data-Deep-Learning/data/intermedi/SD/inpput/second/with_ascii_codes_mapping_drugs.pkl"
                data = load_data(archivo2)
                for i in data:
                    print(i)
                    print("categorias unicas",data[i].nunique())
            else:
                data = self.load_data_clean_data(type_p)
        elif type_p == "procedures":
                   if self.read_archivo_threshold:
                      archivo2 = "/Users/cynthiagarcia/Desktop/Synthetic-Data-Deep-Learning/data/intermedi/SD/inpput/second/procedures/ procedures_total.pkl"
                      data = load_data(archivo2)
                      for i in data:
                        print(i)
                        print("categorias unicas",data[i].nunique())
                   else:
                      data = self.load_data_clean_data(type_p)
   
               
        else:    
            if self.read_archivo_threshold:
                    archivo2 = "/Users/cynthiagarcia/Desktop/Synthetic-Data-Deep-Learning/data/intermedi/SD/inpput/second/diagnosis/diagnosis_total.pkl"
                    data = load_data(archivo2)
                    for i in data:
                        print(i)
                        print("categorias unicas",data[i].nunique())
            else:
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


# each of the following functions is a step in the process of data preprocessing
# the DataPreprocessor class is in charge of coordinating the execution of these functions
# the functions are defined in the preprocess_input1.py file

def procedures_data(type_p, doc_path, admissions_path, patients_path, numerical_cols, n, categorical_cols, 
                    normalize_matrix, log_transformation, encode_categorical, final_preprocessing,
                    columns_to_drop, cols_to_accumulate, feature_accumulative_path, save_accumulate_df,plot_matrix,real,level,read_archivo_threshold):
    name = real
    prepomax = 'std'
    name = "ICD9_CODE"
 
    

    
    
    preprocessor = DataPreprocessor(type_p,
                                    doc_path,
                                    admissions_path, 
                                    patients_path,
                                    categorical_cols,
                                    real,
                                    level, 
                                    numerical_cols,
                                    prepomax,
                                    name,
                                    n,
                                    columns_to_drop,
                                    cols_to_accumulate,
                                    feature_accumulative_path,
                                    save_accumulate_df,
                                    cols_to=None,
                                    normalize_matrix=normalize_matrix, 
                                    log_transformation=log_transformation,
                                    encode_categorical=encode_categorical, 
                                    final_preprocessing=final_preprocessing,
                                    proportion=False,
                                    plot_matrix=plot_matrix,read_archivo_threshold=read_archivo_threshold)
    df_final = preprocessor.run(type_p)
    df_final.to_csv("/Users/cynthiagarcia/Desktop/Synthetic-Data-Deep-Learning/data/intermedi/SD/inpput/second/representation2/procedures_"+real + "_" + type_p + "_non_prepo.csv")
    print(" Procedures Done")
    
    
def drugs_data(type_p, doc_path,
               admissions_path, patients_path, 
               numerical_cols, n, categorical_cols, 
               normalize_matrix, log_transformation, 
               encode_categorical, final_preprocessing,
               columns_to_drop, cols_to_accumulate,
               feature_accumulative_path,
               save_accumulate_df,plot_matrix,real,level,read_archivo_threshold):
    prepomax = 'std'
    name = "DRUG"
 

    save_prproces=True
    preprocessor = DataPreprocessor(
        type_p, doc_path, admissions_path, patients_path, categorical_cols, 
        real, level, numerical_cols, prepomax, name, n, 
        columns_to_drop, cols_to_accumulate, feature_accumulative_path, save_accumulate_df,
        cols_to=None, normalize_matrix=normalize_matrix, 
        log_transformation=log_transformation, encode_categorical=encode_categorical, 
        final_preprocessing=final_preprocessing, proportion=False,plot_matrix=plot_matrix,save_prproces=True,
    read_archivo_threshold=read_archivo_threshold
    )
    #aux = preprocessor.load_data_clean_data(type_p)
    df_final = preprocessor.run(type_p)
    df_final.to_csv("/Users/cynthiagarcia/Desktop/Synthetic-Data-Deep-Learning/data/intermedi/SD/inpput/second/representation2/drugs_" + real + "_" + type_p + "_non_prepo.csv")
    print(" Drugs Done")
    
def diagnosis_data(type_p, doc_path, admissions_path, patients_path, numerical_cols, n, categorical_cols,
                   normalize_matrix, log_transformation,
                   encode_categorical, final_preprocessing, 
                   columns_to_drop, cols_to_accumulate, feature_accumulative_path,
                   save_accumulate_df,plot_matrix,real,level,read_archivo_threshold,):
    type_p = "diagnosis"
    name = "ICD9_CODE"
    prepomax = 'std'

 
    preprocessor = DataPreprocessor(type_p, doc_path, admissions_path, patients_path, categorical_cols, real, level, numerical_cols, prepomax, name, n, 
                                    columns_to_drop, cols_to_accumulate, feature_accumulative_path, save_accumulate_df,
                                    cols_to=None, normalize_matrix=normalize_matrix, log_transformation=log_transformation, 
                                    encode_categorical=encode_categorical, final_preprocessing=final_preprocessing, proportion=False,plot_matrix=plot_matrix,
                                    read_archivo_threshold=read_archivo_threshold)
    df_final = preprocessor.run(type_p)
    df_final.to_csv("/Users/cynthiagarcia/Desktop/Synthetic-Data-Deep-Learning/data/intermedi/SD/inpput/second/representation2/diagnosis_" + real + "_" + type_p + "_non_prepo.csv")
    print(" Diagnosis Done")

    

def run_preprocessing(
    admissions_path, 
    patients_path,
    type_p="diagnosis", 
    normalize_matrix=False,
    log_transformation=False,
    encode_categorical=True, 
    final_preprocessing=False,
    mimic_path='data/raw/MIMIC/',
    numerical_cols=['Age_max',  'LOSRD_avg', 'days from last visit'],
    n=[.88, 0.9,.95, .98, .999],
    categorical_cols=['RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'GENDER','DISCHTIME'],
    MIMIC=Path('data/raw/MIMIC/'),
    columns_to_drop=[],
    cols_to_accumulate=[],
    feature_accumulative_path="",
    save_accumulate_df=False,
    plot_matrix =True,
    real="CCS CODES",level="Otro",
    read_archivo_threshold=True

):
    
    print(f"Process: {type_p}")
    print(f"Standard matrix: {normalize_matrix}")
    print(f"Logarithmic transformation: {log_transformation}")
    print(f"Codify categorical variables: {encode_categorical}")
    print(f"Final preprocessing: {final_preprocessing}")
    print(f"MIMIC path: {MIMIC}")
    print(f"Numerical columns: {numerical_cols}")
    print(f"n values: {n}")
    print(f"Categorical columns: {categorical_cols}")
    
    if type_p == "diagnosis":       
        doc_path = MIMIC/'DIAGNOSES_ICD.csv.gz'  
        list_ral = ['threshold_'+ str(i)  for i in n  ] + ["CCS CODES","LEVE3 CODES"] 
        for real in ["CCS CODES"]:
      
            diagnosis_data(type_p, doc_path, admissions_path, patients_path, numerical_cols, n, categorical_cols, 
                        normalize_matrix, log_transformation, encode_categorical, final_preprocessing,
                        columns_to_drop, cols_to_accumulate, feature_accumulative_path, save_accumulate_df,plot_matrix,real,level,read_archivo_threshold)

    elif type_p == "procedures":
        doc_path = MIMIC/'PROCEDURES_ICD.csv.gz'
        list_ral = ['threshold_'+ str(i)  for i in n  ] + ["CCS CODES"] 
        for real in ["CCS CODES"]:
            procedures_data(type_p, doc_path, admissions_path, patients_path, numerical_cols, n, categorical_cols, 
                            normalize_matrix, log_transformation, encode_categorical, final_preprocessing,
                            columns_to_drop, cols_to_accumulate, feature_accumulative_path, save_accumulate_df,plot_matrix,real,level,read_archivo_threshold)

    elif type_p in ["drugst", "drug2"]:
        doc_path = MIMIC/'PRESCRIPTIONS.csv.gz'
        list_ral = ['threshold_presence_'+ str(i)  for i in n ]
        for real in ['threshold_presence_0.8']:
            drugs_data(type_p, doc_path, admissions_path, patients_path, numerical_cols, n, categorical_cols, 
                    normalize_matrix, log_transformation, encode_categorical, final_preprocessing,
                    columns_to_drop, cols_to_accumulate, feature_accumulative_path, save_accumulate_df,plot_matrix,real,level,read_archivo_threshold)
    else:
        print("Tipo de procesamiento no reconocido.")

if __name__ == "__main__":
    import argparse
    import os
    
    import sys
    sys.path.append('')
    sys.path.append('preprocessing')
    import config 
    #args

    MIMIC = Path('data/raw/MIMIC/')
    
    admissions_path = MIMIC/'ADMISSIONS.csv.gz'
    patients_path = MIMIC /'PATIENTS.csv.gz'
    numerical_cols =  ['Age_max',
            'LOSRD_avg','days from last visit']        
    n = [0.8,.9,.95,.98,.999]
    categorical_cols = [  'RELIGION',
                    'MARITAL_STATUS',  'ETHNICITY','GENDER']
    read_archivo_threshold=True
    plot_matrix=False
    level = "visit"   #"Patient" "visit"
    print(level)
    run_preprocessing(
        admissions_path, 
        patients_path,
        type_p="drugst", #diagnosis, procedures, drugst, drug2
        normalize_matrix=False,
        log_transformation=False,
        encode_categorical=True,
        final_preprocessing=False,
        mimic_path='data/raw/MIMIC/',
        numerical_cols=['Age_max', 'LOSRD_avg', 'days from last visit'],
        n=[0.8,0.85,.9, .95, .98, .999],
        categorical_cols=[ 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'GENDER'],
        MIMIC=Path('data/raw/MIMIC/'),
        columns_to_drop=[],
        cols_to_accumulate=[],
        feature_accumulative_path="",
        save_accumulate_df=False,
        plot_matrix =plot_matrix,level=level,
        read_archivo_threshold=read_archivo_threshold
    )
  


