import numpy as np
import pandas as pd
from datetime import timedelta
from pandas import Timedelta
import os
import sys
import logging
print(os.getcwd())
sys.path.append('../../')
os.chdir("/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/")
import gzip
import pickle
from pandas import Timedelta
import sys
sys.path.append('/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/')

#from evaluation.resemb.resemblance.utilsstats import *
#from evaluation.resemb.resemblance.utilsstats import obtain_dataset_admission_visit_rank, obtain_readmission_ehrs

def obtain_readmission_realdata(total_fetura_valid):
    # Ordenando el DataFrame por 'id_' y 'visit_rank'
    total_fetura_valid = total_fetura_valid.sort_values(by=['id_patient', 'visit_rank'])
    # Crear una nueva columna 'readmission'
    # Comparamos si el siguiente 'visit_rank' es mayor que el actual para el mismo 'id_'
    total_fetura_valid['readmission'] = total_fetura_valid.groupby('id_patient')['visit_rank'].shift(-1).notna().astype(int)  
    return  total_fetura_valid

def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

def replace_zero_days(row):
    if row['visit_rank'] > 1 and row['days from last visit'] == 0:
        return  1 # Reemplazar con un número pequeño, p. ej., 1
    else:
        return row['days from last visit']

def replace_zero_days_mean(row, mean_days_per_patient):
    
    if row['visit_rank'] > 1 and row['days from last visit'] == 0:
        # Check if the patient has more than one visit
        
        return mean_days_per_patient[row.name]  # Replace with the patient's mean
        
    else:
        return row['days from last visit']
    

def generate_patient_ids(real_df, synthetic_df):
    # Analyze the distribution of visits per patient in the real dataset
    visits_per_patient = real_df.groupby('id_patient').size().sort_values(ascending=False)
    
    # Calculate how many times we need to repeat the pattern
    total_synthetic_visits = len(synthetic_df)
    total_real_visits = visits_per_patient.sum()
    repetitions = total_synthetic_visits // total_real_visits
    remaining_visits = total_synthetic_visits % total_real_visits
    
    # Generate patient IDs
    patient_ids = []
    current_id = 1
    
    # Repeat the pattern of visits
    for _ in range(repetitions):
        for num_visits in visits_per_patient:
            patient_ids.extend([current_id] * num_visits)
            current_id += 1
    
    # Add remaining visits
    if remaining_visits > 0:
        for num_visits in visits_per_patient:
            if remaining_visits >= num_visits:
                patient_ids.extend([current_id] * num_visits)
                current_id += 1
                remaining_visits -= num_visits
            else:
                patient_ids.extend([current_id] * remaining_visits)
                break
    
    # Shuffle the patient IDs to randomize the order
    np.random.shuffle(patient_ids)
    
    # Assign these IDs to the synthetic dataset
    synthetic_df['id_patient'] = patient_ids
    
    print(f"Generated {current_id - 1} unique patient IDs for {total_synthetic_visits} synthetic visits.")
    print(f"Original dataset had {len(visits_per_patient)} patients for {total_real_visits} visits.")
    print(f"Pattern of visits per patient was repeated {repetitions} times with {remaining_visits} extra visits.")
    
    return synthetic_df

class EHRDataConstraints:
    """
    Class representing the constraints for EHR data processing.

    This class contains methods to handle and process EHR datasets, including sorting datasets,
    handling categorical data, propagating first visit values, adjusting age and dates,
    removing duplicates, and handling the hospital expire flag.

    Attributes:
        train_ehr_dataset (pandas.DataFrame): The training EHR dataset.
        test_ehr_dataset (pandas.DataFrame): The test EHR dataset.
        synthetic_ehr_dataset (pandas.DataFrame): The synthetic EHR dataset.

    Methods:
        print_shapes(): Prints the shapes of the test, train, and synthetic EHR datasets.
        initiate_processing(): Initiates the processing of the synthetic EHR dataset.
        sort_datasets(): Sorts the train and synthetic EHR datasets.
        handle_categorical_data(): Handles categorical data in the train EHR dataset.
        propagate_first_visit_values(): Propagates the first visit values for each column in the synthetic EHR dataset.
        adjust_age_and_dates(): Adjusts the age and admission dates in the synthetic EHR dataset.
        deal_duplicate_samenumcumusumdayafter(): Removes duplicate rows from the synthetic EHR dataset.
        handle_hospital_expire_flag(): Handles the hospital expire flag in the synthetic EHR dataset.
    """

    def __init__(self, train_ehr_dataset, 
                 test_ehr_dataset,
                 synthetic_ehr_dataset,
                 columns_to_drop,
                 columns_to_drop_syn,
                 cols_continuous ,
                 create_visit_rank_col,
                propagate_fistvisit_categoricaldata,
                adjust_age_and_dates_get,
                get_remove_duplicates,
                get_handle_hospital_expire_flag,
                get_0_first_visit,
                get_sample_synthetic_similar_real,
                create_days_between_visits_by_date_var
                 ,eliminate_negatives_var=True ,
                 type_archivo = 'ARFpkl',
                 invert_normalize = False,
                 subject_continous = False
                 
                 ):
        self.train_ehr_dataset = train_ehr_dataset
        self.test_ehr_dataset = test_ehr_dataset
        self.synthetic_ehr_dataset = synthetic_ehr_dataset
        self.columns_to_drop = columns_to_drop
        self.columns_to_drop_syn = columns_to_drop_syn
        self.cols_continuous = cols_continuous
        self.create_visit_rank_col = create_visit_rank_col
        self.propagate_fistvisit_categoricaldata = propagate_fistvisit_categoricaldata
        self.adjust_age_and_dates_get = adjust_age_and_dates_get
        self.get_remove_duplicates = get_remove_duplicates
        self.get_handle_hospital_expire_flag = get_handle_hospital_expire_flag
        self.get_0_first_visit = get_0_first_visit
        self.get_sample_synthetic_similar_real = get_sample_synthetic_similar_real
        self.eliminate_negatives_var = eliminate_negatives_var
        self.type_archivo = type_archivo
        self.inver_normalize = invert_normalize
        self.subject_continous = subject_continous
        self.create_days_between_visits_by_date_var = create_days_between_visits_by_date_var
        


    def print_shapes(self):
        """
        Prints the shapes of the test, train, and synthetic EHR datasets.

        Returns:
            None
        """
        print(self.test_ehr_dataset.shape)
        print(self.train_ehr_dataset.shape)
        print(self.synthetic_ehr_dataset.shape)

    def initiate_processing(self):
        """
        Initiates the processing of the synthetic EHR dataset.

        This method calls various processing methods in a specific order to process the synthetic EHR dataset.
        The processed synthetic EHR dataset is returned.

        Returns:
            pandas.DataFrame: The processed synthetic EHR dataset.
        """
        #i var a normalized in the synthetic real models
        if self.inver_normalize:
           self.inver_normalization()
        #if var are continous   
        if self.subject_continous:
            self.subject_continous_fun()
         # Change the column name 'SUBJECT_ID' to 'id_patient'        
        self.change_subject_id()       
        if self.get_sample_synthetic_similar_real:    
            self.have_same_patients_similar_num_visits() 
        self.get_admitted_time_datetime()
        if self.create_visit_rank_col:
             self.obtain_readmission_ehrs()
        #self.create_days_between_visits_by_date()          
        self.log_negative_values()
        self.sort_datasets()
        if self.eliminate_negatives_var:
            self.clip_0_negative_value()
        if self.propagate_fistvisit_categoricaldata:    
            self.handle_categorical_data()
            self.propagate_first_visit_values()
        #0 in the firs viit    
        if self.get_0_first_visit:
            self.add_0_between_days_if_visit1()    
        #mean values for days from last visit (fix days from last visit)    
        if self.get_remove_duplicates:   
            self.deal_duplicate_samenumcumusumdayafter()
        #adjust age and dates    
        if self.adjust_age_and_dates_get:    
           self.adjust_age_and_dates()
          
        #adjust hospital expire flag
        if self.get_handle_hospital_expire_flag:    
            self.handle_hospital_expire_flag()
        #elimunate columns    
        self.eliminate_columns()
          
        return self.synthetic_ehr_dataset, self.train_ehr_dataset, self.test_ehr_dataset
    def create_days_between_visits_by_date(self):
        self.synthetic_ehr_dataset['ADMITTIME'] = pd.to_datetime(self.synthetic_ehr_dataset['ADMITTIME'])

# Sort by id_patient and ADMITTIME
        self.synthetic_ehr_dataset = self.synthetic_ehr_dataset.sort_values(by=['id_patient', 'ADMITTIME'])

        # Calculate days between visits
        self.synthetic_ehr_dataset['days_from_last_visit2'] = self.synthetic_ehr_dataset.groupby('id_patient')['ADMITTIME'].diff().dt.days

        # Fill NaN values with 0 for the first visit of each patient
        self.synthetic_ehr_dataset['days_from_last_visit2'] = self.synthetic_ehr_dataset['days_from_last_visit2'].fillna(0).astype(int)

    def subject_continous_fun(self):
        
        self.synthetic_ehr_dataset["SUBJECT_ID"] = self.synthetic_ehr_dataset["SUBJECT_ID"].apply(lambda x: abs(int(round(x))))
        
        
    def inver_normalization(self):
        for column in self.cols_continuous:
            data = self.synthetic_ehr_dataset[column].values
            data_min = self.train_ehr_dataset[column].values.min()
            data_max = self.train_ehr_dataset[column].values.max()
            self.synthetic_ehr_dataset[column] =  data * (data_max - data_min) + data_min

        
    def change_subject_id(self):
        if  self.get_sample_synthetic_similar_real:
            self.train_ehr_dataset   = self.train_ehr_dataset.rename(columns={"SUBJECT_ID":"id_patient"})
            self.test_ehr_dataset   = self.test_ehr_dataset.rename(columns={"SUBJECT_ID":"id_patient"})
            try:
                self.synthetic_ehr_dataset   = self.synthetic_ehr_dataset.rename(columns={"days_between_visits":"days from last visit"})
                self.train_ehr_dataset   = self.train_ehr_dataset.rename(columns={"days_between_visits":"days from last visit"})
                self.test_ehr_dataset   = self.test_ehr_dataset.rename(columns={"days_between_visits":"days from last visit"})
         
            except:
                pass
        else:
       
            self.synthetic_ehr_dataset   = self.synthetic_ehr_dataset.rename(columns={"SUBJECT_ID":"id_patient"})
            self.train_ehr_dataset   = self.train_ehr_dataset.rename(columns={"SUBJECT_ID":"id_patient"})
            self.test_ehr_dataset   = self.test_ehr_dataset.rename(columns={"SUBJECT_ID":"id_patient"})
            try:
                self.synthetic_ehr_dataset   = self.synthetic_ehr_dataset.rename(columns={"days_between_visits":"days from last visit"})
                self.train_ehr_dataset   = self.train_ehr_dataset.rename(columns={"days_between_visits":"days from last visit"})
                self.test_ehr_dataset   = self.test_ehr_dataset.rename(columns={"days_between_visits":"days from last visit"})
  
            except:
                pass

        
    def get_admitted_time_datetime(self):
        if self.type_archivo=='ARFpkl':
            #total_features_synthethic['ADMITTIME'] = total_features_synthethic['year'].astype(str) +"-"+ total_features_synthethic['month'].astype(str) +"-"+ '01'
            self.synthetic_ehr_dataset['ADMITTIME'] = self.synthetic_ehr_dataset['year'].astype(int).astype(str) +"-"+ self.synthetic_ehr_dataset['month'].astype(int).astype(str) +"-"+ '01'
            
        self.synthetic_ehr_dataset['ADMITTIME'] = pd.to_datetime(self.synthetic_ehr_dataset['ADMITTIME'])
        self.synthetic_ehr_dataset = self.synthetic_ehr_dataset.sort_values(by=['id_patient', 'ADMITTIME'])
  
    def obtain_readmission_ehrs(self): 
        self.train_ehr_dataset = obtain_readmission_realdata(self.train_ehr_dataset)
        self.test_ehr_dataset = obtain_readmission_realdata(self.test_ehr_dataset) 
        #obtener readmission
        # este es el caso porque e arf se crearon dos columnas de dmitted time para que pudieran ser categorivas
            
            # Ordena el DataFrame por 'patient_id' y 'visit_date' para garantizar el ranking correcto
        # Agrupa por 'patient_id' y asigna un rango a cada visita
        self.synthetic_ehr_dataset['visit_rank'] = self.synthetic_ehr_dataset.groupby('id_patient')['ADMITTIME'].rank(method='first').astype(int)
        self.synthetic_ehr_dataset = obtain_readmission_realdata(self.synthetic_ehr_dataset) 
   
    def log_negative_values(self):
        """
        Logs the presence of negative values in a DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame to check for negative values.
        """
        # Configura el logging
        logging.basicConfig(level=logging.INFO)

        # Verifica si hay valores negativos en el DataFrame
        non_datetime_cols = self.synthetic_ehr_dataset.select_dtypes(exclude=['datetime64']).columns
        for col in non_datetime_cols:
            self.synthetic_ehr_dataset[col] = self.synthetic_ehr_dataset[col].astype(int)
            
     
        num_valores_negativos = self.synthetic_ehr_dataset[non_datetime_cols][self.synthetic_ehr_dataset[non_datetime_cols] < 0].count().sum()

        if num_valores_negativos > 0:
            logging.info(f'Hay {num_valores_negativos} valores negativos en el DataFrame.')
        else:
            logging.info('No hay valores negativos en el DataFrame.')
    
    def clip_0_negative_value(self):

        # Convertir las columnas de tipo 'category' a numérico
        non_datetime_cols = self.synthetic_ehr_dataset.select_dtypes(exclude=['datetime64']).columns

        # Reemplaza los valores negativos por 0 solo en las columnas que no son de tipo datetime
        for col in non_datetime_cols:
            logging.info(f'Columna: {col} numero de valores negativos {self.synthetic_ehr_dataset[col][self.synthetic_ehr_dataset[col] < 0].count()}')
            self.synthetic_ehr_dataset[col] = self.synthetic_ehr_dataset[col].clip(lower=0)
    
    def sort_datasets(self):
        """
        Sorts the train and synthetic EHR datasets.

        The train EHR dataset is sorted by 'id_patient' and 'ADMITTIME' columns.
        The synthetic EHR dataset is sorted by 'id_patient' and 'visit_rank' columns.

        Returns:
            None
        """
         
        self.train_ehr_dataset.sort_values(by=['id_patient', 'ADMITTIME'], inplace=True)
        self.synthetic_ehr_dataset.sort_values(by=['id_patient', 'visit_rank'], inplace=True)

    def handle_categorical_data(self):
        """
        Handles categorical data in the train EHR dataset.

        This method identifies the categorical columns in the train EHR dataset and returns a list of these columns.

        Returns:
            list: The list of categorical columns in the train EHR dataset.
        """
        categorical_cols = ['ADMITTIME', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'GENDER']
        cols_accounts = []
        for col in categorical_cols:
            cols_f = self.synthetic_ehr_dataset.filter(like=col, axis=1).columns
            cols_accounts.extend(list(cols_f))
        return cols_accounts

    def propagate_first_visit_values(self):
        """
        Propagates the first visit values for each column in the synthetic EHR dataset.

        This method handles categorical data and replaces the values in each column with the first visit value
        for each patient in the dataset.

        Returns:
            None
        """
        cols_accounts = self.handle_categorical_data()
        for column in cols_accounts:
            logging.info(f'Propagating first visit values for column: {column}, number of unique values: {self.synthetic_ehr_dataset[column].nunique()}')
            self.synthetic_ehr_dataset[column] = self.synthetic_ehr_dataset.groupby('id_patient')[column].transform('first')
            logging.info(f"Propagated first visit values for column: {column}, number of transformed values: {self.synthetic_ehr_dataset.groupby('id_patient')[column].nunique()}")
    def adjust_age_and_dates(self):
        """
        Adjusts the age and admission dates in the synthetic EHR dataset.

        This method fills missing values in the 'days from last visit' column with 0,
        calculates the cumulative sum of 'days from last visit' for each patient,
        and adjusts the 'Age_max' and 'ADMITTIME' columns based on the cumulative sum.

        The 'Age_max' values are adjusted by adding the cumulative sum divided by 365
        to the first 'Age_max' value for each patient, for visits with a rank greater than 1.
        For visits with a rank of 1, the 'Age_max' value remains unchanged.

        The 'ADMITTIME' values are adjusted by adding the cumulative sum of 'days from last visit'
        shifted by one day to the first 'ADMITTIME' value for each patient, for visits with a rank greater than 1.
        For visits with a rank of 1, the 'ADMITTIME' value remains unchanged.

        Any 'Age_max' values above 100 are truncated to 89.

        Returns:
            None
        """
        #self.synthetic_ehr_dataset['days from last visit'].fillna(0, inplace=True)
        


        self.synthetic_ehr_dataset['days from last visit_cumsum'] = self.synthetic_ehr_dataset.groupby('id_patient')['days from last visit'].cumsum()
        
        first_age_max = self.synthetic_ehr_dataset.groupby('id_patient')['Age_max'].first()
        first_admission = self.synthetic_ehr_dataset.groupby('id_patient')['ADMITTIME'].first()
   
        
        self.synthetic_ehr_dataset.loc[self.synthetic_ehr_dataset['visit_rank'] > 1, 'Age_max'] = self.synthetic_ehr_dataset['id_patient'].map(first_age_max) + self.synthetic_ehr_dataset.groupby('id_patient')['days from last visit_cumsum'].transform(lambda x: x / 365)
        self.synthetic_ehr_dataset.loc[self.synthetic_ehr_dataset['visit_rank'] == 1, 'Age_max'] = self.synthetic_ehr_dataset['Age_max']
        adjusted_rows = (self.synthetic_ehr_dataset['visit_rank'] > 1).sum()
        total_rows = self.synthetic_ehr_dataset.shape[0]
        logging.info(f'Age_max values adjusted for {adjusted_rows} visits with rank greater than 1 ({adjusted_rows/total_rows:.2%})')
        logging.info(f'Age_max values not modified for {total_rows - adjusted_rows} visits ({(total_rows - adjusted_rows)/total_rows:.2%})')
        logging.info(f"truncated patient over age of 89 to 89 {(self.synthetic_ehr_dataset['Age_max'] > 89).sum() }")
        logging.info(f"truncated patient over age of 89 to 89 {(self.synthetic_ehr_dataset['Age_max'] > 89).sum() / self.synthetic_ehr_dataset.shape[0]}")
              # Truncate 'Age_max' values above 100 to 89
        self.synthetic_ehr_dataset.loc[self.synthetic_ehr_dataset['Age_max'] > 89, 'Age_max'] = 89
          #self.synthetic_ehr_dataset.loc[self.synthetic_ehr_dataset['visit_rank'] > 1, 'ADMITTIME'] = self.synthetic_ehr_dataset['id_patient'].map(first_admission) + self.synthetic_ehr_dataset['days from last visit_cumsum'].shift().apply(lambda x: Timedelta(x, unit='D'))
          
        
        self.synthetic_ehr_dataset.loc[self.synthetic_ehr_dataset['visit_rank'] > 1, 'ADMITTIME'] = self.synthetic_ehr_dataset['id_patient'].map(first_admission) + pd.to_timedelta(self.synthetic_ehr_dataset['days from last visit_cumsum'], unit='D')

        self.synthetic_ehr_dataset.loc[self.synthetic_ehr_dataset['visit_rank'] == 1, 'ADMITTIME'] = self.synthetic_ehr_dataset['ADMITTIME']
      
    def deal_duplicate_samenumcumusumdayafter(self):
        """
        Removes duplicate rows from the synthetic EHR dataset based on specific columns.

        This method identifies duplicate rows in the synthetic EHR dataset based on the columns 'ADMITTIME',
        'id_patient', and 'days from last visit_cumsum'. It then removes these duplicate rows from the dataset.

        Returns:
            None
        """
        mean_days_per_patient = self.synthetic_ehr_dataset.groupby('id_patient')['days from last visit'].transform('mean')

        # Calculate the overall mean of "days from last visit"
        overall_mean_days = self.synthetic_ehr_dataset['days from last visit'].mean()
        # Fill missing values in 33"days from last visit" with the overall mean
        logging.info(f"records modified by mean {self.synthetic_ehr_dataset[(self.synthetic_ehr_dataset['days from last visit']==0)&(self.synthetic_ehr_dataset['visit_rank']>1)].shape[0]}, percentage {self.synthetic_ehr_dataset[(self.synthetic_ehr_dataset['days from last visit']==0)&(self.synthetic_ehr_dataset['visit_rank']>1)].shape[0]/self.synthetic_ehr_dataset.shape[0]}")
        self.synthetic_ehr_dataset['days from last visit'] = self.synthetic_ehr_dataset.apply(replace_zero_days_mean, axis=1, args=(mean_days_per_patient))
        logging.info(f"mean_days_per_patient {self.synthetic_ehr_dataset['days from last visit'].mean()} overall_mean_days {overall_mean_days}")
        
        # fil with adde 1 day instead of 0
        # filtered_rows = self.synthetic_ehr_dataset[(self.synthetic_ehr_dataset['visit_rank'] > 1) & (self.synthetic_ehr_dataset['days from last visit'] == 0)]
        # self.synthetic_ehr_dataset['days from last visit'] = self.synthetic_ehr_dataset.apply(replace_zero_days, axis=1)
        # logging.info(f"the mean od days from lat visit:{ self.synthetic_ehr_dataset['days from last visit'].mean()}" )
        # count = len(filtered_rows)
        # logging.info(f'Filled {count} 0 values in the "days from last visit" column with visit greate than 1., percentage {count/self.synthetic_ehr_dataset.shape[0]}   ')
        # # Coun
        #duplicates = self.synthetic_ehr_dataset.duplicated(subset=['ADMITTIME', 'id_patient', 'days from last visit_cumsum'])
        #logging.info(f'Removing {duplicates.sum()} duplicate rows from the synthetic EHR dataset. percentage {duplicates.sum()/self.synthetic_ehr_dataset.shape[0]}')
        #self.synthetic_ehr_dataset = self.synthetic_ehr_dataset[~duplicates]

    def handle_hospital_expire_flag(self):
        """
        Handles the hospital expire flag in the synthetic EHR dataset.

        This method adjusts the 'HOSPITAL_EXPIRE_FLAG' values in the synthetic EHR dataset based on specific conditions.

        If the 'HOSPITAL_EXPIRE_FLAG' is 1 and the 'ADMITTIME' is not the last admission for a patient,
        the 'HOSPITAL_EXPIRE_FLAG' is set to 0.
        If the 'HOSPITAL_EXPIRE_FLAG' is 1 and the 'days from last visit' is not the maximum value for a patient,
        the 'HOSPITAL_EXPIRE_FLAG' is set to 0.

        Returns:
            None
        """
        last_admission = self.synthetic_ehr_dataset.groupby('id_patient')['ADMITTIME'].transform('max')
        max_days_from_last_visit = self.synthetic_ehr_dataset.groupby('id_patient')['days from last visit'].transform('max')

        shape_last_admission = self.synthetic_ehr_dataset.loc[(self.synthetic_ehr_dataset['HOSPITAL_EXPIRE_FLAG'] == 1) & (self.synthetic_ehr_dataset['ADMITTIME'] != last_admission)].shape[0]
        self.synthetic_ehr_dataset.loc[(self.synthetic_ehr_dataset['HOSPITAL_EXPIRE_FLAG'] == 1) & (self.synthetic_ehr_dataset['ADMITTIME'] != last_admission), 'HOSPITAL_EXPIRE_FLAG'] = 0
        logging.info(f'Adjusted HOSPITAL_EXPIRE_FLAG values based on last admission. {shape_last_admission} percentage {shape_last_admission/self.synthetic_ehr_dataset.shape[0]}')

        shape_dif_days = self.synthetic_ehr_dataset.loc[(self.synthetic_ehr_dataset['HOSPITAL_EXPIRE_FLAG'] == 1) & (self.synthetic_ehr_dataset['days from last visit'] != max_days_from_last_visit)].shape[0]
        self.synthetic_ehr_dataset.loc[(self.synthetic_ehr_dataset['HOSPITAL_EXPIRE_FLAG'] == 1) & (self.synthetic_ehr_dataset['days from last visit'] != max_days_from_last_visit), 'HOSPITAL_EXPIRE_FLAG'] = 0
        logging.info(f'Adjusted HOSPITAL_EXPIRE_FLAG values based on max days between visits. {shape_dif_days} percentage {shape_dif_days/self.synthetic_ehr_dataset.shape[0]}')        
    def add_0_between_days_if_visit1(self):
        non_0_first_visit = self.synthetic_ehr_dataset.loc[(self.synthetic_ehr_dataset['visit_rank'] == 1) & (self.synthetic_ehr_dataset['days from last visit'] > 0.0), 'days from last visit']
        self.synthetic_ehr_dataset.loc[(self.synthetic_ehr_dataset['visit_rank'] == 1) & (self.synthetic_ehr_dataset['days from last visit'] > 0.0), 'days from last visit'] = 0
        logging.info(f'Added 0 days between visits for first visit with non-zero days between visits. {len(non_0_first_visit)}, percentage {len(non_0_first_visit)/self.synthetic_ehr_dataset.shape[0]}')
            
    def have_same_patients_similar_num_visits(self):
     # Analyze the distribution of visits per patient in the real dataset
        self.synthetic_ehr_dataset = generate_patient_ids(self.train_ehr_dataset, self.synthetic_ehr_dataset)
        logging.info(f'Distribution synthetic visits: {self.synthetic_ehr_dataset.groupby("id_patient").size().sort_values(ascending=False)}')
        logging.info(f'Distribution real visits: {self.train_ehr_dataset.groupby("id_patient").size().sort_values(ascending=False)}')   
        
    def eliminate_columns(self):
        self.synthetic_ehr_dataset.drop(columns=self.columns_to_drop_syn, inplace=True)
        self.train_ehr_dataset.drop(columns=self.columns_to_drop, inplace=True)
        self.test_ehr_dataset.drop(columns=self.columns_to_drop, inplace=True)
 
if __name__ == '__main__':
    
    features_path = "data/intermedi/SD/inpput/entire_ceros_tabular_data.pkl"

    #file = 'generated_synthcity_tabular/arftotal_0.2_epochs.pkl'
    file = '/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/synthetic_data_generative_model_arf_per_0.7.pkl'
    valid_perc=.3
    test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset,features = obtain_dataset_admission_visit_rank(sample_patients_path,file,valid_perc,features_path,'ARFpkl')
    ###DRAFT
    #remplazar valores negativos con ero
    
    #synthetic_ehr_dataset = load_pickle(file)     
    
    c = EHRDataConstraints(train_ehr_dataset, test_ehr_dataset, synthetic_ehr_dataset,True)
    c.print_shapes()
    #cols_accounts = c.handle_categorical_data()
    processed_synthetic_dataset = c.initiate_processing()
    c.print_shapes()
    
    aun = synthetic_ehr_dataset[synthetic_ehr_dataset["visit_rank"]>2]
    auz = synthetic_ehr_dataset
    auz["visit_rank"].value_counts()
    i =aun["id_patient"].unique()
    for id_paciente in i[180:190]:
        print(processed_synthetic_dataset[processed_synthetic_dataset["id_patient"]==id_paciente][ ['visit_rank','ADMITTIME','days from last visit_cumsum','HOSPITAL_EXPIRE_FLAG','Age_max']])
    for id_paciente in i[180:190]:
        print(processed_synthetic_dataset[processed_synthetic_dataset["id_patient"]==id_paciente][cols_accounts])
    
    #falt rebisar que cpigo permanece constante en historia de personas-ƒ