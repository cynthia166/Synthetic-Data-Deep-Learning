import pandas as pd

import pandas as pd
from datetime import timedelta
from pandas import Timedelta
import os
import sys
import logging
print(os.getcwd())
sys.path.append('../../')
os.chdir("/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/")

import sys
sys.path.append('/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/')
import evaluation.resemb.resemblance.config
from evaluation.resemb.resemblance.utilsstats import *
from evaluation.resemb.resemblance.utilsstats import obtain_dataset_admission_visit_rank


def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)



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
        remove_duplicates(): Removes duplicate rows from the synthetic EHR dataset.
        handle_hospital_expire_flag(): Handles the hospital expire flag in the synthetic EHR dataset.
    """

    def __init__(self, train_ehr_dataset, test_ehr_dataset, synthetic_ehr_dataset,eliminate_negatives_var = False):
        self.train_ehr_dataset = train_ehr_dataset
        self.test_ehr_dataset = test_ehr_dataset
        self.synthetic_ehr_dataset = synthetic_ehr_dataset
        self.eliminate_negatives_var = eliminate_negatives_var

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
        self.log_negative_values()
        self.sort_datasets()
        if self.eliminate_negatives_var == True:
            
            self.eliminate_negative_values()
        self.handle_categorical_data()
        self.propagate_first_visit_values()
        self.adjust_age_and_dates()
        self.remove_duplicates()
        self.handle_hospital_expire_flag()
        return self.synthetic_ehr_dataset
    
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
    
    def eliminate_negative_values(self):

        # Convertir las columnas de tipo 'category' a numérico
        non_datetime_cols = self.synthetic_ehr_dataset.select_dtypes(exclude=['datetime64']).columns

        # Reemplaza los valores negativos por 0 solo en las columnas que no son de tipo datetime
        for col in non_datetime_cols:
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
            cols_f = self.train_ehr_dataset.filter(like=col, axis=1).columns
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
            self.synthetic_ehr_dataset[column] = self.synthetic_ehr_dataset.groupby('id_patient')[column].transform('first')

    def adjust_age_and_dates(self):
        """
        Adjusts the age and admission dates in the synthetic EHR dataset.

        This method fills missing values in the 'days_between_visits' column with 0,
        calculates the cumulative sum of 'days_between_visits' for each patient,
        and adjusts the 'Age_max' and 'ADMITTIME' columns based on the cumulative sum.

        The 'Age_max' values are adjusted by adding the cumulative sum divided by 365
        to the first 'Age_max' value for each patient, for visits with a rank greater than 1.
        For visits with a rank of 1, the 'Age_max' value remains unchanged.

        The 'ADMITTIME' values are adjusted by adding the cumulative sum of 'days_between_visits'
        shifted by one day to the first 'ADMITTIME' value for each patient, for visits with a rank greater than 1.
        For visits with a rank of 1, the 'ADMITTIME' value remains unchanged.

        Any 'Age_max' values above 100 are truncated to 89.

        Returns:
            None
        """
        self.synthetic_ehr_dataset['days_between_visits'].fillna(0, inplace=True)
        self.synthetic_ehr_dataset['days_between_visits_cumsum'] = self.synthetic_ehr_dataset.groupby('id_patient')['days_between_visits'].cumsum()

        first_age_max = self.synthetic_ehr_dataset.groupby('id_patient')['Age_max'].first()
        first_admission = self.synthetic_ehr_dataset.groupby('id_patient')['ADMITTIME'].first()

        self.synthetic_ehr_dataset.loc[self.synthetic_ehr_dataset['visit_rank'] > 1, 'Age_max'] = self.synthetic_ehr_dataset['id_patient'].map(first_age_max) + (self.synthetic_ehr_dataset.groupby('id_patient')['days_between_visits_cumsum'].shift() / 365)
        self.synthetic_ehr_dataset.loc[self.synthetic_ehr_dataset['visit_rank'] == 1, 'Age_max'] = self.synthetic_ehr_dataset['Age_max']

        # Truncate 'Age_max' values above 100 to 89
        self.synthetic_ehr_dataset.loc[self.synthetic_ehr_dataset['Age_max'] > 100, 'Age_max'] = 89

        self.synthetic_ehr_dataset.loc[self.synthetic_ehr_dataset['visit_rank'] > 1, 'ADMITTIME'] = self.synthetic_ehr_dataset['id_patient'].map(first_admission) + self.synthetic_ehr_dataset['days_between_visits_cumsum'].shift().apply(lambda x: Timedelta(x, unit='D'))
        self.synthetic_ehr_dataset.loc[self.synthetic_ehr_dataset['visit_rank'] == 1, 'ADMITTIME'] = self.synthetic_ehr_dataset['ADMITTIME']

    def remove_duplicates(self):
        """
        Removes duplicate rows from the synthetic EHR dataset based on specific columns.

        This method identifies duplicate rows in the synthetic EHR dataset based on the columns 'ADMITTIME',
        'id_patient', and 'days_between_visits_cumsum'. It then removes these duplicate rows from the dataset.

        Returns:
            None
        """
        duplicates = self.synthetic_ehr_dataset.duplicated(subset=['ADMITTIME', 'id_patient', 'days_between_visits_cumsum'])
        self.synthetic_ehr_dataset = self.synthetic_ehr_dataset[~duplicates]

    def handle_hospital_expire_flag(self):
        """
        Handles the hospital expire flag in the synthetic EHR dataset.

        This method adjusts the 'HOSPITAL_EXPIRE_FLAG' values in the synthetic EHR dataset based on specific conditions.

        If the 'HOSPITAL_EXPIRE_FLAG' is 1 and the 'ADMITTIME' is not the last admission for a patient,
        the 'HOSPITAL_EXPIRE_FLAG' is set to 0.
        If the 'HOSPITAL_EXPIRE_FLAG' is 1 and the 'days_between_visits' is not the maximum value for a patient,
        the 'HOSPITAL_EXPIRE_FLAG' is set to 0.

        Returns:
            None
        """
        last_admission = self.synthetic_ehr_dataset.groupby('id_patient')['ADMITTIME'].transform('max')
        max_days_between_visits = self.synthetic_ehr_dataset.groupby('id_patient')['days_between_visits'].transform('max')

        self.synthetic_ehr_dataset.loc[(self.synthetic_ehr_dataset['HOSPITAL_EXPIRE_FLAG'] == 1) & (self.synthetic_ehr_dataset['ADMITTIME'] != last_admission), 'HOSPITAL_EXPIRE_FLAG'] = 0
        self.synthetic_ehr_dataset.loc[(self.synthetic_ehr_dataset['HOSPITAL_EXPIRE_FLAG'] == 1) & (self.synthetic_ehr_dataset['days_between_visits'] != max_days_between_visits), 'HOSPITAL_EXPIRE_FLAG'] = 0


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
        print(processed_synthetic_dataset[processed_synthetic_dataset["id_patient"]==id_paciente][ ['visit_rank','ADMITTIME','days_between_visits_cumsum','HOSPITAL_EXPIRE_FLAG','Age_max']])
    for id_paciente in i[180:190]:
        print(processed_synthetic_dataset[processed_synthetic_dataset["id_patient"]==id_paciente][cols_accounts])
    
    #falt rebisar que cpigo permanece constante en historia de personas-ƒ