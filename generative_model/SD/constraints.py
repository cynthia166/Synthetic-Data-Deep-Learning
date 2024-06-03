import pandas as pd

import pandas as pd
from datetime import timedelta
from pandas import Timedelta
import os
import sys

print(os.getcwd())
sys.path.append('../../')
os.chdir("/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/")
#from evaluation.resemb.resemblance.metric_stat import *
def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)



class EHRDataConstraints:
    def __init__(self, train_ehr_dataset, test_ehr_dataset, synthetic_ehr_dataset):
        self.train_ehr_dataset = train_ehr_dataset
        self.test_ehr_dataset = test_ehr_dataset
        self.synthetic_ehr_dataset = synthetic_ehr_dataset  # Asegúrate de definir este atributo
        

    def print_shapes(self):
        print(self.test_ehr_dataset.shape)
        print(self.train_ehr_dataset.shape)
        print(self.synthetic_ehr_dataset.shape)

    def initiate_processing(self):
        self.sort_datasets()
        self.handle_categorical_data()
        self.propagate_first_visit_values()
        self.adjust_age_and_dates()
        self.remove_duplicates()
        self.handle_hospital_expire_flag()
        return self.synthetic_ehr_dataset

    def sort_datasets(self):
        self.train_ehr_dataset.sort_values(by=['id_patient', 'ADMITTIME'], inplace=True)
        self.synthetic_ehr_dataset.sort_values(by=['id_patient', 'visit_rank'], inplace=True)

    def handle_categorical_data(self):
        categorical_cols = ['ADMITTIME', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'GENDER']
        cols_accounts = []
        for col in categorical_cols:
            cols_f = self.train_ehr_dataset.filter(like=col, axis=1).columns
            cols_accounts.extend(list(cols_f))
        return cols_accounts

    def propagate_first_visit_values(self):
        cols_accounts = self.handle_categorical_data()
        for column in cols_accounts:
            self.synthetic_ehr_dataset[column] = self.synthetic_ehr_dataset.groupby('id_patient')[column].transform('first')

    def adjust_age_and_dates(self):
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
        duplicates = self.synthetic_ehr_dataset.duplicated(subset=['ADMITTIME', 'id_patient', 'days_between_visits_cumsum'])
        self.synthetic_ehr_dataset = self.synthetic_ehr_dataset[~duplicates]

    def handle_hospital_expire_flag(self):
        last_admission = self.synthetic_ehr_dataset.groupby('id_patient')['ADMITTIME'].transform('max')
        max_days_between_visits = self.synthetic_ehr_dataset.groupby('id_patient')['days_between_visits'].transform('max')

        self.synthetic_ehr_dataset.loc[(self.synthetic_ehr_dataset['HOSPITAL_EXPIRE_FLAG'] == 1) & (self.synthetic_ehr_dataset['ADMITTIME'] != last_admission), 'HOSPITAL_EXPIRE_FLAG'] = 0
        self.synthetic_ehr_dataset.loc[(self.synthetic_ehr_dataset['HOSPITAL_EXPIRE_FLAG'] == 1) & (self.synthetic_ehr_dataset['days_between_visits'] != max_days_between_visits), 'HOSPITAL_EXPIRE_FLAG'] = 0


if __name__ == '__main__':
    
    features_path = "data/intermedi/SD/inpput/entire_ceros_tabular_data.pkl"

    file = 'generated_synthcity_tabular/arftotal_0.2_epochs.pkl'

    test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset,features = obtain_dataset_admission_visit_rank(file,file,valid_perc,features_path)
            
    
    c = EHRDataConstraints(train_ehr_dataset, test_ehr_dataset, synthetic_ehr_dataset)
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