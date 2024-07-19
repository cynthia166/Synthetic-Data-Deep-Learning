import pandas as pd
import numpy as np
import os
os.chdir("/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import sys
sys.path.append('generative_model/SD/model')
from generative_model.SD.model.utils_coupling import *
import logging




class creating_SyntheticSubject:
    def __init__(self, train_ehr_dataset, 
                 test_ehr_dataset,
                 synthetic_ehr_dataset,
                 file_data,
                 folder,
                 name
                ):
        logging.basicConfig(level=logging.INFO)
        self.train_ehr_dataset = train_ehr_dataset
        self.test_ehr_dataset = test_ehr_dataset
        self.synthetic_ehr_dataset = synthetic_ehr_dataset
        self.file_data = file_data
        self.folder =folder
        self.name = name
       
    def initiate_processing(self,):
        logging.info(f'Generating similar synthetic patients with cosine-similitud')
        self.new_synthetic_df   = self.get_most_similar_visit_syntheticehr(self)
       
    def get_most_similar_visit_syntheticehr(self):
        
        feature_columns = [col for col in self.train_ehr_dataset.columns if col not in ['id_patient', 'ADMITTIME', 'visit_rank', 'days from last visit']]
        feature_comunes = self.train_ehr_dataset[feature_columns].columns.intersection(self.synthetic_ehr_dataset.columns)
        # Normalize features
               
        # Create new synthetic dataset with matched visits
        new_synthetic_df=find_similar_admissions(self.train_ehr_dataset, self.synthetic_ehr_dataset, feature_comunes) 
        logging.info(f"Matching complete. ")    
        save_pkl(new_synthetic_df,self.file_data+self.folder+self.name)
        
        return new_synthetic_df
        
                    
    
    