

import pandas as pd
import numpy as np
from scipy import stats
import random
from config_gru import *
from utilsstats import *

class SequenceGenerator:
    def __init__(self, model, real_data_path, 
                 synthetic_data_gru,
                 synthetic_data_arf):
        self.model = model  # This should be your trained GRU model
        self.real_data = pd.read_csv(real_data_path)
        self.synthetic_data_gru = pd.read_csv(synthetic_data_gru)
        self.synthetic_data_arf = pd.read_csv(synthetic_data_arf)
        self.visit_count_distribution = None
        self.days_between_visits_distribution = None

    def fit_distributions(self):
        # Fit distribution for number of visits
        visit_counts = self.real_data.groupby('patient_id').size()
        self.visit_count_distribution = stats.rv_discrete(values=(visit_counts.values, visit_counts.index))

        # Fit distribution for days between visits
        days_between = self.real_data.groupby('patient_id')['days_from_last_visit'].diff().dropna()
        self.days_between_visits_distribution = stats.rv_continuous().fit(days_between)

    def generate_sequences(self, n_patients_with_sequences=7000):
        # Randomly select n_patients_with_sequences
        patients_with_sequences = self.synthetic_data.sample(n=n_patients_with_sequences)
        patients_with_sequences['visit_rank'] = 1

        all_visits = [patients_with_sequences]

        for _, patient in patients_with_sequences.iterrows():
            n_visits = self.visit_count_distribution.rvs()
            
            if n_visits > 1:
                context = patient[self.model.context_columns].values.tolist()
                first_sequence = {
                    'context': context,
                    'data': [patient[self.model.data_columns].values.tolist()]
                }
                
                subsequent_sequences = self.model.generate_subsequent_sequences(first_sequence, n_visits - 1)
                
                for i, seq in enumerate(subsequent_sequences[1:], start=2):
                    visit = pd.DataFrame([seq['data']], columns=self.model.data_columns)
                    visit['patient_id'] = patient['patient_id']
                    visit['visit_rank'] = i
                    all_visits.append(visit)

        # Concatenate all visits
        result = pd.concat(all_visits, ignore_index=True)

        # Add days_from_last_visit
        result['days_from_last_visit'] = result.groupby('patient_id').apply(
            lambda x: self._generate_days_between_visits(len(x))
        ).reset_index(level=0, drop=True)

        # Update age for subsequent visits
        result['age'] = result.apply(
            lambda row: self._update_age(row['age'], row['days_from_last_visit'], row['visit_rank']),
            axis=1
        )

        # Assign demographic variables to subsequent visits
        demographic_cols = [col for col in result.columns if col not in self.model.data_columns + ['visit_rank', 'days_from_last_visit']]
        result[demographic_cols] = result.groupby('patient_id')[demographic_cols].transform('first')

        return result

    def generate_single_visit_patients(self, n_patients_single_visit):
        # Select patients not already used in sequences
        remaining_patients = self.synthetic_data[~self.synthetic_data['patient_id'].isin(self.patients_with_sequences['patient_id'])]
        
        # Randomly select n_patients_single_visit
        single_visit_patients = remaining_patients.sample(n=n_patients_single_visit)
        single_visit_patients['visit_rank'] = 1
        single_visit_patients['days_from_last_visit'] = 0

        return single_visit_patients

    def _generate_days_between_visits(self, n_visits):
        if n_visits == 1:
            return [0]
        else:
            return [0] + list(self.days_between_visits_distribution.rvs(size=n_visits-1))

    def _update_age(self, initial_age, days_since_last_visit, visit_rank):
        if visit_rank == 1:
            return initial_age
        else:
            return initial_age + (days_since_last_visit / 365.25)

    def run(self, n_patients_with_sequences=7000, n_patients_single_visit=3000):
        self.fit_distributions()
        
        # Generate sequences
        patients_with_sequences = self.generate_sequences(n_patients_with_sequences)
        self.patients_with_sequences = patients_with_sequences  # Store this for later use
        
        # Generate single-visit patients
        single_visit_patients = self.generate_single_visit_patients(n_patients_single_visit)
        
        # Concatenate both datasets
        final_dataset = pd.concat([patients_with_sequences, single_visit_patients], ignore_index=True)
        
        return final_dataset

# Usage:
if __name__=="main":
    generator = SequenceGenerator(real_data_path, 
                 synthetic_data_gru,
                 synthetic_data_arf)
    generated_data = generator.run(n_patients_with_sequences=7000, n_patients_single_visit=3000)
    aux = load_pickle(synthetic_data_gru)