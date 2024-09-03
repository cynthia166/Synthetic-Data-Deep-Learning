import numpy as np
import pandas as pd
from datetime import timedelta
from pandas import Timedelta
import os
import sys
import logging
print(os.getcwd())
sys.path.append('../../')
file_principal = os.getcwd()
os.chdir(file_principal )
import gzip
import pickle
from pandas import Timedelta
import sys
sys.path.append(file_principal)
#from evaluation.resemb.resemblance.utilsstats import *
#from evaluation.resemb.resemblance.utilsstats import obtain_dataset_admission_visit_rank, obtain_readmission_ehrs
column_list =column_list = [
    'GENDER_M',
    'GENDER_F',
    'RELIGION_CATHOLIC',
    'RELIGION_Otra',
    'RELIGION_Unknown',
    'MARITAL_STATUS_0',
    'MARITAL_STATUS_DIVORCED',
    'MARITAL_STATUS_LIFE PARTNER',
    'MARITAL_STATUS_MARRIED',
    'MARITAL_STATUS_SEPARATED',
    'MARITAL_STATUS_SINGLE',
    'MARITAL_STATUS_Unknown',
    'MARITAL_STATUS_WIDOWED',
    'ETHNICITY_Otra',
    'ETHNICITY_Unknown',
    'ETHNICITY_WHITE'
]
column_group =column_groups = {
    'GENDER': [col for col in column_list if col.startswith('GENDER_')],
    'RELIGION': [col for col in column_list if col.startswith('RELIGION_')],
    'MARITAL_STATUS': [col for col in column_list if col.startswith('MARITAL_STATUS_')],
    'ETHNICITY': [col for col in column_list if col.startswith('ETHNICITY_')]
}
def obtain_readmission_realdata(total_fetura_valid):
    # Ordenando el DataFrame por 'id_' y 'visit_rank'
    total_fetura_valid = total_fetura_valid.sort_values(by=['id_patient', 'visit_rank'])
    # Crear una nueva columna 'readmission'
    # Comparamos si el siguiente 'visit_rank' es mayor que el actual para el mismo 'id_'
    total_fetura_valid['readmission'] = total_fetura_valid.groupby('id_patient')['visit_rank'].shift(-1).notna().astype(int)  
    return  total_fetura_valid

def load_data(file_path):
    try:
        with gzip.open(file_path, 'rb') as f:
             return pickle.load(f)
    except:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                return data

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
    



import numpy as np
import pandas as pd
import logging

def generate_patient_ids(train_ehr_dataset, synthetic_ehr_dataset):
    # Get the distribution of visits per patient from the real data
    real_visits_per_patient = train_ehr_dataset.groupby('id_patient').size()
    total_real_patients = len(real_visits_per_patient)
    total_synthetic_visits = len(synthetic_ehr_dataset)
    
    # Create a histogram of visits per patient
    hist, bin_edges = np.histogram(real_visits_per_patient, bins='auto', density=True)
    bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
    
    # Create CDF
    cdf = np.cumsum(hist)
    cdf /= cdf[-1]
    
    # Generate random values
    random_values = np.random.rand(total_real_patients)
    
    # Find which bin each random value belongs to
    value_bins = np.searchsorted(cdf, random_values)
    
    # Generate number of visits for each synthetic patient
    visits_per_patient = np.round(bin_midpoints[value_bins]).astype(int)
    
    # Adjust visits_per_patient to match total_synthetic_visits
    total_visits = np.sum(visits_per_patient)
    while total_visits != total_synthetic_visits:
        if total_visits < total_synthetic_visits:
            # Add visits to random patients
            indices = np.random.choice(range(len(visits_per_patient)), total_synthetic_visits - total_visits)
            visits_per_patient[indices] += 1
        else:
            # Remove visits from patients with more than 1 visit
            eligible_indices = np.where(visits_per_patient > 1)[0]
            indices = np.random.choice(eligible_indices, total_visits - total_synthetic_visits)
            visits_per_patient[indices] -= 1
        total_visits = np.sum(visits_per_patient)
    
    # Generate patient IDs
    patient_ids = np.repeat(np.arange(1, len(visits_per_patient) + 1), visits_per_patient)
    
    # Assign patient IDs to synthetic dataset
    synthetic_ehr_dataset['id_patient'] = patient_ids
    
    # Generate visit ranks
    synthetic_ehr_dataset['visit_rank'] = synthetic_ehr_dataset.groupby('id_patient').cumcount() + 1
    
    # Log some information
    logging.info(f"Generated {len(np.unique(patient_ids))} unique patient IDs for {len(synthetic_ehr_dataset)} synthetic visits.")
    logging.info(f"The original dataset had {total_real_patients} patients.")
    logging.info(f"Distribution of visits per patient in synthetic data: {np.bincount(visits_per_patient)}")

    return synthetic_ehr_dataset


# Función adicional para verificar la distribución
def verify_distribution(real_df, synthetic_df):
    real_dist = real_df.groupby('id_patient').size().value_counts(normalize=True).sort_index()
    synth_dist = synthetic_df.groupby('id_patient').size().value_counts(normalize=True).sort_index()
    
    print("Distribución real de visitas por paciente:")
    print(real_dist)
    print("\nDistribución sintética de visitas por paciente:")
    print(synth_dist)
    
    if real_dist.equals(synth_dist):
        print("\nLas distribuciones son exactamente iguales.")
    else:
        print("\nLas distribuciones no son exactamente iguales. Diferencias:")
        print(real_dist.compare(synth_dist))
        
def generate_patient_ids_(real_df, synthetic_df):
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
                 ,eliminate_negatives_var ,
                 get_days_grom_visit_histogram,
                 get_admitted_time,
                 
                 type_archivo ,
                 medication_columns = None,
                 columns_to_drop_sec= None,
                 encoder = None,
                 columnas_demograficas = None,
                 synthetic_type = None
               

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
        self.create_days_between_visits_by_date_var = create_days_between_visits_by_date_var
        self.get_days_grom_visit_histogram  =get_days_grom_visit_histogram
        self.get_admitted_time = get_admitted_time
        self.medication_columns = medication_columns
        self.columns_to_drop_sec = columns_to_drop_sec
        self.encoder = encoder
        self.columnas_demograficas =columnas_demograficas
        self.synthetic_type =synthetic_type
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
        
        if self.type_archivo == "gru_Arf":
            self.round_medical_columns()
            # obtain same number of synthetic patien as  train patient
            self.change_subject_id()
            self.get_admitted_time_datetime()  
            self.sort_datasets()   
            self.obtain_readmission_ehrs()
            self.consider_patients_more1_vist()

               
            

            #self.synthetic_ehr_dataset = self.filter_dataset_to_match()

            #rounde medical columns to nearest integer 
            

            # Change the column name 'SUBJECT_ID' to 'id_patient'                  
            
            self.log_negative_values()
            #eliminate negative values
            self.clip_0_negative_value()
            self.eliminate_columns_seq()
        else:    
            if self.synthetic_type == "sin_var_con":
                self.round_medical_columns()    
                self.decode()
                self.change_subject_id()       
                if self.get_sample_synthetic_similar_real:    
                    self.have_same_patients_similar_num_visits() 

                if self.get_admitted_time:        
                    self.get_admitted_time_datetime()  
                    
                if self.get_days_grom_visit_histogram:
                    self.generate_days_from_visit_histogram()    
            
                #self.create_days_between_visits_by_date()          
                self.log_negative_values()
            
                if self.eliminate_negatives_var:
                    self.clip_0_negative_value()
                if self.propagate_fistvisit_categoricaldata:    
                    self.handle_categorical_data()
                    self.propagate_first_visit_values()
      
  
                      
           
      

            else:
                if self.synthetic_type =="label_decoder":
                    self.decode()
       

       
                # Change the column name 'SUBJECT_ID' to 'id_patient'   
                self.change_subject_id()    

                if self.get_sample_synthetic_similar_real:    
                    self.have_same_patients_similar_num_visits() 
                if self.get_admitted_time:        
                    self.get_admitted_time_datetime()  
                    self.sort_datasets()   
                if self.create_visit_rank_col:
                    self.obtain_readmission_ehrs()    
                    
                if self.get_days_grom_visit_histogram:
                    self.generate_days_from_visit_histogram()    
            
                #self.create_days_between_visits_by_date()          
                self.log_negative_values()
            
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
                if    len(self.columns_to_drop)!=0:
                        self.eliminate_columns()
                
        return self.synthetic_ehr_dataset, self.train_ehr_dataset, self.test_ehr_dataset
    def decode(self):

        def decode_demographic(df, demographic_col, encoder):
                # Get the set of unique values in the column
                unique_values = set(df[demographic_col].unique())
            
                
                # Now apply the inverse transform
                df[demographic_col] = encoder.inverse_transform(df[demographic_col])
                
                return df



    
        base_demographic_columns = [col.replace("_encoded", "") for col in self.columnas_demograficas]

        for col in range(len(self.columnas_demograficas)):
            i = self.columnas_demograficas[col]
            self.train_ehr_dataset = decode_demographic(self.train_ehr_dataset, i, self.encoder[base_demographic_columns[col]])
            self.synthetic_ehr_dataset = decode_demographic(self.synthetic_ehr_dataset, i,  self.encoder[base_demographic_columns[col]])
    
    def consider_patients_more1_vist(self):
        patients_multiple_visits = self.train_ehr_dataset [self.train_ehr_dataset["visit_rank"] > 1]["id_patient"]
        self.train_ehr_dataset = self.train_ehr_dataset[self.train_ehr_dataset['id_patient'].isin(patients_multiple_visits)]
        self.train_ehr_dataset = self.train_ehr_dataset.sort_values(['id_patient', 'visit_rank'])    
        unique_patient_train = self.train_ehr_dataset.id_patient.nunique()
        
        def match_specific_visit_distribution(df_synthetic, train_ehr_dataset, id_col='id_patient', visit_col='visit_rank'):
            """
            Ajusta el dataframe sintético para que coincida con una distribución específica de visitas.
            
            :param df_synthetic: DataFrame con datos sintéticos
            :param visit_distribution: Serie de pandas con la distribución deseada de visitas
            :param id_col: Nombre de la columna que contiene los IDs de pacientes
            :param visit_col: Nombre de la columna que contiene los rangos de visita
            :return: DataFrame sintético ajustado
            """
            visit_distribution = train_ehr_dataset[visit_col].value_counts()
            df_synthetic_matched = pd.DataFrame()
            total_patients_selected = 0

            print("Selección de pacientes por número de visitas:")

            for visit_rank, desired_count in visit_distribution.sort_index().items():
                # Seleccionar pacientes sintéticos con exactamente este número de visitas
                patients_with_visits = df_synthetic.groupby(id_col)[visit_col].nunique()
                eligible_patients = patients_with_visits[patients_with_visits == visit_rank].index
                
                if len(eligible_patients) < desired_count:
                    selected_patients = eligible_patients
                    print(f"  Visitas {visit_rank}: se seleccionaron {len(selected_patients)} de {desired_count} requeridos")
                else:
                    selected_patients = np.random.choice(eligible_patients, size=desired_count, replace=False)
                    print(f"  Visitas {visit_rank}: se seleccionaron {desired_count} de {len(eligible_patients)} disponibles")
                
                # Agregar los datos de los pacientes seleccionados al DataFrame resultante
                patient_data = df_synthetic[df_synthetic[id_col].isin(selected_patients)]
                df_synthetic_matched = pd.concat([df_synthetic_matched, patient_data])
                
                total_patients_selected += len(selected_patients)
            
            print(f"\nTotal de pacientes seleccionados: {total_patients_selected}")
            print(f"Total de pacientes requeridos: {visit_distribution.sum()}")
            
            return df_synthetic_matched.reset_index(drop=True)
        self.synthetic_ehr_dataset =match_specific_visit_distribution(self.synthetic_ehr_dataset, self.train_ehr_dataset, id_col='id_patient', visit_col='visit_rank')
        all_subject_ids = self.synthetic_ehr_dataset['id_patient'].unique()
        selected_ids = np.random.choice(all_subject_ids, size=unique_patient_train, replace=False)
        self.synthetic_ehr_dataset = self.synthetic_ehr_dataset[self.synthetic_ehr_dataset['id_patient'].isin(selected_ids)]
         
        # all_subject_ids = self.synthetic_ehr_dataset['SUBJECT_ID'].unique()
    
        # # Asegurarse de que n no sea mayor que el número total de sujetos
        
        # # Seleccionar n IDs de sujeto aleatorios
        # selected_ids = np.random.choice(all_subject_ids, size=len(patients_multiple_visits), replace=False)
        
        # # Filtrar el DataFrame para incluir solo los sujetos seleccionados
       
    def filter_dataset_to_match(self  ):
        """
        Filter df2 to have approximately the same number of patients and visits as df1.
        
        :param df1: The reference DataFrame
        :param df2: The DataFrame to be filtered
        :param patient_col: Column name for patient IDs
        :param visit_col: Column name for visit IDs
        :return: Filtered version of df2
        """
        df1 = self.train_ehr_dataset
        df2 = self.synthetic_ehr_dataset
        patient_col = "id_patient"
        visit_col ="visit_rank"
        # Count patients and visits in df1
        df1_patient_count = df1[patient_col].nunique()
        df1_visit_count = df1[visit_col].nunique()
        
        # Get patient visit counts in df2
        df2_patient_visits = df2.groupby(patient_col).size().reset_index(name='visit_count')
        
        # Sort patients by visit count (descending) and shuffle within each visit count group
        df2_patient_visits = df2_patient_visits.sort_values('visit_count', ascending=False)
        df2_patient_visits = df2_patient_visits.groupby('visit_count').apply(lambda x: x.sample(frac=1)).reset_index(drop=True)
        
        # Initialize variables for patient and visit selection
        selected_patients = []
        total_visits = 0
        
        # Select patients until we reach or exceed the target counts
        for _, row in df2_patient_visits.iterrows():
            if len(selected_patients) >= df1_patient_count or total_visits >= df1_visit_count:
                break
            selected_patients.append(row[patient_col])
            total_visits += row['visit_count']
        
        # Filter df2 based on selected patients
        filtered_df2 = df2[df2[patient_col].isin(selected_patients)]
        
        # If we have more visits than needed, randomly drop some
        if filtered_df2[visit_col].nunique() > df1_visit_count:
            visits_to_keep = np.random.choice(filtered_df2[visit_col].unique(), df1_visit_count, replace=False)
            filtered_df2 = filtered_df2[filtered_df2[visit_col].isin(visits_to_keep)]
        
        return filtered_df2    


    def round_medical_columns(self):
        """
        Round the specified medical columns to the nearest positive integer.
        If the value is negative, it is set to 0.
        
        Args:
        df (pd.DataFrame): The dataframe containing the medical data
        medical_columns (list): List of column names to be rounded
        
        Returns:
        pd.DataFrame: The dataframe with rounded medical columns
        """
        colu = self.medication_columns + ["year" , "month"] + self.columnas_demograficas
        for col in colu:
            if col in ["year","month"]+self.columnas_demograficas:
                self.synthetic_ehr_dataset[col] = np.where(self.synthetic_ehr_dataset[col] < 0, 1, np.round(self.synthetic_ehr_dataset[col]).astype(int))
                if col == "month":
                     self.synthetic_ehr_dataset["month"] = self.synthetic_ehr_dataset["month"].clip(upper=12)
                     self.synthetic_ehr_dataset["month"] = self.synthetic_ehr_dataset["month"].clip(lower=1)

            else:    
                self.synthetic_ehr_dataset[col] = np.where(self.synthetic_ehr_dataset[col] < 0, 0, np.round(self.synthetic_ehr_dataset[col]).astype(int))
          
        
    def generate_days_from_visit_histogram(self):
        real_days = self.train_ehr_dataset['days from last visit'].values
        non_zero_days = real_days[real_days > 0]
        
        hist, bin_edges = np.histogram(non_zero_days, bins='auto', density=True)
        bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
        
        cdf = np.cumsum(hist)
        cdf /= cdf[-1]
        
        # Vectorized generation of random values
        random_values = np.random.rand(len(self.synthetic_ehr_dataset))
        
        # Vectorized bin search
        value_bins = np.searchsorted(cdf, random_values)
        
        # Generate all days at once
        new_days = np.round(bin_midpoints[value_bins]).astype(int)
        self.synthetic_ehr_dataset['days from last visit'] = new_days
        # Set days for first visits to 0
        logging.info(f"number of first visits that will be change to 0 as they are visit 1 {len(self.synthetic_ehr_dataset[(self.synthetic_ehr_dataset['visit_rank'] == 1) &(self.synthetic_ehr_dataset['days from last visit']!=0)])},  the percentage {len(self.synthetic_ehr_dataset[(self.synthetic_ehr_dataset['visit_rank'] == 1) &(self.synthetic_ehr_dataset['days from last visit']!=0)])/self.synthetic_ehr_dataset.shape[0]}") 
        self.synthetic_ehr_dataset.loc[self.synthetic_ehr_dataset['visit_rank'] == 1, 'days from last visit'] = 0

        
        mask = (self.synthetic_ehr_dataset['visit_rank'] > 1) & (self.synthetic_ehr_dataset['days from last visit'] == 0)
        new_random_values = np.random.rand(len(mask))
        logging.info(f"number of days that is no 1 n has 0  {len(mask)}, the percentage is {len(self.synthetic_ehr_dataset[(self.synthetic_ehr_dataset['visit_rank'] > 1) & (self.synthetic_ehr_dataset['days from last visit'] == 0)])/self.synthetic_ehr_dataset.shape[0]}")
        new_bins = np.searchsorted(cdf, new_random_values)
        self.synthetic_ehr_dataset.loc[mask,'days from last visit']=np.round(bin_midpoints[new_bins]).astype(int)
        
        
    def create_days_between_visits_by_date(self):
        self.synthetic_ehr_dataset['ADMITTIME'] = pd.to_datetime(self.synthetic_ehr_dataset['ADMITTIME'])

# Sort by id_patient and ADMITTIME
        self.synthetic_ehr_dataset = self.synthetic_ehr_dataset.sort_values(by=['id_patient', 'ADMITTIME'])

        # Calculate days between visits
        self.synthetic_ehr_dataset['days_from_last_visit2'] = self.synthetic_ehr_dataset.groupby('id_patient')['ADMITTIME'].diff().dt.days

        # Fill NaN values with 0 for the first visit of each patient
        self.synthetic_ehr_dataset['days_from_last_visit2'] = self.synthetic_ehr_dataset['days_from_last_visit2'].fillna(0).astype(int)


        
    def change_subject_id(self):
        if  self.get_sample_synthetic_similar_real:
            self.train_ehr_dataset   = self.train_ehr_dataset.rename(columns={"SUBJECT_ID":"id_patient"})
            self.test_ehr_dataset   = self.test_ehr_dataset.rename(columns={"SUBJECT_ID":"id_patient"})
            
            #age
            self.test_ehr_dataset   = self.test_ehr_dataset.rename(columns={"Age_max":"Age"})
            self.train_ehr_dataset   = self.train_ehr_dataset.rename(columns={"Age_max":"Age"})
            self.synthetic_ehr_dataset   = self.synthetic_ehr_dataset.rename(columns={"Age_max":"Age"})
            
            try:
                self.synthetic_ehr_dataset   = self.synthetic_ehr_dataset.rename(columns={"days_between_visits":"days from last visit"})
                self.train_ehr_dataset   = self.train_ehr_dataset.rename(columns={"days_between_visits":"days from last visit"})
                self.test_ehr_dataset   = self.test_ehr_dataset.rename(columns={"days_between_visits":"days from last visit"})
                
         
            except:
                pass
        else:
            try:
                self.synthetic_ehr_dataset   = self.synthetic_ehr_dataset.rename(columns={"SUBJECT_ID":"id_patient"})
                self.train_ehr_dataset   = self.train_ehr_dataset.rename(columns={"SUBJECT_ID":"id_patient"})
                self.test_ehr_dataset   = self.test_ehr_dataset.rename(columns={"SUBJECT_ID":"id_patient"})
                #age
                self.test_ehr_dataset   = self.test_ehr_dataset.rename(columns={"Age_max":"Age"})
                self.train_ehr_dataset   = self.train_ehr_dataset.rename(columns={"Age_max":"Age"})
                self.synthetic_ehr_dataset   = self.synthetic_ehr_dataset.rename(columns={"Age_max":"Age"})
            except:
                pass    
            try:
                self.synthetic_ehr_dataset   = self.synthetic_ehr_dataset.rename(columns={"days_between_visits":"days from last visit"})
                self.train_ehr_dataset   = self.train_ehr_dataset.rename(columns={"days_between_visits":"days from last visit"})
                self.test_ehr_dataset   = self.test_ehr_dataset.rename(columns={"days_between_visits":"days from last visit"})
  
            except:
                pass

        
    def get_admitted_time_datetime(self):
       
       
        if self.type_archivo=='ARFpkl' or self.type_archivo=='demo_Arf' or self.type_archivo=='gru_Arf':
            if self.synthetic_type == "sin_var_con"  or self.type_archivo=='demo_Arf':
                self.synthetic_ehr_dataset['ADMITTIME'] = self.synthetic_ehr_dataset['year'].astype(int).astype(str) +"-"+ self.synthetic_ehr_dataset['month'].astype(int).astype(str) +"-"+ '01'
            else:
                #total_features_synthethic['ADMITTIME'] = total_features_synthethic['year'].astype(str) +"-"+ total_features_synthethic['month'].astype(str) +"-"+ '01'
                self.synthetic_ehr_dataset['ADMITTIME'] = self.synthetic_ehr_dataset['year'].astype(int).astype(str) +"-"+ self.synthetic_ehr_dataset['month'].astype(int).astype(str) +"-"+ '01'
                self.synthetic_ehr_dataset['ADMITTIME'] = pd.to_datetime(self.synthetic_ehr_dataset['ADMITTIME'])
            
        else:    
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
        non_datetime_cols = [i for i in non_datetime_cols if i!= "ADMITTIME" and i not in self.columnas_demograficas]
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
        non_datetime_cols = [i for i in non_datetime_cols if i != "ADMITTIME" if i not in self.columnas_demograficas]
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
        if  self.synthetic_type =="sin_var_con":
            pass 
        else:
            self.train_ehr_dataset.sort_values(by=['id_patient', 'ADMITTIME'], inplace=True)
            self.synthetic_ehr_dataset.sort_values(by=['id_patient', 'ADMITTIME'], inplace=True)

    def handle_categorical_data(self):
        """
        Handles categorical data in the train EHR dataset.

        This method identifies the categorical columns in the train EHR dataset and returns a list of these columns.

        Returns:
            list: The list of categorical columns in the train EHR dataset.
        """
        
        cols_accounts = []
        for col in self.columnas_demograficas:
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
            conteo_unico_antes = self.synthetic_ehr_dataset.groupby('id_patient')[column].nunique()
            self.synthetic_ehr_dataset[column] = self.synthetic_ehr_dataset.groupby('id_patient')[column].transform('first')
            conteo_unico_despues = self.synthetic_ehr_dataset.groupby('id_patient')[column].nunique()

            # Paso 4: Comparar los conteos antes y después para determinar el cambio
            cambio = (conteo_unico_antes - conteo_unico_despues).sum()
            logging.info(f'Propagated first visit values for column: {column}, number of transformed values: {cambio}, percentage {cambio/self.synthetic_ehr_dataset.shape[0]}') 

        def correct_one_hot_encoding(df, column_groups):
            """
            Correct one-hot encoding in place, prioritizing the most populated columns.
            
            :param df: pandas DataFrame containing the data
            :param column_groups: dict with group names as keys and lists of column names as values
            :return: DataFrame with corrected one-hot encoding
            """
            for group, columns in column_groups.items():
                # Sort columns by population (sum of 1s) in descending order
                sorted_columns = sorted(columns, key=lambda col: df[col].sum(), reverse=True)
                
                for i, col in enumerate(sorted_columns):
                    if i == 0:  # Most populated column
                        # Keep 1s, change other values in the row to 0
                        mask = df[col] == 1
                        df.loc[mask, sorted_columns[1:]] = 0
                    else:
                        # For subsequent columns
                        mask = (df[col] == 1) & (df[sorted_columns[:i]].sum(axis=1) == 0)
                        df.loc[mask, sorted_columns[i+1:]] = 0
                        
                        # Change 0s to 1s in this column if all previous columns are 0
                        zero_mask = (df[sorted_columns[:i+1]].sum(axis=1) == 0)
                        df.loc[zero_mask, col] = 1
                
                # Ensure at least one column is 1 for each row
                all_zero_rows = (df[columns].sum(axis=1) == 0)
                if all_zero_rows.any():
                    df.loc[all_zero_rows, sorted_columns[0]] = 1
            
            return df
       
        #self.synthetic_ehr_dataset =correct_one_hot_encoding(self.synthetic_ehr_dataset,column_group)
    
    def adjust_age_and_dates(self):
        """
        Adjusts the age and admission dates in the synthetic EHR dataset.

        This method fills missing values in the 'days from last visit' column with 0,
        calculates the cumulative sum of 'days from last visit' for each patient,
        and adjusts the 'Age' and 'ADMITTIME' columns based on the cumulative sum.

        The 'Age' values are adjusted by adding the cumulative sum divided by 365
        to the first 'Age' value for each patient, for visits with a rank greater than 1.
        For visits with a rank of 1, the 'Age' value remains unchanged.

        The 'ADMITTIME' values are adjusted by adding the cumulative sum of 'days from last visit'
        shifted by one day to the first 'ADMITTIME' value for each patient, for visits with a rank greater than 1.
        For visits with a rank of 1, the 'ADMITTIME' value remains unchanged.

        Any 'Age' values above 100 are truncated to 89.

        Returns:
            None
        """
        #self.synthetic_ehr_dataset['days from last visit'].fillna(0, inplace=True)
        


        self.synthetic_ehr_dataset['days from last visit_cumsum'] = self.synthetic_ehr_dataset.groupby('id_patient')['days from last visit'].cumsum()
        
        first_Age = self.synthetic_ehr_dataset.groupby('id_patient')['Age'].first()
        first_admission = self.synthetic_ehr_dataset.groupby('id_patient')['ADMITTIME'].first()
   
        self.synthetic_ehr_dataset = self.synthetic_ehr_dataset.reset_index(drop = True)
        self.synthetic_ehr_dataset.loc[self.synthetic_ehr_dataset['visit_rank'] > 1, 'Age'] = self.synthetic_ehr_dataset['id_patient'].map(first_Age) + self.synthetic_ehr_dataset.groupby('id_patient')['days from last visit_cumsum'].transform(lambda x: x / 365)
        self.synthetic_ehr_dataset.loc[self.synthetic_ehr_dataset['visit_rank'] == 1, 'Age'] = self.synthetic_ehr_dataset['Age']
        adjusted_rows = (self.synthetic_ehr_dataset['visit_rank'] > 1).sum()
        total_rows = self.synthetic_ehr_dataset.shape[0]
        logging.info(f'Age values adjusted for {adjusted_rows} visits with rank greater than 1 ({adjusted_rows/total_rows:.2%})')
        logging.info(f'Age values not modified for {total_rows - adjusted_rows} visits ({(total_rows - adjusted_rows)/total_rows:.2%})')
        logging.info(f"truncated patient over age of 89 to 89 {(self.synthetic_ehr_dataset['Age'] > 89).sum() }")
        logging.info(f"truncated patient over age of 89 to 89 {(self.synthetic_ehr_dataset['Age'] > 89).sum() / self.synthetic_ehr_dataset.shape[0]}")
              # Truncate 'Age' values above 100 to 89
        self.synthetic_ehr_dataset.loc[self.synthetic_ehr_dataset['Age'] > 89, 'Age'] = 89
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
        self.synthetic_ehr_dataset['days from last visit'] = self.synthetic_ehr_dataset.apply(replace_zero_days_mean, axis=1, args=(mean_days_per_patient,))
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

    def eliminate_columns_seq(self):
        
        
        self.train_ehr_dataset.drop(columns=self.columns_to_drop_sec, inplace=True)
        self.test_ehr_dataset.drop(columns=self.columns_to_drop_sec, inplace=True)    
 
if __name__ == '__main__':
    
    features_path = "data/intermedi/SD/inpput/entire_ceros_tabular_data.pkl"

    #file = 'generated_synthcity_tabular/arftotal_0.2_epochs.pkl'
    file = 'C:/Users/cyn_n/Desktop/Synthetic-Data-Deep-Learning/generated_synthcity_tabular/ARF/synthetic_data_generative_model_arf_per_0.7.pkl'
    valid_perc = 0.3
    test_ehr_dataset, train_ehr_dataset, synthetic_ehr_dataset, features = obtain_dataset_admission_visit_rank(sample_patients_path, file, valid_perc, features_path, 'ARFpkl') ###DRAFT
    #remplazar valores negativos con ero
    
    #synthetic_ehr_dataset = load_pickle(file)     
    
    c = EHRDataConstraints(train_ehr_dataset, test_ehr_dataset, synthetic_ehr_dataset,True,)
    c.print_shapes()
    #cols_accounts = c.handle_categorical_data()
    processed_synthetic_dataset = c.initiate_processing()
    c.print_shapes()
    
    aun = synthetic_ehr_dataset[synthetic_ehr_dataset["visit_rank"]>2]
    auz = synthetic_ehr_dataset
    auz["visit_rank"].value_counts()
    i =aun["id_patient"].unique()
    for id_paciente in i[180:190]:
        print(processed_synthetic_dataset[processed_synthetic_dataset["id_patient"]==id_paciente][ ['visit_rank','ADMITTIME','days from last visit_cumsum','HOSPITAL_EXPIRE_FLAG','Age']])
    for id_paciente in i[180:190]:
        print(processed_synthetic_dataset[processed_synthetic_dataset["id_patient"]==id_paciente][cols_accounts])
    
    #falt rebisar que cpigo permanece constante en historia de personas-ƒ