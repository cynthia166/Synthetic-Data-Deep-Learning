import os
import matplotlib.pyplot as plt
import numpy as np
#from pacmap import PaCMAP
import logging
from scipy.stats import wasserstein_distance
import random
from sklearn import preprocessing
file_principal = os.getcwd()
os.chdir(file_principal )
import sys    
sys.path.append('evaluation/resemb/resemblance/utils_stats/')
sys.path.append('evaluation')
from metric_privacy import *
sys.path.append(file_principal )
from evaluation.resemb.resemblance.utilsstats import *
from evaluation.resemb.resemblance.config import *
ruta_actual = os.getcwd()
print(ruta_actual)

set_graph_settings()

class EHRRPrivacyMetrics:
    def __init__(self,list_metric
                 ,test_ehr_dataset,
                 train_ehr_dataset,
                 synthetic_ehr_dataset,
  
                 top_num,
                 columnas_test_ehr_dataset,
                 K=1, #attribute attack
                NUM_TEST_EXAMPLES = 40 ,#membershi attack
                NUM_TOT_EXAMPLES = 40, #membershi attack
                NUM_VAL_EXAMPLES = 20, #membershi attack
                 ):
        self.list_metric = list_metric
        self.test_ehr_dataset = test_ehr_dataset
        self.train_ehr_dataset = train_ehr_dataset
        self.train_ehr_dataset = train_ehr_dataset
        self.synthetic_ehr_dataset = synthetic_ehr_dataset
        self.columnas_test_ehr_dataset = columnas_test_ehr_dataset
        self.top_num = top_num
        self.columnas_test_ehr_dataset =columnas_test_ehr_dataset
        self.K =K
        self.NUM_TEST_EXAMPLES = NUM_TEST_EXAMPLES
        self.NUM_TOT_EXAMPLES = NUM_TOT_EXAMPLES
        self.NUM_VAL_EXAMPLES = NUM_VAL_EXAMPLES
  
        logging.basicConfig(level=logging.INFO)
        
    def evaluate(self):    
        results = {}
        result_final = {}

        if "attributes_attack" in self.list_metric:
            logging.info("Executing Attribute attack")
            
            result = self.attribute_attack_fun(self.synthetic_ehr_dataset,
                                           self.test_ehr_dataset,
                                           self.train_ehr_dataset,
                                           self.K,
                                           self.top_num,
                                           self.columnas_test_ehr_dataset)
            
            results.update(result)

        if "memebership_attack" in list_metric:
            logging.info("Executing Mambership attack")
            result = self.membership_attack_fun()
            results.update(result)

        if "nn_distance_attack" in list_metric:
            logging.info("Executing nn_distance_attack")
            result = self.nn_distance_attack()
            results.update(result)


   
    def attribute_attack_fun(self,synthetic_ehr_dataset,
                                         test_ehr_dataset,
                                          train_ehr_dataset,
                                          K,
                                          top_num,
                                          columnas_test_ehr_dataset):
        
        res =attribute_attack(self.synthetic_ehr_dataset,
                                           self.test_ehr_dataset,
                                           self.train_ehr_dataset,
                                           K,
                                    
                                           self.top_num,
                                           self.columnas_test_ehr_dataset)
    
        print(res)
        return res
    
    def membership_attack_fun(self):
                
                synthetic_ehr_ = change_tosyn_stickers_temporal(self.synthetic_ehr_dataset,self.columnas_test_ehr_dataset,True)

                
                results = membership_attack(self.train_ehr_dataset,
                                            self.test_ehr_dataset,self.synthetic_ehr_dataset,
                                            self.columnas_test_ehr_dataset,self.NUM_TEST_EXAMPLES,self.NUM_TOT_EXAMPLES,self.NUM_VAL_EXAMPLES)
                print(results)
                return results
            
    def nn_distance_attack(self):
            dataset_train = obtain_dataset(self.train_ehr_dataset,self.columnas_test_ehr_dataset)
            dataset_test = obtain_dataset(self.test_ehr_dataset,self.columnas_test_ehr_dataset)

            synthetic_ehr = change_tosyn_stickers_temporal(self.synthetic_ehr_dataset,self.columnas_test_ehr_dataset,True)
            dataset_syn = obtain_dataset(self.synthetic_ehr_dataset,self.columnas_test_ehr_dataset)


            NUM_SAMPLES = min(len(dataset_train), len(dataset_test), len(synthetic_ehr))
            dataset_train = np.random.choice(dataset_train, NUM_SAMPLES)
            dataset_test = np.random.choice(dataset_test, NUM_SAMPLES)
            synthetic_ehr = np.random.choice([p for p in dataset_syn if len(p['visits']) > 0], NUM_SAMPLES)


            nnaar = calc_nnaar(dataset_train, dataset_test, synthetic_ehr,NUM_SAMPLES)
            results = {
                "nn_distance_attack": nnaar
            }
            print(results)
            return results
        

#codigo para attribute attack    
if __name__ == "__main__":
    #"memebership_attack","nn_distance_attack","attributes_attack"
    list_metric = ["nn_distance_attack",]
    top_num = 300
    test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset,features  = load_create_ehr(read_ehr,save_ehr,file_path_dataset,sample_patients_path,file,valid_perc,features_path,name_file_ehr,type_file=type_archivo)
    columnas_test_ehr_dataset =get_cols_diag_proc_drug(train_ehr_dataset)
    print(file)
   

      
                  
       
    train_ehr_dataset,synthetic_ehr_dataset,test_ehr_dataset = make_read_constraints( make_contrains,
                          save_constrains,
                          train_ehr_dataset,
                          test_ehr_dataset,
                          synthetic_ehr_dataset,
                          columns_to_drop,
                          columns_to_drop_syn,
                          type_archivo,
                   
                          cols_continuous,
                           create_visit_rank_col,
                            propagate_fistvisit_categoricaldata,
                            adjust_age_and_dates_get,
                            get_remove_duplicates,
                            get_handle_hospital_expire_flag,
                            get_0_first_visit,
                            get_sample_synthetic_similar_real,
         
                            create_days_between_visits_by_date_var,
                            eliminate_negatives_var,
                            get_days_grom_visit_histogram,
                            get_admitted_time,
                            get_synthetic_subject_clustering,
                            file_path_dataset ,
                            make_read_constraints_name,
                            medication_columns = medication_columns,
                            columns_to_drop_sec = columns_to_drop_sec,
                            encoder =encoder,
                            columnas_demograficas=columnas_demograficas,
                            synthetic_type=synthetic_type)
    
    metrics_e = EHRRPrivacyMetrics(list_metric
                 ,test_ehr_dataset,
                 train_ehr_dataset,
                 synthetic_ehr_dataset,
                 top_num,
                 columnas_test_ehr_dataset,
                 K=1, #attribute attack
                NUM_TEST_EXAMPLES = 40 ,#membershi attack
                NUM_TOT_EXAMPLES = 40, #membershi attack
                NUM_VAL_EXAMPLES = 20, #membershi attack
                 )
    results = metrics_e.evaluate() 