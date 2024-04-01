import sys
import os
sys.path.append('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing')
import config
from privacy.metric_privacy import *
from evaluation.resemblance.metric_stat import *
from sklearn.decomposition import PCA

sys.path.append('/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_models')



def metrics_Ev(total_features,total_features_s,train_test,results):
       
    id =  IdentifiabilityScore()
    id_s = id.compute_score(total_features,total_features_s)
    print("Identifiability Score "+train_test, id_s)
    results["Identifiability Score "+train_test] = id_s
   
    

    # Asume que 'total_features' es tu array de tres dimensiones
    pca = PCA(n_components=2)
    total_features_reduced = pca.fit_transform(total_features.reshape(total_features.shape[0], -1))
    total_features_reduced_s = pca.fit_transform(total_features_s.reshape(total_features_s.shape[0], -1))
    total_features_reduced = total_features_reduced.astype(np.float32)
    total_features_reduced_s = total_features_reduced_s.astype(np.float32)
 
    delta = DeltaPresence()
    delta_s = delta.evaluate(total_features_reduced,total_features_reduced_s)
    print("Delta Presence:", delta_s)
    results["Delta Presence "+train_test] = delta_s
    #test



    evaluator = SyntheticDetectionXGB(n_folds=5, random_state=42)
    score = evaluator.evaluate(total_features_reduced, total_features_reduced_s)
    print("Detection Score:", score)
    results["Detection Score "+train_test] = score



    #Synthetic Data Evaluation    #lpha-precision, beta-recall, and authenticity #Alaa, Ahmed, Boris Van Breugel, Evgeny S. Saveliev, and Mihaela van der Schaar. “How faithful is your synthetic data? 
    # sample-level metrics for evaluating and auditing generative models.” In International Conference on Machine Learning, pp. 290-306. PMLR, 2022.
    #alpha_precision = AlphaPrecision()
    #result = alpha_precision.metrics(features, synthetic_features)
        

    # Ejemplo de cómo llamar a la función plot_tsne
    # Supongamos que X_gt y X_syn son tus arrays de NumPy con los datos.
    # Deberás reemplazar estas líneas con tu carga de datos real.
    synthetic_features_2d = total_features_s.reshape(total_features_s.shape[0], -1)
    features_2d = total_features.reshape(total_features.shape[0], -1)

    
        
    mmd_evaluator = MaximumMeanDiscrepancy(kernel="rbf")
    result =   mmd_evaluator._evaluate(features_2d, synthetic_features_2d)
    print("MaximumMeanDiscrepancy (flattened):", result)
    results["MaximumMeanDiscrepancy(flattened) "+train_test] = result
    result =   mmd_evaluator._evaluate(total_features_reduced, total_features_reduced_s)
    print("MaximumMeanDiscrepancy (PCA):", result)
    results["MaximumMeanDiscrepancy (PCA)"+train_test] = result

    # Example usage:
    # X_gt and X_syn are two numpy arrays representing empirical distributions
    features_1d = total_features.flatten()
    synthetic_features_1d = total_features_s.flatten()


    ks_test = KolmogorovSmirnovTest()
    result = ks_test._evaluate(features_1d, synthetic_features_1d)
    print("Kolmog orov-Smirnov Test:", result)
    results["Kolmogorov-Smirnov Test "+train_test] = result



    score = JensenShannonDistance()._evaluate(total_features, total_features_s)
    print("Jensen-Shannon Distance:", score)
    results["Jensen-Shannon Distance "+train_test] = score
    
    return results    
    
    #plot_marginal_comparison(plt, features_2d, synthetic_features_2d, n_histogram_bins=10, normalize=True)

        
        
def main( total_features_synthethic,total_fetura_valid,total_features_train,dict_res):
    
    results = metrics_Ev(total_features_train,total_features_synthethic,"train",dict_res)
    results = metrics_Ev(total_fetura_valid,total_features_synthethic,"test",results)
    return results

if __name__=='__main__':
    import wandb
    path_features =  "train_sp/"
    path = "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/"
    
    features = load_data(path_features+"train_splitDATASET_NAME_preprotrain_data_features.pkl")
    attributes=load_data(path_features+"train_splitDATASET_NAME_preprotrain_data_attributes.pkl")
    synthetic_attributes = load_data(path_features+'DATASET_NAME_preprosynthetic_attributes.pkl')
    synthetic_features=load_data(path_features+'DATASET_NAME_preprosynthetic_features.pkl')
    features_valid = load_data(path_features+"train_splitDATASET_NAME_preprovalid_data_features.pkl")
    attributes_valid=load_data(path_features+"train_splitDATASET_NAME_preprovalid_data_attributes.pkl")
    config_w = {
         "preprocessed": "True",
         "max_sequence_len": 666,
         "sample_len": 111,
         "batch_size": 100,
         "epochs": 60
        }
    wandb.init(project='SD_generation',config=config_w)


    features_v = features[:attributes_valid.shape[0]]
    attributes_v = attributes[:attributes_valid.shape[0]]   

     # Imprimir la ruta del directorio actual
    N, T, D = features.shape  


    print(features.shape)
    print(attributes.shape)
    print(synthetic_features.shape)
    print(synthetic_attributes.shape)
    
    total_features_synthethic = concat_attributes(synthetic_features, synthetic_attributes)
    total_fetura_valid = concat_attributes(features_valid, attributes_valid)
    total_features_train = concat_attributes(features_v, attributes_v)
    results = {} 


    results = main(total_features_synthethic,total_fetura_valid,total_features_train,results)
    print(results)
    wandb.log( results)
    wandb.finish()
    print("End")
    
    #plot_tsne(plt, features_2d, synthetic_features_2d)
    #plot_marginal_comparison(plt, features_2d, synthetic_features_2d, n_histogram_bins=10, normalize=True)
    #plt.show()
    #plt.savefig('plot.png')
    #plt.close()