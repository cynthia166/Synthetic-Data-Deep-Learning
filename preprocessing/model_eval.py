
import pandas as  pd

from preprocessing.function_mapping import *
from preprocessing.preprocess import *
from features_eng import*

def clustering_ori(X,num_clusters,clustering_method,stri,save,ori,norm_str,real):
    kmeans_labels_l=[]
    ccscodes_thhreshold_l = []
    ccscodes_rand_l = []
    silhouette_avg,davies_bouldin_avg,kmeans_labels,kmeans = cluster_scores(num_clusters,X,clustering_method) 
    kmeans_labels_l.append(kmeans_labels)

    if save == True:
        if stri == "outs_visit":
            joblib.dump(kmeans, 'models_cluster/'+ori+'/visit/'+stri+'_'+norm_str+'_'+real+'_'+clustering_method+'.pkl')
        else:
            joblib.dump(kmeans, 'models_cluster/'+ori+'/patient/'+stri+'_'+norm_str+'_'+real+'_'+clustering_method+'.pkl')
    return silhouette_avg,davies_bouldin_avg
        
        