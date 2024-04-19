
import sys
import os
current_directory = os.getcwd()
os.chdir("/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning")
current_directory = os.getcwd()

print(current_directory)
sys.path.append('preprocessing')
sys.path.append('evaluation')
import config
from evaluation.privacy.metric_privacy import *
from evaluation.resemblance.metric_stat import *
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, f1_score
from xgboost import XGBClassifier  # Specifically
import time 

sys.path.append('generative_models')
import os

# Cambiar el directorio de trabajo actual


# Ahora, cualquier ruta relativa se resolverá con respecto a "/nuevo/directorio"

path_features =  "train_sp/"
#path = "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/"
# The code snippet you provided is defining a variable `dataset_name` as
# `'/train_splitDATASET_NAME_prepro'`. Then, it is loading data files using the `load_data` function

#funciones
def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)
    
def function_models_s(model,K,X,y):
    '''function that trains the predictive model, it is able to stratifi splits the input, it 
    also take sin acounts wtih selection of variables, ussing logistic regression with penanlty l1, '''
    '''
    Inputs:
    model: type of model that will be traines
    sampling: the valid option is under sampling (under), over sampling (over)
    li_feature_selection: if its true it selec features fitting a logistic regression with
    penalty l1, it select features according to importance weights
    kfolds: there are two type of folds, stratified where the classes preserve the percentage
    of samples for each class otherwise it splits dataset in different consecutive folds. 
    K: number of splits to be considered
    output:
    sensitivity;number of true postives /(number of true postives  of false positives plus the number of false positives)
    specificity: true negative /(true negative+false positives)
    precision: number of true| postives /(number of true postives  of false positives plus the number of false negatives)
    accuracy; number of correc prediccition/ total number of prediction
    f1_score: it is the armonic mean of precision and recall 
    '''
    #obtain the kfolfs/ stratifies or consecutive
    skf = StratifiedKFold(n_splits=K, shuffle=False, )
    #initializiate lists   
    y_true = []
    y_pred = []
    y_true_train   =[]
    y_pred_train =[]
     
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # l1-based feature selection
        X_train_new = X_train
        X_test_new = X_test    
        var_moin =0
        var_ini =X_train.shape[1]
        #the Xtrainfolds and Xtest folds are considered 
        rf_clf = model
        
        #train the model 
        rf_clf.fit(X_train_new, y_train)
        #predict with test to obtain the test set error
        y_pred_sub = rf_clf.predict(X_test_new)
        y_true = np.append(y_true, y_test)
        y_pred = np.append(y_pred, y_pred_sub)
        #it prdic with train to obtain the trainig error
        y_pred_sub_train = rf_clf.predict(X_train_new)
        y_true_train = np.append(y_true_train, y_train)
        y_pred_train = np.append(y_pred_train, y_pred_sub_train)
        
        
        xgb1_pred_prob = rf_clf.predict_proba(X_test_new)
        fpr, tpr, thresholds = roc_curve(y_test, xgb1_pred_prob[:, 1])
        
            #obtain roc, and auc score
    try:
        #it obtain metric considered and confussion matrix, metrics for the test set
        rf_conf = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = rf_conf.ravel()
        rf_sen = tp/(tp+fn)
        rf_spe = tn/(tn+fp)
        rf_prec = tp/(tp+fp)
        rf_acc = (tp+tn)/(tp+tn+fp+fn)
        f1 = f1_score(y_true, y_pred, average='macro')
        #metrics for the training set
        rf_conf_t = confusion_matrix(y_true_train, y_pred_train)
        tn_t, fp_t, fn_t, tp_t = rf_conf_t.ravel()
        rf_sen_t = tp_t/(tp_t+fn_t)
        rf_spe_t = tn_t/(tn_t+fp_t)
        rf_prec_t = tp_t/(tp_t+fp_t)
        rf_acc_t = (tp_t+tn_t)/(tp_t+tn_t+fp_t+fn_t)
        f1_t = f1_score(y_true_train, y_pred_train, average='macro')
    except:
        rf_conf = 0
        tn, fp, fn, tp = 0,0,0,0
        rf_sen = 0
        rf_spe = 0
        rf_prec = 0
        rf_acc = 0
        f1 = 0
        
        rf_conf_t = 0
        tn_t, fp_t, fn_t, tp_t = 0,0,0,0
        rf_sen_t = 0
        rf_spe_t = 0
        rf_prec_t = 0
        rf_acc_t = 0
        f1_t = 0


    
    return rf_sen,rf_spe,rf_prec,rf_acc,rf_conf,rf_sen_t,rf_spe_t,rf_prec_t,rf_acc_t,f1,f1_t


def add_names_numpy_array(total_fetura_valid,total_cols):
    
    mapeo_nombres = {total_fetura_valid.columns[i]: total_cols[i] for i in range(len(total_cols))}
    total_fetura_valid = total_fetura_valid.rename(columns=mapeo_nombres)
    return total_fetura_valid   


import numpy as np
def eliminar_0_dataframe(total_features_synthethic,total_fetura_valid,total_features_train,per=0.85):
    # Definir una función que calcula el porcentaje de 0 en una fila
    def porcentaje_ceros(fila):
        return np.mean(fila == 0.0) > per

    # Encontrar las filas que tienen más del 95% de 0
    filas_a_eliminar_synthethic = pd.DataFrame(total_features_synthethic).apply(porcentaje_ceros, axis=1)
    filas_a_eliminar_valid = pd.DataFrame(total_fetura_valid).apply(porcentaje_ceros, axis=1)
    filas_a_eliminar_train = pd.DataFrame(total_features_train).apply(porcentaje_ceros, axis=1)

    # Eliminar esas filas
    total_features_synthethic = pd.DataFrame(total_features_synthethic).loc[~filas_a_eliminar_synthethic]
    total_fetura_valid = pd.DataFrame(total_fetura_valid).loc[~filas_a_eliminar_valid]
    total_features_train = pd.DataFrame(total_features_train).loc[~filas_a_eliminar_train]

    print(total_features_synthethic.shape)
    print(total_fetura_valid.shape)
    print(total_features_train.shape)


    # Obtener el número de filas de cada DataFrame
    num_filas_synthethic = total_features_synthethic.shape[0]
    num_filas_valid = total_fetura_valid.shape[0]
    num_filas_train = total_features_train.shape[0]

    # Encontrar el mínimo
    min_filas = min(num_filas_synthethic, num_filas_valid, num_filas_train)

    print(min_filas)

    # Crear un diccionario con los nombres de los DataFrames y sus números de filas
    num_filas = {
        "synthethic": total_features_synthethic.shape[0],
        "valid": total_fetura_valid.shape[0],
        "train": total_features_train.shape[0]
    }

    # Encontrar el nombre del DataFrame con el mínimo número de filas
    min_filas_df = min(num_filas, key=num_filas.get)

    print(min_filas_df)


    total_fetura_valid = total_fetura_valid[:num_filas[min_filas_df]]
    total_features_synthethic = total_features_synthethic[:num_filas[min_filas_df]]
    total_features_train = total_features_train[:num_filas[min_filas_df]]

    print(total_features_synthethic.shape)
    print(total_fetura_valid.shape)
    print(total_features_train.shape)

    return total_fetura_valid,total_features_synthethic,total_features_train


def get_valid_train_synthetic (path_features, attributes_path_train, features_path_train, features_path_valid, attributes_path_valid):
    attributes= load_data(path_features+attributes_path_train)
    features =load_data(path_features+features_path_train)

    features_valid = load_data(path_features+features_path_valid)
    attributes_valid=load_data(path_features+attributes_path_valid)


    synthetic_attributes = load_data(path_features+synthetic_path_attributes)
    synthetic_features=load_data(path_features+synthetic_path_features)

    config_w = {
            "preprocessed": "True",
            "max_sequence_len": 666,
            "sample_len": 111,
            "batch_size": 100,
            "epochs": 60
        }
    #wandb.init(project='SD_generation',config=config_w)


    features_v = features[:attributes_valid.shape[0]]
    attributes_v = attributes[:attributes_valid.shape[0]]   

        # Imprimir la ruta del directorio actual
    N, T, D = features.shape  


    print(features.shape)
    print(attributes.shape)
    print(synthetic_features.shape)
    print(synthetic_attributes.shape)
    print(attributes_valid.shape)

    #revisar que este productp sea cohere con primer debe ser 0
    total_features_synthethic = concat_attributes(synthetic_features, synthetic_attributes)
    total_fetura_valid = concat_attributes(features_valid, attributes_valid)
    total_features_train = concat_attributes(features_v, attributes_v)


    #aqui tengo que agregar una columnas que sea id_paciente y otra visit_rank

    print(total_features_synthethic.shape)
    print(total_fetura_valid.shape)
    print(total_features_train.shape)
    return total_features_synthethic, total_fetura_valid,total_features_train,attributes



def obtener_dataframe_inicial_denumpyarrray(total_features_synthethic, total_fetura_valid,total_features_train ):
    #funcion que crea dataframe a partir de numpy array
    rango_visitas = np.arange(1, 43)
    visit_rank = np.tile(rango_visitas, (total_features_synthethic.shape[0], 1, 1))
    #

    # Crear un array que vaya de 1 a total_features_synthethic.shape[0] + 1
    id_paciente = np.arange(1, total_features_synthethic.shape[0] + 1)
    # Darle la forma (total_features_synthethic.shape[0], 1, 1)
    id_paciente = id_paciente.reshape(-1, 1, 1)
    # Repetir este array para cada visita
    id_paciente = np.tile(id_paciente, (1, 1, 42))
    print(id_paciente.shape)  # Debería imprimir: (4496, 1, 42)

    #concatenar id_paciente y visit_rank
    total_features_synthethic = np.concatenate((total_features_synthethic, visit_rank, id_paciente), axis=1)
    total_fetura_valid = np.concatenate((total_fetura_valid, visit_rank,id_paciente), axis=1)
    total_features_train = np.concatenate((total_features_train, visit_rank,id_paciente), axis=1)
    print(total_features_synthethic.shape)
    print(total_fetura_valid.shape)
    print(total_features_train.shape)

    # Supongamos que df es tu DataFrame y 'visit_rank' es la última columna

    # Calcular el porcentaje de valores no ceros en cada fila
    lista_dfs_synthetic = []
    lista_dfs_train = []
    lista_dfs_valid = []
    for i in range(total_fetura_valid.shape[0]):
        # Convertir el subarray en un DataFrame y agregarlo a la lista
        df = pd.DataFrame(total_fetura_valid[i])
        lista_dfs_valid.append(df.T)
    for i in range(total_features_train.shape[0]):    
        df1 = pd.DataFrame(total_features_train[i])
        lista_dfs_train.append(df1.T)
    for i in range(total_features_synthethic.shape[0]):    
        df2 = pd.DataFrame(total_features_synthethic[i])
        lista_dfs_synthetic.append(df2.T)

    total_fetura_valid = pd.concat(lista_dfs_valid, ignore_index=True)
    total_features_train = pd.concat(lista_dfs_train, ignore_index=True)
    total_features_synthethic = pd.concat(lista_dfs_synthetic, ignore_index=True)


    print(total_features_synthethic.shape)
    print(total_fetura_valid.shape)
    print(total_features_train.shape)
    
    return total_features_synthethic,total_fetura_valid,total_features_train
    

def crear_syntheti_visit(df):

    try:
        # Paso 1: Identificar todos los id_paciente únicos
        unique_ids = df[681].unique()

        # Paso 2: Asignar un número máximo de consultas aleatorio para cada id_paciente
        np.random.seed(42)  # Para reproducibilidad de los resultados
        max_consultas = np.random.randint(1, 21, size=len(unique_ids))  # Números entre 1 y 20

        # Crear un diccionario para mapear cada id_paciente con su máximo de consultas
        id_to_max_consultas = dict(zip(unique_ids, max_consultas))

        # Paso 3: Mapear estos máximos al DataFrame original
        df['max_consultas'] = df[681].map(id_to_max_consultas)
    except: 
            # Paso 1: Identificar todos los id_paciente únicos
        unique_ids = df['id_patient'].unique()

        # Paso 2: Asignar un número máximo de consultas aleatorio para cada id_paciente
        np.random.seed(42)  # Para reproducibilidad de los resultados
        max_consultas = np.random.randint(1, 21, size=len(unique_ids))  # Números entre 1 y 20

        # Crear un diccionario para mapear cada id_paciente con su máximo de consultas
        id_to_max_consultas = dict(zip(unique_ids, max_consultas))

        # Paso 3: Mapear estos máximos al DataFrame original
        df['max_consultas'] = df['id_patient'].map(id_to_max_consultas)
    

    return df 


# Definir las condiciones y los valores correspondientes
def visitas_readmission(df):
    try:
        condiciones = [
            df['max_consultas'] == df['visit_rank'],
            df['max_consultas'] < df['visit_rank'],
            df['max_consultas'] > df['visit_rank']
        ]
        valores = [0, 1, -1]

        # Crear la nueva columna usando np.select
        df['readmission'] = np.select(condiciones, valores)
    except:
        condiciones = [
            df['max_consultas'] == df[680],
            df['max_consultas'] < df[680],
            df['max_consultas'] > df[680]
        ]
        valores = [0, 1, -1]

        # Crear la nueva columna usando np.select
        df['readmission'] = np.select(condiciones, valores)
    return df

def obtain_readmission_realdata(total_fetura_valid):
    # Ordenando el DataFrame por 'id_' y 'visit_rank'
    total_fetura_valid = total_fetura_valid.sort_values(by=['id_patient', 'visit_rank'])

    # Crear una nueva columna 'readmission'
    # Comparamos si el siguiente 'visit_rank' es mayor que el actual para el mismo 'id_'
    total_fetura_valid['readmission'] = total_fetura_valid.groupby('id_patient')['visit_rank'].shift(-1).notna().astype(int)  
    return  total_fetura_valid

def predict_readmission(total_features_synthethic,result,type_s):
    try:
        X = X.drop(columns=['readmission',"visit_rank","id_patient","max_consultas"])
    except:    
        X = total_features_synthethic.drop(columns=['readmission',"visit_rank","id_patient"]).values
    y = total_features_synthethic['readmission'].to_numpy()
  
    type = "Valid"
    K = 5
    model = XGBClassifier(
                    use_label_encoder=False
            
                    
                )

    model_name = "XGBClassifier"
        
     

    timi_ini =time.time()
    #model = LogisticRegression(penalty='l1', solver='saga')
    rf_sen,rf_spe,rf_prec,rf_acc,rf_conf,rf_sen_t,rf_spe_t,rf_prec_t,rf_acc_t,f1,f1_t = function_models_s(model,K,X,y)            
    time_model = timi_ini-time.time()  
    result['sensitivity_test '+type_s]=rf_sen
    result['specificity_test '+type_s]=rf_spe
    result['precision_test '+type_s]=rf_prec
    result['accuracy_test '+type_s]=rf_acc
    result['sensitivity_train '+type_s]=rf_sen_t
    result['specificity_train '+type_s]=rf_spe_t
    result['precision_train '+type_s]=rf_prec_t
    result['accuracy_train '+type_s]=rf_acc_t
    result['f1_test '+type_s]=f1
    result['f1_train '+type_s]=f1_t
    result['confusion matrix '+type_s]=rf_conf
    result['Classifiers '+type_s]=model_name
    result["time_model "+type_s]=time_model
    result["type "+type_s]=type

    print(result)
    return result

def crear_pred(result,total_features_synthethic,total_fetura_valid,total_features_train):
    result = predict_readmission(total_features_synthethic,result,"Synthetic")


    #filtra aquellas que sean -1 las que no son mayores a max admissions
    # Ejemplo de DataFrame

    result = predict_readmission(total_fetura_valid,result,"Valid")
    result = predict_readmission(total_features_train,result,"Train")

    return result 
 
 
     
def preprocess_data(total_cols,total_features_synthethic,total_cols1,total_fetura_valid,total_features_train):

   
    #crea un random visitas maximas
    total_features_synthethic = crear_syntheti_visit(total_features_synthethic)
    #agrega los nomrbres de columnas
    total_features_synthethic = add_names_numpy_array(total_features_synthethic,total_cols)
    #crea readmission columnas
    total_features_synthethic = visitas_readmission(total_features_synthethic)
    
    
    
    total_features_synthethic = total_features_synthethic[total_features_synthethic['readmission']!=-1]
    total_fetura_valid = total_fetura_valid.drop(columns=[300])
    total_features_train = total_features_train.drop(columns=[300])

    total_fetura_valid = total_fetura_valid[(total_fetura_valid[1]!=0.0)]
    total_features_train = total_features_train[(total_features_train[1]!=0.0)]

    total_fetura_valid = add_names_numpy_array(total_fetura_valid,total_cols1)
    total_features_train = add_names_numpy_array(total_features_train,total_cols1)

    #rbtener readmission
    total_fetura_valid = obtain_readmission_realdata(total_fetura_valid)
    total_features_train = obtain_readmission_realdata(total_features_train)
    # Imprimir la cantidad de ceros en cada fila
    print(total_fetura_valid.shape)
    
    return total_fetura_valid,total_features_train,total_features_synthethic
    
    
if __name__=='__main__':
    attributes_path_train= "non_prepo/DATASET_NAME_non_preprotrain_data_attributes.pkl"
    features_path_train = "non_prepo/DATASET_NAME_non_preprotrain_data_features.pkl"
    features_path_valid = "non_prepo/DATASET_NAME_non_preprovalid_data_features.pkl"
    attributes_path_valid = "non_prepo/DATASET_NAME_non_preprovalid_data_attributes.pkl"
    synthetic_path_attributes = 'non_prepo/DATASET_NAME_non_prepronon_prepo_synthetic_attributes_10.pkl'
    synthetic_path_features = 'non_prepo/DATASET_NAME_non_prepronon_prepo_synthetic_features_10.pkl'

    # se lee los archivos y se obtiene del la longitude de valid
    total_features_synthethic, total_fetura_valid,total_features_train,attributes =  get_valid_train_synthetic (path_features, attributes_path_train, features_path_train, features_path_valid, attributes_path_valid)
    # se transforma de numpy array 3 dimensiones a dataframe
    total_features_synthethic,total_fetura_valid,total_features_train = obtener_dataframe_inicial_denumpyarrray(total_features_synthethic, total_fetura_valid,total_features_train )

    dataset_name = 'DATASET_NAME_non_prepo'
    file_name = "train_sp/non_prepo/DATASET_NAME_non_prepo_non_preprocess.pkl"
    aux = load_data(file_name)
    #aux = load_data(DARTA_INTERM_intput + dataset_name + '_non_preprocess.pkl')
    con_cols = list(aux[0].columns)
    static = pd.read_csv("train_sp/non_prepo/static_data_non_preprocess.csv")
    # Suponiendo que 'total_features_synthethic' es tu DataFrame
    if 'Unnamed' in static.columns:
        static = static.drop(columns=['Unnamed'])
    cat = list(static.columns[1:]) +["visit_rank","id_patient" ,"max_consultas" ]
    del aux
    total_cols =  con_cols+cat 
    cat1 = list(static.columns[2:]) +["visit_rank","id_patient","max_consultas"     ]
    total_cols1 =  con_cols+cat1 
 

        
    result = {   'f1_test':0,
        'f1_train':0,

        'sensitivity_test':0,
        'specificity_test':0,
        'precision_test':0,
        'accuracy_test':0,
        'sensitivity_train':0,
        'specificity_train':0,
        'precision_train':0,
        'accuracy_train':0, 
        'confusion matrix':0,
        'Feature selection':0,
        'Classifiers':0,
        'time_model':0,
        'type':0,

        }     
    total_fetura_valid,total_features_train,total_features_synthethic = preprocess_data(total_cols,total_features_synthethic,total_cols1,total_fetura_valid,total_features_train)
    result  = crear_pred(result,total_features_synthethic,total_fetura_valid,total_features_train)
   