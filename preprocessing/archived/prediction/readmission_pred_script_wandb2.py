import sys
sys.path.append('preprocessing')
from function_pred import *
import pandas as pd
import yaml

import json
import wandb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
    
def replace_nan_with_zero(d):
    """
    Recursively replace NaN values with 0 in a dictionary, including NaN values in arrays.
    
    :param d: Input dictionary
    :return: Dictionary with NaN values replaced by 0
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = replace_nan_with_zero(v)
        elif isinstance(v, list):
            d[k] = [replace_nan_with_zero(x) if isinstance(x, dict) else x for x in v]
            d[k] = [0 if isinstance(x, float) and np.isnan(x) else x for x in d[k]]
        elif isinstance(v, np.ndarray):
            d[k] = np.nan_to_num(v, nan=0.0)
        elif isinstance(v, float) and np.isnan(v):
            d[k] = 0
    return d


def train(X, y, splits, days, name, project_name, param_grid,model):
    try:
        X = X.values
    except:
        pass
    
    try:
        y = y[days + "_READMIT"].to_numpy()
    except:
        pass

    result = {
        'f1_test': 0, 'f1_train': 0,
        'sensitivity_test': 0, 'specificity_test': 0, 'precision_test': 0, 'accuracy_test': 0,
        'sensitivity_train': 0, 'specificity_train': 0, 'precision_train': 0, 'accuracy_train': 0,
        'confusion_matrix': 0, 'auc_train': 0, 'auc_test': 0,
        'Classifiers': 0, 'Mapping': 0,
        'mean_test_scores_folds': 0, 'mean_train_scores_folds': 0,
        'time_model': 0, "prepo": 0,
        "best_params": {}
    }

    start_time = time.time()
    model_results = function_modelov2(X, y, splits, param_grid,model)
    result.update(model_results)
    result['time_model'] = time.time() - start_time
    result['Mapping'] = name
    result['Classifiers'] = str(model)
    result= replace_nan_with_zero(result)
    wandb.init(project=project_name, name=f"experiment_{name}", config=param_grid)
    wandb.log(result)
    wandb.finish()

    return result




def function_modelov2(X, y, splits, param_grid, model):
    train_size = int(len(X) * 0.75)
    val_size = int(len(X) * 0.15)
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    tscv = KFold(n_splits=splits, shuffle=False)

    # Configurar parámetros específicos según el tipo de modelo
    if isinstance(model, XGBClassifier):
        model.set_params(use_label_encoder=False, eval_metric='logloss')
        fit_params = {'eval_set': [(X_val, y_val)], }
    elif isinstance(model, LogisticRegression):
        fit_params = {}
    else:
        fit_params = {}

    # Eliminar parámetros no válidos del param_grid
    valid_params = model.get_params().keys()
    param_grid = {k: v for k, v in param_grid.items() if k in valid_params}

    grid_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        cv=tscv,
        scoring="f1",
        return_train_score=True,
        n_iter=50,  # Ajusta este número según tus necesidades
        n_jobs=-1   # Usa todos los núcleos disponibles
    )

    # Ajustar el modelo
    grid_search.fit(X_train, y_train, **fit_params)

    # Obtener resultados
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    results = {
        'best_params': grid_search.best_params_,
        'mean_test_scores_folds': grid_search.cv_results_['mean_test_score'],
        'mean_train_scores_folds': grid_search.cv_results_['mean_train_score']
    }

    # Calcular métricas
    results.update(calculate_metrics(y_test, y_pred_test, 'test'))
    results.update(calculate_metrics(y_train, y_pred_train, 'train'))

    return results

def calculate_metrics(y_true, y_pred, prefix):
    try:
        conf_matrix = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        return {
            f'{prefix}_confusion_matrix': conf_matrix,
            f'sensitivity_{prefix}': tp / (tp + fn),
            f'specificity_{prefix}': tn / (tn + fp),
            f'precision_{prefix}': tp / (tp + fp),
            f'accuracy_{prefix}': (tp + tn) / (tp + tn + fp + fn),
            f'f1_{prefix}': f1_score(y_true, y_pred, average='macro'),
            f'auc_{prefix}': roc_auc_score(y_true, y_pred)
        }
    except:
        return {f'{prefix}_metrics_error': 'Error calculating metrics'}


def train_ori(X,y, splits, days,name,project_name,param_grid):
    # Inicializa WandB

    try:
        X = X.values
        
    except:
        pass   
    try:
        y = y[days +"_READMIT"].to_numpy()
    except:
        y = y
    ####ENtrenamiento del modelo#####
    # funcion de entrenamiento dem odelo
    #df_res = modelo_df_aux_grid_search(X,y,i,type_reg,model,sampling,li_feature_selection,kfolds,lw,K,models_config,config)

    
    

    import pandas as pd

# Supongamos que df es tu DataFrame


            
    

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
        'auc_train':0,
        'auc_test':0,
        'Classifiers':0,
        'Mapping':0,
        'mean_test_scores_folds':0,
        'mean_train_scores_folds':0,
        'time_model':0,
        "prepo":0,
        "best_learning_rate": 0,
        "best_max_delta_step": 0,
        "best_max_depth": 0,
        "best_min_child_weight": 0,
        "best_n_estimators": 0,
        "best_reg_alpha": 0,
        "best_reg_lambda": 0,
        "best_scale_pos_weight": 0,
        "best_subsample": 0,
 }
     

        
             

    timi_ini =time.time()
    #model = LogisticRegression(penalty='l1', solver='saga')
    
    rf_sen,rf_spe,rf_prec,rf_acc,auc_train,auc_test,rf_conf,rf_sen_t,rf_spe_t,rf_prec_t,rf_acc_t,f1,f1_t,    mean_test_scores_folds ,mean_train_scores_folds,param_grid,best_learning_rate,best_max_delta_step,best_max_depth,best_min_child_weight,best_n_estimators,best_reg_alpha,best_reg_lambda,best_scale_pos_weight,best_subsample= function_models2(X,y,model,splits,wandb)
   
    
    time_model = timi_ini-time.time()  
    result['sensitivity_test']=rf_sen
    result['specificity_test']=rf_spe
    result['precision_test']=rf_prec
    result['accuracy_test']=rf_acc
    result['sensitivity_train']=rf_sen_t
    result['specificity_train']=rf_spe_t
    result['precision_train']=rf_prec_t
    result['accuracy_train']=rf_acc_t
    result['f1_test']=f1
    result['f1_train']=f1_t
    result['confusion matrix']=rf_conf
    result['auc_train']=auc_train
    result['auc_test']=auc_test
    result['Mapping']=name
    result["mean_test_scores_folds"]=mean_test_scores_folds
    result["mean_train_scores_folds"]=mean_train_scores_folds
    result["time_model"]=time_model
    #result["prepo"]=str(prepo)
    result["best_max_depth"] = best_max_depth
    result["best_reg_alpha"] = best_reg_alpha
    result["best_reg_lambda"] = best_reg_lambda
   
    result["best_learning_rate"] = best_learning_rate
    result["best_max_delta_step"] = best_max_delta_step
    result["best_max_depth"] = best_max_depth
    result["best_min_child_weight"] = best_min_child_weight
    result["best_n_estimators"] = best_n_estimators
    result["best_reg_alpha"] = best_reg_alpha
    result["best_reg_lambda"] = best_reg_lambda
    result["best_scale_pos_weight"] = best_scale_pos_weight
    result["best_subsample"] = best_subsample
        #result["name"]=i
    
    wandb.init(project=project_name,name =f"experiment_{name}" ,config = param_grid )

    wandb.log(result)
    wandb.finish()
    
    return result
    
    #df_res = pd.DataFrame(result)
    #df_res_aux = pd.concat([df_res_aux,df_res])
                     
                
# Close the WandB run
        
                
      
    


    #concatenación de dataframes 

# se guarda dataframes    
#df_res_aux.to_csv("./results_pred/results_prediction_"+days+"+non_filtered"+nom_t+".csv")   




def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)         

def function_models2_ori(X,y,model,splits,wandb):
    
    #train_size = int(len(X) * 0.75)  # 75% for training
    #X_train, X_test = X[:train_size], X[train_size:]
    #y_train, y_test = y[:train_size], y[train_size:]
    
    train_size = int(len(X) * 0.75)  # 75% for training
    val_size = int(len(X) * 0.15)  # 15% for validation
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    #tscv = TimeSeriesSplit(n_splits=splits)
    tscv = KFold(n_splits=splits, shuffle=False)

   
    grid_search = RandomizedSearchCV(model, param_grid, cv=tscv,scoring="f1",return_train_score=True)
    grid_search.fit(X_train, y_train,eval_set=[(X_val, y_val)], early_stopping_rounds=3)
    best_params = grid_search.best_params_

# Accede a los valores en el diccionario 'best_params'
    best_learning_rate = best_params['learning_rate']
    best_max_delta_step = best_params['max_delta_step']
    best_max_depth = best_params['max_depth']
    best_min_child_weight = best_params['min_child_weight']
    best_n_estimators = best_params['n_estimators']
    best_reg_alpha = best_params['reg_alpha']
    best_reg_lambda = best_params['reg_lambda']
    best_scale_pos_weight = best_params['scale_pos_weight']
    best_subsample = best_params['subsample']

                                                 
    y_pred_train = grid_search.predict(X_train)
    y_pred = grid_search.predict(X_test)
    mean_test_scores_folds = grid_search.cv_results_['mean_test_score']

# Get the mean train score for each parameter combination
    mean_train_scores_folds = grid_search.cv_results_['mean_train_score']
    results = grid_search.cv_results_

# Registrar los resultados en WandB
    for i in range(len(results['params'])):
        wandb.init(project=project_name )
        
        wandb.log({
            'mean_test_score': results['mean_test_score'][i],
            'mean_train_score': results['mean_train_score'][i],
            'params': results['params'][i]
        })
        
        wandb.finish()

   
    try:
        #it obtain metric considered and confussion matrix, metrics for the test set
        rf_conf = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = rf_conf.ravel()
        rf_sen = tp/(tp+fn)
        rf_spe = tn/(tn+fp)
        rf_prec = tp/(tp+fp)
        rf_acc = (tp+tn)/(tp+tn+fp+fn)
        f1 = f1_score(y_test, y_pred, average='macro')
        #metrics for the training set
        rf_conf_t = confusion_matrix(y_train, y_pred_train)
        tn_t, fp_t, fn_t, tp_t = rf_conf_t.ravel()
        rf_sen_t = tp_t/(tp_t+fn_t)
        rf_spe_t = tn_t/(tn_t+fp_t)
        rf_prec_t = tp_t/(tp_t+fp_t)
        rf_acc_t = (tp_t+tn_t)/(tp_t+tn_t+fp_t+fn_t)
        f1_t = f1_score(y_train, y_pred_train, average='macro')
        auc_test = roc_auc_score(y_test, y_pred)
        auc_train = roc_auc_score(y_train, y_pred_train)
    
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
        auc_train = 0
        auc_test = 0


    
    return rf_sen,rf_spe,rf_prec,rf_acc,auc_train,auc_test,rf_conf,rf_sen_t,rf_spe_t,rf_prec_t,rf_acc_t,f1,f1_t,    mean_test_scores_folds ,mean_train_scores_folds,param_grid,best_learning_rate, best_max_delta_step,best_max_depth,best_min_child_weight,best_n_estimators,best_reg_alpha,best_reg_lambda,best_scale_pos_weight,best_subsample,



    

def parameters_fun(model):
    if model == "xgboost":
        model = XGBClassifier(early_stopping_rounds=3,gpu_id=0)
        param_grid = {
        # Reducir el rango de 'learning_rate' para enfocarse en valores que permitan aprendizaje más lento y estable
        'learning_rate': [0.01, 0.05, 0.1,.5,1],
        
        # 'max_delta_step' puede dejarse en un rango conservador para evitar pasos demasiado grandes en las actualizaciones de peso
        'max_delta_step': [0, 1, 2, 3],
        
        # Limitar la profundidad máxima de los árboles para prevenir modelos excesivamente complejos
        'max_depth': [3, 4, 5, 6, 7,24,],
        
        # Aumentar 'min_child_weight' para requerir más observaciones en cada hoja y así evitar sobreajuste
        'min_child_weight': [3, 4, 5, 6],
        
        # Reducir el número de 'n_estimators' para prevenir la complejidad y fomentar modelos más simples
        'n_estimators': [50, 100, 150,500,1000],
        
        # Incrementar los valores de 'reg_alpha' y 'reg_lambda' para fomentar una mayor regularización L1 y L2, respectivamente
        'reg_alpha': [0.01, 0.1, 1, 10],
        'reg_lambda': [0.01, 0.1, 1, 10,15],
        
        # Ajustar 'scale_pos_weight' basándose en el balance de clases en tus datos
        # Esto es específico al problema y requiere conocimiento previo del balance de clases
        'scale_pos_weight':[1,3.8,2,5,7,10],  # Ejemplo genérico, ajustar según tu conjunto de datos
        
        # 'subsample': Elegir valores menores a 1 para usar menos datos y prevenir sobreajuste
        'subsample': [0.3,0.5, 0.6, 0.7, 0.8],
        }


        '''
        param_grid = {
        # Para 'learning_rate', una lista de valores posibles en una escala logarítmica desde 10^-8 a 10^0
        'learning_rate': [c   1e-1, 1e-0],
   
        # Para 'max_delta_step', una lista de valores enteros desde 0 a 10
        'max_delta_step': list(range(0, 10)),
        
        # Para 'max_depth', una lista de valores enteros desde 1 a 30
        'max_depth': list(range(1, 31)),
        
        # Para 'min_child_weight', una lista de valores enteros desde 0 a 15
        'min_child_weight': list(range(0, 16)),
        
        # Para 'n_estimators', una lista de valores enteros representativos desde 1 a 10000
        # Dada la amplia gama, se eligen valores representativos
        'n_estimators': [1, 10, 100, 1000, 5000, 10000],
        
        # Para 'reg_alpha' (alpha region), una lista de valores desde 0.1 a 1 en pasos definidos
        'reg_alpha': [0.1, 0.5, 1],
        
        # Para 'reg_lambda' (lambda region), una lista de valores desde 0.1 a 1.5 en pasos definidos
        'reg_lambda':[1,0.75],
        
        # Para 'scale_pos_weight', una lista de valores desde 0.1 a 1 en pasos definidos
        'scale_pos_weight': [1,3.8,2,5,7,10],
        
        # Para 'subsample', una lista de valores desde 0.1 a 1 en pasos definidos
        'subsample': [0.1, 0.5, 1]
         }'''

    elif model == "logistic":
            model = LogisticRegression()
            '''param_grid = {	
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Type of regularization to be applied
            'C': np.logspace(-4, 4, 20),  # Inverse of regularization strength
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Algorithm to use in the optimization problem
            'max_iter': [100, 1000, 2500, 5000]  # Maximum number of iterations for the solvers to converge
            } '''   
            param_grid = {
            'penalty': ['l2', 'elasticnet'],  # 'l2' y 'elasticnet' favorecen la regularización
            'C': np.logspace(-2, 2, 5),  # Rango más restringido, enfocado en mayor regularización
            'solver': ['newton-cg', 'saga'],  # Solucionadores compatibles con 'elasticnet' y buenos para 'l2'
            'max_iter': [100, 1000],  # Reducción de opciones para max_iter, enfocándose en valores más comunes
            'l1_ratio': np.linspace(0, 1, 5)  # Solo necesario si 'elasticnet' es elegido; equilibrio entre l1 y l2
        }
    return model ,param_grid       
        




    
 
    
if __name__ == "__main__":
    
    archivo = "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/input/"
    archivo_input_label ="/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/preprocessing2/input/ADMISSIONS.csv.gz"
    category = "diagnosis"
    output_path = "results/models_cluster/"+category+"/prediction/hyperparametertuning/"

  
    project_name =   "Predic_Readmission_diagnosis_Xgboost_kfolds_preproC_new"
    splits = 5

    classifiers = ['logistic', 'xgboost']
    #inputs
        
    days = "30"
    type_a="visit"
    archivo_completa =archivo + category+"_visit"
         
    list_cat_aux= listar_archivos(archivo_completa)
 
    # Initialize an empty list to store all results
    df_final = []
    admission_file = pd.read_csv(archivo_input_label)


    
    for i, name in enumerate(list_cat_aux[-2:-1]):
        print(name)
        for model in classifiers:
            # por cada modelo en cada threshold
            ruta = os.path.join(archivo_completa, name)
            df = pd.read_csv(ruta)
            df = df.iloc[:, 1:]
            #df = df[:2000]
            readmit_df = label_funv2(days,admission_file, df)
            readmit_df.drop(columns='DISCHTIME', inplace=True)
            
            prepo = "std"
            
            X = readmit_df[[col for col in readmit_df.columns if col != f'{days}_READMIT']]
            X = preprocess(X, prepo)
            y = readmit_df[f'{days}_READMIT']
            
            model ,param_grid =parameters_fun(model)
            
            train(X, y, splits, days, name, project_name, param_grid,model)
            
    
    wandb.finish()
        
            
    





    




