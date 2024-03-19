import pandas as pd
def guardar_features(archivo,type_features):
    #archivo'aux/features.csv'
#synthetic_features
    if type_features=="temporal":
        lista_dataframes = [pd.DataFrame(features[i]) for i in range(features.shape[0])]
        df_total_synthetic_features = pd.concat(lista_dataframes, keys=[f'dataframe_{i}' for i in range(len(lista_dataframes))])
        df_total_synthetic_features.to_csv(archivo)
    else;    
        pd.DataFrame(synthetic_attributes).to_csv("aux/synthetic_attributes.csv")


def desconcat_features(archivo,how_desconcat):
    #aux/synthetic_features.csv /'aux/features.csv'
    df = pd.read_csv(archivo, header=[0,1], index_col=0)
    cols_to_drop = df.filter(like='Unnamed', axis=1).columns
    df.drop(cols_to_drop, axis=1, inplace=True)
    if how_desconcat="numpy":
       synthetic_features = df.to_numpy()
    else:
        #same dimension as before
        grouped = df.groupby(df.index)
        synthetic_features = [group.values for _, group in grouped]
        synthetic_features = [arr for arr in synthetic_features if arr.shape == synthetic_features[1].shape]
        synthetic_features = np.stack(synthetic_features)
    return synthetic_features    

def desconcat_attributes(archivo):
    #'aux/synthetic_attributes.csv'  'aux/attributes.csv


    attributes = pd.read_csv(archivo, header=[0,1], index_col=0)
    cols_to_drop = attributes.filter(like='Unnamed', axis=1).columns
    attributes.drop(cols_to_drop, axis=1, inplace=True)
    return attributes
