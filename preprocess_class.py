
from preprocess_input import *
class PreprocessInput:
    def __init__(self):
        # Inicialización de atributos que siempre son necesarios
        self.atributo_general = None

    def concat_archivo_primeto(self,procedures,admi,ruta_archivos,save):
        # concatenacion de la source
        return concat_archivo_primeto(procedures,admi,ruta_archivos,save)

    def diagosis(self, input_2):
        # Operación específica 2
        print(f"Operación específica 2 con {input_2}")

    def ejecutar_flujo(self, usar_metodo_1, input_1=None, usar_metodo_2=False, input_2=None):
        """
        Método gestor que controla la ejecución de otros métodos según las necesidades.
        
        :param usar_metodo_1: Booleano que indica si se debe llamar a metodo_especifico_1.
        :param input_1: Entrada para metodo_especifico_1 si se utiliza.
        :param usar_metodo_2: Booleano que indica si se debe llamar a metodo_especifico_2.
        :param input_2: Entrada para metodo_especifico_2 si se utiliza.
        """
        if usar_metodo_1 and input_1 is not None:
            self.metodo_especifico_1(input_1)
        
        if usar_metodo_2 and input_2 is not None:
            self.metodo_especifico_2(input_2)

# Ejemplo de uso
#concat
procedures = ".s_data\ADMISSIONS.csv.gz"
admi = 's_data\ADMISSIONS.csv.gz'
nom_archivo = 'data\df_non_filtered.parquet'

numerical_cols =  ['Age_max', 'LOSRD_sum',
       'L_1s_last', 'LOSRD_avg','L_1s_last_p1']

categorical_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION',
                'DISCHARGE_LOCATION', 'INSURANCE',  'RELIGION',
                'MARITAL_STATUS', 'ETHNICITY','GENDER']


ruta_archivos = 's_data\*.csv.gz'  # Puedes cambiar '*.csv' por la extensión que desees
save = True #dfale

mi_objeto = PreprocessInput()
df_diagnosis = concat_archivo_primeto(procedures,admi,ruta_archivos,save,nom_archivo)


n = [.88,.95,.98,.999]

d1 = '.\s_data\DIAGNOSES_ICD.csv.gz'
name="ICD9_CODE"
didf_diagnosisa = diagnosis(d1,n,name)

#desconcatenación, y se obtiene threshold y mapping
d2 = '.\s_data\PROCEDURES_ICD.csv.gz' 
prod = procedures(d2,n,name)
prod = limipiar_Codigos(prod)

d1 = '..\s_data\PRESCRIPTIONS.csv.gz'
name1 = "DRUG"
df_drugs = drugs(d1,name1)

name_df = "raw_input.csv"
name_encodeing = "input_onehot_encoding.csv"


#df_final = concat_input(df_drugs, df_diagnosis, df_procedures,numerical_cols,categorical_cols,name_df)
#df_final_encoded = encoding(df_final,categorical_cols)