import subprocess

# Input for clustering_ Admission and Prediction
#python tu_script.py diagnosis --normalize_matrix --log_transformation --encode_categorical
'''Esto significa que estos argumentos serán True por defecto y se establecerán en False cuando 
se proporcionen en la línea de comandos.'''

subprocess.run(["python", "preprocess_class1.py", "diagnosis"])
subprocess.run(["python", "preprocess_class1.py", "procedures"])
subprocess.run(["python", "preprocess_class1.py", "drug2"])

#Input not preprocesed for SD
#python tu_script.py diagnosis --normalize_matrix --log_transformation --encode_categorical
subprocess.run(["python", "preprocess_class1.py", "diagnosis", "--normalize_matrix", "--log_transformation", "--encode_categorical"])
subprocess.run(["python", "preprocess_class1.py", "procedures", "--normalize_matrix", "--log_transformation", "--encode_categorical"])
subprocess.run(["python", "preprocess_class1.py", "drug2", "--normalize_matrix", "--log_transformation", "--encode_categorical"])


#input condat in preprocesin carpet
#"concat","entire_ceros","temporal_state"
#python preprocess_input_SDmodel.py "concat"

#run for dataset padded with 0
#python preprocess_input_SDmodel.py "entire_ceros"

#dataset con shape (n_patient, n_features ,ntime_step), statics features
#python preprocess_input_SDmodel.py "temporal_state"