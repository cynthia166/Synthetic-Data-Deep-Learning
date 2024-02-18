import pandas as pd
import json
import argparse
from function_mapping import *  # Assuming this module contains all the necessary functions.

def main(config):
    # Read the data from the specified CSV file
    df = pd.read_csv(config["archivo"])
    categorical_cols = config["categorical_cols"]
    type_a = config["type_a"]
    stri = config["type_a"]
    # Set pandas option to display all columns (optional)
    pd.set_option('display.max_columns', None)
    
    # Perform data processing based on the configuration
    if config["nom_t"] in ["Drugs", "Diagnosis"]:
        # Process original codes
        nuevo_df_x = desconacat_codes_ori(df, config["ori"])

        # Iterate over the list of categories and process them
        for i, real in enumerate([config["list_cat"][-1]]):
            # Retrieve the corresponding name from nam_p_list
            

            # Perform processing with the imported functions
            # The actual function calls will depend on your processing logic and the functions defined in function_mapping.py
            # Example: X = process_data_function(df, real, nam_p, config)
            
            # Save the processed data to CSV
            output_filename = config["file_save"] + real + "_" + config["type_a"] + "_non_filtered.csv"
            
            archivo = "data/data_preprocess_nonfilteres.csv"
            
            X = input_for_pred_mutualinfo(df,categorical_cols,real,stri,archivo,type_a,nuevo_df_x)
            X.to_csv(output_filename)
            print(f"Saved processed data to {output_filename}")

# Entry point for the script
if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='File path for JSON config file')
    args = parser.parse_args()
    
    # Load the configuration settings from the JSON file
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    
    # Run the main function with the loaded configuration
    main(config)
