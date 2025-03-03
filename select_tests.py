import pandas as pd
import os
import random

def select_random_rows_from_multiple_files(directory_path, output_path, num_rows=100):
    # List to hold dataframes from all CSV files
    all_dfs = []
    
    # Iterate over all CSV files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            all_dfs.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Check if the number of rows in the combined dataframe is less than the number of rows to select
    if len(combined_df) < num_rows:
        raise ValueError("The combined files do not contain enough rows to select the specified number of rows.")
    
    # Filter rows where the "Tokens" column is not empty
    filtered_df = combined_df[combined_df['Tokens'].notna() & (combined_df['Tokens'] != '')]
    
    # Check if the number of rows in the filtered dataframe is less than the number of rows to select
    if len(filtered_df) < num_rows:
        raise ValueError("The filtered files do not contain enough rows to select the specified number of rows.")
    
    # Randomly select rows without replacement from the filtered dataframe
    selected_rows = filtered_df.sample(n=num_rows, replace=False, random_state=random.randint(0, 10000))
    
    # Save the selected rows to a new CSV file
    selected_rows.to_csv(output_path, index=False)

def process_directory(directory_path, output_directory, num_rows=100):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Define the output file path
    output_path = os.path.join(output_directory, "test_selection.csv")
    
    # Select random rows from multiple files
    select_random_rows_from_multiple_files(directory_path, output_path, num_rows)

# Example usage
directory_path = 'training data/'
output_directory = 'sample_tests/'
process_directory(directory_path, output_directory)