""" Combine parquet files by starting number prefix """

import os
import pandas as pd
import re

os.chdir(os.path.dirname(os.path.abspath(__file__)))
folder_path = './landmarks'

combined_dfs = {}

for filename in os.listdir(folder_path):
    if filename.endswith('.parquet'):
        match = re.match(r'(\d+)', filename)
        if match:
            start_number = match.group(1)
            
            file_path = os.path.join(folder_path, filename)
            df = pd.read_parquet(file_path)
            
            if start_number not in combined_dfs:
                combined_dfs[start_number] = df
            else:
                combined_dfs[start_number] = pd.concat([combined_dfs[start_number], df])

for start_number, combined_df in combined_dfs.items():
    output_path = f'./10{start_number}.parquet'
    combined_df.to_parquet(output_path)

print("Parquet files combined successfully.")
