import os
import pandas as pd

current_file_directory = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(current_file_directory, 'landmarks')
output_path = os.path.join(current_file_directory, '101.parquet')

combined_df = pd.DataFrame()
for filename in os.listdir(folder_path):
    if filename.endswith('.parquet'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_parquet(file_path)
        combined_df = pd.concat([combined_df, df])

combined_df.to_parquet(output_path)