""" Auslan data is actually 3-4x faster than Kaggle ASL data """
""" Split data to match Kaggle ASL data and have more data to train on """

import pandas as pd
import os
from typing import List
current_file_directory = os.path.dirname(os.path.abspath(__file__))

""" Update sequence_id to replace the first digit with the split number """
def update_sequence_id(df: pd.DataFrame, split_num: int) -> pd.DataFrame:
    df['sequence_id'] = df['sequence_id'].astype(str).str.replace(r'^\d', str(split_num), regex=True)
    return df


""" Split the parquet file into three groups based on frame number """
def split_parquet_by_frame(session_num: int, num_groups: int = 3) -> None:
    # Load the parquet file based on the session number
    input_file = os.path.join(current_file_directory, 'output_parquet/session_{session_num:02}.parquet')
    df = pd.read_parquet(input_file)
    
    # Split data into three groups based on frame number
    grouped_dfs = [df[df['frame'] % num_groups == i].copy() for i in range(1, num_groups + 1)]

    for i, group in enumerate(grouped_dfs, start=1):
        group['frame'] = group.groupby('sequence_id').cumcount()

        # Update sequence_id
        grouped_dfs[i - 1] = update_sequence_id(group, i)

        # Define output filename
        output_file = os.path.join(current_file_directory, 'landmarks/{i}{session_num:02}.parquet')

        # Save the group into a separate parquet file
        grouped_dfs[i - 1].to_parquet(output_file)
        print(f"Parquet file for session {session_num} saved: {output_file}")

def main(sessions: List[int]) -> None:
    for session in sessions:
        split_parquet_by_frame(session)

if __name__ == '__main__':
    sessions = list(range(1, 15))
    main(sessions)
