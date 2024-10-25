import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import json

from joblib import Parallel, delayed
import multiprocessing as mp
from multiprocessing import cpu_count
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, KFold

import tensorflow as tf
cpu_count()

os.chdir(os.path.dirname(os.path.abspath(__file__)))

train_df = pd.read_csv('./train.csv')
print('train_df')
print(train_df)

temp = pd.read_parquet('./landmarks/101.parquet')
print('temp')
print(temp)
print('temp.loc[temp.index==10101]')
print(temp.loc[temp.index==10101])

ROWS_PER_FRAME = 543
def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

with open('./character_to_prediction_index.json') as json_file:
    LABEL_DICT = json.load(json_file)

selected_columns = temp.drop(columns=['frame']).columns
print('selected_columns')
print(selected_columns)

def load_relevant_data_subset(pq_path):
    return pd.read_parquet(pq_path, columns=selected_columns)

def encode_strings(string_list, mapping=LABEL_DICT):
    return [[mapping[char] for char in string] for string in string_list]
    
def encode_row(row):
    coordinates = load_relevant_data_subset(f'{row.path}')
    coordinates = coordinates.loc[coordinates.index==row.sequence_id].values.astype('float32')
    coordinates_encoded = coordinates.tobytes()
    participant_id = int(row.participant_id)
    sequence_id = int(row.sequence_id)
    phrase = row.phrase.encode('utf-8')
    phrase_encoded = encode_strings([row.phrase])[0]
    record_bytes = tf.train.Example(features=tf.train.Features(feature={
                'coordinates': tf.train.Feature(bytes_list=tf.train.BytesList(value=[coordinates_encoded])),
                'file_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[participant_id])),
                'participant_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[participant_id])),
                'sequence_id':tf.train.Feature(int64_list=tf.train.Int64List(value=[sequence_id])),
                'phrase': tf.train.Feature(bytes_list=tf.train.BytesList(value=[phrase])),
                'phrase_encoded': tf.train.Feature(int64_list=tf.train.Int64List(value=phrase_encoded))
                })).SerializeToString()
    return record_bytes

def process_chunk(chunk, tfrecord_name):
    options = tf.io.TFRecordOptions(compression_type='GZIP', compression_level=9)
    with tf.io.TFRecordWriter(tfrecord_name, options=options) as file_writer:
        for i, row in tqdm(chunk.iterrows()):
            record_bytes = encode_row(row)
            file_writer.write(record_bytes)
            del record_bytes
        file_writer.close()

row = train_df.iloc[0]

N_FILES = len(train_df)
CHUNK_SIZE = 512
N_PART = 1
FOLD = 4
part = 0

class CFG:
    seed = 42
    n_splits = 5

train_folds = train_df.copy()
train_folds['fold']=-1

train_folds = train_folds.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)

kfold = GroupKFold(n_splits=CFG.n_splits) 
print(f'{CFG.n_splits}fold training', len(train_folds), 'samples')
for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(train_folds, groups=train_folds.participant_id)):
    train_folds.loc[valid_idx,'fold'] = fold_idx
    print(f'fold{fold_idx}:', 'train', len(train_idx), 'valid', len(valid_idx))
    
assert not (train_folds['fold']==-1).sum()
assert len(np.unique(train_folds['fold']))==CFG.n_splits
print('train_folds.head()')
print(train_folds.head())

def split_dataframe(df, chunk_size=10000):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
    return chunks

for fold in range(CFG.n_splits):
    rows = train_folds[train_folds['fold'] == fold]
    chunks = split_dataframe(rows, CHUNK_SIZE)
    
    part_size = len(chunks) // N_PART
    last = (part + 1) * part_size if part != N_PART - 1 else len(chunks)
    chunks = chunks[part * part_size:last]

    if not chunks:
        print(f"Empty chunks list for part {part} in fold {fold}")
    N = [len(x) for x in chunks]

    def safe_process_chunk(x, filepath):
        try:
            process_chunk(x, filepath)
        except KeyError as e:
            print(f"KeyError: {e} in fold {fold}, chunk with length {len(x)}")

    _ = Parallel(n_jobs=cpu_count(), timeout=3600)(
        delayed(safe_process_chunk)(x, f'./tfrecords/fold{fold}-{i}-{n}.tfrecords')
        for i, (x, n) in enumerate(zip(chunks, N))
    )
