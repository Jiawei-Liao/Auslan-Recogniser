import pyarrow.parquet as pq
# Open the Parquet file
parquet_file = pq.ParquetFile('../parquet/1019715464.parquet')

# Read the data from the file
data = parquet_file.read()

df = data.to_pandas()

print('hi')
