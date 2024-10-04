0. Put parquet files created by process_data into /landmarks folder
1. Run `docker build -t create_tfr .`
2. Run `docker run -it --rm -v ${pwd}/tfrecords:/app/tfrecords create_tfr`