1. Run `flip_video.py` on any directories of video with left-handedness
2. Run `process_sessions_to_parquet.py`
    a. This will create a folder called output_parquets
3. Run `split_parquet.py`
    a. This will create a folder called landmarks
4. Run `combine_parquets.py`
    a. This will create the final parquet `101.parquet`