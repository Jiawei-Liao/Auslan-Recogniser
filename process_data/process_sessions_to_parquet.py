""" Process video data and save to Parquet files """

import os
import cv2
import mediapipe as mp
import json
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import re
import gc
import numpy as np

""" Process a frame with MediaPipe and extract landmark data """
def process_frame(frame, holistic, selected_columns):
    # Read frame as RGB to process with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)

    # Extract landmark data
    face_landmarks = results.face_landmarks
    pose_landmarks = results.pose_landmarks
    left_hand_landmarks = results.left_hand_landmarks
    right_hand_landmarks = results.right_hand_landmarks

    # Helper function to get landmark value or None if not present
    def get_landmark_value(landmarks, index, axis):
        if landmarks and index < len(landmarks.landmark):
            return getattr(landmarks.landmark[index], axis)
        return None

    # Process extracted data into required format
    frame_data = {}
    for col in selected_columns:
        match = re.match(r"([xyz])_(\w+)_(\d+)", col)
        if match:
            axis, part, index = match.groups()
        else:
            raise ValueError(f"Invalid format: {col}")
        index = int(index)

        if part == 'face':
            value = get_landmark_value(face_landmarks, index, axis)
        elif part == 'pose':
            value = get_landmark_value(pose_landmarks, index, axis)
        elif part == 'left_hand':
            value = get_landmark_value(left_hand_landmarks, index, axis)
        elif part == 'right_hand':
            value = get_landmark_value(right_hand_landmarks, index, axis)
        else:
            value = None

        frame_data[col] = value

    return frame_data

""" Process a video to create parquet file """
""" Split into chunks to avoid eating up too much memory """
def process_video(video_path, json_path, sequence_id, chunk_size=100):
    # Load required columns from JSON file
    with open(json_path, 'r') as f:
        selected_columns = json.load(f)['selected_columns']

    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=2)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for chunk_start in range(0, total_frames, chunk_size):
        chunk_data = {
            'frame': [],
            'sequence_id': []
        }
        for col in selected_columns:
            chunk_data[col] = []

        # Process frames in this chunk
        for frame_num in range(chunk_start, min(chunk_start + chunk_size, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            frame_data = process_frame(frame, holistic, selected_columns)

            chunk_data['frame'].append(np.int16(frame_num))
            chunk_data['sequence_id'].append(np.int64(sequence_id))

            # Add landmark data
            for col in selected_columns:
                value = frame_data[col]
                chunk_data[col].append(np.float32(value) if value is not None else None)

        schema = pa.schema([
            ('frame', pa.int16()),
            ('sequence_id', pa.int64())
        ] + [
            (col, pa.float32()) for col in selected_columns
        ])

        yield chunk_data, schema

        # Force garbage collection after each chunk
        gc.collect()

    # Release video capture
    cap.release()
    holistic.close()

""" Process all videos in a dataset and save to Parquet files """
def process_dataset(dataset_path, json_path, output_dir):
    for session_folder in sorted(os.listdir(dataset_path)):
        session_path = os.path.join(dataset_path, session_folder)
        print(f"Processing session folder: {session_folder}")

        if os.path.isdir(session_path):
            session_number = int(session_folder)
            output_parquet_path = os.path.join(output_dir, f'session_{session_number:02d}.parquet')
            print(f"Output path for session {session_number}: {output_parquet_path}")

            # Create a ParquetWriter for each session
            writer = None

            for video_file in sorted(os.listdir(session_path)):
                print(f"Processing video file: {video_file}")
                if video_file.endswith(".mp4"):
                    video_number = int(os.path.splitext(video_file)[0])
                    sequence_id = int(f'1{session_number:02d}{video_number:02d}')
                    video_path = os.path.join(session_path, video_file)
                    print(f"Processing sequence ID: {sequence_id}")

                    try:
                        # Process the video in chunks
                        for chunk_data, schema in tqdm(process_video(video_path, json_path, sequence_id),
                                               desc=f"Processing {video_file}"):
                            table = pa.Table.from_pydict(chunk_data, schema=schema)

                            if writer is None:
                                # Create a new ParquetWriter if it doesn't exist
                                writer = pq.ParquetWriter(output_parquet_path, schema,
                                                          use_dictionary=True, compression='snappy')

                            writer.write_table(table)

                            # Clear the chunk data and force garbage collection
                            del chunk_data, table
                            gc.collect()

                        print(f"Data for video {video_number} (session {session_number}) appended to {output_parquet_path}")
                    except Exception as e:
                        print(f"Error processing video {video_file}: {str(e)}")

            # Close the ParquetWriter after processing all videos in the session
            if writer:
                writer.close()

        # Force garbage collection after each session
        gc.collect()

if __name__ == '__main__':
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_file_directory, 'dataset')
    json_path = os.path.join(current_file_directory, 'inference_args.json')
    output_dir = os.path.join(current_file_directory, 'output_parquets')

    os.makedirs(output_dir, exist_ok=True)
    process_dataset(dataset_path, json_path, output_dir)
