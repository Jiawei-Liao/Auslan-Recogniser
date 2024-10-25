import cv2
import mediapipe as mp
import pandas as pd
import tensorflow as tf
import numpy as np
import json
from get_the_data import get_data_from_pq, get_the_data


def main():
    # parquet_attempt()
    video_attempt()


def parquet_attempt():
    with open('../models/inference_args.json') as f:
        data = json.load(f)

    # selected_cols = data['selected_columns']

    with open("../models/character_to_prediction_index.json", "r") as f:
        character_map = json.load(f)
    rev_character_map = {j: i for i, j in character_map.items()}

    interpreter = tf.lite.Interpreter(model_path='../models/model.tflite')

    prediction_fn = interpreter.get_signature_runner("serving_default")

    frame = get_data_from_pq('../parquet/1019715464.parquet', 2002968557)
    my_tensor = tf.convert_to_tensor(frame)

    output = prediction_fn(inputs=my_tensor)

    prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output['outputs'], axis=1)])

    print(prediction_str)


def video_attempt():
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose

    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    with open('../models/inference_args.json') as f:
        data = json.load(f)

    selected_cols = data['selected_columns']

    with open("../models/character_to_prediction_index.json", "r") as f:
        character_map = json.load(f)
    rev_character_map = {j: i for i, j in character_map.items()}

    interpreter = tf.lite.Interpreter(model_path='../models/model.tflite')
    prediction_fn = interpreter.get_signature_runner("serving_default")

    # Set up the data frame with the columns
    df = pd.DataFrame(columns=selected_cols)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Sign Language Recognition', frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(rgb_frame)
        results_face_mesh = face_mesh.process(rgb_frame)
        results_pose = pose.process(rgb_frame)

        # Add to the data frame every frame
        df = pd.concat([df, get_the_data(hands=results_hands, face=results_face_mesh, pose=results_pose,
                                         selected_cols=selected_cols)], ignore_index=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # call the prediction_fn on the input data.
    my_tensor = tf.convert_to_tensor(df)
    output = prediction_fn(inputs=my_tensor)

    prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output['outputs'], axis=1)])

    print(prediction_str)

    hands.close()
    face_mesh.close()
    pose.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
