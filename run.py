import cv2
import mediapipe as mp
import pandas as pd
import tensorflow as tf
import numpy as np
import json
from get_the_data import get_the_data
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def video_attempt(video_path):
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose

    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Change from camera to video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    print(f"Video FPS: {fps}")
    print(f"Frame Count: {frame_count}")
    print(f"Duration: {duration} seconds")

    with open('./create_tfr/inference_args.json') as f:
        data = json.load(f)

    selected_cols = data['selected_columns']

    with open("./create_tfr/character_to_prediction_index.json", "r") as f:
        character_map = json.load(f)
    rev_character_map = {j: i for i, j in character_map.items()}

    interpreter = tf.lite.Interpreter(model_path='./train/tflite_models/100epoch_base.tflite')
    prediction_fn = interpreter.get_signature_runner("serving_default")
    
    inputs = interpreter.get_input_details()
    print('{} input(s):'.format(len(inputs)))
    for i in range(0, len(inputs)):
        print('{} {}'.format(inputs[i]['shape'], inputs[i]['dtype']))

    outputs = interpreter.get_output_details()
    print('\n{} output(s):'.format(len(outputs)))
    for i in range(0, len(outputs)):
        print('{} {}'.format(outputs[i]['shape'], outputs[i]['dtype']))

    df = pd.DataFrame(columns=selected_cols)
    print('df initial shape:')
    print(df.shape)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
    
        if frame_count % 2 == 0 or frame_count % 3 == 0:
            continue

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        cv2.imshow('Sign Language Recognition', frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(rgb_frame)
        results_face_mesh = face_mesh.process(rgb_frame)
        results_pose = pose.process(rgb_frame)

        df = pd.concat([df, get_the_data(hands=results_hands, face=results_face_mesh, pose=results_pose,
                                         selected_cols=selected_cols)], ignore_index=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('df shape:')
    print(df.shape)

    my_tensor = tf.convert_to_tensor(df)
    output = prediction_fn(inputs=my_tensor)

    prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output['outputs'], axis=1)])
    print('prediction: ')
    print(prediction_str)

    hands.close()
    face_mesh.close()
    pose.close()
    cap.release()
    cv2.destroyAllWindows()

# Usage
video_path = './process_data/dataset/01/01.mp4'
video_path = './test_alphabet.mp4'
video_attempt(video_path)