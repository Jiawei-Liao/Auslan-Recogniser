import mediapipe as mp
import cv2
import os
import numpy as np
from PIL import Image

def get_joints(image_path):
    # Get image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Use mediapipe to get hand joint positions
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
    results = hands.process(image)

    # Pre define landmarks in case there are missing hands or joints
    all_landmarks = [[(0,0)] * 21, [(0,0)] * 21]

    # Finding if mediapipe found right or left hand first
    rightFirst = False
    try:
        if results.multi_handedness[0].classification[0].index == 0 and results.multi_handedness[0].classification[0].label == "Right":
            rightFirst = True
    except:
        pass
    
    # Verifying dataset as mediapipe may not see every joint due to lighting issues or something else
    try:
        if not (results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2):
            print(image_path, "doesn't have 42 joints or 2 hands, training anyways")
            if len(results.multi_hand_landmarks[0].landmark) != 21:
                print("missing 21 joints for hand 0")
            if len(results.multi_hand_landmarks[1].landmark) != 21:
                print("missing 21 joints for hand 1")
    except:
        pass
    
    # Find joint positions in both hands
    try:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_landmark = [(0,0)] * 21
            for joint_index, landmark in enumerate(hand_landmarks.landmark):
                x = landmark.x
                y = landmark.y
                hand_landmark[joint_index] = (x, y)
            if hand_index == 0:
                all_landmarks[rightFirst] = hand_landmark
            else:
                all_landmarks[not rightFirst] = hand_landmark
    except Exception:
        print("failed to get joint positions:" + image_path)

    all_landmarks_np = np.array(all_landmarks)

    landmark_normalised = (all_landmarks_np * 255).astype(int)
    image = np.concatenate((landmark_normalised, np.zeros_like(landmark_normalised[:, :, :1])), axis=-1)

    return image

def main():
    root_dir = './dataset'
    output_dir = './image_representation'
    for letter in os.listdir(root_dir):
        letter_dir = os.path.join(root_dir, letter)
        output_letter_dir = os.path.join(output_dir, letter)

        if not os.path.exists(output_letter_dir):
            os.makedirs(output_letter_dir)
        for filename in os.listdir(letter_dir):
            image_path = os.path.join(letter_dir, filename)
            output_letter_path = os.path.join(output_letter_dir, filename)
            image_representation = get_joints(image_path)

            cv2.imwrite(output_letter_path, image_representation)
        
if __name__ == "__main__":
    main()