import torch
import cv2
import numpy as np
import mediapipe as mp
from trainer import FingerspellingModel

model_state_dict = torch.load('fingerspelling_model.pth')

model = FingerspellingModel()
model.load_state_dict(model_state_dict)
model.eval()

index_to_letter = {
    0: 'A',
    1: 'B'
}

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    all_landmarks = [[(0,0)] * 21, [(0,0)] * 21]
    rightFirst = False

    try:
        if results.multi_handedness[0].classification[0].index == 0 and results.multi_handedness[0].classification[0].label == "Right":
            rightFirst = True
    except:
        pass

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
        pass
    
    return torch.tensor([all_landmarks], dtype=torch.float32)

def predict_letter(image):
    # Preprocess input
    input_tensor = preprocess(image)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
    
    index = torch.argmax(output)
    alphabet = "ABCDEFGIKLMNOPQRSTUVWXYZ"
    letter = alphabet[index]

    return letter

cap = cv2.VideoCapture(0)  # Use camera
while True:
    ret, frame = cap.read()  # Read frame from camera

    # Call prediction function
    predicted_letter = predict_letter(frame)

    # Display the result on the frame
    cv2.putText(frame, predicted_letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Camera', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()