import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
hand = mp_hands.Hands()

while True:
    success, img = cap.read()
    if success:
        RGB_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = hand.process(RGB_img)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break