import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

# %%
mp_drawing = mp.solutions.drawing_utils  # easier to render
mp_hands = mp.solutions.hands  # hand model

# %%
os.mkdir('../Output Images')


# %%
def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))

            # extracting coordinates
            coords = tuple(np.multiply(  # put in numpy array
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [1920, 1080]).astype(int))  # multiply by the dimensions of the webcam
            output = text, coords
    return output


# %%
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()  # frame is the image

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # recolour the image BGR -> RGB
        image = cv2.flip(image, 1)  # flip on horizontal
        image.flags.writeable = False

        # Detections
        results = hands.process(image)
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # print(results.multi_hand_landmarks)

        # rendering results
        if results.multi_hand_landmarks:  # check if anything in there
            for num, hand in enumerate(results.multi_hand_landmarks):  # loop through each result
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=4, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=4, circle_radius=2)
                                          )

                if get_label(num, hand, results):
                    text, coord = get_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imwrite(os.path.join('../Output Images', '{}.jpg'.format(uuid.uuid1())), image)
        cv2.imshow('Hand tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.read()
cv2.destroyAllWindows()
