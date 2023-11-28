import cv2
import mediapipe as mp
import pandas as pd
import os


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

landmark_data = []  
root_dir = "/home/lnx/Documents/Body-Language-Decoder/sign-language-detector-python/data"


for dir_ in os.listdir(root_dir):
    for image in os.listdir(os.path.join(root_dir, dir_)):
        frame = cv2.imread(os.path.join(root_dir, dir_, image))

        frame = cv2.flip(frame, 1)

        H, W, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_row = {'label':dir_}
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    landmark_row[f"x{idx}"] = landmark.x   
                    landmark_row[f"y{idx}"] = landmark.y   
                landmark_data.append(landmark_row)




df = pd.DataFrame(landmark_data)
df.to_csv('hand_landmarks_dataset.csv', index=False)


