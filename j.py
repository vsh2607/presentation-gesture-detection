import cv2
import mediapipe as mp
import pickle
import pandas as pd
import os


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

landmark_data = []  
label_data = []
root_dir = "/home/lnx/Documents/Body-Language-Decoder/sign-language-detector-python/data"


for dir_ in os.listdir(root_dir):
    for image in os.listdir(os.path.join(root_dir, dir_)):
        frame = cv2.imread(os.path.join(root_dir, dir_, image))

        landmark_positions = []
        x_ = []
        y_ = []
        z_ = []

        frame = cv2.flip(frame, 1)

        H, W, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                    z_.append(landmark.z)
               
                landmark_positions.extend(x_)
                landmark_positions.extend(y_)
                landmark_positions.extend(z_)

                landmark_data.append(landmark_positions)
                label_data.append(dir_)



f = open('data.pickle', 'wb')
pickle.dump({'data': landmark_data, 'labels': label_data}, f)
f.close()


