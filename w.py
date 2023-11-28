import cv2
import mediapipe as mp
import os
import pandas as pd
import pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def extract_landmark_data(image_folder):
    data = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_row = {'image_path': filename}
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        landmark_row[f'landmark_{idx}_x'] = landmark.x
                        landmark_row[f'landmark_{idx}_y'] = landmark.y
                    landmark_row['label'] = 0
                    data.append(landmark_row)
    
    return data

landmark_data = extract_landmark_data('/home/lnx/Documents/Body-Language-Decoder/sign-language-detector-python/data/0')

df = pd.DataFrame(landmark_data)

df.to_csv('hand_landmarks_dataset.csv', index=False)

with open('hand_landmarks_dataset.pkl', 'wb') as f:
    pickle.dump(df, f)

hands.close()
