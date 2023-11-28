import cv2
import mediapipe as mp
import pickle
import os
import numpy as np


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)



model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
landmark_data = []  
label_data = []
# labels_dict= {0:"ctrl+p", 1:"ctrl+l", 2:"ctrl+m"}
labels_dict = {0:"close-palm", 1:"open-palm", 2:"select"}

cap = cv2.VideoCapture(0)



landmark_positions = []
x_ = []
y_ = []

frame = cv2.imread("/home/lnx/Documents/Body-Language-Decoder/sign-language-detector-python/data/1/3.jpg")

H, W, _ = frame.shape
image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = hands.process(image_rgb)

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        for i in range(0, 21):
            x_.append(hand_landmarks.landmark[i].x)
            y_.append(hand_landmarks.landmark[i].y)
                
            x, y = int(hand_landmarks.landmark[i].x * W), int(hand_landmarks.landmark[i].y * H)
            cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)
            
        landmark_positions.extend(x_)
        landmark_positions.extend(y_)
                

cv2.imshow("frame", frame)
prediction = model.predict([np.asarray(landmark_positions)])    
print(int(prediction[0])) 
print(f"x_ : {x_}")
print(f"y_ : {y_}")
print(f"x,y : {landmark_positions}")



cv2.waitKey(0)
cv2.destroyAllWindows()

                
               





