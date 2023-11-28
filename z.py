import cv2
import mediapipe as mp
import pickle
import os
import numpy as np


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)



model_dict = pickle.load(open('./model_hand.p', 'rb'))
model = model_dict['model']

model_dict_gesture = pickle.load(open('./model.p', 'rb'))
model_gesture = model_dict['model']



landmark_data = []  
label_data = []

labels_dict= {0:"right-hand", 1:"left-hand"}
labels_gesture_dict = {0:"close-palm", 1:"open-palm", 2:"select"}
# labels_gesture_dict = {0:"double_select", 1:"ctrl-l", 2:"ctrl-p", 3:"ctrl-e", 4:"ctrl-m", 5:"right-palm", 6:"left-palm", 7:"single_select", 8:"ctrl-i"}
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    H, W, _ = frame.shape   
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
            for i in range(0, 21):
                x_.append(hand_landmarks.landmark[i].x)
                y_.append(hand_landmarks.landmark[i].y)
                z_.append(hand_landmarks.landmark[i].z)
                
                x, y = int(hand_landmarks.landmark[i].x * W), int(hand_landmarks.landmark[i].y * H)
                cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)
               
            landmark_positions.extend(x_)
            landmark_positions.extend(y_)
            landmark_positions.extend(z_)
            
   
            
            prediction_hand = model_gesture.predict([np.asarray(landmark_positions)])
            print(labels_gesture_dict[int(prediction_hand[0])])
            
    cv2.imshow("frame", frame)
    
      
    
    
    
    if cv2.waitKey(1) & 0XFF == ord("q"):
        break

cap.release()     
cv2.destroyAllWindows()           
                
               





