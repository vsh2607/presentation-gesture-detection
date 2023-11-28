import pickle

import cv2
import mediapipe as mp
import numpy as np
import pyautogui


def action(predictionIndex):


    if(predictionIndex == 7):
          index_finger_tip_x = hand_landmarks.landmark[8].x * H
          index_finger_tip_y = hand_landmarks.landmark[8].y * W
          print(f'{index_finger_tip_x}-{index_finger_tip_y}')
          if(index_finger_tip_x > 260):
               pyautogui.press('right')

    
    if(predictionIndex == 8):
          index_finger_tip_x = hand_landmarks.landmark[8].x * H
          index_finger_tip_y = hand_landmarks.landmark[8].y * W
          print(f'{index_finger_tip_x}-{index_finger_tip_y}')
          if(index_finger_tip_x < 200):
               pyautogui.press('left')


    if(predictionIndex == 0):
        pyautogui.hotkey('ctrl', 'p')
    if(predictionIndex == 1):
        pyautogui.hotkey('ctrl', 'l')
    if(predictionIndex == 2):
        pyautogui.hotkey('ctrl', 'i')
    if(predictionIndex == 3):
        pyautogui.hotkey('ctrl', 'e')
    if(predictionIndex == 4):
        pyautogui.hotkey('ctrl', 'm')
        

    if(predictionIndex == 6):

          mouse_x = int(screen_width / W * x_1)
          mouse_y = int(screen_height / H * y_1)
          pyautogui.mouseDown()
          pyautogui.moveTo(mouse_x, mouse_y)

          
    if(predictionIndex == 5):
          mouse_x = int(screen_width / W * x_1)
          mouse_y = int(screen_height / H * y_1)
          pyautogui.mouseUp()
          pyautogui.moveTo(mouse_x, mouse_y)


    


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
screen_width, screen_height = pyautogui.size()

labels_dict = {0: 'pen', 1: 'pointer', 2: 'highlighter', 3:'eraser', 4: 'hide', 5: 'move cursor', 6: 'select cursor', 7: 'right palm', 8: 'left palm'}
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape
    
    #640 w , 480 h : camera
    #1920 w, 1080 h : screen szie
    
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  
                hand_landmarks,  
                mp_hands.HAND_CONNECTIONS,  
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)


            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

         
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10


        if(len(data_aux) <= 42):
            prediction = model.predict([np.asarray(data_aux)])
            confidence = model.predict_proba([np.asarray(data_aux)]).max()

        predicted_character = labels_dict[int(prediction[0])]


        if(confidence > 0.75):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, f'{predicted_character}{confidence}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            action(int(prediction[0]))
            

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()


