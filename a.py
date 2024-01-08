import cv2
import pyautogui
import mediapipe as mp
import pickle
import numpy as np
import time


right_palm_start_time = None
left_palm_start_time = None
palm_duration_threshold = 3


def action(predictionIndex):
    global right_palm_start_time, left_palm_start_time
    
    x__ =  int(hand_landmarks.landmark[8].x * W)
    y__ =  int(hand_landmarks.landmark[8].y * H)

    x_1 = int(x__ * (screen_width/ W))
    y_1 = int(y__ * (screen_height/ H))
    


    if(predictionIndex == 0):
        indices_to_change = [1,2,3,4,5,6,7,8,9, 10]
        for i  in indices_to_change:
            is_action_active[i] = False
        if is_action_active[0] == False:
            print("index 0")
            pyautogui.hotkey('ctrl', 'p')
            is_action_active[0] = True
            
    if(predictionIndex == 1):
        indices_to_change = [0,2,3,4,5,6,7,8,9, 10]
        for i  in indices_to_change:
            is_action_active[i] = False
        if is_action_active[1] == False:            
            print("index 1")
            pyautogui.hotkey('ctrl', 'l')
            is_action_active[1] = True
            
    if(predictionIndex == 2):
        indices_to_change = [0,1,3,4,5,6,7,8,9, 10]
        for i in indices_to_change:
            is_action_active[i] = False
        if is_action_active[2] == False:
            print("index 2")
            pyautogui.hotkey('ctrl', 'i')
            is_action_active[2] = True
            
            
    if(predictionIndex == 3):
        indices_to_change = [0,1,2,4,5,6,7,8,9, 10]
        for i in indices_to_change:
            is_action_active[i] = False
        if is_action_active[3] == False:
            print("index 3")
            pyautogui.hotkey('ctrl', 'e')
            is_action_active[3] = True
            
            
    if(predictionIndex == 4):
        indices_to_change = [0,1,2,3,5,6,7,8,9, 10]
        for i in indices_to_change:
            is_action_active[i] = False
        if is_action_active[4] == False:
            pyautogui.hotkey('ctrl', 'm')
            is_action_active[4] = True
        

    if(predictionIndex == 5):
        indices_to_change = [0,1,2,3,4,6,7,8,9, 10]
        for i in indices_to_change:
            is_action_active[i] = False
        if is_action_active[5] == False:
            is_action_active[5] = True
            print("index 5")
            pyautogui.mouseUp()
        pyautogui.moveTo(x_1, y_1, duration=0.01, _pause = False)    
        cv2.circle(frame,(x__, y__), 10, (255, 0, 0), 3)


    if(predictionIndex == 6):
        indices_to_change = [0,1,2,3,4,5,7,8,9, 10]
        for i in indices_to_change:
            is_action_active[i] = False
        if is_action_active[6] == False:
            is_action_active[6] = True
            print("index 6")
            pyautogui.mouseDown()
        pyautogui.moveTo(x_1, y_1, duration=0.01, _pause = False)            
        cv2.circle(frame,(x__, y__), 10, (255, 0, 0), 3)


    if(predictionIndex == 7):
        indices_to_change = [0,1,2,3,4,5,6,8,9, 10]
        for i in indices_to_change:
            is_action_active[i] = False
        if is_action_active[7] == False:
            is_action_active[7] = True
            pyautogui.press('right')
            print("index 7")
            


    if(predictionIndex == 8):
        indices_to_change = [0,1,2,3,4,5,6,7,9, 10]
        for i in indices_to_change:
            is_action_active[i] = False
        if is_action_active[8] == False:
            is_action_active[8] = True
            pyautogui.press('left')
            print("index 8")
            
            
    if(predictionIndex == 9):
        indices_to_change = [0,1,2,3,4,5,6,7, 8, 10]
        for i in indices_to_change:
            is_action_active[i] = False
        if is_action_active[9] == False:
            is_action_active[9] = True
            print("index 9")
            
    if(predictionIndex == 10):
        indices_to_change = [0,1,2,3,4,5,6,7, 8,9]
        for i in indices_to_change:
            is_action_active[i] = False
        if is_action_active[10] == False:
            is_action_active[10] = True
            print("index 10")
            
            



model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles



hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence= 0.8, max_num_hands = 1)
labels_dict = {0: 'pen', 1: 'pointer', 2: 'highlighter', 3:'eraser', 4: 'hide', 5: 'move cursor', 6: 'select cursor', 7: 'right palm', 8: 'left palm', 9:"none right", 10:"none left"}
fps_start_time = time.time()
fps_frame_count = 0

is_action_active = [False, False, False, False, False,False, False, False, False, False, False]
while True:
    ret, frame = cap.read()
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    screen_width, screen_height = pyautogui.size()

    
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if(results.multi_hand_landmarks):
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
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
        

        if(confidence > 0.55):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, f'{predicted_character}{confidence}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                  
            action(int(prediction[0]))  
         


    fps_frame_count += 1
    if fps_frame_count >= 10:  
        fps_end_time = time.time()
        fps = fps_frame_count / (fps_end_time - fps_start_time)
        print(f"FPS: {fps:.2f}")
        fps_frame_count = 0
        fps_start_time = time.time()


    cv2.imshow('camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
