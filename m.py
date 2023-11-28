import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)


screen_width, screen_height = pyautogui.size()

frame_count = 0
prev_time = 0
prev_x, prev_y = 0, 0  

while True:
    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    target_x = 0
    target_y = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(0, 21):
                x, y = int(hand_landmarks.landmark[i].x * W), int(hand_landmarks.landmark[i].y * H)
                cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)
                
            if frame_count % 3 == 0:   
                
                target_x = int(x * screen_width / W)
                target_y = int(y * screen_height / H)
                    
                steps = 3  
                for step in range(steps):
                    
                    interp_x = prev_x + (target_x - prev_x) * (step + 1) / steps
                    interp_y = prev_y + (target_y - prev_y) * (step + 1) / steps
                    pyautogui.moveTo(interp_x, interp_y, duration=0.1 / steps, _pause=False)
                    time.sleep(0.1 / steps)  
            
                
                prev_x, prev_y = target_x, target_y
               
    
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    frame_count += 1
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()     
cv2.destroyAllWindows()
