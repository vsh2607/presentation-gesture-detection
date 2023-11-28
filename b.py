import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence= 0.5)

while True:
    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame = cv2.flip(frame, 1)
    
    # Define the rectangle coordinates as a percentage of the frame size
    rectangle_width_percentage = 0.8
    rectangle_height_percentage = 0.8
    
    

    x1 = int((1 - rectangle_width_percentage) * W / 2)
    y1 = int((1 - rectangle_height_percentage) * H / 2)
    
    x2 = int((1 + rectangle_width_percentage) * W / 2)
    y2 = int((1 + rectangle_height_percentage) * H / 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    
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
        
            x__ =  int(hand_landmarks.landmark[8].x * x2)
            y__ =  int(hand_landmarks.landmark[8].y *  y2)
            cv2.circle(frame,(x__, y__), 10, (255, 0, 0), 3)
            
        

    
    cv2.imshow("camera", frame)
    
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    