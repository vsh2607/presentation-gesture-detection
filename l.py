import cv2
import mediapipe as mp
import joblib
import pyautogui
from sklearn.neighbors import KNeighborsClassifier

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
knn_model = joblib.load('knn_model_lab.pkl')
landmark_data = []
gesture_dicts = {0: "close_palm", 1: "open_palm", 2: "select"}

frame_count = 0
smooth_factor = 5  # Change this value for different smoothness

def smooth_movement(current_pos, new_pos, factor):
    return (current_pos * (factor - 1) + new_pos) / factor

current_x, current_y = pyautogui.position()

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        landmark_data.clear()  # Clearing landmark_data before appending new data
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * W)
                y = int(landmark.y * H)
                
                cv2.circle(frame, (x, y), 3, (255, 0, 0), 3, -1)
                
                landmark_data.append(landmark.x)
                landmark_data.append(landmark.y)

        if landmark_data:  
            result = knn_model.predict([landmark_data])
            gesture = gesture_dicts[result[0]]
            print(gesture)
            
            if gesture == "select":
                if frame_count % smooth_factor == 0:
                    # Get the coordinates of the index finger (assuming landmarks are in a specific order)
                    index_finger_x = landmark_data[8] * W
                    index_finger_y = landmark_data[9] * H
                    
                    # Smoothly move the cursor to the index finger coordinates
                    current_x = smooth_movement(current_x, index_finger_x, smooth_factor)
                    current_y = smooth_movement(current_y, index_finger_y, smooth_factor)
                    pyautogui.moveTo(int(current_x), int(current_y))

                frame_count += 1
                if frame_count >= 10000:  # Reset frame count to avoid overflow
                    frame_count = 0

    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
