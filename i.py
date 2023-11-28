import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            center_x = int(landmarks[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
            center_y = int(landmarks[mp_hands.HandLandmark.WRIST].y * frame.shape[0])
            
            if center_x < frame.shape[1] // 2:
                hand_side = "Right Hand"
            else:
                hand_side = "Left Hand"

            cv2.putText(frame, hand_side, (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
