import cv2
import mediapipe as mp
import pyautogui
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def move_cursor(x, y):
    screen_width, screen_height = pyautogui.size()
    dest_x = int(x * screen_width)
    dest_y = int(y * screen_height)
    pyautogui.moveTo(dest_x, dest_y)

def calculate_fps(prev_time, frame_count):
    current_time = time.time()
    elapsed_time = current_time - prev_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        return current_time, 0, fps
    return prev_time, frame_count, None

alpha = 0.9
smoothed_x, smoothed_y = None, None

prev_time = time.time()
frame_count = 0
frame_rate = 20
desired_frame_rate = 5
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        for _ in range(frame_rate // desired_frame_rate - 1):
            _, _ = cap.read()

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_x, index_finger_y = index_finger.x, index_finger.y

                # # Apply smoothing
                if smoothed_x is None or smoothed_y is None:
                    smoothed_x, smoothed_y = index_finger_x, index_finger_y
                else:
                    smoothed_x = alpha * index_finger_x + (1 - alpha) * smoothed_x
                    smoothed_y = alpha * index_finger_y + (1 - alpha) * smoothed_y

                # Move cursor based on smoothed finger position
                move_cursor(smoothed_x, smoothed_y)

                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame with landmarks
        cv2.imshow('Hand Tracking', frame)
        frame_count += 1

        # Calculate FPS and display it
        prev_time, frame_count, fps = calculate_fps(prev_time, frame_count)
        if fps:
            print(f"FPS: {fps:.2f}")

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
