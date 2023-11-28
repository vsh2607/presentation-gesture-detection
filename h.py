import cv2
import time

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, you can change this number if you have multiple cameras

timer_delay = 5  # 5 seconds delay (you can change this)

for i in range(timer_delay, 0, -1):
    print(f"Capturing in {i} seconds...")
    time.sleep(1)

ret, frame = cap.read()

if ret:
    cv2.imshow('Captured Image', frame)
    cv2.waitKey(0)  # Waits indefinitely for any key press
    cv2.destroyAllWindows()
else:
    print("Failed to capture image")

cap.release()
