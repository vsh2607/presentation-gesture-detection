import cv2

cap = cv2.VideoCapture(0)  # Replace '0' with the camera index if needed

frame_rate = 28  # Camera's native frame rate
desired_frame_rate = 9  # Target frame rate (every 3rd frame)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('Original Frame', frame)

    for _ in range(frame_rate // desired_frame_rate - 1):
        _, _ = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
