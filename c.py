import cv2
import time

# Set the video source (0 for default camera)
video_source = 0
cap = cv2.VideoCapture(video_source)

# Set the desired frames per second (fps)
desired_fps = 30

# Calculate the time delay between frames based on the desired fps
delay = 1 / desired_fps

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Process every nth frame
    if frame_count % 1 == 0:
        # Your processing code here
        # For example, display the frame
        cv2.imshow('Frame', frame)

    frame_count += 1

    # Wait for a short duration to achieve the desired frames per second
    time.sleep(delay)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
