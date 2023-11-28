import cv2
import time

cap = cv2.VideoCapture(0)  

num_frames = 120 
start_time = time.time()

for i in range(num_frames):
    ret, frame = cap.read()
    cv2.imshow("camera", frame)
    cv2.putText(frame, "test", (250,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()

elapsed_time = end_time - start_time
fps = num_frames / elapsed_time
print(f"Average FPS: {fps}")

cap.release()
cv2.destroyAllWindows()
