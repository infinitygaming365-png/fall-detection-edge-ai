import time
import cv2
from ultralytics import YOLO

model = YOLO("models/best.pt")
cap = cv2.VideoCapture(0)

times = []
frames = 100

for _ in range(frames):
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    model(frame, imgsz=416)
    end = time.time()

    times.append(end - start)

cap.release()

avg_time = sum(times) / len(times)
fps = 1 / avg_time

print("Average inference time:", avg_time)
print("Approx FPS:", fps)