import cv2
import time
import numpy as np
from ultralytics import YOLO

# Load pose model
model = YOLO("models/yolov8n-pose.pt")

CONF_THRESHOLD = 0.5
ANGLE_THRESHOLD = 60
ANGLE_SPIKE_THRESHOLD = 20
HORIZONTAL_CONFIRM_FRAMES = 8

previous_angle = None
collapse_detected = False
horizontal_counter = 0

cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRESHOLD)
    current_angle = None

    if results[0].keypoints is not None:
        for person in results[0].keypoints.xy:

            left_shoulder = person[5]
            right_shoulder = person[6]
            left_hip = person[11]
            right_hip = person[12]

            shoulder_mid = (
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2
            )

            hip_mid = (
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2
            )

            dx = hip_mid[0] - shoulder_mid[0]
            dy = hip_mid[1] - shoulder_mid[1]

            current_angle = np.degrees(np.arctan2(abs(dx), abs(dy)))

    # Stage 1: Detect sudden collapse
    if previous_angle is not None and current_angle is not None:
        angle_change = current_angle - previous_angle
        if angle_change > ANGLE_SPIKE_THRESHOLD:
            collapse_detected = True

    previous_angle = current_angle

    # Stage 2: Confirm horizontal posture
    if collapse_detected and current_angle is not None:
        if current_angle > ANGLE_THRESHOLD:
            horizontal_counter += 1
        else:
            horizontal_counter = 0
            collapse_detected = False

    # Alert
    if horizontal_counter >= HORIZONTAL_CONFIRM_FRAMES:
        cv2.putText(frame,
                    "FALL DETECTED",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255), 3)

    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()