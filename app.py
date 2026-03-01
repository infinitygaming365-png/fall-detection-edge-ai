import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import time

st.set_page_config(page_title="Edge AI Fall Detection", layout="centered")
st.title("🧠 Real-Time Fall Detection (Edge AI Demo)")

# Load model
model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    if "image" in file_type:
        image = Image.open(uploaded_file)
        results = model(image)
        annotated = results[0].plot()
        st.image(annotated, caption="Detection Result", use_column_width=True)

    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        prev_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            results = model(frame, imgsz=416, conf=0.7)
            annotated = results[0].plot()
            end = time.time()

            fps = 1 / (end - start)

            cv2.putText(annotated, f"FPS: {int(fps)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

            stframe.image(annotated, channels="BGR")

        cap.release()