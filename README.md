# Real-Time Pose-Based Fall Detection System

##  Overview

This project implements a real-time fall detection system using pose estimation and spatial-temporal logic.

The system reduces false positives from bending and sitting by using a two-stage detection strategy.

---

##  Problem Motivation

Falls among elderly individuals are a major safety concern.  
Traditional posture-only detection methods often confuse bending or sitting with falling.

This project improves reliability using:

- Body orientation modeling
- Sudden posture change detection
- Sustained horizontal confirmation

---

##  Methodology

The system follows a two-stage approach:

### 1️⃣ Pose Estimation
- YOLOv8-Pose extracts skeletal keypoints in real-time.

### 2️⃣ Body Angle Calculation
- Shoulder and hip midpoints estimate body orientation.

### 3️⃣ Sudden Collapse Detection
- Detects rapid angle change (spike detection).

### 4️⃣ Horizontal Confirmation
- Confirms sustained horizontal posture before triggering alert.

---

##  System Pipeline

Camera → Pose Model → Keypoint Extraction →  
Body Angle Estimation → Sudden Change Detection →  
Temporal Confirmation → Fall Alert

---

##  Features

- Real-time webcam detection
- Two-stage fall logic
- Reduced false positives
- Lightweight implementation
- Runs on standard laptop

---

##  Installation

```bash
pip install -r requirements.txt
