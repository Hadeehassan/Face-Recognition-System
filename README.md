# Face Recognition System

A real-time Face Recognition System built using Python and OpenCV.  
The project detects faces from a webcam feed, collects training data, trains a recognizer model, and performs real-time identification.

---

## Features

- Real-time face detection using Haar Cascade
- Face dataset collection from webcam
- Image preprocessing (resize, grayscale, normalization)
- Face recognition using LBPH algorithm
- Confidence score-based prediction
- Support for unknown face detection
- Simple and modular pipeline

---

## Project Pipeline

1. **Data Collection**
   - Capture face images using webcam
   - Store dataset per person in labeled folders

2. **Preprocessing**
   - Convert to grayscale
   - Resize images to fixed dimensions
   - Apply histogram equalization / filtering

3. **Feature Extraction**
   - LBPH (Local Binary Pattern Histogram)

4. **Model Training**
   - Train LBPH recognizer on collected dataset

5. **Real-Time Recognition**
   - Detect face from webcam stream
   - Predict identity and display confidence score

---

## Technologies Used

- Python
- OpenCV
- NumPy
- LBPH Face Recognizer

---

## Installation

```bash
git clone https://github.com/your-username/face-recognition-system.git
cd face-recognition-system

pip install opencv-python numpy opencv-contrib-python
