import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r"D:\hady\3nd year\Second_term\Supervised_Learning\project\Face-Recognition-System\trainer.yml")

names = ["Abdallah", "Hady", "Nour"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # اكتشاف الوجوه
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # أ) قص منطقة الوجه فقط (ROI)
        face_roi = gray[y:y+h, x:x+w]
        
        face_roi_resized = cv2.resize(face_roi, (200, 200))

        id, confidence = recognizer.predict(face_roi_resized)

        if confidence < 85:
            name = names[id] if id < len(names) else "Unknown"
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255) 

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        text = f"{name} ({round(confidence)})"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition - Press 'q' to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()