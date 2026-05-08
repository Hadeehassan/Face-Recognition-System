import cv2
import numpy as np

# =========================
# Face Detector
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

# =========================
# Load Recognizer
# =========================
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read(
    r"D:\hady\3nd year\Second_term\Supervised_Learning\project\Face-Recognition-System\trainer.yml"
)

# =========================
# Names
# =========================
names = ["Abdallah", "Hady", "Noureldin"]

# =========================
# Face Enhancement
# =========================
def enhance_face(face):
    face = cv2.resize(face, (200, 200))
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    face = clahe.apply(face)
    return face

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        # Crop face
        face_roi = gray[y:y+h, x:x+w]

        # SAME preprocessing as training
        processed_face = enhance_face(face_roi)

        # Predict
        id, confidence = recognizer.predict(processed_face)

        # Lower confidence = better
        if confidence < 65:

            if id < len(names):
                name = names[id]
            else:
                name = "Unknown"

            color = (0, 255, 0)

        else:

            name = "Unknown"

            color = (0, 0, 255)

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            color,
            2
        )

        # Display text
        text = f"{name} ({round(confidence, 1)})"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow(
        "Face Recognition - Press Q to Quit",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()