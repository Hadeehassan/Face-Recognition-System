import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read(r"F:\Github\Face-Recognition-System\model\trainer.yml")

names = ["Abdallah", "Hady", "Nour"]

# Camera
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (200, 200))

    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        # Predict
        id, confidence = recognizer.predict(face)

        if confidence < 70:
            name = names[id] if id < len(names) else "Unknown"
        else:
            name = "Unknown"

        # Rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Put Name
        cv2.putText(
            frame,
            name,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
