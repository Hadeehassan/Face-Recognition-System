import cv2
import os
import time

# 1. SETUP
name = "Hadi"
dataset_path = f"../dataset/{name}"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# 2. LOAD FACE DETECTOR
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 3. CAMERA
cap = cv2.VideoCapture(0)
count = 0

# Delay settings (time between each capture)
last_capture_time = 0
delay = 0.5  # seconds

print(f"Capturing images for {name}. Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale (for detection + saving)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        current_time = time.time()

        # Save image every "delay" seconds
        if current_time - last_capture_time > delay:
            count += 1

            # Crop face from grayscale image
            face_img = gray[y:y+h, x:x+w]

            # Resize
            face_img = cv2.resize(face_img, (200, 200))

            # Save
            cv2.imwrite(f"{dataset_path}/{count}.jpg", face_img)

            last_capture_time = current_time

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show count
    cv2.putText(frame, f"Images: {count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display
    cv2.imshow("Data Collection", frame)

    # Exit conditions
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 80:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print(f"Done! Saved {count} images to {dataset_path}")