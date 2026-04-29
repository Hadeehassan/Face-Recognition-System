
import cv2
import os

# 1. SETUP: Set the name for the person being captured
name = "Hadi" 
dataset_path = f"../dataset/{name}"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# 2. LOAD FACE DETECTOR: Required just to "Crop" the face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. INITIALIZE CAMERA
cap = cv2.VideoCapture(0)
count = 0

print(f"Capturing raw images for {name}. Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect face to know where to crop
    # We use a grayscale version only for detection (doesn't save it as gray)
    temp_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(temp_gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        
        # CROP: Isolate the face area from the ORIGINAL color frame
        raw_face = frame[y:y+h, x:x+w]
        
        # RESIZE: Keep them all the same size (standard practice)
        raw_face = cv2.resize(raw_face, (200, 200))
        
        # SAVE: Saves the raw color photo
        cv2.imwrite(f"{dataset_path}/{count}.jpg", raw_face)

        # Visual feedback
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Raw Images: {count}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Raw Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done! Saved {count} raw images to {dataset_path}")