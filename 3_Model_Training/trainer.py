import cv2
import os
import numpy as np

path = r"D:\hady\3nd year\Second_term\Supervised_Learning\project\Face-Recognition-System\processed_dataset"

recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = []
ids = []
label_map = {}
current_id = 0

for person_name in os.listdir(path):
    person_path = os.path.join(path, person_name)
    if not os.path.isdir(person_path):
        continue

    label_map[current_id] = person_name
    print(f"Training for: {person_name} (ID: {current_id})")

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        gray_img = cv2.imread(image_path, 0)
        
        if gray_img is None: continue
        
        img_numpy = np.array(gray_img, 'uint8')

        detected_faces = detector.detectMultiScale(img_numpy, 1.1, 5)
        
        for (x, y, w, h) in detected_faces:
            faces.append(img_numpy[y:y+h, x:x+w]) # بناخد الوش بس
            ids.append(current_id)

    current_id += 1

if len(faces) > 0:
    recognizer.train(faces, np.array(ids))
    recognizer.write("trainer.yml")
    print("Model Trained Successfully on faces only!")
else:
    print("Error: No faces found in the images. Check your dataset!")