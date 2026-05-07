import cv2
import os
import numpy as np

path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
ids = []

label_map = {}
current_id = 0

for person_name in os.listdir(path):

    person_path = os.path.join(path, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_id] = person_name

    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        gray_img = cv2.imread(image_path , 0)
        img_numpy = np.array(gray_img, 'uint8')

        faces.append(img_numpy)
        ids.append(current_id)

    current_id += 1

recognizer.train(faces, np.array(ids))

recognizer.save("trainer.yml")

print("Model Trained Successfully")
