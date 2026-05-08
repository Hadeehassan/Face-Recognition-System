import cv2
import os
import numpy as np

# =========================
# Dataset Path
# =========================
path = r"D:\hady\3nd year\Second_term\Supervised_Learning\project\Face-Recognition-System\processed_dataset"

# =========================
# Create Recognizer
# =========================
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

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

        # Read grayscale image
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if gray_img is None:
            continue

        # IMPORTANT:
        # image is already cropped face
        faces.append(gray_img)

        ids.append(current_id)

    current_id += 1

if len(faces) > 0:

    recognizer.train(faces, np.array(ids))

    recognizer.write("trainer.yml")

    print("\nModel Trained Successfully!")

    print(f"Total Faces: {len(faces)}")

else:
    print("Error: No images found!")