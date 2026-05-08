import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def enhance_face(face):

    # Resize
    face = cv2.resize(face, (200, 200))

    # CLAHE only
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    face = clahe.apply(face)

    return face

def augment_image(img):

    augmented_images = []

   
    augmented_images.append(img)
    flipped = cv2.flip(img, 1)
    augmented_images.append(flipped)

    rows, cols = img.shape
    for angle in [-10, -5, 5, 10]:

        M = cv2.getRotationMatrix2D(
            (cols / 2, rows / 2),
            angle,
            1
        )

        rotated = cv2.warpAffine(
            img,
            M,
            (cols, rows),
            borderMode=cv2.BORDER_REFLECT
        )

        augmented_images.append(rotated)

    bright = cv2.convertScaleAbs(
        img,
        alpha=1.2,
        beta=20
    )
    augmented_images.append(bright)

    dark = cv2.convertScaleAbs(
        img,
        alpha=0.8,
        beta=-20
    )
    augmented_images.append(dark)
    return augmented_images


input_folder = r"D:\hady\3nd year\Second_term\Supervised_Learning\project\Face-Recognition-System\datasets"

output_folder = r"D:\hady\3nd year\Second_term\Supervised_Learning\project\Face-Recognition-System\processed_dataset"

os.makedirs(output_folder, exist_ok=True)

for person in os.listdir(input_folder):

    person_path = os.path.join(input_folder, person)

    if not os.path.isdir(person_path):
        continue

    save_path = os.path.join(output_folder, person)

    os.makedirs(save_path, exist_ok=True)

    print(f"Processing: {person}")

    for filename in os.listdir(person_path):

        img_path = os.path.join(person_path, filename)

        # Read image
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        # Skip if no face detected
        if len(faces) == 0:
            print(f"No face found in: {filename}")
            continue

        # Take largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])

        x, y, w, h = largest_face

        # Crop face
        face = gray[y:y+h, x:x+w]

        # Enhance face
        enhanced_face = enhance_face(face)

        # Augment
        augmented_list = augment_image(enhanced_face)

        # Save images
        name = os.path.splitext(filename)[0]

        for i, aug_img in enumerate(augmented_list):

            save_name = f"{name}_aug_{i}.jpg"

            cv2.imwrite(
                os.path.join(save_path, save_name),
                aug_img
            )

    print(f"Finished: {person}")

print("\nPreprocessing and Augmentation Complete!")