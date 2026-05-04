import cv2
import os

def preprocess(img):
    img = cv2.resize(img, (200, 200))
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

def augment_image(img):
    augmented_images = [img]

    # Flip
    augmented_images.append(cv2.flip(img, 1))

    rows, cols = img.shape

    # Rotate
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (cols, rows),
                                 borderMode=cv2.BORDER_REPLICATE)
        augmented_images.append(rotated)

    # Brightness
    augmented_images.append(cv2.convertScaleAbs(img, alpha=1.2, beta=20))
    augmented_images.append(cv2.convertScaleAbs(img, alpha=0.8, beta=-20))

    return augmented_images


input_folder = "../dataset"
output_folder = "../processed_dataset"

for person in os.listdir(input_folder):
    person_path = os.path.join(input_folder, person)
    save_path = os.path.join(output_folder, person)
    os.makedirs(save_path, exist_ok=True)

    for filename in os.listdir(person_path):
        img_path = os.path.join(person_path, filename)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = preprocess(img)
        augmented_list = augment_image(img)
        name = os.path.splitext(filename)[0]

        # 3. save
        for i, aug_img in enumerate(augmented_list):
            cv2.imwrite(f"{save_path}/{name}_aug_{i}.jpg", aug_img)

print("Preprocessing and Augmentation complete!")