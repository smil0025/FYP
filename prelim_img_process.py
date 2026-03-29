# create img_processing function that takes in an image of known pixel dimensions (1920 x 1440, what the microscope outputs)
# greyscales the image and reduces it to a specified= size (256x256, standard and square), pad image if nec
# also creates duplicates of the input image that have been rotated 90, 180, 270 degrees,
# and reflected in both the x and y axis, and saves all those versions of the image in a new
# folder (haven't implemented the last two yet)

# maybe also include a sobel edge detection filter if necessary? no gaussian, we don't want
# the images to get any blurrier

# ---------------------------
# IMPORTS
# ---------------------------
import cv2
from PIL import Image
import numpy as np
import os
import random
import shutil
from multiprocessing import Pool, cpu_count

# ---------------------------
# CONFIG- adding classes as we have more training data
# ---------------------------
VALID_CLASSES = ["cotton", "poly_satin", "wool", "poly_chiffon"]
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ---------------------------
# IMAGE PROCESSING FUNCTION- resizing and augmentations
# ---------------------------
def img_processing(input_path, output_dir):
    img = cv2.imread(input_path)

    if img is None:
        print(f"Skipping unreadable file: {input_path}")
        return

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize + pad
    desired_size = 256
    h, w = grey.shape

    scale = desired_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(grey, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded = np.zeros((desired_size, desired_size), dtype=np.uint8)
    x_offset = (desired_size - new_w) // 2
    y_offset = (desired_size - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    pil_img = Image.fromarray(padded)

    base = os.path.splitext(os.path.basename(input_path))[0]

    # ---- ROTATIONS ----
    pil_img.save(os.path.join(output_dir, f"{base}_grey_0.jpg"))

    for angle in [90, 180, 270]:
        rotated = pil_img.rotate(angle, expand=True)
        rotated.save(os.path.join(output_dir, f"{base}_grey_{angle}.jpg"))

    # ---- FLIPS ----
    h_flip = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    h_flip.save(os.path.join(output_dir, f"{base}_grey_hflip.jpg"))

    v_flip = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    v_flip.save(os.path.join(output_dir, f"{base}_grey_vflip.jpg"))

# ---------------------------
# PARALLEL PROCESSING- speeds things up
# ---------------------------
def process_single(args):
    input_path, output_dir = args
    img_processing(input_path, output_dir)

def process_dataset_parallel(input_dir, output_dir):
    tasks = []

    for class_name in VALID_CLASSES:
        class_input_path = os.path.join(input_dir, class_name)

        if not os.path.exists(class_input_path):
            print(f"Missing class folder: {class_name}")
            continue

        class_output_path = os.path.join(output_dir, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        for file in os.listdir(class_input_path):
            if file.lower().endswith(VALID_EXTS):
                input_path = os.path.join(class_input_path, file)
                tasks.append((input_path, class_output_path))

    print(f"Processing {len(tasks)} images using {cpu_count()} cores...")

    with Pool(cpu_count()) as p:
        p.map(process_single, tasks)

    print("Processing complete.")

# ---------------------------
# TRAIN / VAL / TEST SPLIT so data can go on to pytorch
# ---------------------------
def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    for class_name in VALID_CLASSES:
        class_path = os.path.join(input_dir, class_name)

        if not os.path.exists(class_path):
            continue

        files = os.listdir(class_path)
        random.shuffle(files)

        n = len(files)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:]
        }

        for split, split_files in splits.items():
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for f in split_files:
                shutil.copy(
                    os.path.join(class_path, f),
                    os.path.join(split_class_dir, f)
                )

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    input_dir = input("Enter input dataset path: ").strip()
    processed_dir = input("Enter processed output path: ").strip()
    final_dataset_dir = input("Enter final dataset (train/val/test) path: ").strip()

    # Step 1: preprocess + augment
    process_dataset_parallel(input_dir, processed_dir)

    # Step 2: split dataset
    split_dataset(processed_dir, final_dataset_dir)

    print("Pipeline complete.")