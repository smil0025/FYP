# create img_processing function that takes in an image of known pixel dimensions (1920 x 1440, what the microscope outputs)
# greyscales the image and reduces it to a specified= size (256x256, standard and square), pad image if nec
# also creates duplicates of the input image that have been rotated 90, 180, 270 degrees,
# and reflected in both the x and y axis, and saves all those versions of the image in a new
# folder (haven't implemented the last two yet)

# maybe also include a sobel edge detection filter if necessary? no gaussian, we don't want
# the images to get any blurrier

import cv2
from PIL import Image
import numpy as np
import os

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

    # ---- SAVE STANDARD + ROTATIONS ----
    pil_img.save(os.path.join(output_dir, f"{base}_grey_0.jpg"))

    for angle in [90, 180, 270]:
        rotated = pil_img.rotate(angle, expand=True)
        rotated.save(os.path.join(output_dir, f"{base}_grey_{angle}.jpg"))

    # ---- FLIPS ----
    h_flip = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    h_flip.save(os.path.join(output_dir, f"{base}_grey_hflip.jpg"))

    v_flip = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    v_flip.save(os.path.join(output_dir, f"{base}_grey_vflip.jpg"))

    # ---- SOBEL (saved separately for now) ----
    sobelx = cv2.Sobel(padded, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(padded, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)

    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
    sobel = sobel.astype(np.uint8)

    Image.fromarray(sobel).save(os.path.join(output_dir, f"{base}_sobel.jpg"))