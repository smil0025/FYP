# create img_processing function that takes in an image of known pixel dimensions (1920 x 1440, what the microscope outputs)
# greyscales the image and reduces it to a specified= size (256x256, standard and square), pad image if nec
# also creates duplicates of the input image that have been rotated 90, 180, 270 degrees,
# and reflected in both the x and y axis, and saves all those versions of the image in a new
# folder (haven't implemented the last two yet)

# maybe also include a sobel edge detection filter if necessary? no gaussian, we don't want
# the images to get any blurrier

# import libraries
import cv2
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

def img_processing(input_path):
    """
    Loads an RGB JPG image, converts to greyscale, resizes to 256x256 with padding,
    rotates 90, 180, 270 degrees, and saves all versions as original_grey_<angle>.jpg
    """
    # Load image with OpenCV (BGR format)
    img = cv2.imread(input_path)

    # Convert to greyscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Resize to 256x256 with padding ---
    desired_size = 256
    h, w = grey.shape

    # Compute scale factor
    scale = desired_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize while keeping aspect ratio
    resized = cv2.resize(grey, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create padded 256x256 canvas
    padded = np.zeros((desired_size, desired_size), dtype=np.uint8)

    # Center the resized image
    x_offset = (desired_size - new_w) // 2
    y_offset = (desired_size - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # Convert to PIL for rotation + saving
    pil_img = Image.fromarray(padded)

    # Extract base filename
    base = os.path.splitext(os.path.basename(input_path))[0]

    # Save greyscale resized image
    pil_img.save(f"{base}_grey_0.jpg")

    # Rotation angles
    angles = [90, 180, 270]

    for angle in angles:
        rotated = pil_img.rotate(angle, expand=True)
        rotated.save(f"{base}_grey_{angle}.jpg")

 # Test img_processing function on one of our images   
input_file = "test_img.JPG"
img_processing(input_file)

# check img_processing is working correctly by displaying the processed images
def display_subplots(input_path):
    base = os.path.splitext(os.path.basename(input_path))[0]

    # Load the processed images
    img_0 = Image.open(f"{base}_grey_0.jpg")
    img_90 = Image.open(f"{base}_grey_90.jpg")
    img_180 = Image.open(f"{base}_grey_180.jpg")
    img_270 = Image.open(f"{base}_grey_270.jpg")

    # Set up 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    images = [img_0, img_90, img_180, img_270]
    titles = ["Grey (0°)", "Grey (90°)", "Grey (180°)", "Grey (270°)"]

    for ax, image, title in zip(axes.flatten(), images, titles):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

display_subplots("test_img.JPG")
