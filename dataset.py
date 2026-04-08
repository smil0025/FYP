import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

# ---------------------------
# GABOR FILTER BANK
# ---------------------------
def build_gabor_kernels():
    kernels = []

    ksize = 21
    sigma = 5.0
    gamma = 0.5

    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    lambdas = [10.0]  # you can expand later

    for theta in thetas:
        for lambd in lambdas:
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma,
                theta,
                lambd,
                gamma,
                psi=0,
                ktype=cv2.CV_32F
            )
            kernels.append(kernel)

    return kernels


# ---------------------------
# DATASET CLASS
# ---------------------------
class FabricDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []

        self.class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

        for cls in self.class_names:
            class_path = os.path.join(root_dir, cls)

            for file in os.listdir(class_path):
                path = os.path.join(class_path, file)
                self.image_paths.append(path)
                self.labels.append(self.class_to_idx[cls])

        # build once (important!)
        self.gabor_kernels = build_gabor_kernels()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # ---------------------------
        # SOBEL
        # ---------------------------
        sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        sobel = cv2.magnitude(sobelx, sobely)

        # ---------------------------
        # GABOR FILTERS
        # ---------------------------
        gabor_features = []
        for kernel in self.gabor_kernels:
            filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
            gabor_features.append(filtered)

        # ---------------------------
        # NORMALIZATION
        # ---------------------------
        img = img.astype(np.float32) / 255.0
        sobel = sobel / 255.0

        # normalize each gabor channel independently (important!)
        gabor_norm = []
        for g in gabor_features:
            g = np.abs(g)  # magnitude helps stability
            g = (g - g.min()) / (g.max() - g.min() + 1e-6)
            gabor_norm.append(g)

        # ---------------------------
        # STACK CHANNELS
        # ---------------------------
        combined = np.stack([img, sobel] + gabor_norm, axis=0)

        return torch.tensor(combined, dtype=torch.float32), self.labels[idx]
    