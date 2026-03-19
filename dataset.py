# dataset class for pytorch, also includes dataset splitting function

import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Sobel
        sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        sobel = cv2.magnitude(sobelx, sobely)

        # Normalize
        img = img / 255.0
        sobel = sobel / 255.0

        # Stack channels → [2, H, W]
        combined = np.stack([img, sobel], axis=0)

        return torch.tensor(combined, dtype=torch.float32), self.labels[idx]