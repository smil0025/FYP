import matplotlib.pyplot as plt
import numpy as np
from dataset import FabricDataset

def visualize_random_sample(dataset_path):
    # Load dataset
    dataset = FabricDataset(dataset_path)

    # Pick random sample
    gamma = 0.5
    print("Dataset length:", len(dataset))
    idx = np.random.randint(len(dataset))
    print(type(dataset))
    idx = np.random.randint(len(dataset))
    x, label = dataset[idx]

    x = x.numpy()  # [C, H, W]

    # Titles for channels
    titles = [
        "Original",
        "Sobel",
        "Gabor 0°",
        "Gabor 45°",
        "Gabor 90°",
        "Gabor 135°"
    ]

    # Plot 3x2 grid
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    for i, ax in enumerate(axes.flat):
        if i < x.shape[0]:
            ax.imshow(x[i], cmap='gray')
            ax.set_title(titles[i] if i < len(titles) else f"Channel {i}")
        ax.axis("off")

    plt.suptitle(f"Class: {dataset.class_names[label]}")
    plt.tight_layout()
    plt.show()

 # -------------------------
# Call the function here
# -------------------------
if __name__ == "__main__":
    visualize_random_sample("C:\\Users\\saski\\Documents\\hold\\output_v3\\train")