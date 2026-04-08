# evaluate.py

import torch
from torch.utils.data import DataLoader
from dataset import FabricDataset
from model import SimpleCNN
import torch
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Setup
# -----------------------------
test_dir = "processed/test"

# Temporary dataset to get class names dynamically
temp_dataset = FabricDataset(test_dir)

# Recreate model architecture
model = SimpleCNN(num_classes=len(temp_dataset.class_names))
model.load_state_dict(torch.load("fabric_cnn.pth"))
model.eval()  # important for dropout/batchnorm

# -----------------------------
# Evaluation function
# -----------------------------
def evaluate_model(model, test_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_dataset = FabricDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=16)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

# -----------------------------
# Run evaluation
# -----------------------------
if __name__ == "__main__":
    evaluate_model(model, test_dir)

    # confusion matrix

def evaluate_model_no_sklearn(model, test_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_dataset = FabricDataset(test_dir)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    all_labels = []
    all_preds = []

    # Gather predictions
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)


    # Per-class accuracy
    class_names = test_dataset.class_names
    for i, class_name in enumerate(class_names):
        idxs = (all_labels == i)
        correct = np.sum(all_preds[idxs] == all_labels[idxs])
        total = np.sum(idxs)
        class_acc = 100 * correct / total if total > 0 else 0
        print(f"{class_name}: {class_acc:.2f}% ({correct}/{total})")

    # Confusion matrix
    test_dataset = FabricDataset(test_dir)
    num_classes = len(test_dataset.class_names)

    model = SimpleCNN(num_classes=num_classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Loop over data to add text
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")

    plt.title("Confusion Matrix")
    plt.colorbar(im)
    plt.show()

# Recreate model first
num_classes = len(temp_dataset.class_names)  # cotton vs poly_satin
model = SimpleCNN(num_classes=num_classes)
model.load_state_dict(torch.load("fabric_cnn.pth"))
model.eval()

# Evaluate
evaluate_model_no_sklearn(model, test_dir="processed/test")

    # visualisation tool

def visualize_predictions(model, test_dir, n=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_dataset = FabricDataset(test_dir)
    indices = np.random.choice(len(test_dataset), n, replace=False)

    for idx in indices:
        img, label = test_dataset[idx]
        img_input = img.unsqueeze(0).to(device)  # add batch dim

        with torch.no_grad():
            output = model(img_input)
            _, pred = torch.max(output, 1)

        plt.imshow(img[0].numpy(), cmap="gray")
        plt.title(f"True: {test_dataset.class_names[label]}, Pred: {test_dataset.class_names[pred.item()]}")
        plt.axis('off')
        plt.show()