# train.py (or continue in same file)
# ---------------------------
# Imports
# ---------------------------
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import FabricDataset  # your dataset class
from model import SimpleCNN        # your CNN class

# ---------------------------
# Training curves plotting
# ---------------------------
def plot_training_curves(losses, val_accuracies):
    epochs = range(1, len(losses)+1)
    plt.figure(figsize=(12,5))

    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, losses, 'b-o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    # Validation Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, val_accuracies, 'g-o', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# ---------------------------
# Training function
# ---------------------------
def train_model(train_dir, val_dir, epochs=10, batch_size=16, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_dataset = FabricDataset(train_dir)
    val_dataset = FabricDataset(val_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = SimpleCNN(num_classes=len(train_dataset.class_names)).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Track metrics per epoch
    losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        # --- Training ---
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)

        # --- Validation ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

    # Plot training curves
    plot_training_curves(losses, val_accuracies)

    return model

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    dataset_root = input("Enter dataset root (train/val/test): ").strip()
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")

    # Train model
    model = train_model(train_dir, val_dir, epochs=10)

    # Save model
    model_path = "fabric_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")