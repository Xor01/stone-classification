import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

from transforms import get_transforms
from dataset import get_dataloaders

# 1. Automatic path detection to avoid missing file errors
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train():
    # 2. Select hardware device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # 3. Load dataset transforms and dataloaders
    train_tf, val_tf = get_transforms()
    train_loader, val_loader, _, class_names = get_dataloaders(
        DATA_DIR, batch_size=16, train_tf=train_tf, val_tf=val_tf
    )
    num_classes = len(class_names)

    # 4. Load Pretrained ConvNeXt-Tiny model
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    model = models.convnext_tiny(weights=weights)

    # 5. Modify classification head for our 9 rock classes
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    model = model.to(device)

    # 6. Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # 7. Start training loop
    epochs = 5
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

        train_acc = correct.double() / total

        # Validation Phase
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)

        val_acc = val_correct.double() / val_total
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Save the best model state
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_save_path = MODELS_DIR / "model.pt"
            torch.save(model.state_dict(), model_save_path)
            print(f"💾 Saved best model to: {model_save_path}")

    # 8. Save class labels mapping to models/labels.json for Backend
    labels_map = {str(i): name.strip() for i, name in enumerate(class_names)}
    labels_file = MODELS_DIR / "labels.json"
    with open(labels_file, "w") as f:
        json.dump(labels_map, f, indent=2)
    print(f"📄 Saved class labels map to: {labels_file}")

if __name__ == "__main__":
    train()