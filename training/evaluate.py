import json
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score

from dataset import get_dataloaders
from transforms import get_transforms

# 1. Path configuration relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "model.pt"
LABELS_PATH = MODELS_DIR / "labels.json"

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Running evaluation on device: {device}")

    # 2. Verify model and labels exist
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError("❌ model.pt or labels.json not found in models/ directory!")

    # 3. Load labels mapping
    with open(LABELS_PATH, "r") as f:
        labels_map = json.load(f)
    num_classes = len(labels_map)
    class_names = [labels_map[str(i)] for i in range(num_classes)]

    # 4. Load Test DataLoader
    _, val_tf = get_transforms()
    _, _, test_loader, _ = get_dataloaders(
        DATA_DIR, batch_size=16, val_tf=val_tf
    )

    # 5. Initialize ConvNeXt-Tiny and load trained weights
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    model = models.convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # 6. Perform inference on test set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    # 7. Calculate accuracy and detailed metrics
    acc = accuracy_score(all_targets, all_preds)
    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        output_dict=True
    )

    print(f"\n🎯 Overall Test Accuracy: {acc * 100:.2f}%\n")

    # 8. Export evaluation summary to reports/model_metrics.json
    metrics_summary = {
        "test_accuracy": round(float(acc), 4),
        "model_architecture": "ConvNeXt-Tiny",
        "detailed_report": report
    }

    metrics_file = REPORTS_DIR / "model_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"📊 Saved evaluation report to: {metrics_file}")

if __name__ == "__main__":
    evaluate()