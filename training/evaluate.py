import json
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score

from dataset import get_datasets
from transforms import get_preprocessing_fn

# Path configuration relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LABELS_PATH = MODELS_DIR / "labels.json"


def evaluate(model_path=None):
    """Evaluate a trained Keras model on the test set.

    Args:
        model_path: Path to a .keras model file. If None, evaluates the
                     first experiment_*.keras file found in models/.
    """
    print(f"🔍 Running evaluation")

    # 1. Find model file
    if model_path is None:
        keras_files = sorted(MODELS_DIR.glob("experiment_*.keras"))
        if not keras_files:
            raise FileNotFoundError(
                "❌ No experiment_*.keras files found in models/ directory!"
            )
        model_path = experiment_01.keras  # Use the latest experiment
        print(f"   Using model: {model_path.name}")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model not found: {model_path}")

    # 2. Load labels mapping
    if not LABELS_PATH.exists():
        raise FileNotFoundError("❌ labels.json not found in models/ directory!")

    with open(LABELS_PATH, "r") as f:
        labels_map = json.load(f)
    num_classes = len(labels_map)
    class_names = [labels_map[str(i)] for i in range(num_classes)]
    print(f"   Classes ({num_classes}): {class_names}")

    # 3. Determine model type from filename or labels for preprocessing
    # Try to read the experiment report for model name
    exp_id = model_path.stem  # e.g. "experiment_01"
    report_path = REPORTS_DIR / f"{exp_id}.json"
    model_name = "EfficientNetB0"  # default
    if report_path.exists():
        with open(report_path, "r") as f:
            report = json.load(f)
        model_name = report.get("model", "EfficientNetB0")

    print(f"   Model architecture: {model_name}")

    # 4. Load test dataset
    _, _, test_ds_raw, _ = get_datasets(
        str(DATA_DIR), batch_size=16, image_size=(224, 224)
    )

    # Apply preprocessing (no augmentation for test)
    preprocess_fn = get_preprocessing_fn(model_name)

    def _preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = preprocess_fn(image)
        return image, label

    test_ds = test_ds_raw.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # 5. Load trained model
    model = keras.models.load_model(str(model_path))
    print(f"   Model loaded successfully\n")

    # 6. Perform inference on test set
    all_preds = []
    all_targets = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        preds = np.argmax(predictions, axis=1)
        all_preds.extend(preds)
        all_targets.extend(labels.numpy())

    # 7. Calculate accuracy and detailed metrics
    acc = accuracy_score(all_targets, all_preds)
    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        output_dict=True,
    )

    # Print human-readable report
    print(classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
    ))

    print(f"\n🎯 Overall Test Accuracy: {acc * 100:.2f}%\n")

    # 8. Export evaluation summary
    metrics_summary = {
        "experiment_id": exp_id,
        "model_architecture": model_name,
        "model_path": str(model_path),
        "test_accuracy": round(float(acc), 4),
        "detailed_report": report,
    }

    metrics_file = REPORTS_DIR / f"{exp_id}_test_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"📊 Saved evaluation report to: {metrics_file}")


if __name__ == "__main__":
    # Accept optional model path as command-line argument
    path = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate(path)