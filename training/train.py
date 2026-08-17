import os
import sys
import json
import time
import traceback
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

from dataset import get_datasets
from transforms import get_augmentation_layer, get_preprocessing_fn
from model import build_model, set_trainable_layers

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Experiment Definitions
# ============================================================================
EXPERIMENTS = [
    # {
    #     "id": "experiment_01",
    #     "model": "EfficientNetB0",
    #     "strategy": "partial_fine_tuning",
    #     "image_size": (224, 224),
    #     "batch_size": 16,
    #     "epochs": 20,
    #     "learning_rate": 1e-4,
    #     "optimizer": "Adam",
    #     "dropout": 0.3,
    #     "trainable_layers": 50,
    #     "early_stopping_patience": 5,
    #     "augmentation": {
    #         "horizontal_flip": True,
    #         "rotation": 0.055,   # ±20°
    #         "zoom": 0.15,
    #         "brightness": 0.1,
    #         "contrast": 0.1,
    #     },
    # },
    # {
    #     "id": "experiment_02",
    #     "model": "EfficientNetB0",
    #     "strategy": "gradual_unfreezing",
    #     "image_size": (224, 224),
    #     "batch_size": 16,
    #     "dropout": 0.2,
    #     "optimizer": "Adam",
    #     "early_stopping_patience": 5,
    #     "augmentation": {
    #         "horizontal_flip": True,
    #         "rotation": 0.055,
    #         "zoom": 0.15,
    #         "brightness": 0.1,
    #         "contrast": 0.1,
    #     },
    #     "stages": [
    #         {"epochs": 5, "trainable_layers": 0, "learning_rate": 1e-3},
    #         {"epochs": 5, "trainable_layers": 20, "learning_rate": 1e-4},
    #         {"epochs": 10, "trainable_layers": 50, "learning_rate": 3e-5},
    #     ],
    # },
    # {
    #     "id": "experiment_03",
    #     "model": "EfficientNetB0",
    #     "strategy": "full_fine_tuning",
    #     "image_size": (224, 224),
    #     "batch_size": 16,
    #     "epochs": 25,
    #     "learning_rate": 1e-5,
    #     "optimizer": "Adam",
    #     "dropout": 0.3,
    #     "trainable_layers": -1,  # all layers
    #     "early_stopping_patience": 5,
    #     "augmentation": {
    #         "horizontal_flip": True,
    #         "rotation": 0.055,
    #         "zoom": 0.15,
    #         "brightness": 0.1,
    #         "contrast": 0.1,
    #     },
    # },
    # {
    #     "id": "experiment_04",
    #     "model": "ResNet50",
    #     "strategy": "partial_fine_tuning",
    #     "image_size": (224, 224),
    #     "batch_size": 16,
    #     "epochs": 15,
    #     "learning_rate": 1e-4,
    #     "optimizer": "Adam",
    #     "dropout": 0.2,
    #     "trainable_layers": 35,
    #     "early_stopping_patience": 5,
    #     "augmentation": {
    #         "horizontal_flip": True,
    #         "rotation": 0.055,
    #         "zoom": 0.15,
    #         "brightness": 0.1,
    #         "contrast": 0.1,
    #     },
    # },
    {
        "id": "experiment_05",
        "model": "ConvNeXtTiny",
        "strategy": "partial_fine_tuning",
        "image_size": (224, 224),
        "batch_size": 16,
        "epochs": 20,
        "learning_rate": 1e-4,
        "optimizer": "Adam",
        "dropout": 0.2,
        "trainable_layers": 50,
        "early_stopping_patience": 5,
        "augmentation": {
            "horizontal_flip": True,
            "rotation": 0.055,   # ±20°
            "zoom": 0.15,
            "brightness": 0.1,
            "contrast": 0.1,
        },
    },
    
    {
        "id": "experiment_06",
        "model": "ConvNeXtTiny",
        "strategy": "gradual_unfreezing",
        "image_size": (224, 224),
        "batch_size": 16,
        "dropout": 0.2,
        "optimizer": "Adam",
        "early_stopping_patience": 5,
        "augmentation": {
            "horizontal_flip": True,
            "rotation": 0.055,
            "zoom": 0.15,
            "brightness": 0.1,
            "contrast": 0.1,
        },
        "stages": [
            {"epochs": 5, "trainable_layers": 0, "learning_rate": 1e-3},
            {"epochs": 5, "trainable_layers": 20, "learning_rate": 1e-4},
            {"epochs": 10, "trainable_layers": 50, "learning_rate": 3e-5},
        ],
    },
]


# ============================================================================
# Custom Callback for detailed progress logging
# ============================================================================
class TrainingProgressCallback(keras.callbacks.Callback):
    """Prints detailed per-epoch progress with timing and ETA."""

    def __init__(self, experiment_id, experiment_idx, total_experiments,
                 total_epochs, start_epoch_offset=0):
        super().__init__()
        self.experiment_id = experiment_id
        self.experiment_idx = experiment_idx
        self.total_experiments = total_experiments
        self.total_epochs = total_epochs
        self.start_epoch_offset = start_epoch_offset  # for gradual unfreezing stages
        self.epoch_times = []
        self.experiment_start_time = None
        self.epoch_start_time = None
        self.best_val_acc = 0.0

    def on_train_begin(self, logs=None):
        if self.experiment_start_time is None:
            self.experiment_start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        global_epoch = self.start_epoch_offset + epoch + 1
        print(f"\n{'=' * 60}")
        print(f"Experiment {self.experiment_idx:02d}/{self.total_experiments:02d}"
              f" - {self.experiment_id}")
        print(f"Epoch {global_epoch}/{self.total_epochs}")
        print(f"{'=' * 60}")

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        elapsed = time.time() - self.experiment_start_time

        global_epoch = self.start_epoch_offset + epoch + 1

        train_loss = logs.get("loss", 0)
        train_acc = logs.get("accuracy", 0)
        val_loss = logs.get("val_loss", 0)
        val_acc = logs.get("val_accuracy", 0)

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = self.total_epochs - global_epoch
        if len(self.epoch_times) >= 2:
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_str = _format_time(eta_seconds)
        else:
            eta_str = "calculating..."

        print(f"\nTrain:")
        print(f"  Loss:     {train_loss:.4f}")
        print(f"  Accuracy: {train_acc * 100:.2f}%")
        print(f"\nValidation:")
        print(f"  Loss:     {val_loss:.4f}")
        print(f"  Accuracy: {val_acc * 100:.2f}%")
        print(f"\nBest Validation Accuracy: {self.best_val_acc * 100:.2f}%")
        print(f"\nEpoch Time:              {epoch_time:.1f}s")
        print(f"Elapsed Time:            {_format_time(elapsed)}")
        print(f"Estimated Remaining Time: {eta_str}")


def _format_time(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins:02d}m {secs:02d}s"


# ============================================================================
# Augmentation wrapper for tf.data pipeline
# ============================================================================
def _apply_augmentation_and_preprocessing(ds, augmentation_layer, preprocess_fn,
                                          is_training=False):
    """Apply augmentation (training only) and model-specific preprocessing.

    Args:
        ds: tf.data.Dataset of (image, label) batches (images are uint8).
        augmentation_layer: Keras Sequential augmentation layer.
        preprocess_fn: Model-specific preprocessing function.
        is_training: Whether to apply augmentation.

    Returns:
        tf.data.Dataset with preprocessed float32 images.
    """
    def _process(image, label):
        image = tf.cast(image, tf.float32)
        if is_training and augmentation_layer is not None:
            image = augmentation_layer(image, training=True)
        image = preprocess_fn(image)
        return image, label

    return ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)


# ============================================================================
# Report Generation
# ============================================================================
def _save_report(experiment_config, history_dict, best_epoch, best_val_acc,
                 best_val_loss, training_time, model_path):
    """Save experiment report as JSON."""
    total_epochs_completed = len(history_dict.get("loss", []))
    avg_epoch_time = training_time / max(total_epochs_completed, 1)

    # Build configuration subset for report
    config_report = {
        "image_size": list(experiment_config["image_size"]),
        "batch_size": experiment_config["batch_size"],
        "optimizer": experiment_config["optimizer"],
        "dropout": experiment_config["dropout"],
    }

    if "epochs" in experiment_config:
        config_report["epochs_requested"] = experiment_config["epochs"]
        config_report["learning_rate"] = experiment_config["learning_rate"]
    elif "stages" in experiment_config:
        config_report["stages"] = experiment_config["stages"]

    report = {
        "experiment_id": experiment_config["id"],
        "model": experiment_config["model"],
        "strategy": experiment_config["strategy"],
        "configuration": config_report,
        "result": {
            "epochs_completed": total_epochs_completed,
            "best_epoch": best_epoch,
            "best_val_accuracy": round(float(best_val_acc), 4),
            "best_val_loss": round(float(best_val_loss), 4),
        },
        "timing": {
            "training_time_seconds": round(training_time, 1),
            "average_epoch_time_seconds": round(avg_epoch_time, 1),
        },
        "history": {
            k: [round(float(v), 4) for v in vals]
            for k, vals in history_dict.items()
        },
        "model_path": str(model_path),
    }

    report_path = REPORTS_DIR / f"{experiment_config['id']}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report_path


# ============================================================================
# Print experiment completion summary
# ============================================================================
def _print_completion(experiment_config, epochs_completed, total_epochs,
                      best_epoch, best_val_acc, best_val_loss, training_time,
                      model_path, report_path, next_experiment_id=None):
    """Print a clear summary after an experiment finishes."""
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT {experiment_config['id'].upper()} COMPLETED")
    print(f"{'=' * 60}")
    print(f"\nModel:    {experiment_config['model']}")
    print(f"Strategy: {experiment_config['strategy']}")
    print(f"\nEpochs completed: {epochs_completed}/{total_epochs}")
    print(f"Best epoch:       {best_epoch}")
    print(f"\nBest Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Best Validation Loss:     {best_val_loss:.4f}")
    print(f"\nTraining Time: {_format_time(training_time)}")
    print(f"\nModel saved:  {model_path}")
    print(f"Report saved: {report_path}")
    if next_experiment_id:
        print(f"\nStarting {next_experiment_id}...")
    print(f"{'=' * 60}\n")


# ============================================================================
# Run a single standard experiment (partial / full fine-tuning)
# ============================================================================
def _run_standard_experiment(config, train_ds, val_ds, num_classes,
                             experiment_idx, total_experiments):
    """Run a standard (non-gradual) experiment."""
    exp_id = config["id"]
    model_path = MODELS_DIR / f"{exp_id}.keras"

    # Build model
    model = build_model(
        model_name=config["model"],
        num_classes=num_classes,
        dropout=config["dropout"],
        image_size=config["image_size"],
    )

    # Set trainable layers
    set_trainable_layers(model, config["trainable_layers"])

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Print trainable parameter summary
    total_params = model.count_params()
    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {total_params - trainable_params:,}\n")

    # Callbacks
    progress_cb = TrainingProgressCallback(
        experiment_id=exp_id,
        experiment_idx=experiment_idx,
        total_experiments=total_experiments,
        total_epochs=config["epochs"],
    )

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    early_stop_cb = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=config["early_stopping_patience"],
        restore_best_weights=True,
        verbose=1,
    )

    # Train
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["epochs"],
        callbacks=[progress_cb, checkpoint_cb, early_stop_cb],
        verbose=1,
    )
    training_time = time.time() - start_time

    # Extract results
    history_dict = history.history
    val_accs = history_dict.get("val_accuracy", [])
    val_losses = history_dict.get("val_loss", [])
    best_epoch_idx = int(np.argmax(val_accs)) if val_accs else 0
    best_val_acc = val_accs[best_epoch_idx] if val_accs else 0
    best_val_loss = val_losses[best_epoch_idx] if val_losses else 0
    best_epoch = best_epoch_idx + 1
    epochs_completed = len(val_accs)

    # Early stopping message
    if epochs_completed < config["epochs"]:
        print(f"\nEarly stopping triggered.")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")

    return history_dict, best_epoch, best_val_acc, best_val_loss, \
        epochs_completed, config["epochs"], training_time, model_path


# ============================================================================
# Run a gradual unfreezing experiment
# ============================================================================
def _run_gradual_experiment(config, train_ds, val_ds, num_classes,
                            experiment_idx, total_experiments):
    """Run a gradual unfreezing experiment with multiple stages."""
    exp_id = config["id"]
    model_path = MODELS_DIR / f"{exp_id}.keras"
    stages = config["stages"]
    total_epochs = sum(s["epochs"] for s in stages)

    # Build model
    model = build_model(
        model_name=config["model"],
        num_classes=num_classes,
        dropout=config["dropout"],
        image_size=config["image_size"],
    )

    # Aggregate history across stages
    full_history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    global_best_val_acc = 0.0
    global_best_val_loss = float("inf")
    global_best_epoch = 0
    epoch_offset = 0

    start_time = time.time()

    for stage_idx, stage in enumerate(stages, 1):
        print(f"\n{'*' * 60}")
        print(f"Stage {stage_idx}/{len(stages)} — "
              f"Trainable layers: {stage['trainable_layers']}, "
              f"LR: {stage['learning_rate']}")
        print(f"{'*' * 60}")

        # Unfreeze layers for this stage
        set_trainable_layers(model, stage["trainable_layers"])

        # Recompile with new learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=stage["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Print trainable summary
        total_params = model.count_params()
        trainable_params = sum(
            tf.size(w).numpy() for w in model.trainable_weights
        )
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters:    {total_params - trainable_params:,}\n")

        # Callbacks
        progress_cb = TrainingProgressCallback(
            experiment_id=exp_id,
            experiment_idx=experiment_idx,
            total_experiments=total_experiments,
            total_epochs=total_epochs,
            start_epoch_offset=epoch_offset,
        )
        # Share experiment start time across stages
        if stage_idx > 1:
            progress_cb.experiment_start_time = start_time
            progress_cb.epoch_times = []
            progress_cb.best_val_acc = global_best_val_acc

        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )

        early_stop_cb = keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=config["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=stage["epochs"],
            callbacks=[progress_cb, checkpoint_cb, early_stop_cb],
            verbose=1,
        )

        # Accumulate history
        stage_history = history.history
        for key in full_history:
            full_history[key].extend(stage_history.get(key, []))

        # Update global best
        stage_val_accs = stage_history.get("val_accuracy", [])
        stage_val_losses = stage_history.get("val_loss", [])
        for i, (acc, loss) in enumerate(zip(stage_val_accs, stage_val_losses)):
            if acc > global_best_val_acc:
                global_best_val_acc = acc
                global_best_val_loss = loss
                global_best_epoch = epoch_offset + i + 1

        epoch_offset += len(stage_val_accs)

    training_time = time.time() - start_time
    epochs_completed = len(full_history["loss"])

    return full_history, global_best_epoch, global_best_val_acc, \
        global_best_val_loss, epochs_completed, total_epochs, \
        training_time, model_path


# ============================================================================
# Main training entry point
# ============================================================================
def train():
    print(f"\n{'#' * 60}")
    print(f"# Rock Classification — TensorFlow/Keras Training Pipeline")
    print(f"# {len(EXPERIMENTS)} experiments queued")
    print(f"{'#' * 60}\n")

    # Detect hardware
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"🚀 GPU detected: {gpus[0].name}")
        # Allow memory growth to avoid OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️  No GPU detected — training on CPU (this will be slow)")

    # Load datasets (shared across experiments with same image_size/batch_size)
    # We'll reload per experiment to handle different image sizes if needed
    first_config = EXPERIMENTS[0]
    train_ds_raw, val_ds_raw, test_ds_raw, class_names = get_datasets(
        str(DATA_DIR),
        batch_size=first_config["batch_size"],
        image_size=first_config["image_size"],
    )
    num_classes = len(class_names)
    print(f"\n📋 Classes ({num_classes}): {class_names}")

    # Save labels once
    labels_map = {str(i): name for i, name in enumerate(class_names)}
    labels_file = MODELS_DIR / "labels.json"
    with open(labels_file, "w") as f:
        json.dump(labels_map, f, indent=2)
    print(f"📄 Saved class labels to: {labels_file}\n")

    # Run each experiment
    for exp_idx, config in enumerate(EXPERIMENTS, 1):
        exp_id = config["id"]
        model_path = MODELS_DIR / f"{exp_id}.keras"
        next_id = EXPERIMENTS[exp_idx]["id"] if exp_idx < len(EXPERIMENTS) else None

        print(f"\n{'#' * 60}")
        print(f"# Starting {exp_id} ({exp_idx}/{len(EXPERIMENTS)})")
        print(f"# Model: {config['model']} | Strategy: {config['strategy']}")
        print(f"{'#' * 60}\n")

        try:
            # Reload datasets if image_size or batch_size differs
            if (config["image_size"] != first_config["image_size"] or
                    config["batch_size"] != first_config["batch_size"]):
                train_ds_raw, val_ds_raw, _, _ = get_datasets(
                    str(DATA_DIR),
                    batch_size=config["batch_size"],
                    image_size=config["image_size"],
                )

            # Build augmentation and preprocessing
            aug_layer = get_augmentation_layer(config.get("augmentation"))
            preprocess_fn = get_preprocessing_fn(config["model"])

            # Apply augmentation + preprocessing to datasets
            train_ds = _apply_augmentation_and_preprocessing(
                train_ds_raw, aug_layer, preprocess_fn, is_training=True
            )
            val_ds = _apply_augmentation_and_preprocessing(
                val_ds_raw, None, preprocess_fn, is_training=False
            )

            # Run the experiment
            if config["strategy"] == "gradual_unfreezing":
                result = _run_gradual_experiment(
                    config, train_ds, val_ds, num_classes, exp_idx, len(EXPERIMENTS)
                )
            else:
                result = _run_standard_experiment(
                    config, train_ds, val_ds, num_classes, exp_idx, len(EXPERIMENTS)
                )

            (history_dict, best_epoch, best_val_acc, best_val_loss,
             epochs_completed, total_epochs, training_time, model_path) = result

            # Save report immediately
            report_path = _save_report(
                config, history_dict, best_epoch, best_val_acc,
                best_val_loss, training_time, model_path,
            )

            # Print completion summary
            _print_completion(
                config, epochs_completed, total_epochs, best_epoch,
                best_val_acc, best_val_loss, training_time,
                model_path, report_path, next_id,
            )

            # Clear session to free memory between experiments
            keras.backend.clear_session()

        except Exception as e:
            print(f"\n{'!' * 60}")
            print(f"! EXPERIMENT {exp_id} FAILED")
            print(f"{'!' * 60}")
            print(f"Error: {e}")
            traceback.print_exc()
            print(f"\nPrevious experiment results are preserved.")
            print(f"Continuing to next experiment...\n")

            # Clear session even on failure
            keras.backend.clear_session()
            continue

    # Final summary
    print(f"\n{'#' * 60}")
    print(f"# ALL EXPERIMENTS COMPLETED")
    print(f"{'#' * 60}")
    print(f"\nModels saved in: {MODELS_DIR}")
    print(f"Reports saved in: {REPORTS_DIR}")

    # List saved files
    for f in sorted(MODELS_DIR.glob("*")):
        print(f"  ✓ {f.name}")
    for f in sorted(REPORTS_DIR.glob("*")):
        print(f"  ✓ {f.name}")


if __name__ == "__main__":
    train()