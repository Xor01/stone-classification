import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class RockDataset(Dataset):
    """Load either Roboflow CSV exports or folder-per-class exports."""

    def __init__(self, split_dir, transform=None):
        self.split_dir = Path(split_dir)
        self.transform = transform

        csv_path = self.split_dir / "_classes.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            self.class_names = list(df.columns[1:])
            self.image_paths = [str(self.split_dir / name) for name in df["filename"]]
            one_hot = df.iloc[:, 1:].values
            self.labels = np.argmax(one_hot, axis=1)
            return

        class_dirs = sorted(
            [p for p in self.split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")],
            key=lambda p: p.name,
        )
        if not class_dirs:
            raise FileNotFoundError(
                f"No class folders found under {self.split_dir}. "
                "Expected folders like 'Basalt', 'Clay', etc."
            )

        self.class_names = [p.name for p in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.image_paths = []
        self.labels = []

        for class_name in self.class_names:
            class_dir = self.split_dir / class_name
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    self.image_paths.append(str(image_path))
                    self.labels.append(self.class_to_idx[class_name])

        if not self.image_paths:
            raise FileNotFoundError(
                f"No image files found under {self.split_dir}. "
                "Check that your Roboflow export contains image files in class folders."
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_dataloaders(data_dir, batch_size=16, train_tf=None, val_tf=None):
    """Create Train/Validation/Test loaders for either CSV or folder-based Roboflow exports."""
    train_ds = RockDataset(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = RockDataset(os.path.join(data_dir, "valid"), transform=val_tf)
    test_ds = RockDataset(os.path.join(data_dir, "test"), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds.class_names

