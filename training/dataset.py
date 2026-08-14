import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class RockDataset(Dataset):
    """Custom PyTorch Dataset for loading rock images and labels from Roboflow CSV."""
    def __init__(self, split_dir, transform=None):
        self.split_dir = split_dir
        self.transform = transform
        
        # 1. Load the CSV file containing image filenames and one-hot labels
        csv_path = os.path.join(split_dir, "_classes.csv")
        df = pd.read_csv(csv_path)
        
        # 2. Extract class names and build full image file paths
        self.class_names = list(df.columns[1:])
        self.image_paths = [os.path.join(split_dir, name) for name in df["filename"]]
        
        # 3. Convert One-Hot encoded columns into integer class indices
        one_hot = df.iloc[:, 1:].values
        self.labels = np.argmax(one_hot, axis=1)

    def __len__(self):
        # Return total number of images in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Fetch a single image and convert it to RGB format
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        # Apply image transformations if provided
        if self.transform:
            image = self.transform(image)

        # Return transformed image tensor and label tensor
        return image, torch.tensor(label, dtype=torch.long)


def get_dataloaders(data_dir, batch_size=16, train_tf=None, val_tf=None):
    """Helper function to create Train, Validation, and Test DataLoaders."""
    # Create RockDataset instances for each split
    train_ds = RockDataset(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds   = RockDataset(os.path.join(data_dir, "valid"), transform=val_tf)
    test_ds  = RockDataset(os.path.join(data_dir, "test"), transform=val_tf)

    # Wrap Datasets into DataLoaders for batching and shuffling
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds.class_names

