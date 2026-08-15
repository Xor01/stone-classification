import os
import pandas as pd
import numpy as np
import tensorflow as tf


def _load_split_info(split_dir):
    """Read _classes.csv from a Roboflow split directory.
    
    Returns:
        image_paths: list of absolute file paths
        labels: numpy array of integer class indices
        class_names: list of class name strings
    """
    csv_path = os.path.join(split_dir, "_classes.csv")
    df = pd.read_csv(csv_path)

    # Extract class names from CSV columns (skip 'filename')
    class_names = [col.strip() for col in df.columns[1:]]

    # Build full image file paths
    image_paths = [os.path.join(split_dir, name.strip()) for name in df["filename"]]

    # Convert one-hot encoded columns into integer class indices
    one_hot = df.iloc[:, 1:].values
    labels = np.argmax(one_hot, axis=1)

    return image_paths, labels, class_names


def _load_and_resize_image(file_path, label, image_size):
    """Load a JPEG image from disk and resize it."""
    raw = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, image_size)
    # Cast to uint8 for compatibility with Keras preprocessing layers
    image = tf.cast(image, tf.uint8)
    return image, label


def get_datasets(data_dir, batch_size=16, image_size=(224, 224)):
    """Create train, validation, and test tf.data.Dataset objects.

    Args:
        data_dir: Path to the data directory containing train/, valid/, test/ splits.
        batch_size: Batch size for all datasets.
        image_size: Tuple (height, width) for resizing images.

    Returns:
        train_ds: tf.data.Dataset (shuffled, batched, prefetched)
        val_ds: tf.data.Dataset (batched, prefetched)
        test_ds: tf.data.Dataset (batched, prefetched)
        class_names: List of class name strings
    """
    # Load split information
    train_paths, train_labels, class_names = _load_split_info(
        os.path.join(data_dir, "train")
    )
    val_paths, val_labels, _ = _load_split_info(
        os.path.join(data_dir, "valid")
    )
    test_paths, test_labels, _ = _load_split_info(
        os.path.join(data_dir, "test")
    )

    # Build tf.data.Dataset for each split
    def _make_dataset(paths, labels, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
        ds = ds.map(
            lambda p, l: _load_and_resize_image(p, l, image_size),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _make_dataset(train_paths, train_labels, shuffle=True)
    val_ds = _make_dataset(val_paths, val_labels, shuffle=False)
    test_ds = _make_dataset(test_paths, test_labels, shuffle=False)

    return train_ds, val_ds, test_ds, class_names
