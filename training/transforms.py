import tensorflow as tf
from tensorflow import keras


def get_augmentation_layer(config=None):
    """Build a Keras Sequential augmentation layer from config.

    Args:
        config: dict with optional keys:
            - horizontal_flip (bool, default True)
            - rotation (float, fraction of 2π, default 0.055 ≈ ±20°)
            - zoom (float, default 0.15)
            - brightness (float, default 0.1)
            - contrast (float, default 0.1)

    Returns:
        A tf.keras.Sequential layer that applies random augmentation.
    """
    if config is None:
        config = {}

    layers = []

    if config.get("horizontal_flip", True):
        layers.append(keras.layers.RandomFlip("horizontal"))

    rotation = config.get("rotation", 0.055)  # ±20° ≈ 20/360
    if rotation > 0:
        layers.append(keras.layers.RandomRotation(rotation))

    zoom = config.get("zoom", 0.15)
    if zoom > 0:
        layers.append(keras.layers.RandomZoom((-zoom, zoom)))

    brightness = config.get("brightness", 0.1)
    if brightness > 0:
        layers.append(keras.layers.RandomBrightness(brightness))

    contrast = config.get("contrast", 0.1)
    if contrast > 0:
        layers.append(keras.layers.RandomContrast(contrast))

    return keras.Sequential(layers, name="augmentation")


def get_preprocessing_fn(model_name):
    """Return the appropriate Keras preprocess_input function for a model.

    Args:
        model_name: One of 'EfficientNetB0', 'ResNet50', 'ConvNeXtTiny'.

    Returns:
        A callable that preprocesses image tensors for the specified model.
    """
    if model_name == "EfficientNetB0":
        return tf.keras.applications.efficientnet.preprocess_input
    elif model_name == "ResNet50":
        return tf.keras.applications.resnet50.preprocess_input
    elif model_name == "ConvNeXtTiny":
        return tf.keras.applications.convnext.preprocess_input
    else:
        raise ValueError(f"Unknown model name: {model_name}")