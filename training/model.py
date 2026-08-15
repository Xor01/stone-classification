import tensorflow as tf
from tensorflow import keras


def build_model(model_name, num_classes, dropout=0.3, image_size=(224, 224)):
    """Build a transfer learning model with a custom classification head.

    Args:
        model_name: 'EfficientNetB0' or 'ResNet50'.
        num_classes: Number of output classes.
        dropout: Dropout rate before the final Dense layer.
        image_size: Tuple (height, width) for input shape.

    Returns:
        A tf.keras.Model (not compiled — caller handles compilation).
    """
    input_shape = (*image_size, 3)

    # Load pretrained backbone without top classification layers
    if model_name == "EfficientNetB0":
        backbone = keras.applications.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )
    elif model_name == "ResNet50":
        backbone = keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Freeze all backbone layers by default
    backbone.trainable = False

    # Build the full model
    inputs = keras.Input(shape=input_shape)
    x = backbone(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model


def set_trainable_layers(model, num_layers=None):
    """Unfreeze backbone layers for fine-tuning.

    The model is expected to have structure: Input → Backbone → GAP → Dropout → Dense.
    The backbone is model.layers[1].

    Args:
        model: A Keras Model built by build_model().
        num_layers: Number of layers to unfreeze from the end of the backbone.
            - None or 0: keep all frozen
            - negative or >= total layers: unfreeze all
    """
    backbone = model.layers[1]  # The backbone (EfficientNetB0 or ResNet50)

    if num_layers is None or num_layers == 0:
        backbone.trainable = False
        return

    backbone.trainable = True
    total = len(backbone.layers)

    if num_layers < 0 or num_layers >= total:
        # Unfreeze all layers
        for layer in backbone.layers:
            layer.trainable = True
    else:
        # Freeze early layers, unfreeze last num_layers
        for layer in backbone.layers[: total - num_layers]:
            layer.trainable = False
        for layer in backbone.layers[total - num_layers :]:
            layer.trainable = True
