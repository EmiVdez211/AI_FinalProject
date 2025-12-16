from __future__ import annotations

import tensorflow as tf


def build_model(num_classes: int, image_size: tuple[int, int]) -> tf.keras.Model:
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")

    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3), name="image")

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    x = data_augmentation(inputs)

    # MobileNetV2 preprocess_input equivalent:
    # x in [0,255] -> x/127.5 - 1 => [-1,1]
    x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescale")(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="beverage_classifier")
