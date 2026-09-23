"""TensorFlow model builders with lazy optional imports."""

from __future__ import annotations


def _keras():
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is optional. Install requirements-train.txt to build models."
        ) from exc
    return keras


def build_classifier():
    keras = _keras()
    return keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ], name="fashion_classifier")


def build_autoencoder(latent_dim: int = 32):
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    keras = _keras()
    inputs = keras.layers.Input(shape=(784,))
    encoded = keras.layers.Dense(128, activation="relu")(inputs)
    encoded = keras.layers.Dense(64, activation="relu")(encoded)
    latent = keras.layers.Dense(latent_dim, activation="relu", name="latent")(encoded)
    decoded = keras.layers.Dense(64, activation="relu")(latent)
    decoded = keras.layers.Dense(128, activation="relu")(decoded)
    outputs = keras.layers.Dense(784, activation="sigmoid")(decoded)
    return keras.Model(inputs, outputs, name="fashion_autoencoder")


def build_dcgan_generator(noise_dim: int = 100):
    if noise_dim <= 0:
        raise ValueError("noise_dim must be positive")
    keras = _keras()
    return keras.Sequential([
        keras.layers.Input(shape=(noise_dim,)),
        keras.layers.Dense(7 * 7 * 128),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Reshape((7, 7, 128)),
        keras.layers.Conv2DTranspose(128, 4, strides=2, padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Conv2DTranspose(64, 4, strides=2, padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Conv2D(1, 7, padding="same", activation="tanh"),
    ], name="dcgan_generator")


def build_dcgan_discriminator():
    keras = _keras()
    return keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(64, 4, strides=2, padding="same"),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(128, 4, strides=2, padding="same"),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name="dcgan_discriminator")
