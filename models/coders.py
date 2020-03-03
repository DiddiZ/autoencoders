import numpy as np
import tensorflow as tf


def encoder_linear(input_shape, latent_dim):
    """Fully connected linear encoder."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim, activation=None),
        ],
    )


def decoder_linear(latent_dim, output_shape):
    """Fully connected linear decoder."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=latent_dim),
            tf.keras.layers.Dense(np.prod(output_shape), activation=None),
            tf.keras.layers.Reshape(output_shape),
        ],
    )


def encoder_fc(input_shape, latent_dim):
    """Fully connected linear encoder."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation=tf.nn.relu),
            tf.keras.layers.Dense(1024, activation=tf.nn.relu),
            tf.keras.layers.Dense(latent_dim, activation=None),
        ],
    )


def decoder_fc(latent_dim, output_shape):
    """Fully connected linear decoder."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=latent_dim),
            tf.keras.layers.Dense(1028, activation=tf.nn.relu),
            tf.keras.layers.Dense(4096, activation=tf.nn.relu),
            tf.keras.layers.Dense(np.prod(output_shape), activation=None),
            tf.keras.layers.Reshape(output_shape),
        ],
    )


def encoder_conv_64x64(channels, latent_dim):
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(64, 64, channels)),
            # (64, 64, channels) -> (32, 32, 32)
            tf.keras.layers.Conv2D(32, 5, 2, "same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (32, 32, 32) -> (16, 16, 64)
            tf.keras.layers.Conv2D(64, 3, 2, "same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (16, 16, 64) -> (8, 8, 128)
            tf.keras.layers.Conv2D(128, 3, 2, "same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (16, 16, 64) -> (latent_dim, )
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim, activation=None),
        ],
    )


def decoder_conv_64x64(latent_dim, channels):
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim, )),
            # (latent_dim, ) -> (8, 8, 128)
            tf.keras.layers.Dense(8 * 8 * 128, activation=None),
            tf.keras.layers.Reshape([8, 8, 128]),
            # (8, 8, 128) -> (16, 16, 64)
            tf.keras.layers.Conv2DTranspose(64, 3, 2, "same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (16, 16, 64) -> (32, 32, 32)
            tf.keras.layers.Conv2DTranspose(32, 3, 2, "same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # (32, 32, 32) -> (64, 64, channels)
            tf.keras.layers.Conv2DTranspose(channels, 5, 2, "same"),
        ],
    )
