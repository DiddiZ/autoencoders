import tensorflow as tf


def transform_sequences(transformations):
    """Applies a list of transformations to each sequence.

    Returns a new data set, consisting of applying all transfromations to all examples.
    If the original examples are to be kept, pass the identity function as transformation.
    """
    @tf.function
    def apply_transformations(image, label):
        return tf.data.Dataset.from_tensor_slices(
            (
                [fn(image) for fn in transformations],
                [label for fn in transformations],
            )
        )

    return lambda dataset: dataset.interleave(apply_transformations, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def default_transformations():
    """Set of image transformations.

    Includes identity, vertical and horizontal flip, rotation to left and right."""
    return [
        lambda img: img,  # Identity
        tf.image.flip_left_right,  # hflip
        tf.image.flip_up_down,  # vflip
        lambda img: tf.image.rot90(img, k=1),  # Rotation left
        lambda img: tf.image.rot90(img, k=3),  # Rotation right
    ]