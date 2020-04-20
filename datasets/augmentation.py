import tensorflow as tf


def augment(transformations):
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
