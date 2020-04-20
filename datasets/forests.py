import numpy as np
import tensorflow as tf
from datasets.sources.scryfall import search, get_image


def load_forest_dataset(shape=None, shuffle_seed=0):
    """Collection of artworks of MTG forests.

    Image data is fetched from the Scryfall API.
    At the time of writing, there are 212 unique artworks available,
    though this number grows as new sets are released.

    The art crops are not uniform in size.

    Args:
        shape: Size to resize to. `None` for original size.
        shuffle_seed: Seed used to shuffle the datased intially.

    Returns:
        dataset
        N: length of datase
        shape: Shape of the dataset entries. `None` if not uniform.
    """
    @tf.function
    def load_image(img_file, label):
        # Read image from file and decode
        img = tf.io.read_file(img_file)
        img = tf.image.decode_jpeg(img, channels=shape[-1] if shape is not None else 0)

        # Normalize tp [0, 1]
        img = tf.image.convert_image_dtype(img, tf.float32)

        # Resize input image
        if shape is not None:
            img = tf.image.resize(img, shape[:2])

        return img, label

    img_files = [get_image(forest["image_uris"]["art_crop"]) for forest in search("%21forest", unique="art")]
    np.random.RandomState(0).shuffle(img_files)  # Initial, determinisitc shuffling
    labels = [4 for img_file in img_files]  # WUBRG

    N = len(img_files)
    dataset = tf.data.Dataset.from_tensor_slices((img_files, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset, N, shape
