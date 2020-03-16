import tensorflow as tf
from datasets.sources.scryfall import search, get_image


def load_forest_dataset():
    @tf.function
    def load_image(img_file, label):
        # Read image from file and decode
        img = tf.io.read_file(img_file)
        img = tf.io.decode_image(img)

        # Normalize tp [0, 1]
        img = tf.image.convert_image_dtype(img, tf.float32)

        return img, label

    img_files = [get_image(forest["image_uris"]["art_crop"]) for forest in search("%21forest", unique="art")]
    labels = [4 for img_file in img_files]  # WUBRG

    N = len(img_files)
    dataset = tf.data.Dataset.from_tensor_slices((img_files, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset, N
