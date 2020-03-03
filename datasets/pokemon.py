from pathlib import Path
import tensorflow as tf


def load_pokemon_dataset(shape=(256, 256, 3), image_folder="pokemon"):
    @tf.function
    def load_image(img_file, label):
        # Read image from file and decode
        img = tf.io.read_file(img_file)
        img = tf.image.decode_png(img, channels=shape[-1])

        # Normalize tp [0, 1]
        img = tf.image.convert_image_dtype(img, tf.float32)

        # Resize input image (RGB)
        if shape[:2] != (256, 256):
            img = tf.image.resize(img, shape[:2])

        return img, label

    img_paths = [child for child in Path(image_folder).iterdir() if child.name[-4:] == ".png"]
    img_files = [str(child) for child in img_paths]
    labels = [child.name[:-4] for child in img_paths]

    N = len(img_files)
    dataset = tf.data.Dataset.from_tensor_slices((img_files, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset, N
