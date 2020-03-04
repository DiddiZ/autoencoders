import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # Reduce TF log level

from pathlib import Path
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datasets.augmentation import transform_sequences, default_transformations
from datasets.pokemon import load_pokemon_dataset
from models.autoencoder import Autoencoder, VAE, VAE_GAN
from models.coders import encoder_linear, decoder_linear, encoder_fc, decoder_fc, encoder_conv_64x64, decoder_conv_64x64

if __name__ == "__main__":
    shape = (64, 64, 4)
    latent_dim = 128
    learn_rate = 2e-5
    beta_kl = 1e-1
    epochs = 1000
    batch_size = 64
    log_dir = "E:/log/"
    checkpoint_interval = 100

    dataset, N = load_pokemon_dataset(shape)

    N_train = 768
    N_test = N - N_train
    train_dataset = dataset.take(N_train)
    test_dataset = dataset.skip(N_train)

    transformations = default_transformations()
    train_dataset = train_dataset.apply(transform_sequences(transformations)).cache().shuffle(
        N_train * len(transformations)
    ).batch(batch_size).prefetch(2)
    test_dataset = test_dataset.cache().shuffle(N_test).batch(N_test)

    # model = VAE(
    #     # Autoencoder(
    #     #encoder=encoder_linear(shape, latent_dim),
    #     #decoder=decoder_linear(latent_dim, shape),
    #     #encoder=encoder_fc(shape, latent_dim * 2),
    #     #decoder=decoder_fc(latent_dim, shape),
    #     encoder=encoder_conv_64x64(shape[-1], latent_dim * 2),
    #     decoder=decoder_conv_64x64(latent_dim, shape[-1]),
    #     beta_kl=beta_kl,
    # )

    model = VAE_GAN(
        encoder=encoder_conv_64x64(shape[-1], latent_dim * 2),
        decoder=decoder_conv_64x64(latent_dim, shape[-1]),
        discriminator=encoder_conv_64x64(shape[-1], 1),
        beta_kl=beta_kl,
        l=5,
    )

    # Initialize optimizer
    opt = tf.optimizers.Adam(learn_rate)

    # Construct descriptive name
    run_name = time.strftime("%Y%m%d-%H%M%S") + f",dZ={latent_dim},lr={learn_rate},beta_kl={beta_kl}"
    Path(log_dir + run_name).mkdir(parents=True, exist_ok=True)

    writer = tf.summary.create_file_writer(log_dir + run_name)
    with writer.as_default():
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            model.reset_metrics()

            # Train batches
            for i, (image, label) in enumerate(train_dataset):
                model.train(opt, image)

            pbar.set_description(
                "Loss: %.6f/Rec: %.6f" % (
                    model.metric_loss.result(),
                    model.metric_loss_rec.result(),
                )
            )

            # Log training summaries
            for metric in model.metrics:
                tf.summary.scalar(metric.name, metric.result(), step=epoch)

            # Log train set reconstruction
            image, _ = tf.data.experimental.get_single_element(train_dataset.take(1))
            reconstructed, z, _, _ = model(image)
            tf.summary.image('train/target', image, step=epoch, max_outputs=3)
            tf.summary.image('train/reconstructed', reconstructed, step=epoch, max_outputs=3)

            # Log train set reconstruction
            image, _ = tf.data.experimental.get_single_element(test_dataset.take(1))
            reconstructed, z, _, _ = model(image)
            tf.summary.image('test/target', image, step=epoch, max_outputs=3)
            tf.summary.image('test/reconstructed', reconstructed, step=epoch, max_outputs=3)

            tf.summary.histogram('test/latent', z, step=epoch)
            tf.summary.scalar('test/loss_rec', tf.reduce_mean(tf.square(image - reconstructed)), step=epoch)

            # Log random sample reconstruction
            tf.summary.image('random', model.sample(6), step=epoch, max_outputs=6)

            writer.flush()

            # Save model
            if (epoch + 1) % checkpoint_interval == 0:
                model.save_weights(log_dir + run_name + ("/checkpoint%03d" % epoch))
