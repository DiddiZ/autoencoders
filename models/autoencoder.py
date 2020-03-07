import tensorflow as tf


class Autoencoder(tf.keras.models.Model):
    """Basic autoencoder consisitng of a encoder and a decoder part."""
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        # Metrics
        self.metric_loss = tf.metrics.Mean('train/loss')
        self.metric_loss_rec = tf.metrics.Mean('train/loss_rec')

    @tf.function
    def call(self, input_image, training=False):
        """Forward pass through the autoencoder.

        Returns:
            reconstruction: Output of the AE
            z: Latent representation at the bottle neck
        """
        # Get latent represenation for input
        z = self.encoder(input_image, training=training)

        # Reconstruct from latent representation
        reconstruction = tf.nn.sigmoid(self.decoder(z, training=training))

        return reconstruction, z

    @tf.function
    def train(self, opt, input_image):
        with tf.GradientTape() as tape:
            reconstructed, z = self(input_image, training=True)
            loss_rec = tf.reduce_mean(tf.square(input_image - reconstructed))
            loss = loss_rec

        gradients = tape.gradient(loss, self.trainable_variables)
        opt.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.metric_loss(loss)
        self.metric_loss_rec(loss_rec)


class VAE(Autoencoder):
    """Variational autoencoder."""
    def __init__(self, encoder, decoder, beta_kl):
        """Initialize the model.

        Args:
            beta_kl: Weight of the KL-divergence in the loss term
        """
        super().__init__(encoder, decoder)

        assert encoder.output.shape[-1] % 2 == 0
        self.latent_dim = encoder.output.shape[-1] // 2
        self.beta_kl = beta_kl

        # Metrics
        self.metric_loss_kl = tf.metrics.Mean('train/loss_kl')

    @tf.function
    def call(self, input_image, training=False):
        """Forward pass through the autoencoder.

        Returns:
            reconstruction: Output of the AE
            z: Latent representation at the bottle neck
            z_mean: Mean of the latent representation
            z_log_var: Log-variance of the latent representation
        """
        # Get latent represenation for input
        encoded = self.encoder(input_image, training=training)
        z_mean = encoded[:, :self.latent_dim]
        z_log_var = encoded[:, self.latent_dim:]

        # sample from latent space
        epsilon = tf.random.normal(z_log_var.shape)
        z = z_mean + tf.exp(z_log_var) * epsilon

        # Reconstruct from latent representation
        reconstruction = tf.nn.sigmoid(self.decoder(z, training=training))

        return reconstruction, z, z_mean, z_log_var

    @tf.function
    def sample(self, N):
        """Reconstruct random samples from the latent space.

        Returns:
            reconstruction: Output of the AE
        """
        # Random sample
        z = tf.random.normal((N, self.latent_dim))

        # Reconstruct from latent representation
        reconstruction = tf.nn.sigmoid(self.decoder(z))

        return reconstruction

    @tf.function
    def train(self, opt, input_image):
        with tf.GradientTape() as tape:
            reconstructed, _, z_mean, z_log_var = self(input_image, training=True)

            loss_rec = tf.reduce_mean(tf.square(input_image - reconstructed))
            loss_kl = 0.5 * tf.reduce_mean(tf.exp(z_log_var) + tf.square(z_mean) - z_log_var - 1)
            loss = loss_rec + self.beta_kl * loss_kl

        opt.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))

        # Update metrics
        self.metric_loss(loss)
        self.metric_loss_rec(loss_rec)
        self.metric_loss_kl(loss_kl)


class VAE_GAN(VAE):
    """VAE-GAN

    See:
        https://arxiv.org/abs/1512.09300
    """
    def __init__(self, encoder, decoder, discriminator, beta_kl, l):
        """Initialize the model.

        Args:
            beta_kl: Weight of the KL-divergence in the encoder loss term.
            l: Index of intermediate layer of the discriminator used for reconstruction loss.
        """
        super().__init__(encoder, decoder, beta_kl)

        # Expose l-th layer of discriminator
        self.discriminator = tf.keras.Model(
            inputs=discriminator.input,
            outputs=[discriminator.layers[l].output, discriminator.output],
        )

        # Metrics
        self.metric_loss_enc = tf.metrics.Mean('train/loss_enc')
        self.metric_loss_dec = tf.metrics.Mean('train/loss_dec')
        self.metric_loss_dis = tf.metrics.Mean('train/loss_dis')
        self.metric_acc_real = tf.metrics.Accuracy('train/acc_real')
        self.metric_acc_fake = tf.metrics.Accuracy('train/acc_fake')

    @tf.function
    def train(self, opt, input_image):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as dis_tape:
            reconstruction, _, z_mean, z_log_var = self(input_image, training=True)
            reconstruction_random = self.sample(tf.shape(input_image)[0])

            # Run distriminator on target images
            dis_real_inter, dis_real_logits = self.discriminator(input_image, training=True)
            # Run distriminator on generated images
            dis_fake_inter, dis_fake_logits = self.discriminator(reconstruction, training=True)
            _, dis_fake2_logits = self.discriminator(reconstruction_random, training=True)

            # MSE at l-th discriminator layer
            loss_rec = tf.reduce_mean(tf.square(dis_real_inter - dis_fake_inter))
            # KL-divergence of the latent space
            loss_kl = 0.5 * tf.reduce_mean(tf.exp(z_log_var) + tf.square(z_mean) - z_log_var - 1)
            # Cross-entropy of real images
            loss_real = tf.reduce_mean(cross_entropy(0.1 * tf.ones_like(dis_real_logits), dis_real_logits))
            # Cross-entropy of generated images
            loss_fake = (
                tf.reduce_mean(cross_entropy(0.9 * tf.ones_like(dis_fake_logits), dis_fake_logits)) +
                tf.reduce_mean(cross_entropy(0.9 * tf.ones_like(dis_fake2_logits), dis_fake2_logits))
            ) / 2
            # Ability of generator to fool the discriminator
            loss_fool = tf.reduce_mean(cross_entropy(tf.zeros_like(dis_fake_logits), dis_fake_logits))

            tf.reduce_mean(tf.math.log(1 + tf.exp(-dis_fake_logits)))

            loss_enc = loss_rec + self.beta_kl * loss_kl
            loss_dec = loss_rec + loss_fool
            loss_dis = loss_real + loss_fake

        # Update encoder
        opt.apply_gradients(
            zip(
                enc_tape.gradient(loss_enc, self.encoder.trainable_variables),
                self.encoder.trainable_variables,
            )
        )

        # Update decoder
        opt.apply_gradients(
            zip(
                dec_tape.gradient(loss_dec, self.decoder.trainable_variables),
                self.decoder.trainable_variables,
            )
        )

        # Update discriminator
        opt.apply_gradients(
            zip(
                dis_tape.gradient(loss_dis, self.discriminator.trainable_variables),
                self.discriminator.trainable_variables,
            )
        )

        # Update metrics
        self.metric_loss_rec(loss_rec)
        self.metric_loss_kl(loss_kl)
        self.metric_loss_enc(loss_enc)
        self.metric_loss_dec(loss_dec)
        self.metric_loss_dis(loss_dis)
        self.metric_acc_real(tf.zeros_like(dis_real_logits), tf.math.round(tf.sigmoid(dis_real_logits)))
        self.metric_acc_fake(tf.ones_like(dis_fake_logits), tf.math.round(tf.sigmoid(dis_fake_logits)))
