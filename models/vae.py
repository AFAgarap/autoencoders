"""TensorFlow 2.0 implementation of variational autoencoder"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Abien Fred Agarap'

import tensorflow as tf


class VAE(tf.keras.Model):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()
        self.encoder = Encoder(
                intermediate_dim=kwargs['intermediate_dim'],
                latent_dim=kwargs['latent_dim']
                )
        self.decoder = Decoder(
                intermediate_dim=kwargs['intermediate_dim'],
                original_dim=kwargs['original_dim']
                )

    @tf.function
    def call(self, features):
        z_mean, z_log_var, latent_code = self.encoder(features)
        reconstructed = self.decoder(latent_code)
        kl_divergence = -5e-2 * tf.reduce_sum(tf.exp(z_log_var) + tf.square(z_mean) - 1 - z_log_var)
        self.add_loss(kl_divergence)
        return reconstructed


class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
                units=kwargs['intermediate_dim'],
                activation=tf.nn.relu
                )
        self.z_mean_layer = tf.keras.layers.Dense(
                units=kwargs['latent_dim']
                )
        self.z_log_var_layer = tf.keras.layers.Dense(
                units=kwargs['latent_dim']
                )
        self.sampling = Sampling()

    def call(self, features):
        activation = self.hidden_layer(features)
        z_mean = self.z_mean_layer(activation)
        z_log_var = self.z_log_var_layer(activation)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
                units=kwargs['intermediate_dim'],
                activation=tf.nn.relu
                )
        self.output_layer = tf.keras.layers.Dense(
                units=kwargs['original_dim'],
                activation=tf.nn.sigmoid
                )

    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        output = self.output_layer(activation)
        return output


class Sampling(tf.keras.layers.Layer):
    def call(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dimension = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(
                shape=(batch, dimension),
                mean=0.,
                stddev=1.
                )
        return z_mean + epsilon * tf.exp(0.5 * z_log_var)
