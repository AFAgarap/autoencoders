from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Abien Fred Agarap'

import tensorflow as tf


class CVAE(tf.keras.Model):
    def __init__(self, **kwargs):
        pass

    @tf.function
    def call(self, features):
        pass

class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(
                input_shape=kwargs['input_shape']
                )
        self.conv_layer_1 = tf.keras.layers.Conv2D(
                filters=6,
                kernel_size=5,
                strides=(2, 2),
                activation=tf.nn.relu
                )
        self.conv_layer_2 = tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=5,
                strides=(2, 2),
                activation=tf.nn.relu
                )
        self.flatten = tf.keras.layers.Flatten()
        self.z_mean_layer = tf.keras.layers.Dense(
                units=kwargs['latent_dim']
                )
        self.z_log_var_layer = tf.keras.layers.Dense(
                units=kwargs['latent_dim']
                )
        self.sampling = Sampling()

    def call(self, features):
        features = self.input_layer(features)
        activation = self.conv_layer_1(features)
        activation = self.conv_layer_2(activation)
        activation = self.flatten(activation)
        z_mean = self.z_mean_layer(activation)
        z_log_var = self.z_log_var_layer(activation)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(
                input_shape=(kwargs['latent_dim'], )
                )
        self.hidden_layer_1 = tf.keras.layers.Dense(
                units=(7 * 7 * 6),
                activation=tf.nn.relu
                )
        self.reshape_layer = tf.keras.layers.Reshape(target_shape=(7, 7, 6))
        self.convt_layer_1 = tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=5,
                strides=(2, 2),
                padding='same',
                activation=tf.nn.relu
                )
        self.convt_layer_2 = tf.keras.layers.Conv2DTranspose(
                filters=6,
                kernel_size=5,
                strides=(2, 2),
                padding='same',
                activation=tf.nn.relu
                )
        self.output_layer = tf.keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=5,
                strides=(1, 1),
                padding='same',
                activation=tf.nn.sigmoid
                )

    def call(self, features):
        features = self.input_layer(features)
        activation = self.hidden_layer_1(features)
        activation = self.reshape_layer(activation)
        activation = self.convt_layer_1(activation)
        activation = self.convt_layer_2(activation)
        output = self.output_layer(activation)
        return output



