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
        self.input_layer = tf.keras.layers.InputLayer(input_shape=kwargs['input_shape'])
        self.conv_layer_1 = tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation=tf.nn.relu)
        self.pool_layer_1 = tf.keras.layers.MaxPooling2D(padding='same')
        self.conv_layer_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation=tf.nn.relu)
        self.pool_layer_2 = tf.keras.layers.MaxPooling2D(padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.hidden_layer_1 = tf.keras.layers.Dense(units=120, activation=tf.nn.relu)
        self.hidden_layer_2 = tf.keras.layers.Dense(units=84, activation=tf.nn.relu)
        self.z_mean_layer = tf.keras.layers.Dense(units=kwargs['latent_dim'])
        self.z_log_var_layer = tf.keras.layers.Dense(units=kwargs['latent_dim'])
        self.sampling = Sampling()

    def call(self, features):
        features = self.input_layer(features)
        activation = self.conv_layer_1(features)
        activation = self.pool_layer_1(activation)
        activation = self.conv_layer_2(activation)
        activation = self.pool_layer_2(activation)
        activation = self.flatten(activation)
        activation = self.hidden_layer_1(activation)
        activation = self.hidden_layer_2(activation)
        z_mean = self.z_mean_layer(activation)
        z_log_var = self.z_log_var_layer(activation)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z
