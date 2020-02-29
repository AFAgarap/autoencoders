"""Implementation of vanila autoencoder in TensorFlow 2.0 Subclassing API"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "1.0.0"
__author__ = "Abien Fred Agarap"

from functools import partial
import tensorflow as tf


dense = partial(
    tf.keras.layers.Dense, activation=tf.nn.relu, kernel_initializer="he_normal"
)


class Encoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        self.encoder_layer_1 = dense(units=500)
        self.encoder_layer_2 = dense(units=500)
        self.encoder_layer_3 = dense(units=2000)
        self.code_layer = tf.keras.layers.Dense(
            units=kwargs["code_dim"], activation=tf.nn.sigmoid
        )

    def call(self, features):
        activation = self.encoder_layer_1(features)
        activation = self.encoder_layer_2(activation)
        activation = self.encoder_layer_3(activation)
        code = self.code_layer(activation)
        return code


class Decoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()
        self.decoder_layer_1 = dense(units=2000)
        self.decoder_layer_2 = dense(units=500)
        self.decoder_layer_3 = dense(units=500)
        self.reconstruction_layer = tf.keras.layers.Dense(
            units=kwargs["input_shape"], activation=tf.nn.sigmoid
        )

    def call(self, code):
        activation = self.decoder_layer_1(code)
        activation = self.decoder_layer_2(activation)
        activation = self.decoder_layer_3(activation)
        reconstruction = self.reconstruction_layer(activation)
        return reconstruction


class Autoencoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Autoencoder, self).__init__()
        self.loss = []
        self.encoder = Encoder(code_dim=kwargs["code_dim"])
        self.decoder = Decoder(input_shape=kwargs["input_shape"])

    def call(self, features):
        code = self.encoder(features)
        reconstructed = self.decoder(code)
        return reconstructed
