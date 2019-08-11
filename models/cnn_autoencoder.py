from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Abien Fred Agarap'

import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_layer_1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.activation_1 = tf.keras.layers.Activation('relu')
        self.max_pool_1 = tf.keras.layers.MaxPool2D((2, 2))
        self.conv_layer_2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.activation_2 = tf.keras.layers.Activation('relu')
        self.max_pool_2 = tf.keras.layers.MaxPool2D((2, 2))
        self.conv_layer_3 = tf.keras.layers.Conv2D(16, (3, 3), padding='same')
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()
        self.activation_3 = tf.keras.layers.Activation('relu')
        self.code = tf.keras.layers.MaxPool2D((2, 2))

    def call(self, features):
        activation = self.conv_layer_1(features)
        activation = self.batch_norm_1(activation)
        activation = self.activation_1(activation)
        activation = self.max_pool_1(activation)
        activation = self.conv_layer_2(activation)
        activation = self.batch_norm_2(activation)
        activation = self.activation_2(activation)
        activation = self.max_pool_2(activation)
        activation = self.conv_layer_3(activation)
        activation = self.batch_norm_3(activation)
        activation = self.activation_3(activation)
        code = self.code(activation)
