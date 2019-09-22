"""Implementation of convolutional autoencoder in TensorFlow 2.0"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Richard Ricardo'

import tensorflow as tf


class LeNet(tf.keras.Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.loss = []
		
        """Encoder"""
        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=6,
            kernel_size=(5, 5),
            strides=(2, 2),
            activation=tf.nn.relu,
            input_shape=(28, 28, 1))
        self.max_pool_1 = tf.keras.layers.MaxPool2D((2, 2), padding='same')
        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(5, 5),
            strides=(2, 2),
            activation=tf.nn.relu)
        self.max_pool_2 = tf.keras.layers.MaxPool2D((2, 2), padding='same')
        """Decoder"""
        self.convtrans_layer_1 = tf.keras.layers.Conv2DTranspose(
            filters=6,
            kernel_size=(5, 5),
            strides=(2, 2),
            activation=tf.nn.relu)
        self.convtrans_layer_2 = tf.keras.layers.Conv2DTranspose(
            filters=16,
            kernel_size=(5, 5),
            strides=(2, 2),
            activation=tf.nn.relu)
        self.convtrans_layer_3 = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation=tf.nn.relu)

    def call(self, input_features):
      activation = self.conv_layer_1(input_features)
      activation = self.max_pool_1(activation)
      activation = self.conv_layer_2(activation)
      activation = self.max_pool_2(activation)
      activation = self.convtrans_layer_1(activation)
      activation = self.convtrans_layer_2(activation)
      activation = self.convtrans_layer_3(activation)
      return activation