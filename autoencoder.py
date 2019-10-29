"""Implementation of feed forward autoencoder in TensorFlow 2.0"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Richard Ricardo'

import tensorflow as tf
from tensorflow.keras.layers import Dense


class Autoencoder(tf.keras.Model):
    def __init__(self, neurons, original_dim):
        super(Autoencoder, self).__init__()
        self.loss = []
        self.neurons = neurons
        self.encoder_hidden_layers = []
        self.decoder_hidden_layers = []
        self.original_dim = original_dim

        """Encoder"""
        for neuron in self.neurons:
          self.encoder_hidden_layers.append(Dense(units=neuron, activation=tf.nn.relu))
        """Decoder"""
        for neuron in self.neurons[::-1]:
          self.decoder_hidden_layers.append(Dense(units=neuron, activation=tf.nn.relu))
        self.decoder_output_layer = Dense(units=original_dim, activation=tf.nn.relu)

    def call(self, input_features):
      """Encoder"""
      activation = self.encoder_hidden_layers[0](input_features)
      if len(self.neurons) > 1:
        for layer in range(len(self.neurons) - 1):
          layer = layer + 1
          activation = self.encoder_hidden_layers[layer](activation)
      """Decoder"""
      activation = self.decoder_hidden_layers[0](activation)
      if len(self.neurons) > 1:
        for layer in range(len(self.neurons) - 1):
          layer = layer + 1
          activation = self.decoder_hidden_layers[layer](activation)
      return self.decoder_output_layer(activation)