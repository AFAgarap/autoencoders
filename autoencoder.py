"""Implementation of feed forward autoencoder in TensorFlow 2.0"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Richard Ricardo'

import tensorflow as tf
from tensorflow.keras.layers import Dense
from keras.models import Model

class Autoencoder(tf.keras.Model):
    def __init__(self, intermediate_dim, code_dim, original_dim, num_hidden_layers, encoder_decrements, decoder_increments):
        super(Autoencoder, self).__init__()
        self.loss = []
        self.encoder_decrements = encoder_decrements
        self.decoder_increments = decoder_increments
        self.intermediate_dim = intermediate_dim
        self.code_dim = code_dim
        self.original_dim = original_dim
        self.num_hidden_layers = num_hidden_layers
        self.encoder_hidden_layers = []
        self.decoder_hidden_layers = []
        """Encoder"""
        for layer in range(self.num_hidden_layers):
          self.encoder_hidden_layers.append(Dense(units=self.intermediate_dim, activation=tf.nn.relu))
          self.intermediate_dim = self.intermediate_dim - self.encoder_decrements
        self.encoder_output_layer = Dense(units=code_dim, activation=tf.nn.relu)
        """Decoder"""
        for layer in range(self.num_hidden_layers):
          self.decoder_hidden_layers.append(Dense(units=self.code_dim, activation=tf.nn.relu))
          self.code_dim = self.code_dim - self.decoder_increments
        self.decoder_output_layer = Dense(units=original_dim, activation=tf.nn.relu)
        
    def call(self, input_features):
      """Encoder"""
      activation = self.encoder_hidden_layers[0](input_features)
      if self.num_hidden_layers > 1:
        for layer in range(self.num_hidden_layers - 1):
          layer = layer + 1
          activation = self.encoder_hidden_layers[layer](activation)
      activation = self.encoder_output_layer(activation)           
      """Decoder"""
      activation = self.decoder_hidden_layers[0](activation)
      if self.num_hidden_layers > 1:
        for layer in range(self.num_hidden_layers - 1):
          layer = layer + 1
          activation = self.decoder_hidden_layers[layer](activation)
      return self.decoder_output_layer(activation)

def loss(reconstructed, original):
    return tf.reduce_mean(tf.square(tf.subtract(reconstructed, original)))

def train(loss, model, opt, original):
    with tf.GradientTape() as tape:
        reconstructed = model(original)
        reconstruction_error = loss(reconstructed, original)
    gradients = tape.gradient(reconstruction_error, model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)
    return reconstruction_error

def train_loop(model, loss, dataset, epochs):
  opt = tf.optimizers.Adam(learning_rate=1e-3)
  for epoch in range(epochs):
      epoch_loss = 0
      for batch_features in dataset:
          loss_values = train(loss, model, opt, batch_features)
          epoch_loss += loss_values
      model.loss.append(tf.reduce_mean(epoch_loss))
      print('Epoch {}/{}. Loss: {}'.format(epoch + 1, epochs, tf.reduce_mean(epoch_loss)))