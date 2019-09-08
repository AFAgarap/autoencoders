"""Implementation of feed forward autoencoder in TensorFlow 2.0"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Richard Ricardo'

import numpy as np
import tensorflow as tf
import datetime
from tensorflow.keras.layers import Dense


class Autoencoder(tf.keras.Model):

    def __init__(
        self,
        intermediate_dim,
        code_dim,
        original_dim,
        num_hidden_layers,
        encoder_decrements,
        decoder_increments,
        learning_rate=1e-3,
        training=True,
        ):

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
        self.learning_rate = learning_rate
        if training is False:
            self.learning_rate = 1e-3
        self.optimizer = \
            tf.optimizers.Adam(learning_rate=self.learning_rate)

        for layer in range(self.num_hidden_layers):
            self.encoder_hidden_layers.append(Dense(units=self.intermediate_dim,
                    activation=tf.nn.relu))
            self.intermediate_dim = self.intermediate_dim \
                - self.encoder_decrements
        self.encoder_output_layer = Dense(units=code_dim,
                activation=tf.nn.relu)
        for layer in range(self.num_hidden_layers):
            self.decoder_hidden_layers.append(Dense(units=self.code_dim,
                    activation=tf.nn.relu))
            self.code_dim = self.code_dim - self.decoder_increments
        self.decoder_output_layer = Dense(units=original_dim,
                activation=tf.nn.relu)

    def call(self, input_features):
        """Encoder"""

        activation = self.encoder_hidden_layers[0](input_features)
        if self.num_hidden_layers > 1:
            for layer in range(self.num_hidden_layers - 1):
                layer = layer + 1
                activation = \
                    self.encoder_hidden_layers[layer](activation)
        activation = self.encoder_output_layer(activation)
        activation = self.decoder_hidden_layers[0](activation)
        if self.num_hidden_layers > 1:
            for layer in range(self.num_hidden_layers - 1):
                layer = layer + 1
                activation = \
                    self.decoder_hidden_layers[layer](activation)
        return self.decoder_output_layer(activation)

    def opt(self):
        return self.optimizer


def loss(reconstructed, original):
    return tf.reduce_mean(tf.square(tf.subtract(reconstructed,
                          original)))


def train(
    loss,
    model,
    opt,
    original,
    ):

    with tf.GradientTape() as tape:
        reconstructed = model(original)
        reconstruction_error = loss(reconstructed, original)
    gradients = tape.gradient(reconstruction_error,
                              model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)

    train_loss(reconstruction_error)

    return reconstruction_error


def train_loop(
    model,
    loss,
    dataset,
    epochs,
    ):

    opt = model.opt()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_features in dataset:
            loss_values = train(loss, model, opt, batch_features)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(),
                                  step=epoch)
            epoch_loss += loss_values
        model.loss.append(tf.reduce_mean(epoch_loss))
        print('Epoch {}/{}. Loss: {}'.format(epoch + 1, epochs,
              tf.reduce_mean(epoch_loss)))
        train_loss.reset_states()


def log_images(test_features, model):

  # Sets up a timestamped log directory.
    logdir = 'logs/train_data' \
        + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

  # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)

    originals = []
    reconstructed = []

    number = 10  # Number of digits to display

    for index in range(number):

		# Save original images
        originals.append(test_features[index].reshape(28, 28))

		# Save reconstruction images
        reconstructed.append(model(test_features)[index].numpy().reshape(28,
                             28))

    originals = np.array(originals)
    reconstructed = np.array(reconstructed)

    with file_writer.as_default():
        images = np.reshape(originals, (-1, 28, 28, 1))
        tf.summary.image('Original', images, max_outputs=10, step=0)

    with file_writer.as_default():
        images = np.reshape(reconstructed, (-1, 28, 28, 1))
        tf.summary.image('Reconstructed', images, max_outputs=10,
                         step=0)


train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/train_data/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
