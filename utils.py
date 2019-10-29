"""Utility script for autoencoder experiments in TensorFlow 2.0"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Richard Ricardo'

import tensorflow as tf


def loss(reconstructed, original):
    return tf.reduce_mean(tf.square(tf.subtract(reconstructed, original)))

def train_step(loss, model, opt, original):
    with tf.GradientTape() as tape:
        reconstructed = model(original)
        reconstruction_error = loss(reconstructed, original)
    gradients = tape.gradient(reconstruction_error, model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)

    return reconstruction_error

def train(model, opt, loss, dataset, epochs):
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_features in dataset:
            loss_values = train_step(loss, model, opt, batch_features)
            epoch_loss += loss_values
        model.loss.append(tf.reduce_mean(epoch_loss))
        print('Epoch {}/{}. Loss: {}'.format(epoch + 1, epochs, tf.reduce_mean(epoch_loss)))