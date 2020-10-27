# A Tutorial on Autoencoders
# Copyright (C) 2020  Abien Fred Agarap and Richard Ralph Ricardo
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Also add information on how to contact you by electronic and paper mail.
#
# If your software can interact with users remotely through a computer
# network, you should also make sure that it provides a way for users to
# get its source.  For example, if your program is a web application, its
# interface could display a "Source" link that leads users to an archive
# of the code.  There are many ways you could offer source, and different
# solutions will be better for different programs; see section 13 for the
# specific requirements.
#
# You should also get your employer (if you work as a programmer) or school,
# if any, to sign a "copyright disclaimer" for the program, if necessary.
# For more information on this, and how to apply and follow the GNU AGPL, see
# <http://www.gnu.org/licenses/>.
"""TensorFlow 2.0 implementation of variational autoencoder"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "1.0.0"
__author__ = "Abien Fred Agarap"

import tensorflow as tf


class VAE(tf.keras.Model):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()
        self.encoder = Encoder(
            intermediate_dim=kwargs["intermediate_dim"], latent_dim=kwargs["latent_dim"]
        )
        self.decoder = Decoder(
            intermediate_dim=kwargs["intermediate_dim"],
            original_dim=kwargs["original_dim"],
        )

    @tf.function
    def call(self, features):
        z_mean, z_log_var, latent_code = self.encoder(features)
        reconstructed = self.decoder(latent_code)
        kl_divergence = -5e-2 * tf.reduce_sum(
            tf.exp(z_log_var) + tf.square(z_mean) - 1 - z_log_var
        )
        self.add_loss(kl_divergence)
        return reconstructed


class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            units=kwargs["intermediate_dim"], activation=tf.nn.relu
        )
        self.z_mean_layer = tf.keras.layers.Dense(units=kwargs["latent_dim"])
        self.z_log_var_layer = tf.keras.layers.Dense(units=kwargs["latent_dim"])
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
            units=kwargs["intermediate_dim"], activation=tf.nn.relu
        )
        self.output_layer = tf.keras.layers.Dense(
            units=kwargs["original_dim"], activation=tf.nn.sigmoid
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
        epsilon = tf.random.normal(shape=(batch, dimension), mean=0.0, stddev=1.0)
        return z_mean + epsilon * tf.exp(0.5 * z_log_var)
