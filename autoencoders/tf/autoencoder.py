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
"""TensorFlow 2.0 implementation of vanilla autoencoder"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import tensorflow as tf

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


dense = partial(
    tf.keras.layers.Dense, activation=tf.nn.relu, kernel_initializer="he_normal"
)


class Encoder(tf.keras.Model):
    """
    The encoder model of a fully connected autoencoder model.
    This architecture is based on Salakhutdinov & Hinton (2007)
    [http://proceedings.mlr.press/v2/salakhutdinov07a.html]
    """

    def __init__(self, **kwargs):
        """
        Constructs the encoder model.

        Parameters
        ----------
        kwargs
        code_dim: int
            The dimensionality of the latent code representation.

        """
        super(Encoder, self).__init__()
        self.encoder_layer_1 = dense(units=500)
        self.encoder_layer_2 = dense(units=500)
        self.encoder_layer_3 = dense(units=2000)
        self.code_layer = tf.keras.layers.Dense(
            units=kwargs["code_dim"], activation=tf.nn.sigmoid
        )

    def call(self, features):
        """
        The forward pass for the encoder model.

        Parameters
        ----------
        features: array or array-like object
            The input features to encode.

        Returns
        -------
        code: array or array-like object
            The latent code representation of the features.
        """
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
