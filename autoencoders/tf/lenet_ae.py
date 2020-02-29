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
"""TensorFlow 2.0 implementation of LeNet-based autoencoder"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "1.0.0"
__author__ = "Abien Fred Agarap"

import tensorflow as tf


class CAE(tf.keras.Model):
    def __init__(self, **kwargs):
        super(CAE, self).__init__()
        self.encoder = Encoder(
            input_shape=kwargs["input_shape"], latent_dim=kwargs["latent_dim"]
        )
        self.decoder = Decoder(latent_dim=kwargs["latent_dim"])

    def call(self, features):
        code = self.encoder(features)
        reconstructed = self.decoder(code)
        return reconstructed


class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=kwargs["input_shape"])
        self.conv_1_layer_1 = tf.keras.layers.Conv2D(
            filters=6, kernel_size=(5, 5), strides=(2, 2), activation=tf.nn.relu
        )
        self.conv_1_layer_2 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=(5, 5), strides=(2, 2), activation=tf.nn.sigmoid
        )

    def call(self, features):
        activation = self.conv_1_layer_1(features)
        code = self.conv_1_layer_2(activation)
        return code


class Decoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()
        self.convt_1_layer_1 = tf.keras.layers.Conv2DTranspose(
            filters=16, kernel_size=(5, 5), strides=(2, 2), activation=tf.nn.relu
        )
        self.convt_1_layer_2 = tf.keras.layers.Conv2DTranspose(
            filters=5, kernel_size=(5, 5), activation=tf.nn.relu
        )
        self.convt_2_layer_1 = tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=(5, 5), strides=(1, 1), activation=tf.nn.sigmoid
        )

    def call(self, features):
        activation = self.convt_1_layer_1(features)
        activation = self.convt_1_layer_2(activation)
        output = self.convt_2_layer_1(activation)
        return output
