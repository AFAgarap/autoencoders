# A Tutorial on Autoencoders
# Copyright (C) 2019  Abien Fred Agarap and Richard Ralph Ricardo
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
"""TensorFlow implementation of a feed-forward neural network"""
import tensorflow as tf

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


class NeuralNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.hidden_layers = [
            tf.keras.layers.Dense(
                units=unit, activation=tf.nn.relu, kernel_initializer="he_normal"
            )
            for unit in kwargs["units"]
        ]
        self.output_layer = tf.keras.layers.Dense(units=kwargs["num_classes"])

    def call(self, features):
        activations = {}
        for index, layer in enumerate(self.hidden_layers):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations[index - 1])
        logits = self.output_layer(activations[len(activations) - 1])
        return logits
