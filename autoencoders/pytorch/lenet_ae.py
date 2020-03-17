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
"""PyTorch implementation of LeNet-based autoencoder"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


class LeNetAE(torch.nn.Module):
    def __init__(self, **kwargs):
        super(LeNetAE, self).__init__()
        self.encoder_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=kwargs["input_dim"], out_channels=6, kernel_size=5, stride=(2, 2)),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=(2, 2))
        ])
        self.decoder_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=16, out_channels=6, kernel_size=5, stride=(2, 2)),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=(2, 2)),
            torch.nn.Conv2d(in_channels=16, out_channels=kwargs["input_dim"], kernel_size=4, stride=(2, 2))
        ])

    def forward(self, features):
        encoder_activations = {}
        for index, encoder_layer in enumerate(self.encoder_layers):
            if index == 0:
                encoder_activations[index] = torch.relu(encoder_layer(features))
            else:
                encoder_activations[index] = encoder_layer(encoder_activations[index - 1])
        code = torch.sigmoid(encoder_activations[len(encoder_activations) - 1])
        decoder_activations = {}
        for index, decoder_layer in enumerate(self.decoder_layers):
            if index == 0:
                decoder_activations[index] = torch.relu(decoder_layer(code))
            elif index == len(self.decoder_layers) - 1:
                decoder_activations[index] = torch.sigmoid(decoder_layer(decoder_activations[index - 1]))
            else:
                decoder_activations[index] = torch.relu(decoder_layer(decoder_activations[index - 1]))
        reconstruction = decoder_activations[len(decoder_activations) - 1]
        return reconstruction