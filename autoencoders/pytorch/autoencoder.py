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
"""PyTorch implementation of a vanilla autoencoder"""
import torch


class Autoencoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_layers = torch.nn.ModuleList([
            torch.nn.Linear(in_features=kwargs["input_shape"], out_features=500),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=500, out_features=500),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=500, out_features=2000),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2000, out_features=kwargs["code_dim"]),
            torch.nn.Sigmoid()
        ])
        self.decoder_layers = torch.nn.ModuleList([
            torch.nn.Linear(in_features=kwargs["code_dim"], out_features=2000),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2000, out_features=500),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=500, out_features=500),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=500, out_features=kwargs["input_shape"]),
            torch.nn.Sigmoid()
        ])

    def forward(self, features):
        activations = {}
        for index, encoder_layer in enumerate(self.encoder_layers):
            if index == 0:
                activations[index] = encoder_layer(features)
            else:
                activations[index] = encoder_layer(activations[index - 1])
        code = activations[len(activations) - 1]
        activations = {}
        for index, decoder_layer in enumerate(self.decoder_layers):
            if index == 0:
                activations[index] = decoder_layer(code)
            else:
                activations[index] = decoder_layer(activations[index - 1])
        reconstruction = activations[len(activations) - 1]
        return reconstruction


def train_step(model, optimizer, loss, batch_features):
    optimizer.zero_grad()
    outputs = model(batch_features)
    train_loss = loss(outputs, batch_features)
    train_loss.backward()
    optimizer.step()
    return train_loss


def train(model, optimizer, loss, train_loader, epochs, device):
    train_loss = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_features, _ in train_loader:
            batch_features = batch_features.view(-1, 784).to(device)
            step_loss = train_step(model, optimizer, loss, batch_features)
            epoch_loss += step_loss.item()
        epoch_loss = epoch_loss / len(train_loader)
        train_loss.append(epoch_loss)
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, epoch_loss))