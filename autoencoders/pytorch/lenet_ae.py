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
            torch.nn.Conv2d(in_channels=16, out_channels=kwargs["input_dim"], kernel_size=5, stride=(2, 2))
        ])

    def forward(self, features):
        encoder_activations = {}
        for index, encoder_layer in enumerate(self.encoder_layers):
            if index == 0:
                encoder_activations[index] = encoder_layer(features)
            else:
                encoder_activations[index] = encoder_layer(encoder_activations[index - 1])
        code = encoder_activations[len(encoder_activations) - 1]
        decoder_activations = {}
        for index, decoder_layer in enumerate(self.decoder_layers):
            if index == 0:
                decoder_activations[index] = decoder_layer(code)
            else:
                decoder_activations[index] = decoder_layer(decoder_activations[index - 1])
        reconstruction = decoder_activations[len(decoder_activations) - 1]
        return reconstruction

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=(2, 2)
        )
        self.conv_layer_2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=(2, 2)
        )

    def forward(self, features):
        activation = self.conv_layer_1(features)
        activation = F.relu(activation)
        activation = self.conv_layer_2(activation)
        code = F.relu(activation)
        return code


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convt_layer_1 = nn.ConvTranspose2d(
            in_channels=16, out_channels=6, kernel_size=5, stride=(2, 2)
        )
        self.convt_layer_2 = nn.ConvTranspose2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=(2, 2)
        )
        self.convt_layer_3 = nn.ConvTranspose2d(
            in_channels=16, out_channels=1, kernel_size=4, stride=(1, 1)
        )

    def forward(self, features):
        activation = self.convt_layer_1(features)
        activation = F.relu(activation)
        activation = self.convt_layer_2(activation)
        activation = F.relu(activation)
        activation = self.convt_layer_3(activation)
        reconstructed = F.relu(activation)
        return reconstructed


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, features):
        code = self.encoder(features)
        reconstructed = self.decoder(code)
        return reconstructed


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=4
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss().to(device)

print(model)
print([parameter.size() for parameter in model.parameters()])

epochs = 20

for epoch in range(epochs):
    loss = 0
    for batch_features, _ in train_loader:
        batch_features = batch_features.to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = criterion(outputs, batch_features)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()
    loss = loss / len(train_loader)
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

with torch.no_grad():
    number = 10
    plt.figure(figsize=(20, 4))
    for index in range(number):
        # display original
        ax = plt.subplot(2, number, index + 1)
        plt.imshow(test_dataset.data[index].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, number, index + 1 + number)
        test_data = test_dataset.data[index]
        test_data = test_data.to(device)
        test_data = test_data.float()
        test_data = test_data.view(-1, 1, 28, 28)
        output = model(test_data)
        plt.imshow(output.cpu().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
