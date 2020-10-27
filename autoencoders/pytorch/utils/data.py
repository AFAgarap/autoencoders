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
"""Utility functions module"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import torch
import torchvision


def load_dataset(name: str = "mnist") -> Tuple[object, object]:
    """
    Returns a tuple of torchvision dataset objects.

    Parameter
    ---------
    name : str
        The name of the dataset to load. Current choices:
            1. mnist (MNIST)
            2. fashion_mnist (FashionMNIST)
            3. emnist (EMNIST/Balanced)

    Returns
    -------
    Tuple[object, object]
        A tuple consisting of the training dataset and the test dataset.
    """
    if name == "mnist":
        train_dataset = torchvision.datasets.MNIST(
            root="~/torch_datasets", train=True, download=True
        )
        test_dataset = torchvision.datasets.MNIST(
            root="~/torch_datasets", train=False, download=True
        )
    elif name == "fashion_mnist":
        train_dataset = torchvision.datasets.FashionMNIST(
            root="~/torch_datasets", train=True, download=True
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root="~/torch_datasets", train=False, download=True
        )
    elif name == "emnist":
        train_dataset = torchvision.datasets.EMNIST(
            root="~/torch_datasets", train=True, split="balanced", download=True
        )
        test_dataset = torchvision.datasets.EMNIST(
            root="~/torch_datasets", train=False, split="balanced", download=True
        )
    return train_dataset, test_dataset


def create_dataloader(
    dataset: object, batch_size: int = 16, shuffle: bool = True, num_workers: int =
    0
) -> torch.utils.dataloader.DataLoader:
    """
    Returns a data loader object, ready to be used by a model.

    Parameters
    ----------
    dataset : object
        The dataset from `torchvision.datasets`.
    batch_size : int
        The mini-batch size for the data loading. Default is [16].
    shuffle : bool
        Whether to shuffle dataset or not. Default is [True].
    num_workers : int
        The number of subprocesses to use for data loading. Default is [0].

    Returns
    -------
    data_loader : torch.utils.dataloader.DataLoader
        The data loader object to be used by a model.
    """
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return data_loader
