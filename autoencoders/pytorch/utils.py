"""Utility functions module"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import torch
import torchvision

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


def load_dataset() -> Tuple[object, object]:
    train_dataset = torchvision.datasets.MNIST(
            root="~/torch_datasets", train=True, download=True
            )
    test_dataset = torchvision.datasets.MNIST(
            root="~/torch_datasets", train=False, download=True
            )
    return (train_dataset, test_dataset)


def create_dataloader(
    dataset, batch_size, shuffle, num_workers
) -> torch.utils.dataloader.DataLoader:
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return data_loader
