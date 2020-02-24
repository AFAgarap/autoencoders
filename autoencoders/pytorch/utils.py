"""Utility functions module"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


def create_dataloader(
    dataset, batch_size, shuffle, num_workers
) -> torch.utils.dataloader.DataLoader:
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return data_loader
