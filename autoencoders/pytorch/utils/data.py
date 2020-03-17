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
