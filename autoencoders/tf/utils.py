from typing import Tuple
import numpy as np
import tensorflow_datasets as tfds


def load_tfds(name: str = "mnist") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns a data set from `tfds`.

    For this experiment, the options are:
        1. mnist
        2. fashion_mnist
        3. emnist/letters
        4. cifar10
        5. svhn_cropped

    Parameters
    ----------
    name : str
        The name of the TensorFlow data set to load.

    Returns
    -------
    train_features : np.ndarray
        The train features.
    test_features : np.ndarray
        The test features.
    train_labels : np.ndarray
        The train labels.
    test_labels : np.ndarray
        The test labels.
    """
    train_dataset = tfds.load(name=name, split=tfds.Split.TRAIN, batch_size=-1)
    train_dataset = tfds.as_numpy(train_dataset)

    train_features = train_dataset["image"]
    train_labels = train_dataset["label"]

    train_features = train_features.astype("float32")
    train_features = train_features.reshape(-1, np.prod(train_features.shape[1:]))
    train_features = train_features / 255.0

    test_dataset = tfds.load(name=name, split=tfds.Split.TEST, batch_size=-1)
    test_dataset = tfds.as_numpy(test_dataset)

    test_features = test_dataset["image"]
    test_labels = test_dataset["label"]

    test_features = test_features.astype("float32")
    test_features = test_features.reshape(-1, np.prod(test_features.shape[1:]))
    test_features = test_features / 255.0

    return train_features, test_features, train_labels, test_labels
