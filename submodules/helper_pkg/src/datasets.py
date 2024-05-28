###############################
# Imports # Imports # Imports #
###############################

import numpy as np
import torch
from torch import float32
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pathlib
import os

from helper_pkg.src.helpers import get_config
import helper_pkg.src.models as models

#########################################
# Dataset Functions # Dataset Functions #
#########################################

def load_wrapped_datasets(new_labels=None):
    # Load config file and values
    config = get_config()

    if new_labels is not None:
        pass
    else:
        new_labels = config['new_labels']

    # Load raw dataset
    print(f"Using dataset {config['dataset']}")
    train_ds, test_ds = get_datasets(config['dataset'])

    if new_labels == 'none':
        # Original labels
        train_ds_wrapped, test_ds_wrapped = wrap_datasets_shuffle_class(train_ds, test_ds, device=config['device'])
    elif new_labels == 'shuffle_class':
        # Shuffle classes
        new_labels = np.random.choice(10, 10, replace=False)
        train_ds_wrapped, test_ds_wrapped = wrap_datasets_shuffle_class(train_ds, test_ds, new_labels=new_labels, device=config['device'])
    elif new_labels == 'shuffle_full':
        # Shuffle all labels
        train_ds_wrapped, test_ds_wrapped = wrap_datasets_shuffle_labels(train_ds, test_ds, new_labels=new_labels, device=config['device'])
    else:
        raise Exception(f"load_wrapped_datasets: new_labels has value \'{new_labels}\' which is unsupported")

    return train_ds_wrapped, test_ds_wrapped

def get_datasets(dataset, batch_size = None, shuffle = None):
    """
    Returns the training and testing datasets for either CIFAR10 or MNIST

    dataset: either 'CIFAR10' or 'MNIST'
    """
    # Make sure dataset is viable
    if dataset not in ["MNIST", "CIFAR10"]:
        raise Exception(f"Dataset {dataset} must be MNIST or CIFAR10")

    # Load config file and values
    config = get_config()
    if batch_size is None:
        batch_size = config[f"batch_size"]
    if shuffle is None:
        shuffle = config[f"shuffle"]

    # Load dataset
    # Download and load dataset
    base_dir = pathlib.Path().resolve() # get current file path
    dataset_dir = os.path.join(base_dir, "Datasets") # where to save dataset

    # Normalize data
    if config['dataset'] == 'CIFAR10' and config['scale'] == 1:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    if dataset == "CIFAR10":
        training_data = datasets.CIFAR10(
            root=dataset_dir,
            train=True,
            download=True,
            transform=transform
        )

        test_data = datasets.CIFAR10(
            root=dataset_dir,
            train=False,
            download=True,
            transform=transform
        )
    elif dataset == "MNIST":
        training_data = datasets.MNIST(
            root=dataset_dir,
            train=True,
            download=True,
            transform=transform
        )

        test_data = datasets.MNIST(
            root=dataset_dir,
            train=False,
            download=True,
            transform=transform
        )

    train_dl = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
    test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl

def wrap_datasets_shuffle_labels(train_dl, test_dl, new_labels=None, device=None):
    """
    Given datasets, wrap them to optimize them for training
    """
    # Load config file and values
    config = get_config()
    if device is None:
        device = config["device"]

    class WrappedDataLoader:
        def __init__(self, dl, func):
            self.dl = dl
            self.func = func

        def __len__(self):
            return len(self.dl)

        def __iter__(self):
            batches = iter(self.dl)
            for b in batches:
                # *b makes it so that the input to func is two variables: the images and the labels
                # yield is like return except it stops execution until the object is "grabbed"
                yield (self.func(*b)) 

    def preprocess(x, y):
        # Create dummy MLP
        mlp_d, mlp_d_loss_fn, mlp_d_optimizer = models.create_MLP(name="MLP_D")
        x = x.to(device).to(float32)
        with torch.no_grad():
            y = mlp_d(x)
        y = y.argmax(1)
        return x, y

    train_dl = WrappedDataLoader(train_dl, preprocess)
    test_dl = WrappedDataLoader(test_dl, preprocess)

    return train_dl, test_dl

def wrap_datasets_shuffle_class(train_dl, test_dl, new_labels=None, device=None):
    """
    Given datasets, wrap them to optimize them for training
    """
    # Load config file and values
    config = get_config()
    if device is None:
        device = config["device"]

    if new_labels is None:
        new_labels = np.arange(10)
    f = lambda x: new_labels[x]

    class WrappedDataLoader:
        def __init__(self, dl, func):
            self.dl = dl
            self.func = func

        def __len__(self):
            return len(self.dl)

        def __iter__(self):
            batches = iter(self.dl)
            for b in batches:
                # *b makes it so that the input to func is two variables: the images and the labels
                # yield is like return except it stops execution until the object is "grabbed"
                yield (self.func(*b)) 

    def preprocess(x, y):
        y = y.apply_(f) # Shuffle labels if necessary
        return x.to(device).to(float32), y.to(device)

    train_dl = WrappedDataLoader(train_dl, preprocess)
    test_dl = WrappedDataLoader(test_dl, preprocess)

    return train_dl, test_dl
