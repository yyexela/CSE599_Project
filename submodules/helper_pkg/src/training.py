###############################
# Imports # Imports # Imports #
###############################

import torch
import os
import numpy as np
from helper_pkg.src.helpers import get_config
import helper_pkg.src.models as models
import helper_pkg.src.datasets as datasets

#####################################
# Generic Helpers # Generic Helpers #
#####################################

def get_optimizer(optimizer_str, parameters, lr, betas=(0.9,0.999),\
                  epsilon=1e-8, weight_decay=0):
    if optimizer_str == 'AdamW':
        optimizer = torch.optim.AdamW(parameters, lr=lr, betas=betas)
    elif optimizer_str == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=betas, eps=epsilon, weight_decay=weight_decay)
    elif optimizer_str == 'RMSprop':
        optimizer = torch.optim.RMSprop(parameters, lr=lr)
    elif optimizer_str == 'Adagrad':
        optimizer = torch.optim.Adagrad(parameters, lr=lr)
    else:
        raise Exception(f"Optimizer \"{optimizer_str}\" not valid, use one of \"Adam\", \"AdamW\", \"RMSprop\" or \"Adagrad\"")
    return optimizer

def get_scheduler(optimizer, scheduler_str):
    if scheduler_str == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9, last_epoch=-1)
    elif scheduler_str == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=0, last_epoch=-1)
    else:
        raise Exception(f"Scheduler \"{scheduler_str}\" not valid, use one of \"StepLR\" or \"CosineAnnealingLR\"")
    return scheduler

###################################
# TRAINING LOOPS # TRAINING LOOPS #
###################################

def train_multiple_epochs(model, loss_fn, optimizer, train_ds_wrapped, test_ds_wrapped, model_name="model"):
    # Load config file and values
    config = get_config()

    print(f"Training {model.name} on {config['dataset']}")

    # Train MLP
    train_acc = list()
    test_acc = list()
    train_loss = list()
    test_loss = list()
    alphas = list()

    for epoch in range(config['epochs']):
        alpha = min(2*(epoch+1)/config['epochs'],1.)
        print(f"Epoch {epoch+1} Alpha {alpha}\n---------------------------------")

        acc, loss = train_epoch(train_ds_wrapped, model, loss_fn, optimizer, alpha, batch_size=config['batch_size'])
        train_loss.append(loss.cpu().detach().item())
        train_acc.append(acc)

        print()
        
        acc, loss = test_loop(test_ds_wrapped, model, loss_fn, alpha)
        test_acc.append(acc)
        test_loss.append(loss)
        alphas.append(alpha)

        print()

    print("Done!\n")

    return train_acc, test_acc, train_loss, test_loss, alphas

def train_epoch(dataloader, model, loss_fn, optimizer, alpha=1., batch_size=None):
    # Load config file and values
    config = get_config()
    if batch_size is None:
        batch_size = config["batch_size"]

    num_batches = len(dataloader) # Total number of batches
    size = 0 # Calculate size of dataset
    train_loss_sum = 0 # Running training loss
    correct = 0 # Running number of correctly identified images

    # Iterate over entire dataset
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        if model.name == "MLP_E" or model.name == "CNN_E":
            beta = 1.-alpha
            pred = alpha*pred[0] + beta*pred[1]
        y = torch.nn.functional.one_hot(y, num_classes=10)
        y = y.to(torch.float32)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute statistics
        train_loss_sum += loss
        correct += int((pred.argmax(1) == torch.argmax(y, dim=1)).type(torch.float).sum().item())
        size += X.shape[0]

        # Prints
        loss = loss.item()
        print(f"loss: {loss:>7f} [batch {batch+1} of {num_batches}]")

    # Print results
    train_loss_avg = train_loss_sum / num_batches
    correct_pct = 100. * correct / size
    print()
    print(f"Train error:")
    print(f"  Correct:  {correct} of {size}")
    print(f"  Accuracy: {(correct_pct):>0.2f}%")
    print(f"  Avg loss: {train_loss_avg:>0.6f}")
    return (correct_pct, train_loss_avg)

def test_loop(dataloader, model, loss_fn, alpha=1.):
    size = 0 # Calculate size of dataset
    num_batches = len(dataloader) # Total number of batches
    test_loss_sum = 0 # Running testing loss
    correct = 0 # Running number of correctly identified images

    # Iterate over dataset and infer results using the model
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # Evaluate model
            pred = model(X)
            if model.name == "MLP_E" or model.name == "CNN_E":
                beta = 1.-alpha
                pred = alpha*pred[0] + beta*pred[1]
            y = torch.nn.functional.one_hot(y, num_classes=10)
            y = y.to(torch.float32)

            # Compute statistics
            test_loss_sum += loss_fn(pred, y).item()
            correct += int((pred.argmax(1) == torch.argmax(y, dim=1)).type(torch.float).sum().item())
            size += X.shape[0]

    # Print results
    test_loss_avg = test_loss_sum / num_batches
    correct_pct = 100.* correct / size
    print(f"Test error:")
    print(f"  Correct:  {correct} of {size}")
    print(f"  Accuracy: {(correct_pct):>0.2f}%")
    print(f"  Avg loss: {test_loss_avg:>0.6f}")
    return (correct_pct, test_loss_avg)
