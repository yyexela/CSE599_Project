# Imports
import argparse
import torch
from torch import nn
import numpy as np
import pathlib
import os
import sys

# Add custom package to import path and import it
file_dir = pathlib.Path().resolve()
pkg_dir = os.path.join(file_dir, "submodules")
sys.path.insert(0, pkg_dir)
sys.path.insert(0, os.path.join(pkg_dir, "helper_pkg"))
from helper_pkg.src import *

def main(args=None):

    # Load config file
    config = helpers.get_config()

    # Move things a layer down
    if args is not None:
        # Update hyperparameter dictionary from command line arguments
        helpers.update_dict_from_args(config, args)
    config['input_dim'] = config[config['dataset']]['input_dim']
    config['hidden_layers'] = config[config['dataset']]['hidden_layers']
    config['hidden_dim'] = config[config['dataset']]['hidden_dim']
    config['output_dim'] = config[config['dataset']]['output_dim']
    config['scale'] = config[config['dataset']]['scale']
    config['input_channels'] = config[config['dataset']]['input_channels']
    helpers.print_hyperparameters(config)

    # Make directories where necessary
    pathlib.Path(os.path.join(file_dir, config['top_dir'])).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(file_dir, config['top_dir'], 'Datasets')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(file_dir, config['top_dir'], 'Plots')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(file_dir, config['top_dir'], 'Saved_Data')).mkdir(parents=True, exist_ok=True)

    # Set seeds
    np.random.seed(config['np_seed'])
    torch.manual_seed(config['torch_seed'])

    # Make sure Torch is installed and see if a GPU is available
    print(f"GPU Available? {torch.cuda.is_available()}\n")
    device = config['device']
    print(f"Using device {device}")

    # Dictionaries containing lists of useful values that we'll save after training
    train_loss = {'original': list(), 'extended': list(), 'continuous': list()}
    test_loss = {'original': list(), 'extended': list(), 'continuous': list()}
    train_acc = {'original': list(), 'extended': list(), 'continuous': list()}
    test_acc = {'original': list(), 'extended': list(), 'continuous': list()}
    alphas = {'original': None, 'extended': list(), 'continuous': None}

    # Dictionary mapping long names to short names
    names_d = {'original': f"{config['model']}_O", 'extended': f"{config['model']}_E", 'continuous': f"{config['model']}_C"}

    # Create Continuous model: Parameters are never reset
    model_c, model_c_loss_fn, model_c_optimizer = models.create_model(name=f"{config['model']}_C")

    # Train a newly initialized model from scratch to get weights for extended model
    # Create model
    model_o, model_o_loss_fn, model_o_optimizer = models.create_model(name=f"{config['model']}_O")
    # Load datasets
    train_ds_wrapped, test_ds_wrapped = datasets.load_wrapped_datasets(new_labels='none')
    # Train
    training.train_multiple_epochs(model_o, model_o_loss_fn, model_o_optimizer, train_ds_wrapped, test_ds_wrapped)

    # So that code doesn't break later
    model_e = None

    # Run restarts procedure
    for restart_num in range(config['restarts']):

        # Load datasets
        train_ds_wrapped, test_ds_wrapped = datasets.load_wrapped_datasets()
 
        # Print information
        print(f"Training restart number {restart_num+1}")
        print()

        # Create Extended model using model trained on previous objective
        if model_e is None:
            # Previous objective doesn't exist so we use a dummy model trained before the loop
            model_e, model_e_loss_fn, model_e_optimizer = models.create_model_E(old_model=model_o, name=f"{config['model']}_E")
        else:
            # Previous objective: use expanded model from before
            model_e, model_e_loss_fn, model_e_optimizer = models.create_model_E(old_model=model_e, name=f"{config['model']}_E")

        # Create New Original model
        model_o, model_o_loss_fn, model_o_optimizer = models.create_model(name=f"{config['model']}_O")

        # Train New model
        model_o_train_acc, model_o_test_acc, model_o_train_loss, model_o_test_loss, model_o_alphas =\
            training.train_multiple_epochs(model_o, model_o_loss_fn, model_o_optimizer, train_ds_wrapped, test_ds_wrapped)
        # Save results
        train_acc['original'].append(model_o_train_acc)
        test_acc['original'].append(model_o_test_acc)
        train_loss['original'].append(model_o_train_loss)
        test_loss['original'].append(model_o_test_loss)

        # Train Continuous model
        model_c_train_acc, model_c_test_acc, model_c_train_loss, model_c_test_loss, model_c_alphas =\
            training.train_multiple_epochs(model_c, model_c_loss_fn, model_c_optimizer, train_ds_wrapped, test_ds_wrapped)
        # Save results
        train_acc['continuous'].append(model_c_train_acc)
        test_acc['continuous'].append(model_c_test_acc)
        train_loss['continuous'].append(model_c_train_loss)
        test_loss['continuous'].append(model_c_test_loss)

        # Train Extended model
        model_e_train_acc, model_e_test_acc, model_e_train_loss, model_e_test_loss, model_e_alphas =\
            training.train_multiple_epochs(model_e, model_e_loss_fn, model_e_optimizer, train_ds_wrapped, test_ds_wrapped)
        # Save results
        train_acc['extended'].append(model_e_train_acc)
        test_acc['extended'].append(model_e_test_acc)
        train_loss['extended'].append(model_e_train_loss)
        test_loss['extended'].append(model_e_test_loss)
        alphas['extended'].append(model_e_alphas)

        # Save Data
        save_fn = f"{config['model']}_o_{restart_num}.torch"
        helpers.save_model(save_fn, model_o_train_loss, model_o_test_loss, model_o_train_acc, model_o_test_acc)

        save_fn = f"{config['model']}_C_{restart_num}.torch"
        helpers.save_model(save_fn, model_c_train_loss, model_c_test_loss, model_c_train_acc, model_c_test_acc)

        save_fn = f"{config['model']}_E_{restart_num}.torch"
        helpers.save_model(save_fn, model_e_train_loss, model_e_test_loss, model_e_train_acc, model_e_test_acc, model_e_alphas)

        # Create and Save Plots
        helpers.make_restart_plot(train_loss, test_loss, train_acc, test_acc, alphas, names_d, restart_num)

    # Save combined losses
    save_fn = f"final_losses.torch"
    helpers.save_final_losses(save_fn, train_loss, test_loss, train_acc, test_acc)

    # Make final plots
    helpers.make_final_plots({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "config": config 
        })

if __name__ == '__main__':
    # To allow num_workers>0
    torch.multiprocessing.set_start_method('spawn', force=True)

    # To allow CLIs
    parser = argparse.ArgumentParser()
    parser.add_argument('--np_seed', type=float)
    parser.add_argument('--torch_seed', type=float)
    parser.add_argument('--device', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--shuffle', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--restarts', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--top_dir', type=str)
    parser.add_argument('--SGD_method', type=str)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--beta_1', type=float)
    parser.add_argument('--beta_2', type=float)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--new_labels', type=str)
    parser.add_argument('--input_channels', type=int)
    parser.add_argument('--input_dim', type=int)
    parser.add_argument('--hidden_layers', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--output_dim', type=int)
    parser.add_argument('--scale', type=int)

    args = parser.parse_args()

    main(args)
