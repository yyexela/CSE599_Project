###############################
# Imports # Imports # Imports #
###############################

import os
import yaml
import pathlib
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add custom package to import path and import it
file_dir = pathlib.Path().resolve()
pkg_dir = os.path.join(file_dir, "submodules")
sys.path.insert(0, pkg_dir)

#######################################
# Config Functions # Config Functions #
#######################################

def get_loader() -> yaml.SafeLoader:
    """
    Makes it so we can put directories and sub-directories in YAML

    ex:
    dataset_dir: &dataset_dir "/Datasets"
    original_data_dir: !join [*dataset_dir, "Original"]

    config['original_data_dir'] yields "/Datasets/Original"
    """
    loader = yaml.SafeLoader

    # define custom tag handler
    # (https://stackoverflow.com/questions/5484016/how-can-i-do-string-concatenation-or-string-replacement-in-yaml)
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return os.path.join(*seq)

    ## register the tag handler
    loader.add_constructor('!join', join)

    return loader

def init_config() -> dict:
    """
    Initializes global dictionary of yaml config file
    Reads `../config/config.yaml` and sets it to the global `config` dictionary,
    which is returned in subsequent `get_config` calls
    """
    global config

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'config', "config.yaml"), 'r') as config_file:
        config = yaml.load(config_file, Loader=get_loader())

    return config

def update_config(new_path: list[str]) -> dict:
    """
    Updates global dictionary of to new yaml config file from specified path `new_path`
    
    Tested
    """
    global config
    with open(new_path, 'r') as config_file:
        test_config = yaml.load(config_file, Loader=get_loader())
    # Update individual keys, keeping old ones
    for key in list(test_config.keys()):
        config[key] = test_config[key]
    return config

def get_config() -> dict:
    """
    Returns current global config dictionary
    This is initialized with `init_config` to the default config file and
    updated with `update_config`

    Tested
    """
    global config
    return config

#######################################
# Saving Functions # Saving Functions #
#######################################

def save_model(save_fn, train_loss, test_loss, train_acc, test_acc, alphas=None):
    save_dir = os.path.join(file_dir,\
                            config['top_dir'],
                            "Saved_Data")
    save_abs = os.path.join(save_dir, save_fn)
    os.makedirs(save_dir, mode=0o777, exist_ok=True)
    torch.save({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "alphas": alphas,
            "config": config 
        }, save_abs)
    print(f"Saved data to \"{save_abs}\"")

def save_final_losses(save_fn, train_loss, test_loss, train_acc, test_acc):
    save_dir = os.path.join(file_dir,\
                            config['top_dir'],\
                            "Saved_Data")
    save_abs = os.path.join(save_dir, save_fn)
    os.makedirs(save_dir, mode=0o777, exist_ok=True)
    torch.save({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "config": config 
        }, save_abs)
    print(f"Saved data to \"{save_abs}\"")

###########################################
# Plotting Functions # Plotting Functions #
###########################################

def make_restart_plot(train_loss, test_loss, train_acc, test_acc, alphas, names_d, restart_num):
    for split in ["train", "test"]:
        for model in ["original", "extended", "continuous"]:
            # Get data
            loss = (train_loss if split == 'train' else test_loss)[model][-1]
            acc = (train_acc if split == 'train' else test_acc)[model][-1]
            a_s = alphas[model]
            if a_s is not None:
                a_s = a_s[-1]
            abrev = names_d[model]
            # Plot losses
            plot_fn = f"{abrev}_{restart_num}_{split}_loss.pdf"
            plot_and_save(loss, f"{abrev} {split.title()} Loss Per Epoch", "Epoch", f"{split.title()} loss", plot_fn, a_s)
            # Plot accuracies
            plot_fn = f"{abrev}_{restart_num}_{split}_acc.pdf"
            plot_and_save(acc, f"{abrev} {split.title()} Accuracy Per Epoch", "Epoch", f"{split.title()} accuracy (%)", plot_fn, a_s)

    return None

def plot_and_save(data, title: str, x_axis: str, y_axis: str, plot_fn: str, alphas=None):
    plot_dir = os.path.join(file_dir, config['top_dir'], "Plots")
    # Make plots
    lines = []
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(x_axis)
    ax1.set_ylabel(y_axis)
    lines.append(ax1.plot(data, color='black', label=y_axis)[0])

    # Add alphas if necessary
    if alphas is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Alpha")
        lines.append(ax2.plot(alphas, color='blue', label="Alpha")[0])

    labels = [l.get_label() for l in lines]
    plt.title(title)
    plt.legend(lines, labels)

    # Save plot
    fig.tight_layout()
    plot_abs = os.path.join(plot_dir, plot_fn)
    plt.savefig(plot_abs)
    print(f"Saved plot to \"{plot_abs}\"")

    # Show plot
    plt.show()
    plt.close()

def plot_sample(dataset, train_dl, plot_fn):
    """
    Plot a 4x4 grid sapmle of CIFAR10 images
    """
    plot_dir = os.path.join(file_dir, config['top_dir'], "Plots")

    # Make sure dataset is viable
    if dataset not in ["MNIST", "CIFAR10"]:
        raise Exception(f"Dataset {dataset} must be MNIST or CIFAR10")

    # Number of images we'll plot
    ncols = 4
    nrows = 4

    # Load dataset if not given
    batch = next(iter(train_dl))
    cifar10_labels = get_cifar10_labels()

    print("batch shape:")
    print(batch[0].shape)
    print(batch[1].shape)

    # Create our figure as a grid
    fig1, f1_axes = plt.subplots(ncols=ncols, nrows=nrows, constrained_layout=True, figsize=(3.5*ncols,3.5*nrows), facecolor='white')
    fig1.set_dpi(50.0)

    # Plot the modified images
    for c in range(ncols):
        for r in range(nrows):
                image = batch[0][r*ncols+c]
                if c+r == 0:
                    print(f"Image size: {image.shape}")
                label_real = batch[1][r*ncols+c]
                image_show = torch.permute(image, (1,2,0)) # Change ordering of dimensions

                ax = f1_axes[r][c]
                ax.set_axis_off()
                fsize = 60
                if dataset == "CIFAR10":
                    ax.set_title(f"{cifar10_labels[label_real]}", fontsize=fsize)
                else:
                    ax.set_title(f"{label_real}", fontsize=fsize)
                ax.imshow(np.asarray(image_show.to('cpu')), cmap='gray')

    # Save plot
    plot_abs = os.path.join(plot_dir, plot_fn)
    plt.savefig(plot_abs)
    print(f"Saved plot to \"{plot_abs}\"")
    plt.show()

def plot_all_restarts(original_data, continuous_data, extended_data, ylabel, loc='upper left'):
    plot_dir = os.path.join(file_dir, config['top_dir'], "Plots")

    # Load config file
    restarts = config["restarts"]
    epochs = config["epochs"]
    vlines = [epochs*(idx+1) for idx in range(restarts)]
    vlines.append(0)

    plt.figure(figsize=(12,4))
    for i, vline in enumerate(vlines):
        if i == 0:
            plt.axvline(vline, linestyle='dashed', color='gray', label='Restarts')
        else:
            plt.axvline(vline, linestyle='dashed', color='gray')
    plt.plot(original_data, color='black', label="Original")
    plt.plot(continuous_data, color='orange', label="Continuous")
    plt.plot(extended_data, color='blue', label="Extended")
    plt.title(f"{ylabel} for Original, Continuous, and\nExtended Models")
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend(loc=loc)
    plot_fn=f"all_restarts_{ylabel}.pdf"
    plot_abs = os.path.join(plot_dir, plot_fn)
    plt.savefig(plot_abs)
    print(f"Saved plot to \"{plot_abs}\"")
    plt.show()
    plt.close()

def plot_one_restart(original_data, continuous_data, extended_data, ylabel, loc='upper left'):
    plot_dir = os.path.join(file_dir, config['top_dir'], "Plots")

    plt.plot(original_data, color='black', label="Original")
    plt.plot(continuous_data, color='orange', label="Continuous")
    plt.plot(extended_data, color='blue', label="Extended")
    plt.title(f"{ylabel} for Original, Continuous, and\nExtended Models")
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend(loc=loc)
    plot_fn=f"one_restart_{ylabel}.pdf"
    plot_abs = os.path.join(plot_dir, plot_fn)
    plt.savefig(plot_abs)
    plt.show()
    plt.close()

def make_final_plots(final_losses):
    # Full plots
    original_train_loss = np.hstack(final_losses['train_loss']["original"])
    continuous_train_loss = np.hstack(final_losses['train_loss']["continuous"])
    extended_train_loss = np.hstack(final_losses['train_loss']["extended"])
    plot_all_restarts(original_train_loss, continuous_train_loss, extended_train_loss, "Training Loss")

    original_test_loss = np.hstack(final_losses['test_loss']["original"])
    continuous_test_loss = np.hstack(final_losses['test_loss']["continuous"])
    extended_test_loss = np.hstack(final_losses['test_loss']["extended"])
    plot_all_restarts(original_test_loss, continuous_test_loss, extended_test_loss, "Testing Loss")

    original_train_acc = np.hstack(final_losses['train_acc']["original"])
    continuous_train_acc = np.hstack(final_losses['train_acc']["continuous"])
    extended_train_acc = np.hstack(final_losses['train_acc']["extended"])
    plot_all_restarts(original_train_acc, continuous_train_acc, extended_train_acc, "Training Acc", loc='lower left')

    original_test_acc = np.hstack(final_losses['test_acc']["original"])
    continuous_test_acc = np.hstack(final_losses['test_acc']["continuous"])
    extended_test_acc = np.hstack(final_losses['test_acc']["extended"])
    plot_all_restarts(original_test_acc, continuous_test_acc, extended_test_acc, "Testing Acc", loc='lower left')

    # Final restart
    original_train_loss = final_losses['train_loss']["original"][-1]
    continuous_train_loss = final_losses['train_loss']["continuous"][-1]
    extended_train_loss = final_losses['train_loss']["extended"][-1]
    plot_one_restart(original_train_loss, continuous_train_loss, extended_train_loss, "Training Loss")

    original_test_loss = final_losses['test_loss']["original"][-1]
    continuous_test_loss = final_losses['test_loss']["continuous"][-1]
    extended_test_loss = final_losses['test_loss']["extended"][-1]
    plot_one_restart(original_test_loss, continuous_test_loss, extended_test_loss, "Testing Loss")

    original_train_acc = final_losses['train_acc']["original"][-1]
    continuous_train_acc = final_losses['train_acc']["continuous"][-1]
    extended_train_acc = final_losses['train_acc']["extended"][-1]
    plot_one_restart(original_train_acc, continuous_train_acc, extended_train_acc, "Training Acc", loc='lower left')

    original_test_acc = final_losses['test_acc']["original"][-1]
    continuous_test_acc = final_losses['test_acc']["continuous"][-1]
    extended_test_acc = final_losses['test_acc']["extended"][-1]
    plot_one_restart(original_test_acc, continuous_test_acc, extended_test_acc, "Testing Acc", loc='lower left')

#######################################
# Helper Functions # Helper Functions #
#######################################

def get_cifar10_labels():
    """
    Mapping from number to string for CIFAR10 labels
    """
    cifar10_labels = ["airplane",
                      "automobile",
                      "bird",
                      "cat",
                      "deer",
                      "dog",
                      "frog",
                      "horse",
                      "ship",
                      "truck"]
    return cifar10_labels

def update_dict_from_args(d, args):
    if args.np_seed is not None:
        d['np_seed'] = args.np_seed
    if args.torch_seed is not None:
        d['torch_seed'] = args.torch_seed
    if args.device is not None:
        d['device'] = args.device
    if args.batch_size is not None:
        d['batch_size'] = args.batch_size
    if args.shuffle is not None:
        d['shuffle'] = args.shuffle
    if args.epochs is not None:
        d['epochs'] = args.epochs
    if args.restarts is not None:
        d['restarts'] = args.restarts
    if args.dataset is not None:
        d['dataset'] = args.dataset
    if args.model is not None:
        d['model'] = args.model
    if args.top_dir is not None:
        d['top_dir'] = args.top_dir
    if args.SGD_method is not None:
        d['SGD_method'] = args.SGD_method
    if args.learning_rate is not None:
        d['learning_rate'] = args.learning_rate
    if args.beta_1 is not None:
        d['beta_1'] = args.beta_1
    if args.beta_2 is not None:
        d['beta_2'] = args.beta_2
    if args.epsilon is not None:
        d['epsilon'] = args.epsilon
    if args.weight_decay is not None:
        d['weight_decay'] = args.weight_decay
    if args.new_labels is not None:
        d['new_labels'] = args.new_labels
    if args.input_channels is not None:
        d['input_channels'] = args.input_channels
    if args.input_dim is not None:
        d['input_dim'] = args.input_dim
    if args.hidden_layers is not None:
        d['hidden_layers'] = args.hidden_layers
    if args.hidden_dim is not None:
        d['hidden_dim'] = args.hidden_dim
    if args.output_dim is not None:
        d['output_dim'] = args.output_dim
    if args.scale is not None:
        d['scale'] = args.scale
    return d

def print_hyperparameters(hp_dict: dict[str, str]) -> None:
    """
    Print given hyperparameters
    
    Tested
    """
    for key in hp_dict.keys():
        print(f"{key}: {hp_dict[key]}")

    return None
