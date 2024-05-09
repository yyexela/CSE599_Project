###############################
# Imports # Imports # Imports #
###############################

import copy
import torch
from torch import nn

from helper_pkg.src.helpers import get_config
import helper_pkg.src.training as training

###############################
# General # General # General #
###############################

def create_model(name="Model"):
    config = get_config()

    if config['model'] == 'MLP':
        return create_MLP(name)
    elif config['model'] == 'CNN':
        return create_CNN(name)
    else:
        raise Exception(f"create_model: invalid model {config['model']}")

def create_model_E(old_model, name="Model"):
    config = get_config()

    if config['model'] == 'MLP':
        return create_MLP_E(old_model, name)
    elif config['model'] == 'CNN':
        return create_CNN_E(old_model, name)
    else:
        raise Exception(f"create_model_E: invalid model {config['model']}")

###############################
# CNN # CNN # CNN # CNN # CNN #
###############################

def create_CNN(name="CNN"):
    # Load config file and values
    config = get_config()

    cnn = CNN(
        config['input_channels'],
        config['input_dim'],
        config['output_dim'],
        name)\
    .to(config['device']).to(torch.float32)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = training.get_optimizer(
        config['SGD_method'],
        cnn.parameters(),
        config['learning_rate'],
        (config['beta_1'],config['beta_2']),
        float(config['epsilon']),
        config['weight_decay']
    )

    total_params = sum(p.numel() for p in cnn.parameters())
    print(f"{name} Total Parameters: {total_params}\n")

    return cnn, loss_fn, optimizer

def create_CNN_E(old_model, name="CNN_E"):
    # Load config file and values
    config = get_config()

    cnn_e = CNN_Expanded(
        config['input_channels'],
        config['input_dim'],
        config['output_dim'],
        old_model,
        name)\
    .to(config['device']).to(torch.float32)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = training.get_optimizer(
        config['SGD_method'],
        filter(lambda p: p.requires_grad, cnn_e.parameters()),
        config['learning_rate'],
        (config['beta_1'],config['beta_2']),
        float(config['epsilon']),
        config['weight_decay']
    )

    total_params = sum(p.numel() for p in cnn_e.parameters())
    trainable_params = sum(p.numel() for p in cnn_e.parameters() if p.requires_grad)
    print(f"{name} Trainable Parameters: {trainable_params} of {total_params}\n")

    return cnn_e, loss_fn, optimizer

###############################
# MLP # MLP # MLP # MLP # MLP #
###############################

def create_MLP(name="MLP"):
    # Load config file and values
    config = get_config()

    mlp = MLP(
        config['input_dim'],
        config['hidden_layers'],
        config['hidden_dim'],
        config['output_dim'],
        name)\
    .to(config['device']).to(torch.float32)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = training.get_optimizer(
        config['SGD_method'],
        mlp.parameters(),
        config['learning_rate'],
        (config['beta_1'],config['beta_2']),
        float(config['epsilon']),
        config['weight_decay']
    )

    if name != "MLP_D":
        total_params = sum(p.numel() for p in mlp.parameters())
        print(f"{name} Total Parameters: {total_params}\n")

    return mlp, loss_fn, optimizer

def create_MLP_E(old_model, name="MLP_E"):
    # Load config file and values
    config = get_config()

    mlp_e = MLP_Expanded(
        config['input_dim'],
        config['hidden_layers'],
        config['hidden_dim'],
        config['output_dim'],
        old_model,
        name)\
    .to(config['device']).to(torch.float32)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = training.get_optimizer(
        config['SGD_method'],
        filter(lambda p: p.requires_grad, mlp_e.parameters()),
        config['learning_rate'],
        (config['beta_1'],config['beta_2']),
        float(config['epsilon']),
        config['weight_decay']
    )

    total_params = sum(p.numel() for p in mlp_e.parameters())
    trainable_params = sum(p.numel() for p in mlp_e.parameters() if p.requires_grad)
    print(f"{name} Trainable Parameters: {trainable_params} of {total_params}\n")

    return mlp_e, loss_fn, optimizer

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_dim, output_dim, name):
        # See https://realpython.com/python-super/ for info about super
        super(MLP, self).__init__()
        # Linear part
        modules = []
        modules.append(nn.Linear(input_dim, hidden_dim))
        modules.append(nn.ReLU())
        for _ in range(hidden_layers-1):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_dim, output_dim))
        # Combine it all
        self.classifier = nn.Sequential(*modules)
        # Identify it
        self.name = name
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
    def get_label(self, logits):
        pred_prob = nn.Softmax(dim=1)(logits)
        y_pred = pred_prob.argmax(1)
        return y_pred

class MLP_Expanded(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_dim, output_dim, old_model, name):
        # See https://realpython.com/python-super/ for info about super
        super(MLP_Expanded, self).__init__()

        # Copy old model and freeze layers
        self.old_model = copy.deepcopy(old_model)
        for _, p in old_model.named_parameters():
            p.requires_grad = False
        if old_model.name == 'MLP_E':
            self.old_model = self.old_model.new_model

        # Create new part
        self.new_model = MLP(input_dim, hidden_layers, hidden_dim, output_dim, name)
        self.name = name
        
    def forward(self, x):
        # Pass through old model
        x1 = self.old_model(x)
        # Pass through new model
        x2 = self.new_model(x)

        return x1, x2
        
    def get_label(self, logits):
        pred_prob = nn.Softmax(dim=1)(logits)
        y_pred = pred_prob.argmax(1)
        return y_pred

#############################
# CIFAR10 CNN # CIFAR10 CNN #
#############################

class CNN_Expanded(nn.Module):
    def __init__(self, input_channels, input_dim, output_dim, old_model, name):
        # See https://realpython.com/python-super/ for info about super
        super(CNN_Expanded, self).__init__()

        # Copy old model and freeze layers
        self.old_model = copy.deepcopy(old_model)
        for _, p in old_model.named_parameters():
            p.requires_grad = False
        if old_model.name == 'CNN_E':
            self.old_model = self.old_model.new_model

        # Create new part
        self.new_model = CNN(input_channels, input_dim, output_dim, name)
        self.name = name
        
    def forward(self, x):
        # Pass through old model
        x1 = self.old_model(x)
        # Pass through new model
        x2 = self.new_model(x)

        return x1, x2
        
    def get_label(self, logits):
        pred_prob = nn.Softmax(dim=1)(logits)
        y_pred = pred_prob.argmax(1)
        return y_pred

class CNN(nn.Module):
    def __init__(self, input_channels, input_dim, output_dim, name):
        # See https://realpython.com/python-super/ for info about super
        super(CNN, self).__init__()
        # CNN part
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,out_channels=2*input_channels,kernel_size=3,padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=2*input_channels,out_channels=4*input_channels,kernel_size=3,padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # Linear part
        self.classifier = nn.Sequential(
            nn.Linear(int(input_dim/4), int(input_dim/8)),
            nn.ReLU(inplace=True),
            nn.Linear(int(input_dim/8), int(input_dim/8)),
            nn.ReLU(inplace=True),
            nn.Linear(int(input_dim/8), output_dim),
        )
        self.name = name
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
    def get_label(self, logits):
        pred_prob = nn.Softmax(dim=1)(logits)
        y_pred = pred_prob.argmax(1)
        return y_pred

#####################################
# Generic Helpers # Generic Helpers #
#####################################

def save_model(epoch, model, optimizer,\
                     train_acc, test_acc, train_loss, test_loss,\
                     save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_loss': train_loss,
        'test_loss': test_loss,
        }, save_path)
