#imports
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from utility import load_data, process_image
from functions import build_classifier, validation, train_model, test_model, save_model, load_model

from workspace_utils import active_session
with active_session():

    # Set up calculation device
    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory',
                        action = 'store',
                        default = 'flowers',
                        help = 'Enter path to training data: ')

    parser.add_argument('--arch', 
                        action = 'store', 
                        dest = 'pretrained_model',
                        default = 'vgg11', 
                        help = 'Enter pretrained model to use. Else default is set to VGG-11: ')

    parser.add_argument('--save_dir', 
                        action = 'store', 
                        dest = 'save_directory', 
                        default = 'checkpoint.pth', 
                        help = 'Enter location to save Checkpoint: ')

    parser.add_argument('--learning_rate', 
                        action = 'store', 
                        dest = 'lr', 
                        type = float,
                        default = 0.002, 
                        help = 'Enter learning_rate for training model: ')

    parser.add_argument('--dropout',
                        action = 'store', 
                        dest = 'dropout', 
                        type = float, 
                        default = 0.2, 
                        help = 'Enter dropout for training model: ')

    parser.add_argument('--hidden_units', 
                        action = 'store', 
                        dest = 'hunits', 
                        type = int, 
                        default = 4096, help = 'Enter of hidden unints for training model: ')

    parser.add_argument('--epochs', 
                        action = 'store', 
                        dest = 'number_epochs', 
                        type = int, 
                        default = 2, 
                        help = 'Enter number of epochs for training model: ')

    parser.add_argument('--gpu', 
                        action = "store_true", 
                        default = True, 
                        help = 'Turn GPU mode on or off: ')

    results = parser.parse_args()

    data_dir = results.data_directory

    pt_model  = results.pretrained_model

    save_dir = results.save_directory

    learning_rate = results.lr

    dropout = results.dropout

    hidden_units = results.hunits

    epochs = results.number_epochs

    gpu = results.gpu

    # Load Data
    train_data, valid_data, test_data, trainloader, validloader, testloader = load_data(data_dir)

    #model = models.pt_model(pretrained = True)
    model = getattr(models,pt_model)(pretrained = True)
    #loaded_model = load_checkpoint(model, save_dir)

    # Build classifier
    input_units = model.classifier[0].in_features
    build_classifier(model, input_units, hidden_units, dropout)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    # Train model
    model, optimizer = train_model(model, epochs, trainloader, validloader, criterion, optimizer, gpu)

    # Test model
    test_model(model, testloader, gpu)

    # Save model
    save_model(model, train_data, optimizer, save_dir, epochs)