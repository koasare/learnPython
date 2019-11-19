#imports
import argparse

import numpy as np

import pandas as pd

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from utility import load_data, process_image
from functions import network, validation, train_model, test_model, save_model, load_checkpoint, predict

# Set up calculation device
parser.add_argument('data_directory', action = 'store', help = 'Enter path to training data: ')

parser.add_argument('--arch', action = 'store', dest = 'pretrained_model',
                    default = 'vgg11', help = 'Enter pretrained model to use. Else default is set to VGG-11: ')

parser.add_argument('--save_dir', action = 'store', dest = 'save_directory', default = 'checkpoint.pth', help = 'Enter location to save Checkpoint: ')

parser.add_argument('--learning_rate', action = 'store', dest = 'lr', type = 'int', defualt = 0.001, help = 'Enter learning_rate for training model: ')

parser.add_argument('--dropout', action = 'store', dest = 'drpt', type = 'int', defualt = 0.05, help = 'Enter dropout for training model: ')

parser.add_argument('--hidden_units', action = 'store', dest = 'size', type = 'int', defualt = 500, help = 'Enter of hidden unints for training model: ')

parser.add_argument('--epochs', action = 'store', dest = 'number_epochs', type = 'int', defualt = 3, help = 'Enter number of epochs for training model: ')

parser.add_argument('--gpu,' action = "store_true", default = False, help = 'Turn GPU mode on or off: ')

results = parser.parse_args()

data_dir = result.data_directory

save_dir = result.save_directory

learning_rate = results.lr

dropout = results.drpt

hidden_units = results.unints

epochs = results.number_epochs

gpu = results.gpu
