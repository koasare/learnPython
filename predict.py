#imports
import argparse

import numpy as np

import pandas as pd

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

from utility import load_data, process_image
from functions import test_model, load_model, predict

parser = argparse.ArgumentParser()

parser.add_argument('--image_path', 
                    action = 'store', 
                    default = '../ImageClassifier/flowers/test/11/image_03115.jpg', 
                    help = 'Enter path to image: ')

parser.add_argument('--save_dir', 
                    action = 'store', 
                    dest = 'save_directory', 
                    default = 'checkpoint.pth', 
                    help = 'Enter location to save checkpoint in: ')

parser.add_argument('--arch', 
                    action = 'store', 
                    dest = 'pretrained_model', 
                    default = 'vgg11', 
                    help = 'Enter pretrained model to use: ')

parser.add_argument('--top_k', 
                    action = 'store', 
                    dest = 'topk', 
                    type = int, 
                    default = 5, 
                    help = 'Enter number of top most likely classes to view: ')

parser.add_argument('--cat_to_name', 
                    action = 'store', 
                    dest = 'cat_name_dir', 
                    default = 'cat_to_name.json', 
                    help = 'Enter path to image: ')

parser.add_argument('--gpu', 
                    action = "store_true", 
                    default = False, 
                    help = 'Turn GPU mode on or off: ')

# Parser results
results = parser.parse_args()

save_dir = results.save_directory

pt_model = results.pretrained_model

image = results.image_path

top_k = results.topk

gpu = results.gpu

cat_names = results.cat_name_dir

with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)
    
model = getattr(models,pt_model)(pretrained = True)

# Load model
loaded_model = load_model(model, save_dir, gpu)

# Preprocess image
processed_image = process_image(image)

# Define top K likely classes with probabilities
probs, classes = predict(processed_image, loaded_model, top_k, gpu)

# Define names for Classes
names = [cat_to_name[i] for i in classes]

# Print out top K classes and probabilities
print(f"Top {top_k} classes are: {classes}, with assocatied probabilities: {probs}")

# Print out most likely output
print(f"The most likely outcome is a: '{names[0]} ({round(probs[0]*100, 2)}%)'")
