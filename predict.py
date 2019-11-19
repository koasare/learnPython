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

parser = argparse.ArgumentParser(description = 'Use neural network to make predictions of images')

parser.add_argument('--image_path', action = 'store', default = '../flowers/test/11/image_03115.jpg', help = 'Enter path to image: ')

parser.add_argument('--save_dir', action = 'store', dest = 'save_directory', default = 'checkpoint.pth', help = 'Enter location to save checkpoint in: ')

parser.add_argument('--arch', action = 'store', dest = 'pretrained_model', default = 'vgg11', help = 'Enter pretrained model to use: ')

parser.add_argument('--top_k', action = 'store', dest = 'topk', type = int, default = 5, help = 'Enter number of top most likely classes to view: ')

parser.add_argument('--cat_to_name', action = 'store', dest = 'cat_name_directory' default = 'cat_to_name.json', help = 'Enter path to image: ')

parser.add_argument('--gpu,' action = "store_true", default = False, help = 'Turn GPU mode on or off: ')

results = parser.parse_args()

save_dir = results.save_directory

image = results.image_path

top_k = results.topk

gpu = results.gpu

cat_names = results.cat_name_dir

with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

pt_model = results.pretrained_model
model = models.pt_model(pretrained = True)

# Load model
loaded_model = load_checkpoint(model, save_dir, gpu)

# Preprocess image
processed_image = process_image(image)

if gpu == store_true:
    processed_image = process_image.to('cuda')
else:
    pass

probs, classes = predict(processed_image, loaded_model, top_k, gpu)

print(probs)
print(classes)

names = []
for i in classes:
    names += [cat_to_name[i]]

print(f"This flower is a: '{names[0]}', with a {round(probs[0]*100, 4)}% confidence.")
