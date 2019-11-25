#Imports
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms, models

def load_data(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Training transforms: random rotation, resize and Flip
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    # validation transforms
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    # test transforms
    test_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])


    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, valid_transforms)
    test_data = datasets.ImageFolder(test_dir, test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(train_data, batch_size=64)

    return train_data, valid_data, test_data, trainloader, validloader, testloader

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    # Resize and crop image
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    # transform image for network model
    image_tensor = transform(im)
    
    #convert to numpy array
    image_array = np.array(image_tensor)
    
    # convert to torch tensor
    image_tensor = torch.from_numpy(image_array).type(torch.FloatTensor)
    
    # insert new axis at index 0
    image_input = image_tensor.unsqueeze_(0)
    
    return image_input