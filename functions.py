#Imports
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

# classifier function
def network(model, input_size, hidden_size, dropout):
    # Freeze parameters so we don't backpropagation
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 512)),
                              ('relu1', nn.ReLU()),
                              ('dropout', nn.Dropout(0,2)),
                              ('fc2', nn.Linear(512, 102)),
                              ('output', nn.LogSoftmax(dim = 1))
                              ]))

    model.classifier = classifier
    return model

def validation(model, validloader, criterion):
    valid_loss = 0
    accuracy = 0

    model.to(device)

    for ii, (inputs, labels) in enumerate(validloader):

        inputs, labels = inputs.to(device), labels.to(device)

        # Feed forward
        output = model.forward(inputs)

        #calculate loss
        batch_loss = criterion(output, labels)
        valid_loss += batch_loss.item()

        #calculate probabilties and Accuracy
        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(1, dim = 1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return valid_loss, accuracy

def train_model(model, epochs, trainloader, validloader, criterion, optimizer):
    steps = 0
    print_every = 5

    model.to(device)

    for epoch in range(epochs):
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            #get inputs
            inputs, labels = inputs.to(device), labels.to(device)

            #zero parameter gradients
            optimizer.zero_grad()

            #forward pass, backward pass + optimize
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # validation steps
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)

                print(f"Epoch {epoch+1}/{epochs}| "
                      f"Train loss: {running_loss/print_every:.3f}| "
                      f"Validation loss: {valid_loss/len(validloader):.3f}| "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")

            running_loss = 0
            model.train()

    return model, optimizer

def test_model():
    return

def save_model():
    return

def load_model():
    return

def predict():
    return
