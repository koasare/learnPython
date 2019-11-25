#Imports
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict


# Define build classifier function
def build_classifier(model, input_units, hidden_units, dropout):
    '''
    Define build classifier using existing pretrained models.
    Freeze model parameters and modify classifier for local use.
    inputs: pretrained model, number of inputs, inputs for hidden layer, dropout
    outputs: customized model
    '''
    
    # Freeze parameters so we don't backpropagation
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_units, hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('dropout', nn.Dropout(p = dropout)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim = 1))
                              ]))

    model.classifier = classifier
    return model

# Define Validation Function
def validation(model, validloader, criterion, gpu):
    '''
    Model validation.
    inputs: model, validation loader, loss function, GPU mode
    outputs: validation loss, model accuracy
    '''
    
    valid_loss = 0
    accuracy = 0

    if gpu == True:
        model.to('cuda')
    else:
        pass

    for ii, (images, labels) in enumerate(validloader):
        if gpu == True:
            images, labels = images.to('cuda'), labels.to('cuda')
        else:
            pass

        # Feed forward
        output = model.forward(images)

        #calculate loss
        batch_loss = criterion(output, labels)
        valid_loss += batch_loss.item()

        #calculate probabilties and Accuracy
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim = 1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return valid_loss, accuracy

def train_model(model, epochs, trainloader, validloader, criterion, optimizer, gpu):
    '''
    train model
    inputs: model, epochs, trainloader, validloader, criterion, optimizer, gpu
    outputs: model, optimizer
    '''
    
    steps = 0
    print_every = 25
    running_loss = 0

    if gpu == True:
        model.to('cuda')
    else:
        pass

    for epoch in range(epochs):
        for ii, (images, labels) in enumerate(trainloader):
            steps += 1

            #get inputs
            if gpu == True:
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                pass
            
            #zero parameter gradients
            optimizer.zero_grad()

            #forward pass, backward pass + optimize gradient step
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # validation steps
            if steps % print_every == 0:
                
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, gpu)

                print(f"Epoch {epoch+1}/{epochs}| "
                      f"Train loss: {running_loss/print_every:.3f}| "
                      f"Validation loss: {valid_loss/len(validloader):.3f}| "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")

            running_loss = 0
            model.train()

    return model, optimizer

def test_model(model, testloader, gpu):
    correct = 0
    total = 0

    if gpu == True:
        model.to('cuda')
    else:
        pass

    with torch.no_grad():
        for ii, (images, labels) in enumerate(testloader):
            
            if gpu == True:
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                pass

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images is: %d%%' % (100 * correct / total))

def save_model(model, train_data, optimizer, save_dir, epochs):
    
    # save mapping of classes into indices
    checkpoint = {'model': model,
                  'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': train_data.class_to_idx,
                  'optimizer_state': optimizer.state_dict,
                  'number_epochs': epochs}
    return torch.save(checkpoint, save_dir)

def load_model(model, save_dir, gpu):
    '''
        This function loads the checkpoint
        and rebuilds the network model
        input: filepath
        output: network model
    '''
    
    #load the saved file
    if gpu == True:
        checkpoint = torch.load(save_dir)
    else:
        checkpoint = torch.load(save_dir, map_location = 'cpu')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def predict(processed_image, loaded_model, topk, gpu):
    #set model to eval, turn gradients off
    loaded_model.eval()
    
    if gpu == True:
        loaded_model.to('cuda')
    else:
        loaded_model.cpu()
        
    with torch.no_grad():
        output = loaded_model.forward(processed_image)
    
    #calc probability
    probs = torch.exp(output)
    top_probs = probs.topk(topk)[0]
    top_index = probs.topk(topk)[1]
    
    # converting probabilities and outputs to lists
    top_probs_list = np.array(top_probs)[0]
    top_index_list = np.array(top_index[0])
    
    #loading index and class mapping
    class_to_idx = loaded_model.class_to_idx
    # Inverting  index-class dictionary
    idx_to_class = {x: y for y, x in class_to_idx.items()}
    
    # converting index list to class list
    top_classes_list = []
    for index in top_index_list:
        top_classes_list += [idx_to_class[index]]
    
    return top_probs_list, top_classes_list