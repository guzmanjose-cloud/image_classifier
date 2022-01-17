from train1 import main_function, load_checkpoint, process_image
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd


main_function()

load_checkpoint = load_checkpoint('checkpoint.pth')

if torch.cuda.is_available():
     device = torch.device('cuda')
else:
    device = torch.device("cpu")
    print("using", device, "device")

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets


data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
testing_data = datasets.ImageFolder(test_dir, transform = valid_transforms)
# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(validation_data, batch_size = 64, shuffle = True)
Testloader = torch.utils.data.DataLoader(validation_data, batch_size = 64, shuffle = True)

vgg = models.vgg16(pretrained=True)
vgg

for param in vgg.parameters():
    param.requires_grad = False

from collections import OrderedDict

# Define a new, untrainted feed-forward network as a classifier, using ReLU activations and dropout
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 512, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(512, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

vgg.classifier = classifier


def predict(image_path, model, topk=5,):
    
    global vgg
    #Processing image
    image = process_image(image_path)
    image = image.float().unsqueeze_(0)
    image = image.to(device)
    vgg = vgg.to(device)  
    #Creating prediction score
    with torch.no_grad():
            
        output = vgg.forward(image)
        
    prediction = F.softmax(output.data, dim = 1)
    
    probs, indices = prediction.topk(topk)
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]
    
    
    return probs, indices


test_image = (test_dir + '/1/' + 'image_06743.jpg')
probs, classes = predict(test_image, vgg)
print(probs)
print(classes)


predict(test_image, vgg)