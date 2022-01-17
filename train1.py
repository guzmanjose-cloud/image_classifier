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



def main_function():

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

    import json

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # TODO: Build and train your network
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

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(vgg.classifier.parameters(), lr=0.001)
    

    epochs = 10

    for e in range(epochs):

        training_loss = 0

        step = 0

        train_losses, valid_losses = [], []
        for images, labels in dataloaders:
            images, labels = images.to(device), labels.to(device)
            vgg.to(device)

            step += 1

            # TODO: Training pass

            optimizer.zero_grad()

            output = vgg.forward(images)

            loss = criterion(output, labels)



            loss.backward()

            optimizer.step()



            training_loss += loss.item()

            #print(f"Training loss: {running_loss/len(dataloaders)}")

            #print(f"Step: {step} Training loss: {running_loss/step}")

            vgg.eval()
            valid_loss = 0
            accuracy = 0


        #else:
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    vgg.to(device)
                    log_ps = vgg(images)
                    valid_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            train_losses.append(training_loss/len(dataloaders))
            valid_losses.append(valid_loss/len(validloader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(training_loss/len(dataloaders)),
                  "valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "valid Accuracy: {:.3f}".format(accuracy/len(validloader)))
            vgg.train()

    correct = accuracy/len(validloader)
    total = 0

    with torch.no_grad():
        for data in Testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = vgg(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    
def process_image(image):
    
    
    picture = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    np_array = transform(picture).float()
    
    return np_array

    processed_image = (test_dir + '/43/' + 'image_02365.jpg')
    processed_image = process_image(processed_image)
    
    # TODO: Save the checkpoint 
    vgg.class_to_idx = image_datasets.class_to_idx
    checkpoint = {'vgg': vgg,
                  'state_dict': vgg.state_dict(),
                  'class_to_idx': image_datasets.class_to_idx,
                  'epochs': epochs,
                  'accuracy': accuracy,
                  'train_losses': train_losses,
                  'valid_loss': valid_losses,
                  'optimizer_state': optimizer.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')
    


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    vgg = checkpoint['vgg']
    vgg.load_state_dict(checkpoint['state_dict'])
    vgg.class_to_idx = checkpoint['class_to_idx']
    vgg.epochs = checkpoint['epochs']
    vgg.accuracy = checkpoint['accuracy']
    vgg.train_losses = checkpoint['train_losses']
    vgg.valid_loss = checkpoint['valid_loss']
    vgg.optimizer_state = checkpoint['optimizer_state']
    return vgg

    
    loaded_checkpoint = load_checkpoint('checkpoint.pth')
    #print(loaded_checkpoint)
    
#main_function()
load_checkpoint('checkpoint.pth')
