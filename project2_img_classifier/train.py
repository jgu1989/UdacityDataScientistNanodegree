# Imports here
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import argparse
import load




def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device='cpu'):
    steps = 0

    # change to cuda
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e + 1, epochs),
                      "Loss: {:.4f}".format(running_loss / print_every))

                running_loss = 0

def check_accuracy_on_test(model, testloader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def save_checkpoint(model):
    model.class_to_idx = dataloaders['train'].dataset.class_to_idx
    checkpoint = {'input_size': [3, 224, 224],
                  'output_size': 102,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'num_epochs': 3}
    torch.save(checkpoint, 'checkpoint.pth')




# parser = argparse.ArgumentParser(description='Train image classifier')
# parser.add_argument('--save_dir', type=str,  required=True, help='Directory to save checkpoint')
# parser.add_argument('--arch', type=str, required=True, help='Architecture')
# parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
# parser.add_argument('--hidden_units')
# parser.add_argument('--epochs', type=int,  required=True, help='Number of epochs')
# parser.add_mutually_exclusive_group().add_argument('--gpu', action='store_true', help='Use GPU')
#
# args = parser.parse_args()

dataloaders = load.load_data('flowers')

model = models.densenet121(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 500)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(500, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

do_deep_learning(model, dataloaders['train'], 10, 100, criterion, optimizer, 'cuda')
check_accuracy_on_test(model, dataloaders['test'],'cuda')