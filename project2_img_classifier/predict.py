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
from collections import OrderedDict
from process import process_image
import json



def load_model(path):
    checkpoint = torch.load(path)
#     checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model = models.densenet121()
    print(model)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1024, 500)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(500, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint['class_to_idx']

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    im = torch.FloatTensor(process_image(image_path)).unsqueeze(0)
    im = im.to(device)
    #     print(im)
    #     with torch.no_grad():
    #         outputs = model(im)
    model.eval()
    outputs = model.forward(im)
    #     print(outputs.data)
    probs = torch.nn.functional.softmax(outputs.data)
    return probs.topk(topk)

def plot(image_path, cat_to_name, device = 'cpu'):
    # image_path = 'flowers/test/28/image_05230.jpg'
    # image_path = 'flowers/test/1/image_06743.jpg'

    im = Image.open(image_path)
    probs, idx = predict(image_path, model, 5, device)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    top_class = [idx_to_class[x] for x in idx.tolist()[0]]
    names = [cat_to_name[k] for k in top_class]
    print(names)
    print(probs)

    f, (ax1, ax2) = plt.subplots(figsize=(8, 8), ncols=1, nrows=2)
    ax1.imshow(im)
    # ax1.imshow(image_path)
    ax2.barh(names, probs.tolist()[0])

parser = argparse.ArgumentParser(description='Predict the class')
parser.add_argument('input', type=str, metavar='', help='Path of image')
parser.add_argument('chp', type=str, metavar='', help='Path of checkpoint')
parser.add_argument('--top_k', type=int, metavar='', required=True, help='Top K most likely classes')
parser.add_argument('--category_names', type=str, metavar='', help='Use a mapping of categories to real names')
parser.add_argument('--gpu', metavar='', help='Use GPU')

args = parser.parse_args()


# load model
model, class_to_idx = load_model('checkpoint.pth')


# make predictions
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
image_path = 'flowers/test/28/image_05230.jpg'
plot(image_path, cat_to_name, 'cuda')