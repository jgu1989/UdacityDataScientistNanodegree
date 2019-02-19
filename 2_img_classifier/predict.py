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
    
#     model = models.densenet121()
    model = checkpoint['model']
#     model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint['class_to_idx']

def predict(image_path, model, topk, device, class_to_idx, cat_to_name):
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
    probs, idx = probs.topk(topk)
    idx_to_class = { v : k for k,v in class_to_idx.items()}
    top_class = [idx_to_class[x] for x in idx.tolist()[0]]
    names = [cat_to_name[k] for k in top_class]
    return probs.tolist()[0], names

parser = argparse.ArgumentParser(description='Predict the class')
parser.add_argument('--input', type=str, metavar='', default='flowers/test/28/image_05230.jpg', help='Path of image')
parser.add_argument('--chp', type=str, metavar='', default='checkpoint.pth', help='Path of checkpoint')
parser.add_argument('--top_k', type=int, metavar='', default=5, help='Top K most likely classes')
parser.add_argument('--category_names', type=str, metavar='', default='cat_to_name.json', help='Use a mapping of categories to real names')
parser.add_mutually_exclusive_group().add_argument('--gpu', action='store_true', help='Use GPU')

args = parser.parse_args()


# load model
model, class_to_idx = load_model(args.chp)
# print(model)

# make predictions
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
image_path = args.input
if args.gpu:
    device = 'cuda'
else:
    device = 'cpu'
print(predict(image_path, model, args.top_k, device, class_to_idx, cat_to_name))