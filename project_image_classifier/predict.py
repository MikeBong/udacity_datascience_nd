# Import packages
import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import time
import json
from collections import OrderedDict

from functions import load_data, process_image, load_checkpoint, predict, test_model 


parser = argparse.ArgumentParser(description='Use neural network model to predict classification of input image.')

parser.add_argument('--image_path', action='store',
                    default = './flowers/test/8/image_03291.jpg',
                    help='Enter path (with ".jpg") where target image is stored.')

parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_directory', default = 'checkpoint.pth',
                    help = 'Enter location to save model checkpoint for future use in.')

parser.add_argument('--top_k', action='store',
                    dest='topk', type=int, default = 5,
                    help='Enter top most likely classes to view, default set to 5.')

parser.add_argument('--cat_to_name', action='store',
                    dest='cat_name_dir', default = 'cat_to_name.json',
                    help='Enter name of integer to category mapping file.')

parser.add_argument('--gpu', action="store_true", default=True,
                    help='Turn GPU mode on or off, default set to off.')

results = parser.parse_args()

save_dir = results.save_directory
image = results.image_path
topk = results.topk
gpu_mode = results.gpu
cat_names = results.cat_name_dir

# Generate dictionary mapping the integer encoded categories to the actual flower names
with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

# Load model
loaded_model = load_checkpoint(save_dir, gpu_mode)

# Preprocess image - assumes jpeg format
processed_image = process_image(image)

if gpu_mode == True:
    processed_image = processed_image.to('cuda')
else:
    pass

# Carry out prediction
probs, classes = predict(processed_image, loaded_model, topk, gpu_mode)

# Converting classes to names
classes_names = []
for i in classes:
    classes_names += [cat_to_name[i]]

# Print results   
print('Prediction detailed results -->')
print('-----------------------------------')
print('Probabilities: {}'.format(probs))
print('Class number: {}'.format(classes))
print('Class name: {}'.format(classes_names))
print('-----------------------------------')
print('')

# Print name of predicted flower with highest probability
print(f"This flower has the highest likelihood of being a: '{classes_names[0]}' at a probability of {round(probs[0]*100,4)}% ")
print('')