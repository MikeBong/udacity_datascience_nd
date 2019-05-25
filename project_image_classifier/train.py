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

from functions import load_data, build_classifier, validation, train_model, test_model, save_model, load_checkpoint

parser = argparse.ArgumentParser(description='Train the neural network for image classification.')

parser.add_argument('--data_directory', action = 'store',
                    dest = 'data_directory', default = './flowers',
                    help = 'Enter path to access training data set.')

parser.add_argument('--arch', action='store',
                    dest = 'pretrained_model', default = 'vgg16',
                    help= 'Enter pretrained model to use; this classifier can currently work with\
                           VGG model. The default is VGG-16.')

parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_directory', default = 'checkpoint.pth',
                    help = 'Enter location to save model checkpoint for future use in.')

parser.add_argument('--learning_rate', action = 'store',
                    dest = 'l_rate', type=int, default = 0.001,
                    help = 'Enter learning rate used for training the model, default set to 0.001.')

parser.add_argument('--dropout', action = 'store',
                    dest='dropout_rate', type=int, default = 0.05,
                    help = 'Enter dropout rate for training the model, default set to 0.05.')

parser.add_argument('--hidden_units', action = 'store',
                    dest = 'hidden_units', type=int, default = 512,
                    help = 'Enter number of hidden units in classifier, default set to 512.')

parser.add_argument('--epochs', action = 'store',
                    dest = 'num_epochs', type=int, default = 3,
                    help = 'Enter number of epochs to use for training, default set to 3.')

parser.add_argument('--gpu', action = "store_true", default = True,
                    help = 'Turn GPU mode on or off, default set to "off".')

results = parser.parse_args()

data_dir = results.data_directory
save_dir = results.save_directory
learning_rate = results.l_rate
dropout_rate = results.dropout_rate
hidden_unit_count = results.hidden_units
epochs = results.num_epochs
gpu_mode = results.gpu

arch = results.pretrained_model

# Load and preprocess data 
train_loader, test_loader, validate_loader, train_data, test_data, validate_data = load_data(data_dir)

# Load pretrained model
pre_train_model = results.pretrained_model
model = getattr(models,pre_train_model)(pretrained=True)

# Build and attach new classifier
input_unit_count = model.classifier[0].in_features
build_classifier(model, input_unit_count, hidden_unit_count, dropout_rate)

# Using a NLLLoss as output is LogSoftmax
criterion = nn.NLLLoss()
                    
# Using Adam optimiser algorithm - uses concept of momentum to add fractions 
# to previous gradient descents to minimize loss function
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

# Train model
model, optimizer = train_model(model, epochs, train_loader, validate_loader, criterion, optimizer, gpu_mode)

# Test model
test_model(model, test_loader, gpu_mode)
                    
# Save model
save_model(model, pre_train_model, train_data, optimizer, save_dir, epochs)