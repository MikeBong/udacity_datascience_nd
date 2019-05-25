# Import packages
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

#--------------------------------------------------------------------------------------------------
# Function to load data, perform initial processing on images
def load_data(data_dir):
    # TODO: Define your transforms for the training, validation, and testing sets
    # Split data transforms into 3: train_transforms, test_transforms, validate_transforms

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    validate_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    
    # TODO: Load the datasets with ImageFolder
    # Separate datasets by use: train, test, validate
#     data_dir = './flowers'
    if data_dir == None:
        data_dir = './flowers'
    else:
        data_dir
        
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'    
    
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    validate_data = datasets.ImageFolder(valid_dir, transform = validate_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # Separate dataloaders by use: train, test, validate

    # Note: Set shuffle=True to ensure order of images does not impact model

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle = True)
    validate_loader = torch.utils.data.DataLoader(validate_data, batch_size = 32, shuffle = True)
    
    return train_loader, test_loader, validate_loader, train_data, test_data, validate_data

#--------------------------------------------------------------------------------------------------
# Function: Transform/process image
# def process_image(image):
#     ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
#         returns an Numpy array
#     '''
    
#     # TODO: Process a PIL image for use in a PyTorch model
    
#     # Use PIL to load image
#     pil_image = Image.open(image)
    
#     # Define transformation to the loaded image 
#     # - resize to 256 pixels
#     # - center crop 224x224
#     # - normalise using means[0.485, 0.456, 0.406] and std devs[0.229, 0.224, 0.225]
#     transform = transforms.Compose([transforms.Resize(256),
#                                     transforms.CenterCrop(224),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize([0.485, 0.456, 0.406],
#                                                          [0.229, 0.224, 0.225])])
    
    # Apply transformation to loaded image
#     pil_image_transformed = transform(pil_image)
    
    
#     return pil_image_transformed 

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Use PIL to load image
    pil_image = Image.open(image)

    # Required transformation to the loaded image 
    # - resize to 256 pixels
    # - center crop 224x224
    # - normalise using means[0.485, 0.456, 0.406] and std devs[0.229, 0.224, 0.225]     

    # Resize to targeted size    
    width, height = pil_image.size
    size = 256
    shortest_side = min(width, height)
    pil_image_resized = pil_image.resize((int((width/shortest_side)*size) , int((height/shortest_side)*size)))

    # Perform a center crop
    cent_crop_size = 224
    cent_crop_value = 0.5 * (size - cent_crop_size)                                                                      
    pil_image_cropped =pil_image_resized.crop((cent_crop_value,
                                              cent_crop_value,
                                              size-cent_crop_value,
                                              size-cent_crop_value))                                         
    # Convert to array
    pil_image_array = np.array(pil_image_cropped)
    pil_image_np = pil_image_array/255
    
    # Normalisation of image                                         
    normalize_mean = np.array([0.485, 0.456, 0.406])
    normalize_std = np.array([0.229, 0.224, 0.225])                                    
    pil_image_np_normalized = (pil_image_np - normalize_mean) / normalize_std
                                         
    # Apply transformation to loaded image
    transform = transforms.Compose([transforms.ToTensor()])
    pil_image_transformed = transform(pil_image_np_normalized)    
    
    return pil_image_transformed
#--------------------------------------------------------------------------------------------------
# Function: Build a new classifier model
def build_classifier(model, input_unit_count, hidden_unit_count, dropout_rate):
    # Freeze weights of pretrained model, to prevent us from backproppagating through/updating them
    for param in model.parameters():
        param.requires_grad = False

    # Specify the new model classifier
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_unit_count, hidden_unit_count)),
                                            ('relu', nn.ReLU()),
                                            ('dropout1', nn.Dropout(dropout_rate)), 
                                            ('fc2', nn.Linear(hidden_unit_count, 102)),
                                            ('output', nn.LogSoftmax(dim=1))
                                           ]))

    # Replace the model classifier
    model.classifier = classifier
    return model

#--------------------------------------------------------------------------------------------------
# Function: Used to validate model
def validation(model, validate_loader, criterion, gpu_mode):
    valid_loss = 0
    accuracy = 0
    
    if gpu_mode == True:
    # change model to work with cuda
        model.to('cuda')
    else:
        pass
    # Iterate over data from validloader
    for ii, (images, labels) in enumerate(validate_loader):
        
        if gpu_mode == True:
        # Change images and labels to work with cuda
            images, labels = images.to('cuda'), labels.to('cuda')
        else:
            pass
        
        # Forward pass image though model for prediction
        output = model.forward(images)
        # Calculate loss
        valid_loss += criterion(output, labels).item()
        # Calculate probability
        ps = torch.exp(output)
        
        # Calculate accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

#--------------------------------------------------------------------------------------------------
# Function: Used to validate model
def train_model(model, epochs, train_loader, validate_loader, criterion, optimizer, gpu_mode):
    print('Model training commenced -->')
    print('-----------------------------------')
#     epochs = 5
    steps = 0
    print_every_x_step = 5

    if gpu_mode == True:
    # change to cuda
        model.to('cuda')
    else:
        pass
    
    for e in range(epochs):
    #     since = time.time()
        running_loss = 0

        # Iterating over data to carry out training step
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1

            if gpu_mode == True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                pass

            # zeroing parameter gradients
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Carrying out validation step
            if steps % print_every_x_step == 0:
                # setting model to evaluation mode during validation
                model.eval()

                # Gradients are turned off as no longer in training
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validate_loader, criterion, gpu_mode)

                print("Epoch: {}/{}, Step: {} -->".format(e+1, epochs, steps))
                print("Training Loss: {}".format(round(running_loss/print_every_x_step,3)))
                print("Valid Loss: {}".format(round(valid_loss/len(validate_loader),3)))
                print("Valid Accuracy: {}".format(round(float(accuracy/len(validate_loader)),3)))
                print("")

                running_loss = 0

                # Turning training back on
                model.train()

    return model, optimizer

#--------------------------------------------------------------------------------------------------
# Function: Used to test model
def test_model(model, test_loader, gpu_mode):
    correct = 0
    total = 0
    
    if gpu_mode == True:
        model.to('cuda')
    else:
        pass

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            
            if gpu_mode == True:
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                pass
            
            # Calculate probabilities of what label fits the image 
            outputs = model(images)
            
            # Convert probabilities into actual predicted label
            _, predicted = torch.max(outputs.data, 1)
            
            # Count total number of images
            total += labels.size(0)
            
            # Count instances where predicted label == correct label
            correct += (predicted == labels).sum().item()
    print('-----------------------------------')        
    print('Model training complete - testing complete -->')
    print('Model accuracy - applied to test images: %d %%' % (100 * correct / total))
    print('')
#--------------------------------------------------------------------------------------------------
# Function: Used to save model
def save_model(model, model_arch, train_data, optimizer, save_dir, epochs):
   
    model.class_to_idx = train_data.class_to_idx
    
    # Define checkpoint features to be saved
    checkpoint = {'classifier': model.classifier,
                  'opt_state':optimizer.state_dict,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx,
                 'num_epochs': epochs,
                 'arch': model_arch}
    
    if save_dir == None:
        save_dir = './checkpoint.pth'
    else:
        save_dir
    # Saving the checkpoint
    print('Model saved as {}'.format(save_dir))
    print('')
    
    return torch.save(checkpoint, save_dir)

#--------------------------------------------------------------------------------------------------
# Function: Used to load model (checkpoint)
def load_checkpoint(save_dir, gpu_mode):
    # Loan the checkpoint
    if gpu_mode == True:
        checkpoint = torch.load(save_dir)
    else:
        checkpoint = torch.load(save_dir, map_location=lambda storage, loc: storage)
    
    # Rebuild the model    
    model = getattr(models,checkpoint['arch'])(pretrained=True)
    
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    print('Model loaded from {}'.format(save_dir))
    print('')
    
    return model

#--------------------------------------------------------------------------------------------------
# Function: Used to perform class prediction
def predict(processed_image, loaded_model, topk, gpu_mode):   
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    # Setting model to evaluation mode and turning off gradients
    loaded_model.eval()
    
    if gpu_mode == True:
        loaded_model.to('cuda')
    else:
        loaded_model.cpu()
  
    processed_image_unsqueezed = processed_image.unsqueeze_(0)
    processed_image_torch = processed_image_unsqueezed.float()

    # Feeding the input image through the model
    with torch.no_grad():
        output_result = loaded_model.forward(processed_image_torch)

    # Calc probabilities
    probs_raw = torch.exp(output_result)
    probs_topk = probs_raw.topk(topk)[0]
    index_topk = probs_raw.topk(topk)[1]

    # Converting probabilities and outputs to lists
    probs_topk_list = np.array(probs_topk)[0]
    index_topk_list = np.array(index_topk[0])    
    
    # Loading the class to index, then inverting into index to class
    class_to_idx = loaded_model.class_to_idx
    idx_to_class = {x: y for y, x in class_to_idx.items()}        

    # Converting index list to class list
    classes_topk_list = []
    for index in index_topk_list:
        classes_topk_list += [idx_to_class[index]]
        
    return probs_topk_list, classes_topk_list
#     return probs_topk_list, np.array(classes_topk_list[0])
#--------------------------------------------------------------------------------------------------





