# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 21:48:01 2022

@author: User
"""
# The MNIST datasets are hosted on yann.lecun.com that has moved under CloudFlare protection
# Run this script to enable the datasets download
# Reference: https://github.com/pytorch/vision/issues/1938
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
TRAIN = False
TEST = True
SHOW_IMG = False
if SHOW_IMG:
    BATCH_SIZE_VAL = 20
    SHOW_MAX_NUM = 4
    shuffle = True
else:
    BATCH_SIZE_VAL = 1
    SHOW_MAX_NUM = 800
    shuffle = False
# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# load the training and test datasets
'''
train_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=False,
                                  download=True, transform=transform)
'''


IMAGE_SIZE_W, IMAGE_SIZE_H = 128,128
TRAIN_DATA_DIR = "/home/ali/YOLOV5/runs/detect/f_384_2min/crops_left"
VAL_DATA_DIR = TRAIN_DATA_DIR
DEFEAT_DATA_DIR = "/home/ali/YOLOV5/runs/detect/f_384_2min/defeat"
BATCH_SIZE = 32
size = (IMAGE_SIZE_H,IMAGE_SIZE_W)
img_data = torchvision.datasets.ImageFolder(TRAIN_DATA_DIR,
                                            transform=transforms.Compose([
                                                transforms.Resize(size),
                                                #transforms.RandomHorizontalFlip(),
                                                #transforms.Scale(64),
                                                transforms.CenterCrop(size),
                                             
                                                transforms.ToTensor()
                                                ])
                                            )

train_loader = torch.utils.data.DataLoader(img_data, batch_size=BATCH_SIZE,shuffle=True,drop_last=False)
print('train_loader length : {}'.format(len(train_loader)))


#size = (IMAGE_SIZE,IMAGE_SIZE)
img_test_data = torchvision.datasets.ImageFolder(VAL_DATA_DIR,
                                            transform=transforms.Compose([
                                                transforms.Resize(size),
                                                #transforms.RandomHorizontalFlip(),
                                                #transforms.Scale(64),
                                                transforms.CenterCrop(size),
                                                transforms.ToTensor()
                                                ])
                                            )

print('img_test_data length : {}'.format(len(img_test_data)))

test_loader = torch.utils.data.DataLoader(img_test_data, batch_size=BATCH_SIZE_VAL,shuffle=shuffle,drop_last=False)
print('test_loader length : {}'.format(len(test_loader)))

# Create training and test dataloaders

img_defeat_data = torchvision.datasets.ImageFolder(DEFEAT_DATA_DIR,
                                            transform=transforms.Compose([
                                                transforms.Resize(size),
                                                #transforms.RandomHorizontalFlip(),
                                                #transforms.Scale(64),
                                                transforms.CenterCrop(size),
                                                transforms.ToTensor()
                                                ])
                                            )

print('img_defeat_data length : {}'.format(len(img_defeat_data)))

#BATCH_SIZE_VAL = 1
defeat_loader = torch.utils.data.DataLoader(img_defeat_data, batch_size=BATCH_SIZE_VAL,shuffle=shuffle,drop_last=False)
print('defeat_loader length : {}'.format(len(defeat_loader)))



num_workers = 0
# how many samples per batch to load
#batch_size = 20

# prepare data loaders
#train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

import matplotlib.pyplot as plt
#%matplotlib inline
    
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# get one image from the batch
img = np.squeeze(images[0])

#fig = plt.figure(figsize = (5,5)) 
#ax = fig.add_subplot(111)
#ax.imshow(img)
#ax.imshow(img, cmap='gray')

import torch.nn as nn
import torch.nn.functional as F

'''
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
                dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, 
                   return_indices=False, ceil_mode=False)

torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
                         output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
'''
'''
# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)


    def forward(self, x):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
                
        return x
    '''
# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d( 4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(16,  3, 2, stride=2)


    def forward(self, x):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # compressed representation
        x = F.relu(self.conv4(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.sigmoid(self.t_conv4(x))
                
        return x
#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# initialize the NN
model = ConvAutoencoder().to(device)
print(model)

# specify loss function
criterion = nn.MSELoss()

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if TRAIN:
    # number of epochs to train the model
    n_epochs = 5
    _lowest_loss = 100.0
    import os
    SAVE_MODEL_DIR = r"/home/ali/AutoEncoder-Pytorch/model"
    if not os.path.exists(SAVE_MODEL_DIR):
        os.makedirs(SAVE_MODEL_DIR)
        
    SAVE_MODEL_PATH = os.path.join(SAVE_MODEL_DIR,"AE_3_best_2.pt")
    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for images, _  in train_loader:
            # _ stands in for labels, here
            # no need to flatten images
            images = images.to(device)
            #images, _ = data
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            # calculate the loss
            loss = criterion(outputs, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*images.size(0)
                
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))
        
        if train_loss < _lowest_loss:
            save_model = epoch+1
            _lowest_loss = train_loss
            print('Start save model !')
            
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print('save model weights complete with loss : %.3f' %(train_loss))
            
#BATCH_SIZE_VAL = 20


if TEST:
    show_num = 0
    positive_loss, defeat_loss = [],[]
    print('Start test :')
    modelPath = r"/home/ali/AutoEncoder-Pytorch/model/AE_3_best_2.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    #model.eval()
    model = ConvAutoencoder()
    #model = torch.load(modelPath).to(device)
    model.load_state_dict(torch.load(modelPath))
    print('load model weight from {} success'.format(modelPath))
    print('VAL_DATA_DIR : {}'.format(VAL_DATA_DIR))
    # obtain one batch of test images
    dataiter = iter(test_loader)
    while(show_num < SHOW_MAX_NUM):
        images, labels = dataiter.next()
        print('{} Start positvie AE:'.format(show_num))
        # get sample outputs
        output = model(images)
        loss = criterion(output, images)
        positive_loss.append( (loss*IMAGE_SIZE_H*IMAGE_SIZE_W).detach().numpy())
        print('loss : {}'.format(loss*IMAGE_SIZE_H*IMAGE_SIZE_W))
        #print('finish AE')
        # prep images for display
        images = images.numpy()
        #print('images : \n {}'.format(images))
        # output is resized into a batch of iages
        output = output.view(BATCH_SIZE_VAL, 3, IMAGE_SIZE_H, IMAGE_SIZE_W)
        #output = output.view(BATCH_SIZE, 3, 28, 28)
        # use detach when it's an output that requires_grad
        output = output.detach().numpy()
        if SHOW_IMG:
            # plot the first ten input images and then reconstructed images
            fig, axes = plt.subplots(nrows=2, ncols=15, sharex=True, sharey=True, figsize=(25,4))
            
            # input images on top row, reconstructions on bottom
            for images, row in zip([images, output], axes):
                #print(len(images))
                #print(len(row))
                for img, ax in zip(images, row):
                    #print(img)
                    #print(np.shape(img))
                    #print(np.shape(np.squeeze(img)))
                    #img = img[-1::]
                    img = img[:,:,::-1].transpose((2,1,0))
                    #print(np.shape(img))
                    #print(np.shape(np.squeeze(img)))
                    #ax.imshow(np.squeeze(img), cmap='gray')
                    ax.imshow(img)
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
           
        show_num+=1
    show_num = 0
    dataiter = iter(defeat_loader)
    while(show_num < SHOW_MAX_NUM):
        images, labels = dataiter.next()
        print('{} Start defeat AE:'.format(show_num))
        # get sample outputs
        output = model(images)
        loss = criterion(output, images)
        defeat_loss.append( (loss*IMAGE_SIZE_H*IMAGE_SIZE_W).detach().numpy())
        print('loss : {}'.format(loss*IMAGE_SIZE_H*IMAGE_SIZE_W))
        #print('finish defeat AE')
        # prep images for display
        images = images.numpy()
        #print('images : \n {}'.format(images))
        # output is resized into a batch of iages
        output = output.view(BATCH_SIZE_VAL, 3, IMAGE_SIZE_H, IMAGE_SIZE_W)
        #output = output.view(BATCH_SIZE, 3, 28, 28)
        # use detach when it's an output that requires_grad
        output = output.detach().numpy()
        if SHOW_IMG:
            # plot the first ten input images and then reconstructed images
            fig, axes = plt.subplots(nrows=2, ncols=15, sharex=True, sharey=True, figsize=(100,16))
            
            # input images on top row, reconstructions on bottom
            for images, row in zip([images, output], axes):
           
                for img, ax in zip(images, row):
         
                    img = img[:,:,::-1].transpose((2,1,0))
               
                    ax.imshow(img)
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
        
        show_num+=1
        
    if not SHOW_IMG: 
        # Importing packages
        import matplotlib.pyplot as plt2
        
        # Define data values
        x = [i for i in range(SHOW_MAX_NUM)]
        y = positive_loss
        z = defeat_loss
        print(x)
        print(positive_loss)
        print(defeat_loss)
        # Plot a simple line chart
        #plt2.plot(x, y)
        
        # Plot another line on the same chart/graph
        #plt2.plot(x, z)
        
        plt2.scatter(x,y)
        plt2.scatter(x,z) 
        plt2.show()