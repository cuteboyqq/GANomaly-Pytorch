# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 13:51:45 2022

@author: User
"""
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from network import *

def test():
  
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
   
    
    IMAGE_SIZE_W, IMAGE_SIZE_H = 32,32
    TRAIN_DATA_DIR = "/home/ali/YOLOV5/runs/detect/f_384_2min/crops"
    VAL_DATA_DIR = TRAIN_DATA_DIR
    DEFEAT_DATA_DIR = "/home/ali/YOLOV5/runs/detect/f_384_2min/defeat"
    size = (IMAGE_SIZE_H,IMAGE_SIZE_W)



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
    
    
    # specify loss function
    criterion = nn.MSELoss()
    
    
    show_num = 0
    positive_loss, defeat_loss = [],[]
    print('Start test :')
    modelPath = r"/home/ali/AutoEncoder-Pytorch/model/AE_3_best_2.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    #model.eval()
    #model = ConvAutoencoder()
    model = NetG()
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
        outputs = model(images)
        gen_imag, latent_i, latent_o = outputs
        # calculate the loss
      
        loss_con = criterion(gen_imag, images)
        #loss_enc = criterion(z1, z2)
        loss_enc = criterion(latent_i, latent_o)
        loss = loss_enc + 50*loss_con
        positive_loss.append( (loss*IMAGE_SIZE_H*IMAGE_SIZE_W).detach().numpy())
        print('loss : {}'.format(loss*IMAGE_SIZE_H*IMAGE_SIZE_W))
        #print('finish AE')
        # prep images for display
        images = images.numpy()
        #print('images : \n {}'.format(images))
        # output is resized into a batch of iages
       
        outputs = gen_imag.view(BATCH_SIZE_VAL, 3, IMAGE_SIZE_H, IMAGE_SIZE_W)
      
        # use detach when it's an output that requires_grad
        outputs = outputs.detach().numpy()
        if SHOW_IMG:
            # plot the first ten input images and then reconstructed images
            fig, axes = plt.subplots(nrows=2, ncols=15, sharex=True, sharey=True, figsize=(25,4))
            
            # input images on top row, reconstructions on bottom
            for images, row in zip([images, outputs], axes):
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
        outputs = model(images)
        gen_imag, latent_i, latent_o = outputs
        # calculate the loss
     
        loss_con = criterion(gen_imag, images)
     
        loss_enc = criterion(latent_i, latent_o)
        loss = loss_enc + 50*loss_con
        defeat_loss.append( (loss*IMAGE_SIZE_H*IMAGE_SIZE_W).detach().numpy())
        print('loss : {}'.format(loss*IMAGE_SIZE_H*IMAGE_SIZE_W))
        #print('finish defeat AE')
        # prep images for display
        images = images.numpy()
        #print('images : \n {}'.format(images))
        # output is resized into a batch of iages
     
        output = gen_imag.view(BATCH_SIZE_VAL, 3, IMAGE_SIZE_H, IMAGE_SIZE_W)
     
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
        
if __name__=="__main__":
    test()