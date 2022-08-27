# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 13:45:49 2022

@author: User
"""
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import torch
from network import *



def train():
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # initialize the NN
    #model = ConvAutoencoder().to(device)
    model = NetG().to(device)
    print(model)

    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # number of epochs to train the model
    
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
            gen_imag, latent_i, latent_o = outputs
            # calculate the loss
            loss_con = criterion(gen_imag, images)
            loss_enc = criterion(latent_i, latent_o)
            loss = loss_enc + 50*loss_con
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
            
if __name__=="__main__":
    train()