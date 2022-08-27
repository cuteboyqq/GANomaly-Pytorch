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
import os
import torchvision.transforms as transforms
import torchvision


def main():
    IMAGE_SIZE_W, IMAGE_SIZE_H = 32,32
    TRAIN_DATA_DIR = "/home/ali/YOLOV5/runs/detect/f_384_2min/crops"
    BATCH_SIZE = 64
    
    train(IMAGE_SIZE_H,
              IMAGE_SIZE_W,
              TRAIN_DATA_DIR,
              BATCH_SIZE)

def train(IMAGE_SIZE_H = 32,
          IMAGE_SIZE_W = 32,
          TRAIN_DATA_DIR = "/home/ali/YOLOV5/runs/detect/f_384_2min/crops",
          BATCH_SIZE = 64
          ):
    
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
    
    
    
    ''' use gpu if available'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NetG().to(device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    _lowest_loss = 100.0
    SAVE_MODEL_DIR = r"/home/ali/AutoEncoder-Pytorch/model"
    if not os.path.exists(SAVE_MODEL_DIR):
        os.makedirs(SAVE_MODEL_DIR)
        
    SAVE_MODEL_PATH = os.path.join(SAVE_MODEL_DIR,"AE_3_best_2.pt")
    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0   
        
        for images, _  in train_loader: 
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            gen_imag, latent_i, latent_o = outputs
            ''' calculate the loss'''
            loss_con = criterion(gen_imag, images)
            loss_enc = criterion(latent_i, latent_o)
            loss = loss_enc + 50*loss_con
            '''backward pass: compute gradient of the loss with respect to model parameters'''
            loss.backward()
            '''perform a single optimization step (parameter update)'''
            optimizer.step()
            '''update running training loss'''
            train_loss += loss.item()*images.size(0)
                
            
        '''print avg training statistics''' 
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        
        if train_loss < _lowest_loss:
            _lowest_loss = train_loss
            print('Start save model !')
            
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print('save model weights complete with loss : %.3f' %(train_loss))
            
if __name__=="__main__":
    main()