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
from network import network
import os
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn

def main():
    IMAGE_SIZE_W, IMAGE_SIZE_H = 128,128
    TRAIN_DATA_DIR = r"C:\factory_data\2022-08-26\f_384_2min\crops"
    BATCH_SIZE = 64
    SAVE_MODEL_DIR = r"C:\GitHub_Code\AE\AutoEncoder-Pytorch\runs\train"
    n_epochs = 20
    train(IMAGE_SIZE_H,
          IMAGE_SIZE_W,
          TRAIN_DATA_DIR,
          BATCH_SIZE,
          SAVE_MODEL_DIR,
          n_epochs)

def train(IMAGE_SIZE_H = 32,
          IMAGE_SIZE_W = 32,
          TRAIN_DATA_DIR = "/home/ali/YOLOV5/runs/detect/f_384_2min/crops",
          BATCH_SIZE = 64,
          SAVE_MODEL_DIR = r"/home/ali/AutoEncoder-Pytorch/model",
          n_epochs = 20
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
    '''load model'''
    model = network.NetG(isize=IMAGE_SIZE_H, nc=3, nz=100, ngf=64, ndf=64, ngpu=1, extralayers=0).to(device)
    print(model)
    print('IMAGE_SIZE_H:{}\n IMAGE_SIZE_W:{}\n TRAIN_DATA_DIR:{}\n BATCH_SIZE:{}\n SAVE_MODEL_DIR:{}\n n_epochs:{}\n'.format(IMAGE_SIZE_H,
                                                                                                                             IMAGE_SIZE_W,
                                                                                                                             TRAIN_DATA_DIR,
                                                                                                                             BATCH_SIZE,
                                                                                                                             SAVE_MODEL_DIR,
                                                                                                                             n_epochs))
    ''' set loss function '''
    criterion = nn.MSELoss()
    ''' set optimizer function '''
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    _lowest_loss = 100.0
    
    if not os.path.exists(SAVE_MODEL_DIR):
        os.makedirs(SAVE_MODEL_DIR)
        
    SAVE_MODEL_PATH = os.path.join(SAVE_MODEL_DIR,"AE_3_best_2.pt")
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0   
        for images, _  in train_loader: 
            images = images.to(device)
            '''initial optimizer'''
            optimizer.zero_grad()
            '''inference'''
            outputs = model(images)
            ''' compute loss '''
            loss = compute_loss(outputs,images,criterion)
            ''' loss back propagation '''
            loss.backward()
            ''' optimize weight & bias '''
            optimizer.step()
            ''' sum loss '''
            train_loss += loss.item()*images.size(0)
    
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        
        if train_loss < _lowest_loss:
            _lowest_loss = train_loss
            print('Start save model !')
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print('save model weights complete with loss : %.3f' %(train_loss))
            
def compute_loss(outputs,images,criterion):
    gen_imag, latent_i, latent_o = outputs
    loss_con = criterion(gen_imag, images)
    loss_enc = criterion(latent_i, latent_o)
    loss = loss_enc + 50*loss_con
    return loss
    
if __name__=="__main__":
    main()