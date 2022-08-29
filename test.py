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
from network import network
from util import loss
from util import plot
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

def get_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    #'/home/ali/datasets/train_video/NewYork_train/train/images'
    parser.add_argument('-noramldir','--normal-dir',help='image dir',default=r"/home/ali/YOLOV5/runs/detect/f_384_2min/crops")
    parser.add_argument('-abnoramldir','--abnormal-dir',help='image dir',default= r"/home/ali/YOLOV5/runs/detect/f_384_2min/defeat")
    parser.add_argument('-imgsize','--img-size',type=int,help='image size',default=64)
    parser.add_argument('-batchsize','--batch-size',type=int,help='train batch size',default=64)
    parser.add_argument('-savedir','--save-dir',help='save model dir',default="/home/ali/AutoEncoder-Pytorch/runs/train")
    parser.add_argument('-model','--model',help='model path',default= "/home/ali/AutoEncoder-Pytorch/runs/train/AE_3_best_2.pt")
    parser.add_argument('-viewimg','--view-img',action='store_true',help='view images')
    return parser.parse_args()    


def main():
    args = get_args()
    
    test(args.img_size,
        args.img_size,
        args.normal_dir,
        args.abnormal_dir,
        True, #args.view_img,
        args.model)

def test(IMAGE_SIZE_W=32,
         IMAGE_SIZE_H=32,
         VAL_DATA_DIR="/home/ali/YOLOV5/runs/detect/f_384_2min/crops",
         DEFEAT_DATA_DIR="/home/ali/YOLOV5/runs/detect/f_384_2min/defeat",
         SHOW_IMG=True,
         modelPath = r"/home/ali/AutoEncoder-Pytorch/model/AE_3_best_2.pt"
         ):
  
    if SHOW_IMG:
        BATCH_SIZE_VAL = 10
        SHOW_MAX_NUM = 5
        shuffle = True
    else:
        BATCH_SIZE_VAL = 1
        SHOW_MAX_NUM = 2000
        shuffle = False
    # convert data to torch.FloatTensor
   
    
    test_loader = data_loader(IMAGE_SIZE_H,
                              IMAGE_SIZE_W,
                              VAL_DATA_DIR,
                              BATCH_SIZE_VAL,
                              shuffle)
 
    defeat_loader = data_loader(IMAGE_SIZE_H,
                                IMAGE_SIZE_W,
                                DEFEAT_DATA_DIR,
                                BATCH_SIZE_VAL,
                                shuffle)
    # specify loss function
    criterion = nn.MSELoss()
    show_num = 0
    positive_loss, defeat_loss = [],[]
    print('Start test :') 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    #model.eval()
    #model = ConvAutoencoder()
    model = network.NetG(isize=IMAGE_SIZE_H, nc=3, nz=400, ngf=64, ndf=64, ngpu=1, extralayers=0)
    #model = torch.load(modelPath).to(device)
    model.load_state_dict(torch.load(modelPath))
    print('load model weight from {} success'.format(modelPath))
    print('VAL_DATA_DIR : {}'.format(VAL_DATA_DIR))
    
    positive_loss = infer(test_loader,SHOW_MAX_NUM,model,criterion,positive_loss,
            IMAGE_SIZE_H,IMAGE_SIZE_W,BATCH_SIZE_VAL,SHOW_IMG,'positive')
    
    defeat_loss = infer(defeat_loader,SHOW_MAX_NUM,model,criterion,defeat_loss,
            IMAGE_SIZE_H,IMAGE_SIZE_W,BATCH_SIZE_VAL,SHOW_IMG,'defect')
        
    if not SHOW_IMG: 
        plot.plot_loss_distribution(SHOW_MAX_NUM,positive_loss,defeat_loss)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def compute_loss(outputs,images,criterion):
    gen_imag, latent_i, latent_o = outputs
    loss_con = loss.l2_loss(images, gen_imag)
    loss_enc = loss.l1_loss(latent_i, latent_o)
    loss_sum = loss_enc + 50*loss_con
    return loss_sum




def infer(data_loader,
          SHOW_MAX_NUM,
          model,
          criterion,
          loss_list,
          IMAGE_SIZE_H,
          IMAGE_SIZE_W,
          BATCH_SIZE_VAL,
          SHOW_IMG,
          data_type
          ):
    show_num = 0
    dataiter = iter(data_loader)
    while(show_num < SHOW_MAX_NUM):
        images, labels = dataiter.next()
        print('{} Start {} AE:'.format(show_num,data_type))
        # get sample outputs
        outputs = model(images)
        gen_imag, latent_i, latent_o = outputs
        loss = compute_loss(outputs,images,criterion)
        loss_list.append(loss.detach().numpy())
        print('loss : {}'.format(loss))
        
        unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        images = unorm(images)
        outputs = unorm(outputs)
        
        images = images.numpy()
        outputs = gen_imag.view(BATCH_SIZE_VAL, 3, IMAGE_SIZE_H, IMAGE_SIZE_W)
        outputs = outputs.detach().numpy()
        if SHOW_IMG:
            plot.plot_images(images,outputs)             
        show_num+=1
    return loss_list

def data_loader(IMAGE_SIZE_H=32,
                IMAGE_SIZE_W=32,
                VAL_DATA_DIR=r'C:\factory_data\2022-08-26\f_384_2min\defeat',
                BATCH_SIZE_VAL=20,
                shuffle=True):
    size = (IMAGE_SIZE_H,IMAGE_SIZE_W)
    img_test_data = torchvision.datasets.ImageFolder(VAL_DATA_DIR,
                                                transform=transforms.Compose([
                                                    transforms.Resize(size),
                                                    #transforms.RandomHorizontalFlip(),
                                                    #transforms.Scale(64),
                                                    transforms.CenterCrop(size),
                                
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #GANomaly parameter
                                                    ])
                                                )

    print('img_test_data length : {}'.format(len(img_test_data)))

    test_loader = torch.utils.data.DataLoader(img_test_data, batch_size=BATCH_SIZE_VAL,shuffle=shuffle,drop_last=False)
    print('test_loader length : {}'.format(len(test_loader)))
    
    return test_loader


        
if __name__=="__main__":
    
    main()
    