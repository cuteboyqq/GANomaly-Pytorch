#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:59:04 2022

@author: ali
"""

from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from network.network import NetG, NetD, weights_init
from network.loss import l2_loss




class Ganomaly(nn.Module):
    """GANomaly Class
    """

    @property
    def name(self): return 'Ganomaly'
    

    
    def __init__(self,model_dir='/home/ali/AutoEncoder-Pytorch/runs/train/',batchsize=64):
        super(Ganomaly, self).__init__()
        
        self.batchsize = batchsize
        self.isize = 64
        self.lr = 2e-4
        self.beta1 = 0.5
        self.isTrain = True
        self.resume = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0
        

        ##
        # Create and initialize networks.
        self.netg = NetG().to(self.device)
        self.netd = NetD().to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        ##
        if self.resume != '':
            print("\nLoading pre-trained networks.")
            #self.iter = torch.load(os.path.join(self.resume, 'netG.pt'))['epoch']
            #self.netg.load_state_dict(torch.load(os.path.join(self.resume, 'netG.pt'))['state_dict'])
            #self.netd.load_state_dict(torch.load(os.path.join(self.resume, 'netD.pt'))['state_dict'])
            self.netg.load_state_dict(torch.load(os.path.join(self.resume, 'netG.pt')))
            self.netd.load_state_dict(torch.load(os.path.join(self.resume, 'netD.pt')))
            print("\tDone.\n")

        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.batchsize, 3, self.isize, self.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(self.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.batchsize, 3, self.isize, self.isize), dtype=torch.float32, device=self.device)
        self.real_label = torch.ones (size=(self.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.batchsize,), dtype=torch.float32, device=self.device)
        ##
        # Setup optimizer
        if self.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

    ##
    def forward_g(self,x):
        """ Forward propagate through netG
        """
        self.fake, self.latent_i, self.latent_o = self.netg(x)

    ##
    def forward_d(self,x):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(x)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())
    
    
    ##
    def backward_g(self,x):
        """ Backpropagate through netG
        """
        self.err_g_adv = self.l_adv(self.netd(x)[1], self.netd(self.fake)[1])
        self.err_g_con = self.l_con(self.fake, x)
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)
        self.err_g = self.err_g_adv * 1 + \
                     self.err_g_con * 50 + \
                     self.err_g_enc * 1
        self.err_g.backward(retain_graph=True)
        
        return self.err_g

    ##
    def backward_d(self):
        """ Backpropagate through netD
        """
        # Real - Fake Loss
        #print(len(self.pred_real))
        #print(len(self.real_label))
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()
        
        return self.err_d

    ##
    def reinit_d(self):
        """ Re-initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('   Reloading net d')
        
        
    
    
    
    def forward(self,x):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g(x)
        self.forward_d(x)

        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        error_g = self.backward_g(x)
        self.optimizer_g.step()

        # netd
        self.optimizer_d.zero_grad()
        error_d = self.backward_d()
        self.optimizer_d.step()
        if self.err_d.item() < 1e-5: self.reinit_d()
        
        return error_g, error_d, self.fake, self.netg, self.netd #error_d
    '''
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()
        epoch_iter = 0
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            # self.optimize()
            self.optimize_params()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes, fixed = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed)

        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))
        # self.visualizer.print_current_errors(self.epoch, errors)

    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_one_epoch()
            res = self.test()
            if res[self.opt.metric] > best_auc:
                best_auc = res[self.opt.metric]
                self.save_weights(self.epoch)
            self.visualizer.print_current_performance(res, best_auc)
        print(">> Training model %s.[Done]" % self.name)
        '''