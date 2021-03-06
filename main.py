#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import configparser
import logging
import os

import torch
import torch.optim as optim

import datasets
import trainer
from model import NeuralProcess

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

if __name__ == "__main__":

    # prepare figure directory
    save_img_path = './fig'
    os.makedirs(save_img_path) if not os.path.isdir(save_img_path) else None            
            
    # configurations
    logging.basicConfig(format='[NP] %(levelname)s: %(message)s', level=logging.INFO)
    cfg = configparser.ConfigParser()
    cfg.read('config.ini')
    train_cfg = dict(zip([key for key, _ in cfg.items('train')], \
                         [int(val) if val.isdigit() else val for _, val in cfg.items('train')]))
    print('config:', train_cfg)
    # parameters from config
    context_size = train_cfg['context_size']
    x_dim = train_cfg['x_dim']
    h_dim = train_cfg['h_dim']
    r_dim = train_cfg['r_dim']
    z_dim = train_cfg['z_dim']
    y_dim = train_cfg['y_dim']
    batch_size = train_cfg['batch_size']
    n_iter = train_cfg['n_iter']
    n_epoch = train_cfg['n_epoch']
    n_display = train_cfg['n_display'] 
    
    device = torch.device('cuda') if torch.cuda.device_count() > 0 else torch.device('cpu')
    
    # load dataloader
    data_loader = datasets.load_mnist(batch_size=batch_size)
    # data_loader = datasets.load_celeba(batch_size=batch_size)
    
    model = NeuralProcess(x_dim=x_dim, h_dim=h_dim, r_dim=r_dim, z_dim=z_dim, y_dim=y_dim, device=device)
    optimizer = optim.Adam(model.parameters(), lr=4e-3)
    # print(model)
    
    ModelTrainer = trainer.NPTrainer(model=model, context_size=context_size, optimizer=optimizer, device=device)
    ModelTrainer.train(data_loader=data_loader, n_epoch=n_epoch, n_iter=n_iter, test_for_every=n_display)

    # import sys; sys.exit(0)
