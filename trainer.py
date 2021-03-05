import logging
import os
import gc

import torch
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm, trange
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dotmap import DotMap

import numpy as np

logger = logging.getLogger(__name__)

from model import NeuralProcess
import utils
import viz_utils

class NPTrainer(object):
    def __init__(self, model, context_size, optimizer, device='cuda'):
        self.model = model
        self.context_size = context_size
        self.device = device
        self.optimizer = optimizer
        self.epoch_losses = []
        
        print('running on:', self.device)
        
    def train(self, data_loader, n_epoch, n_iter, test_for_every):
        gc.collect()
        torch.cuda.empty_cache()
        tsne = TSNE()
        fig = plt.figure()
        ax = fig.add_subplot()
        # colors = cm.rainbow(np.linspace(0,1,len(data_loader)))
        for epoch in range(n_epoch):
            epoch_loss = 0.
            t = tqdm(data_loader, total=len(data_loader))
            for batch_idx, data in enumerate(t):
                t.set_description('batch %d/%d' % (batch_idx,len(data_loader)))
                
                # image statistics
                batch_img, batch_label = data # [batch_size, 1, 28, 28]
                batch_size, num_channel, height, width = batch_img.size()
                n_pixels = height * width
                
                # randomly select context and target indices
                indices = torch.randperm(n_pixels)
                context_indices = indices[:self.context_size]
                target_indices = indices[self.context_size:]

                # prepare empty grid and pixel vector for the image
                all_x = torch.stack([torch.Tensor([i, j]) for i in range(height) for j in range(width)]).repeat(batch_size, 1, 1)
                all_y = batch_img.view(batch_size, -1)

                # split into context and target
                context_x, context_y = all_x[:, context_indices], all_y[:, context_indices]
                target_x, target_y = all_x[:, target_indices], all_y[:, target_indices]

                # upload to device
                self.model = self.model.to(self.device)
                batch_img = batch_img.to(self.device)
                context_x = context_x.to(self.device)
                context_y = context_y.to(self.device)
                target_x = target_x.to(self.device)
                target_y = target_y.to(self.device)
                
                context_y = context_y.unsqueeze(2)
                target_y = target_y.unsqueeze(2)
                query = (context_x, context_y, target_x)

                # training
                # ** need to iterate for ~1000 for performance
                t_iter = tqdm(range(n_iter), total=n_iter)
                for iter in t_iter:
                    self.optimizer.zero_grad()
                    self.model.train()
                    
                    context_mu, logits, p_y_pred, q_target, q_context, loss = self.model(query, target_y)                
                    context_pred_logits, target_pred_logits = logits['context'], logits['target']
                    # logger.info('epoch: {:3d} | batch: {:3d} | iter: {:4d} | loss: {:6.3f}'.format(epoch, batch_idx, iter, loss))
                    
                    try:
                        loss.backward()
                        self.optimizer.step()
                        epoch_loss += loss.item()
                        # print(' [train] epoch loss: %.3f' % (epoch_loss/int(iter+1)))
                    except ValueError as ve:
                        print('[ERROR] (train mode, but test mode loss output) loss error:', loss)
                        
                    # update progress bar
                    t_iter.set_description('iter: %d/%d / loss: %.3f' % (iter,n_iter,(epoch_loss/int(iter+1))))
                    
                    ''' evaluate '''
                    if iter % test_for_every == 0: # train mode
                        # print('  image generated at iteration', iter)
                        self.model.eval()
                        context_mu, logits, _, _, _, _ = self.model(query)                
                        context_pred_logits, target_pred_logits = logits['context'], logits['target']
                        
                        ''' visualize latent vector by classes '''
                        params = DotMap()
                        params.batch_size = batch_size
                        params.epoch = epoch
                        params.iter = iter
                        params.save_path = f'./fig/latent/latent (batch: {batch_size}, r: {self.model.r_dim}, z: {self.model.z_dim})'
                        # params.save_path = f'./fig/latent/latent_epoch_{epoch}_iter_{iter}.png'
                        # params.save_path = f'./fig/latent/latent_epoch_0_bsz300.png'
                        # params.save_path = f'./fig/latent/_latent_bsz_{batch_size}_r_{self.model.r_dim}_z_{self.model.z_dim}.png'
                        viz_utils.visualize_latent(tsne, context_mu, batch_label, ax, params)
                        
                        ''' visualize logit images '''
                        images = torch.tensor([], device='cpu')
                        for i in range(batch_size):
                            index = torch.tensor([i]).to(self.device)
                            target_pred_logits_i = torch.index_select(target_pred_logits, 0, index)
                            context_pred_logits_i = torch.index_select(context_pred_logits, 0, index)
                            target_y_i = torch.index_select(target_y.squeeze(), 0, index)
                            context_y_i = torch.index_select(context_y.squeeze(), 0, index)
                            
                            # target image
                            target_img = torch.zeros(n_pixels, device=self.device)
                            target_img[target_indices] = target_y_i.squeeze()
                            target_img[context_indices] = context_y_i.squeeze()
                            target_img = target_img.view(height, width)                            

                            # context image
                            context_img = torch.zeros(n_pixels, device=self.device)
                            context_img[context_indices] = context_y_i.squeeze()
                            context_img = context_img.view(height, width)

                            # clip for single-channel MNIST
                            target_pred_logits_i = utils.clip_tensor(target_pred_logits_i, min=0, max=1)
                            context_pred_logits_i = utils.clip_tensor(context_pred_logits_i, min=0, max=1)
                            
                            # predicted context and targets overlayed together
                            predicted_img = torch.zeros(n_pixels, device=self.device)
                            predicted_img[target_indices] = target_pred_logits_i.squeeze()
                            predicted_img[context_indices] = context_pred_logits_i.squeeze()
                            # predicted_img[context_indices] = context_y_i.squeeze()
                            predicted_img = predicted_img.view(height, width)
                            
                            img_to_view = torch.cat([target_img, context_img, predicted_img], dim=0)
                            images = torch.cat([images, img_to_view.cpu()], dim=1)
                            
                            vutils.save_image(images, f'fig/img_epoch_{epoch}.png', normalize=True)
                
