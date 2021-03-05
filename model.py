#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
            
import modules
import utils
class NeuralProcess(nn.Module):

    def __init__(self, x_dim=2, h_dim=400, r_dim=128, z_dim=128, y_dim=1, device='cuda'):
        super(NeuralProcess, self).__init__()
        
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
                
        self.encoder = modules.Encoder(x_dim=self.x_dim, y_dim=self.y_dim, r_dim=self.r_dim, h_dim=self.h_dim)
        self.latent_encoder = modules.LatentEncoder(r_dim=self.r_dim, z_dim=self.z_dim, h_dim=self.h_dim)
        self.decoder = modules.Decoder(x_dim=self.x_dim, y_dim=self.y_dim, z_dim=self.z_dim, h_dim=self.h_dim)

        self.device = device
    def forward(self, query, target_y=None):
        # representation r is the latent for regular NP
        # sampling z:
        # - training: from posterior (encoded contexts+targets)
        # - testing: from prior (encoded contexts)
                
        context_x, context_y, target_x = query
        if self.training:
            all_x = torch.cat((context_x, target_x), dim=1)
            all_y = torch.cat((context_y, target_y), dim=1)    
        # print('\t[DATA] context:',context_x.shape, context_y.shape, 'target:', target_x.shape, target_y.shape)
        # [num_samples, context_size(num_points), x_dim]
        
        batch_size, num_targets, x_dim = target_x.shape
        _, _, y_dim = context_y.shape
        _, num_contexts, _ = context_x.shape
        
        # latent encoding from contexts
        context_repr = self.encoder(context_x, context_y)
        prior, context_mu, context_sigma = self.latent_encoder(context_repr)
        
        ''' sample z (latent representation) '''
        # train
        if self.training and target_y is not None:
            all_repr = self.encoder(all_x, all_y)
            posterior, all_mu, all_sigma = self.latent_encoder(all_repr)
            
            # set distributions for kl divegence ( target (posterior) || context (prior) )
            q_target = posterior
            q_context = prior
            
            # sample from encoded distribution using reparam trick
            z_samples = q_target.rsample() # sampled from context+target encoded vector
        # test
        elif not self.training and target_y is None:
            # set distributions accordingly
            q_target = None
            q_context = prior
            
            # sample from encoded distribution using reparam tricks
            z_samples = q_context.rsample()
            
            # alternative to sample() method
            # z = torch.rand_like(std) * std + mean
            
        ''' decode '''
        # generation (context only)
        context_pred_mu, context_pred_sigma = self.decoder(context_x, z_samples)
        context_pred_logits = utils.logits_from_pred_mu(context_pred_mu, batch_size, self.device)
        
        target_pred_mu, target_pred_sigma = self.decoder(target_x, z_samples)
        target_pred_logits = utils.logits_from_pred_mu(target_pred_mu, batch_size, self.device)
        
        logits = {}        
        logits.update({'context': context_pred_logits, 'target': target_pred_logits})
        
        # distribution for the predicted/generated target parameters    
        p_y_pred = torch.distributions.normal.Normal(target_pred_mu, target_pred_sigma)
        
        ''' set distributions and compute loss '''
        if self.training:
            # loss about the distributions from batch images
            loss = utils.compute_loss(p_y_pred, target_y, q_target, q_context, logits['target'])
        else:
            loss = None
            
        return context_mu, logits, p_y_pred, q_target, q_context, loss    