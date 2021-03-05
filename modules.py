#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import utils

def make_layer(input_dim, output_dim):
    return nn.Sequential(
                # linear layer expects [batch_size, n_features]
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim), # requires number of points in the input tensor to be > 1
                nn.ReLU(inplace=True),
            )
    # simpler alternative
    # return nn.Linear(input_dim, output_dim)

# encoder: x, y --> r
class Encoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, h_dim):
        super(Encoder, self).__init__()
        
        self.input_dim = x_dim + y_dim
        self.hidden_dim = h_dim
        self.output_dim = r_dim
        
        self.layers = nn.Sequential(
                make_layer(self.input_dim, self.hidden_dim),
                make_layer(self.hidden_dim, self.hidden_dim),
                make_layer(self.hidden_dim, self.output_dim), # nn.Linear(self.hidden_dim, self.output_dim),
                )
   
    def forward(self, x, y):
        batch_size, num_points, _ = x.size() # [batch_size, num_points, 1]
        
        input = torch.cat((x, y), dim=-1)
        # flatten
        input_flattened = input.view(batch_size * num_points, self.input_dim)
        r = self.layers(input_flattened)
        # unflatten
        r = r.view(batch_size, num_points, self.output_dim)
        
        # print('[ENCODER] encoder_input:', input.shape)
        # print('[ENCODER] encoder_output:', r.shape)
        return r 

# private class
class _Aggregator(nn.Module):
    '''
    representation r --> latent z
    '''
    def __init__(self, r_dim, h_dim, z_dim):
        super(_Aggregator, self).__init__()
        
        self.input_dim = r_dim
        self.hidden_dim = h_dim
        self.output_dim = z_dim
        
        self.layers = nn.Sequential(
                        nn.Linear(self.input_dim, self.hidden_dim),
                        nn.LeakyReLU(),
                        nn.Linear(self.hidden_dim, self.output_dim),
                        )
    
    def mean_aggregate(self, r):
        return r.mean(dim=1)
    
    def forward(self, r):
        r = self.mean_aggregate(r)
        z = self.layers(r)
        return z

class LatentEncoder(nn.Module):
    '''
    representation r --> latent vector z (distribution, mu, sigma)
    - invokes Aggregator to mean aggregate and encode representation into latent vecto
    - q ( z | x_context, y_context )
    '''
    def __init__(self, r_dim=400, z_dim=1, h_dim=400):
        super(LatentEncoder, self).__init__()
        
        self.input_dim = r_dim
        self.hidden_dim = h_dim
        self.output_dim = z_dim
        
        self.mean_aggregater = _Aggregator(r_dim=self.input_dim, h_dim=self.hidden_dim, z_dim=self.output_dim)
        self.variance_aggregator = _Aggregator(r_dim=self.input_dim, h_dim=self.hidden_dim, z_dim=self.output_dim)
        
    def forward(self, encoder_output, sample=False):
        
        mu = self.mean_aggregater(encoder_output)
        logvar = self.variance_aggregator(encoder_output) 
        sigma = utils.logvar_to_std(logvar)
        # print('[z ENCODER] [AGGREGATE] z mu:', mu.shape, '/ sigma:', sigma.shape)
        
        latent_dist = torch.distributions.normal.Normal(mu, sigma)
        if sample:
            z_samples = latent_dist.rsample()
            return latent_dist, z_samples, mu, sigma
        # mu, sigma: [batch_size, z_dim
        return latent_dist, mu, sigma
class Decoder(nn.Module):
    '''
    x, latent representation z --> y (distribution, mu, sigma)
    '''
    def __init__(self, x_dim, y_dim, z_dim, h_dim):
        super(Decoder, self).__init__()
        
        self.input_dim = x_dim + z_dim
        self.hidden_dim = h_dim
        self.output_dim = y_dim * 2 # mu, sigma
        
        self.xr_to_hidden = nn.Sequential(
                                make_layer(self.input_dim, self.hidden_dim),
                                make_layer(self.hidden_dim, self.hidden_dim),
                                make_layer(self.hidden_dim, self.hidden_dim),
                                make_layer(self.hidden_dim, self.hidden_dim),
                                nn.Linear(self.hidden_dim, self.output_dim)
                            )
    
    def forward(self, x, latent):
        # print('[DECODER] latent:', latent.shape, 'x:', x.shape)
        
        # make x, z have dimensions x+z dim such as (batch_size, num_points, x_dim + z_dim)
        batch_size, num_x, _ = x.shape
        latent = latent.unsqueeze(1).repeat(1, num_x, 1)
        xz_input = torch.cat((x, latent), dim=-1)
        # flatten
        xz_input_flattened = xz_input.view(batch_size * num_x, self.input_dim)
        hidden = self.xr_to_hidden(xz_input_flattened)
        # unflatten
        hidden = hidden.view(batch_size, num_x, self.output_dim)
        
        # pass through fully connected layer for mean and standard deviation
        # mu, sigma: (784, 1)
        mu, logsigma = torch.split(hidden, split_size_or_sections=1, dim=2)        
        sigma = utils.logstd_to_std(logsigma)
        
        # device
        mu = mu.cpu()
        sigma = sigma.cpu()
                
        # print('[DECODER] x: {}, z: {}'.format(x.shape, latent.shape))
        # print('[DECODER] x,z input:', xz_input.shape)
        # print('[DECODER] hidden:',hidden.shape)
        # print('[DECODER] y mu:',mu.shape, 'sigma:',sigma.shape)
                
        return mu, sigma
    