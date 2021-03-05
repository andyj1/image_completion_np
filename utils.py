#!/usr/bin/python
# -*- coding: utf-8 -*-
 
import torch
import torch.nn.functional as F
import numpy as np

import sys

# for kullback-leibler divergence computation
sys.setrecursionlimit(1500)

CRED    = '\033[91m'
CBLUE   = '\033[94m'
CEND    = '\033[0m'

def compute_loss(p_y_pred, target_y, q_target, q_context, target_pred_logits):
    
    ''' original proposed loss by DeepMind'''
    log_likelihood = p_y_pred.log_prob(target_y.cpu()).mean(dim=0).sum() # Compute the log-likelihood of n x under m univariate normal distributions
    kl = torch.distributions.kl.kl_divergence(q_target, q_context).mean(dim=0).sum()
    orig_loss = log_likelihood - kl
    orig_loss *= -1
    
    ''' bce loss adopted from https://github.com/yamad07/NeuralProcess '''
    # method 1: define and apply
    # criterion = torch.nn.BCELoss(reduction='mean')
    # loss += criterion(target_pred_logits, target_y.cpu())
    # method 2: apply at runtime
    bce_loss = F.binary_cross_entropy(target_pred_logits, target_y)
    
    loss = bce_loss # orig_loss
    # print(f'loss: {CRED}{loss.item():10.5f}{CEND} / kl: {CBLUE}{kl.item():10.5f}{CEND}')
    
    return loss

# not used
def get_loss(target_pred_mu, target_pred_sigma, all_mu, all_sigma, context_mu, context_sigma, target_pred_logits, target_y):
    p_y_pred = torch.distributions.normal.Normal(target_pred_mu, target_pred_sigma)
    log_likelihood = p_y_pred.log_prob(target_y).mean(dim=0).sum()
    kl = neg_kl_div(context_mu, context_sigma, all_mu, all_sigma).mean(dim=0).sum()
    kl = KL_divergence(context_mu, context_sigma, all_mu, all_sigma).mean(dim=0).sum()
    loss = log_likelihood - kl
    # print(loss)
    return loss

''' kl divergence '''
def neg_kl_div(c_mu, c_std, a_mu, a_std):
    c_std = torch.exp(0.5 * c_std) # convert log var to std
    a_std = torch.exp(0.5 * a_std)
    return  torch.log(c_std / a_std) + \
        ((a_std ** 2 + (a_mu - c_mu) ** 2) / (2 * c_std ** 2)) - 0.5
            
def KL_divergence(mu_p, sigma_p, mu_q, sigma_q):
    ''' KL(p || q) '''
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    kl_div = torch.log(sigma_q / sigma_p) + ( sigma_p**2 + (mu_p - mu_q)**2 ) / ( 2*sigma_q**2) - 1/2
    return kl_div.mean(0).sum()

# https://discuss.pytorch.org/t/vae-example-reparametrize/35843
# logVariance = log($\sigma^2$) = 2 * log(sigma)
# logStdDev = 2 * log(sigma) / 2 = 0.5 * 2 * log(sigma) = log(sigma)
def logvar_to_std(logvar):
    logstd = (1/2) * logvar
    std = torch.exp(logstd)
    
    # as per original deepmind code (in LatentEncoder)
    # used in LatentEncoder
    std = 0.1 + 0.9 * torch.sigmoid(logvar)
    
    # === reparameterization trick ===
    # "Empirical Evaluation of Neural Process Objectives" and "Attentive Neural Processes"
    # reparameterization trick
    # sigma = 0.1 + 0.9 * sigma
    # z_sample = self.std_normal.sample() * sigma + mu
    # z_sample = z_sample.unsqueeze(0)
    
    # uniform distribution
    # z_samples = torch.rand_like(std) * std + mu
    # normal distribution
    return std

def logstd_to_std(logstd):
    # used in Decoder
    activation = torch.nn.Sigmoid()
    bounded_std = 0.1 + 0.9 * activation(logstd)
    return bounded_std

def logits_from_pred_mu(pred_mu, batch_size, device):
    if not pred_mu.is_cuda: 
        pred_mu = pred_mu.to(device)
    logits = torch.tensor([]).to(device)
    for i in range(batch_size):
        index = torch.tensor([i]).to(device)
        indexed_pred_mu = torch.index_select(pred_mu, 0, index)
        new_logits = torch.sigmoid(indexed_pred_mu)
        logits = torch.cat([logits, new_logits], dim=0)
    return logits

def clip_tensor(input:torch.Tensor = torch.tensor([]), min=0, max=1):
    output = torch.clip(input, 0, 1)
    return output