import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *


def Our_New_loss(res, labels, adj, domain, weights):

    class_weight, beta, ent_weight, y_w, _ = weights
    rl = recons_loss(res['a_recons'], adj)
    
    kly = kl_loss(res['ymu'], res['ylv'])

    ent_loss = max_entropy(res['y'])

    if domain == 0:
        class_loss = F.cross_entropy(input=res['cls_output'], target=labels, weight=class_weight)
    else:
        class_loss = 0
  
    loss = rl + beta * kly + y_w * class_loss + ent_weight * ent_loss
    loss = torch.maximum(loss, torch.zeros_like(loss))
    return loss


def recons_loss(recons, adjs):
    batch_size, n_node, _ = recons.shape
    total_node = batch_size * n_node * n_node
    n_edges = adjs.sum()
    device = adjs.device

    if n_edges == 0: 
        pos_weight = torch.zeros(()).to(device)
    else:
        pos_weight = float(total_node - n_edges) / n_edges

    norm = float(total_node) / (2 * (total_node - n_edges))

    rl = norm * F.binary_cross_entropy_with_logits(input=recons, target=adjs, pos_weight=pos_weight, reduction='mean')

    rl = torch.maximum(rl, torch.zeros_like(rl))

    return rl


def kl_loss(mu, lv):
    n_node = mu.shape[1]
    kld = -0.5 / n_node * torch.mean(torch.sum(1 + 2 * lv - mu.pow(2) - lv.exp().pow(2), dim=-1))
    return kld

def max_entropy(x):
    ent = 0.693148 + torch.mean(torch.sigmoid(x) * F.logsigmoid(x))
    return ent

def learning_rate_adjust(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def learning_rate_decay(optimizer, decay_rate=0.99):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def clip_gradient(optimizer, grad_clip=0.1):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

