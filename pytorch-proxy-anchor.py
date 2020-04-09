import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init

class proxy_anchor_loss(nn.Module):
    '''
    ref: https://arxiv.org/abs/2003.13911
    official pytorch codes: https://github.com/tjddus9597/Proxy-Anchor-CVPR2020
    '''
    def __init__(self, input_dim, n_classes, scale, margin):
        super(proxy_anchor_loss, self).__init__()
        self.proxy = Parameter(torch.Tensor(input_dim, n_classes))
        self.n_classes = n_classes
        self.alpha = scale
        self.delta = margin
        init.kaiming_normal_(self.proxy, mode='fan_out')

    def forward(self, embeddings, target):
        embeddings_l2 = F.normalize(embeddings, p=2, dim=1)
        proxy_l2 = F.normalize(self.proxy, p=2, dim=0)
        
        # N, dim, cls

        sim_mat = embeddings_l2.matmul(proxy_l2) # (N, cls)
        
        pos_target = F.one_hot(target, self.n_classes).float()
        neg_target = 1.0 - pos_target
        
        pos_mat = (sim_mat - self.delta) * pos_target
        neg_mat = (sim_mat + self.delta) * neg_target

        pos_term = 1.0 / torch.unique(target).shape[0] * torch.sum(torch.log(1.0 + torch.sum(torch.exp(-self.alpha * pos_mat), axis=0)))
        neg_term = 1.0 / self.n_classes * torch.sum(torch.log(1.0 + torch.sum(torch.exp(self.alpha * neg_mat), axis=0)))

        loss = pos_term + neg_term

        return loss
