import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init

class proxy_anchor_loss(nn.Module):
    def __init__(self, input_dim, n_classes, scale, margin):
        super(Proxy_loss, self).__init__()
        self.proxy = Parameter(torch.Tensor(input_dim, n_classes))
        self.n_classes = n_classes
        self.alpha = scale
        self.delta = margin
        init.kaiming_uniform_(self.proxy, a=math.sqrt(5))

    def forward(self, input, target):
        # input already l2_normalized
        self.proxy_l2 = F.normalize(self.proxy, p=2, dim=0)
        
        # N, dim, cls

        sim_mat = input.matmul(self.proxy_l2) # (N, cls)
        
        pos_target = F.one_hot(target, self.n_classes).float()
        neg_target = 1.0 - pos_target
        
        pos_mat = sim_mat * pos_target
        neg_mat = sim_mat * neg_target

        pos_term = 1.0 / torch.unique(target).shape[0] * torch.sum(torch.log(1.0 + torch.sum(torch.exp(-self.alpha * (pos_mat - self.delta)), axis=1)))
        neg_term = 1.0 / self.n_classes * torch.sum(torch.log(1.0 + torch.sum(torch.exp(self.alpha * (neg_mat + self.delta)), axis=1)))

        loss = pos_term + neg_term

        return loss, self.alpha
