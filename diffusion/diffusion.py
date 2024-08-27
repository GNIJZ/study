import math
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self,dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim=dim

    def forward(self,x):
        device=x.device
        half_dim=self.dim//2
        emb=math.log(1000)/(half_dim-1)
        emb=torch.exp(torch.arange(half_dim,device=device)*-emb)
        emb=x[:,None]*emb[None,:]
        emb=torch.cat((emb.sin(),emb.cos()),dim=-1)
        return emb

# 对于 x = 1.0
# sin([1.0000, 0.1000, 0.0100, 0.0010])
# cos([1.0000, 0.1000, 0.0100, 0.0010])
# 得到：
# [0.8415, 0.0998, 0.0100, 0.0010, 0.5403, 0.9950, 1.0000, 1.0000]

# 对于 x = 2.0
# sin([2.0000, 0.2000, 0.0200, 0.0020])
# cos([2.0000, 0.2000, 0.0200, 0.0020])
# 得到：
# [0.9093, 0.1987, 0.0200, 0.0020, -0.4161, 0.9801, 1.0000, 1.0000]

class MLP(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim,device,t_dim):
        super(MLP, self).__init__()

        self.t_dim = t_dim
        self.device = device
        self.a_dim = state_dim


        self.time_mlp=nn.Sequential(
                SinusoidalPosEmb(t_dim),
                nn.Linear(t_dim,t_dim*2),
                nn.Mish(),
                nn.Linear(t_dim*2,t_dim)
        )
        input_dim=state_dim+action_dim+t_dim
        self.mid_layer=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Mish(),
        )
        self.final_layer=nn.Linear(hidden_dim,action_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,x,time,state):
        t_emb=self.time_mlp(time)
        x=torch.cat((x,state,t_emb),dim=1)
        x=self.mid_layer(x)
        x=self.final_layer(x)
        return x

























