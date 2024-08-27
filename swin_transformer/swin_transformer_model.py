import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np


from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
# from timm.models.layers import DropPath, trunc_normal_
# import logging
# from mmcv.utils import get_logger
# from mmcv.runner import load_checkpoint


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super(MLP, self).__init__()
        hidden_layers = hidden_features or in_features
        out_features= out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_layers)
        self.gelu=nn.GELU()
        self.fc2 = nn.Linear(hidden_layers, out_features)
        self.dropout=nn.Dropout(drop)
    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x=self.dropout(x)
        x = self.dropout(self.fc2(x))
        return  x

def window_partition(x,window_size):
    B,H,C,W=x.shape
    x=x.view(B,H//window_size,window_size,W//window_size,window_size,C)
    windows=x.permute(0,1,3,2,4,5).contiguous().view(-1,window_size,window_size,C)
    return windows

def window_reverse(windows,window_size,H,W):
    """
     Args:
        windows: (num_windows * B, window_size, window_size, C).
        window_size(int): Window size.
        H(int): Height of image.
        W(int): Width of image.

    Returns:
        x: (B, H, W, C).
    """
    B=int(windows.shape[0]/(H*W/window_size/window_size))
    x=windows.view(B,H//window_size,W//window_size,window_size,window_size,-1)
    x=x.permute(0,1,3,2,4,5).contiguous().view(B,H,W,-1)
    return x
class WindowAttention(nn.Module):
    def __init__(self,dim,window_size,num_heads,qkv_bias=True,qk_scale=None, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.dim=dim
        self.window_size=window_size # Wh,Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        #划分windows_size的大小的窗口
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)























