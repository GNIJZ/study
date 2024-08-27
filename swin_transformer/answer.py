"""

relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]

tensor([[[0],
         [0],
         [0],
         [1],
         [1],
         [1],
         [2],
         [2],
         [2]],

        [[0],
         [1],
         [2],
         [0],
         [1],
         [2],
         [0],
         [1],
         [2]]])

tensor([[[0, 0, 0, 1, 1, 1, 2, 2, 2]],

        [[0, 1, 2, 0, 1, 2, 0, 1, 2]]])

relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]

tensor([[[ 0,  0,  0, -1, -1, -1, -2, -2, -2],
         [ 0,  0,  0, -1, -1, -1, -2, -2, -2],
         [ 0,  0,  0, -1, -1, -1, -2, -2, -2],
         [ 1,  1,  1,  0,  0,  0, -1, -1, -1],
         [ 1,  1,  1,  0,  0,  0, -1, -1, -1],
         [ 1,  1,  1,  0,  0,  0, -1, -1, -1],
         [ 2,  2,  2,  1,  1,  1,  0,  0,  0],
         [ 2,  2,  2,  1,  1,  1,  0,  0,  0],
         [ 2,  2,  2,  1,  1,  1,  0,  0,  0]],

        [[ 0, -1, -2,  0, -1, -2,  0, -1, -2],
         [ 1,  0, -1,  1,  0, -1,  1,  0, -1],
         [ 2,  1,  0,  2,  1,  0,  2,  1,  0],
         [ 0, -1, -2,  0, -1, -2,  0, -1, -2],
         [ 1,  0, -1,  1,  0, -1,  1,  0, -1],
         [ 2,  1,  0,  2,  1,  0,  2,  1,  0],
         [ 0, -1, -2,  0, -1, -2,  0, -1, -2],
         [ 1,  0, -1,  1,  0, -1,  1,  0, -1],
         [ 2,  1,  0,  2,  1,  0,  2,  1,  0]]])

"""
import torch
window_size=[2,2]
coords_h = torch.arange(window_size[0])
coords_w = torch.arange(window_size[1])
coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
relative_coords[:, :, 1] += window_size[1] - 1
relative_coords[:, :, 0] *= 2 * window_size[1] - 1
relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

print(relative_position_index)