'''
Author: LJL . 
Date: 2025-04-16 19:13:11
LastEditors: LJL . 
LastEditTime: 2025-04-19 21:34:36
FilePath: /xPatch/models/xPatch.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import math

from layers.decomp import DECOMP
from layers.my_decomp import MYDECOMP # myadd
from layers.network import Network
from layers.my_network import MYNetwork # myadd
from layers.FAN import FAN # myadd
# from layers.network_mlp import NetworkMLP # For ablation study with MLP-only stream
# from layers.network_cnn import NetworkCNN # For ablation study with CNN-only stream
from layers.revin import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # Parameters
        seq_len = configs.seq_len   # lookback window L
        pred_len = configs.pred_len # prediction length (96, 192, 336, 720)
        c_in = configs.enc_in       # input channels

        # Patching
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in,affine=True,subtract_last=False)
        
        # 性能更差了
        # self.fan = FAN(seq_len=seq_len, pred_len=pred_len, enc_in=c_in)

        # Moving Average
        self.ma_type = configs.ma_type
        alpha = configs.alpha       # smoothing factor for EMA (Exponential Moving Average)
        beta = configs.beta         # smoothing factor for DEMA (Double Exponential Moving Average)

        # self.decomp = DECOMP(self.ma_type, alpha, beta) 原来的
        self.decomp = MYDECOMP(self.ma_type, alpha, beta)

        # self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch) 原来的
        self.net = MYNetwork(seq_len, pred_len, patch_len, stride, padding_patch, c_in)

        # self.net_mlp = NetworkMLP(seq_len, pred_len) # For ablation study with MLP-only stream
        # self.net_cnn = NetworkCNN(seq_len, pred_len, patch_len, stride, padding_patch) # For ablation study with CNN-only stream

    def forward(self, x):
        # x: [Batch, Input, Channel]
        batch_size = x.shape[0]   # 第一个维度 (Batch)
        input_size = x.shape[1]   # 第二个维度 (Input)
        channel_size = x.shape[2] # 第三个维度 (Channel)

        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':   # If no decomposition, directly pass the input to the network
            # x = self.net(x, x) 原来的
            x = self.net(x,x,x) # 如果不进行分解 创建三个相同的输入 myadd
            # x = self.net_mlp(x) # For ablation study with MLP-only stream
            # x = self.net_cnn(x) # For ablation study with CNN-only stream
        else:
            # seasonal_init, trend_init = self.decomp(x) 原来的
            # x = self.net(seasonal_init, trend_init) 原来的

            seasonal_init, trend_init, residual_init = self.decomp(x) # myadd
            # print("seasonal_init 的形状：", seasonal_init.shape)
            # print("trend_init 的形状：", trend_init.shape)
            # print("residual_init 的形状：", residual_init.shape)

            # print("x处理前的形状", x.shape)
            
            x = self.net(seasonal_init,trend_init,residual_init)
            # print("x处理后的形状", x.shape)

        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x
    
