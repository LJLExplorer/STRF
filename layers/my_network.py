'''
Author: LJL . 
Date: 2025-04-16 19:12:54
LastEditors: LJL . 
LastEditTime: 2025-04-22 10:27:15
FilePath: /xPatch_2/layers/my_network.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import torch
from torch import nn
import torch.nn.functional as F
# from layers.Adaptive_Spectral_Block import Adaptive_Spectral_Block
# from layers.FAN import FAN # myadd
# from layers.ICB import ICB # myadd
# from layers.WaveletBlock import WaveletBlock


class MYNetwork(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,c_dim):
        super(MYNetwork, self).__init__()

        # Parameters
        self.pred_len = pred_len

        # Non-linear Stream 季节性
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len)//stride + 1
        self.c_dim = c_dim #myadd
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            self.patch_num += 1

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)
        
        # CNN Depthwise
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream
        self.fc2 = nn.Linear(self.dim, patch_len)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Linear Stream 趋势性
        # MLP
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # # 残差流处理 （新增）
        # self.fan = FAN(seq_len=seq_len, pred_len=pred_len, enc_in=c_dim)
        
        # # 使用Adaptive_Spectral_Block自适应减轻噪声 c_dim = C
        # self.asb = Adaptive_Spectral_Block(dim=self.c_dim)
        
        # self.icb = ICB(in_features=self.c_dim, hidden_features=self.c_dim, drop=0.1)
        
        # self.wavelet_block = WaveletBlock(seq_len=seq_len, wavelet='db4', level=2)
        # self.wavelet_block = WaveletBlock(seq_len=seq_len, wavelet='haar', level=2)
        

        # # 因为残差通常包含高频信息，我们使用简单的MLP进行处理
        self.fc_r1 = nn.Linear(seq_len, pred_len * 2)
        self.gelu_r1 = nn.GELU()
        # self.selu_r1 = nn.SELU()
        self.fc_r2 = nn.Linear(pred_len * 2, pred_len)

        

        # Streams Concatination
        # self.fc8 = nn.Linear(pred_len * 2, pred_len)
        # 修改为合并三个流而不是两个
        self.fc8 = nn.Linear(pred_len * 3, pred_len)
        
        # 添加最终的残差校正层
        self.residual_correction = nn.Linear(pred_len, pred_len) # myadd last

    # def forward(self, s, t):
    def forward(self, s, t, r):

        # x: [Batch, Input, Channel]
        # s - seasonality
        # t - trend
        # r - residual (残差)
        
        # 首先对r残差流进行去噪处理再进入mlp
        # 使用减轻噪声模块
        # xin zhu shi
        # print("原始r:", torch.mean(r), torch.std(r))
        # r = self.asb(r)
        # print("ASB后r:", torch.mean(r), torch.std(r))
        # # 使用fan先解决非平稳性问题，最近提出了可逆实例规一化，以通过某些统计指标（例如均值和方差）减轻趋势的影响。
        # # t = self.fan(t, mode='n')  
        # r = self.fan(r, mode='n') 
        # print("FAN后r:", torch.mean(r), torch.std(r))
        # r = self.icb(r)
        # print("ICB后r:", torch.mean(r), torch.std(r))
                
        
        s = s.permute(0,2,1) # to [Batch, Channel, Input]
        t = t.permute(0,2,1) # to [Batch, Channel, Input]
        r = r.permute(0,2,1) # to [Batch, Channel, Input] myadd
        
        
        # Channel split for channel independence
        B = s.shape[0] # Batch size
        C = s.shape[1] # Channel size
        I = s.shape[2] # Input size
        s = torch.reshape(s, (B*C, I)) # [Batch and Channel, Input]
        t = torch.reshape(t, (B*C, I)) # [Batch and Channel, Input]
        r = torch.reshape(r, (B*C, I)) # [Batch and Channel, Input] myadd
        
        # NEW: 对残差部分应用小波变换处理
        # r = self.wavelet_block(r)  # 使用小波变换处理残差

        # Non-linear Stream
        # Patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [Batch and Channel, Patch_num, Patch_len]
        
        # Patch Embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual Stream
        res = self.fc2(res)
        s = s + res

        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Flatten Head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        # Linear Stream

        # MLP
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)
 
        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)
        
        # t_original = t.clone()
        # t = self.wavelet_block(t)  # 使用小波变换处理残差
        # t = t + t_original

        # 残差流处理（新增）
        
        # 使用简单的MLP网络处理
        # r = self.wavelet_block(r)  # 使用小波变换处理残差  
        r = self.fc_r1(r)
        r = self.gelu_r1(r)
        # r = self.selu_r1(r)
        r = self.fc_r2(r)
        
        
 
        
        # print("s 的形状：", s.shape)
        # print("t 的形状：", t.shape)
        # print("r 的形状：", r.shape)

         # 合并三个流
        x = torch.cat((s, t, r), dim=1)  # 现在是三个预测结果的拼接
        x = self.fc8(x)  # 将三个流合并为最终预测
        
        
        


        # # Streams Concatination
        # x = torch.cat((s, t), dim=1)
        # x = self.fc8(x)

        # Channel concatination
        x = torch.reshape(x, (B, C, self.pred_len)) # [Batch, Channel, Output]

        x = x.permute(0,2,1) # to [Batch, Output, Channel]
        

        return x