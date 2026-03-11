'''
Author: LJL . 
Date: 2025-04-16 19:12:56
LastEditors: LJL . 
LastEditTime: 2025-04-16 19:19:41
FilePath: /xPatch/layers/waveletSeasonalDecomposer.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import numpy as np

# 小波变换法得到季节性和残差
class WaveletSeasonalDecomposer(nn.Module):
    def __init__(self, wavelet='db4', level=3, mode='periodical'):
        """
        使用小波变换提取季节性成分
        Args:
            wavelet: 小波类型
            level: 分解级别
            mode: 边界处理模式
        """
        super(WaveletSeasonalDecomposer, self).__init__()
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        
    def forward(self, detrended):
        """
        从去趋势的时间序列提取季节性成分
        Args:
            detrended: [Batch, Seq_len] 去趋势后的时间序列
        Returns:
            seasonal: [Batch, Seq_len] 季节性成分
            residual: [Batch, Seq_len] 残差成分
        """
        batch_size, seq_len = detrended.shape
        device = detrended.device
        
        # 确保序列长度适合多级小波分解
        pad_len = 0
        if seq_len % (2**self.level) != 0:
            pad_len = int(2**self.level * np.ceil(seq_len / 2**self.level)) - seq_len
        
        # 创建输出张量
        seasonal = torch.zeros_like(detrended)
        
        for i in range(batch_size):
            x = detrended[i].cpu().detach().numpy()
            
            # 如果需要，添加填充
            if pad_len > 0:
                x_padded = np.pad(x, (0, pad_len), mode='reflect')
            else:
                x_padded = x
            
            # 执行小波分解
            coeffs = pywt.wavedec(x_padded, self.wavelet, mode=self.mode, level=self.level)
            
            # 提取季节性成分（保留细节系数）
            seasonal_coeffs = [np.zeros_like(coeffs[0])]  # 将近似系数置零
            seasonal_coeffs.extend(coeffs[1:])  # 保留所有细节系数
            
            # 重构季节性成分
            seasonal_np = pywt.waverec(seasonal_coeffs, self.wavelet, mode=self.mode)
            
            # 移除填充（如果有）并转回PyTorch
            seasonal_np = seasonal_np[:seq_len]
            seasonal[i] = torch.tensor(seasonal_np, device=device)
        
        # 计算残差
        residual = detrended - seasonal
        
        return seasonal, residual