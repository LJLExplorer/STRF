import pytorch_lightning as pl
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))  # * 0.5)
        self.adaptive_filter = adaptive_filter  # 设置 adaptive_filter 选项

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # 计算频域中的能量
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # 将能量展平，并计算中位数
        flat_energy = energy.view(B, -1)  # 展平 H 和 W 到单一维度
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # 计算中位数
        median_energy = median_energy.view(B, 1)  # 重塑维度匹配原始形状

        # 归一化能量
        normalized_energy = energy / (median_energy + 1e-6)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # 沿时间维度应用 FFT
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            # 自适应高频掩码
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # 应用逆 FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # 重塑回原始形状

        return x