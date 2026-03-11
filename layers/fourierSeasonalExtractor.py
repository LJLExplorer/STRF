import torch
import torch.nn as nn
import numpy as np

# 傅里叶分析法 计算得到季节性和残差
class FourierSeasonalExtractor(nn.Module):
    def __init__(self, period=None, n_harmonics=3):
        """
        Args:
            period: 季节性周期长度，如果为None则自动检测
            n_harmonics: 使用的谐波数量
        """
        super(FourierSeasonalExtractor, self).__init__()
        self.period = period
        self.n_harmonics = n_harmonics
        
    def _detect_period(self, x):
        """自动检测时间序列的主要周期"""
        # 将张量转为numpy进行FFT分析
        x_np = x.cpu().detach().numpy()
        
        # 对每个批次和通道执行FFT
        batch_size, seq_len = x_np.shape
        periods = []
        
        for i in range(batch_size):
            # 计算功率谱
            fft_values = np.abs(np.fft.rfft(x_np[i]))
            power_spectrum = fft_values**2
            
            # 跳过零频率
            freqs = np.fft.rfftfreq(seq_len, 1)
            mask = freqs > 0
            freqs = freqs[mask]
            power_spectrum = power_spectrum[mask]
            
            # 找到功率谱的最大值对应的频率（主频率）
            if len(power_spectrum) > 0:
                main_freq_idx = np.argmax(power_spectrum)
                main_freq = freqs[main_freq_idx]
                # 将频率转换为周期
                period = int(1.0 / main_freq) if main_freq > 0 else seq_len
                period = min(period, seq_len // 2)  # 周期不超过序列长度的一半
            else:
                period = seq_len // 2
                
            periods.append(period)
        
        # 返回最常见的周期
        return max(set(periods), key=periods.count)
    
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
        
        # 检测或使用预设周期
        period = self.period if self.period is not None else self._detect_period(detrended)
        period = min(period, seq_len // 2)  # 确保周期合理
        
        # 为每个批次创建季节性成分
        seasonal = torch.zeros_like(detrended)
        
        for i in range(batch_size):
            x = detrended[i]
            
            # 转为numpy进行FFT分析
            x_np = x.cpu().detach().numpy()
            
            # 计算FFT
            fft_values = np.fft.rfft(x_np)
            
            # 创建频率掩码，只保留与季节性相关的频率
            freqs = np.fft.rfftfreq(seq_len, 1)
            mask = np.zeros_like(freqs, dtype=bool)
            
            # 保留基本频率及其谐波
            for harmonic in range(1, self.n_harmonics + 1):
                target_freq = harmonic / period
                # 找到最接近目标频率的FFT频率索引
                idx = np.argmin(np.abs(freqs - target_freq))
                if idx > 0 and idx < len(mask):
                    mask[idx] = True
            
            # 应用掩码，保留谐波，过滤掉其他频率
            filtered_fft = np.zeros_like(fft_values, dtype=complex)
            filtered_fft[mask] = fft_values[mask]
            
            # 逆FFT重建季节性成分
            seasonal_np = np.fft.irfft(filtered_fft, n=seq_len)
            seasonal[i] = torch.tensor(seasonal_np, device=device)
        
        # 计算残差
        residual = detrended - seasonal
        
        return seasonal, residual