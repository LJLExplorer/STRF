import torch
import torch.nn as nn
import torch.nn.functional as F

# 季节性解卷积法 得到季节性和残差
class SeasonalDecomposer(nn.Module):
    def __init__(self, period=None, smoothing_window=None):
        """
        Args:
            period: 季节性周期长度，如果为None则自动检测
            smoothing_window: 平滑窗口大小，默认为period//2
        """
        super(SeasonalDecomposer, self).__init__()
        self.period = period
        self.smoothing_window = smoothing_window
        
    def _autocorrelation(self, x):
        """计算自相关以检测周期"""
        # 将批次展平
        batch_size, seq_len = x.shape
        corr_len = min(seq_len // 2, 100)  # 不计算太长的自相关
        
        periods = []
        for i in range(batch_size):
            series = x[i].cpu().detach().numpy()
            corr = np.correlate(series, series, mode='full')
            corr = corr[corr.size//2:corr.size//2 + corr_len]
            
            # 寻找自相关峰值（跳过0滞后）
            peaks = [j for j in range(2, len(corr)-1) if corr[j] > corr[j-1] and corr[j] > corr[j+1]]
            
            if peaks:
                # 取最强的周期
                period = peaks[np.argmax([corr[p] for p in peaks])]
                periods.append(period)
            else:
                # 默认周期
                periods.append(7)  # 默认为7（常见的周周期）
        
        # 返回最常见的周期
        return max(set(periods), key=periods.count) if periods else 7
    
    def _seasonal_mean(self, x, period):
        """计算季节性平均值"""
        batch_size, seq_len = x.shape
        n_periods = seq_len // period
        device = x.device
        
        # 将序列重塑为(batch, n_periods, period)
        if n_periods > 0:
            # 截断序列以匹配完整周期
            truncated_len = n_periods * period
            x_reshaped = x[:, :truncated_len].reshape(batch_size, n_periods, period)
            
            # 沿周期轴计算平均值
            seasonal_pattern = torch.mean(x_reshaped, dim=1)  # [batch, period]
            
            # 重复模式以匹配原始序列长度
            repeats_needed = seq_len // period + (1 if seq_len % period > 0 else 0)
            seasonal = seasonal_pattern.repeat(1, repeats_needed)[:, :seq_len]  # [batch, seq_len]
            
            return seasonal
        else:
            # 处理序列短于周期的情况
            return torch.zeros_like(x)
    
    def forward(self, detrended):
        """
        从去趋势的时间序列提取季节性成分
        Args:
            detrended: [Batch, Seq_len] 去趋势后的时间序列
        Returns:
            seasonal: [Batch, Seq_len] 季节性成分
            residual: [Batch, Seq_len] 残差成分
        """
        # 检测或使用预设周期
        period = self.period if self.period is not None else self._autocorrelation(detrended)
        
        # 确定平滑窗口大小
        smoothing_window = self.smoothing_window if self.smoothing_window is not None else max(period // 2, 1)
        
        # 提取初始季节性模式
        seasonal_raw = self._seasonal_mean(detrended, period)
        
        # 计算平滑的季节性成分（使用卷积平滑）
        if smoothing_window > 1:
            # 创建高斯卷积核
            kernel_size = min(smoothing_window * 2 + 1, period)
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # 确保奇数
            
            sigma = smoothing_window / 3.0
            grid = torch.arange(kernel_size, device=detrended.device) - (kernel_size - 1) / 2
            kernel = torch.exp(-(grid**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            
            # 应用平滑卷积
            padding = (kernel_size - 1) // 2
            seasonal = F.conv1d(
                seasonal_raw.unsqueeze(1),
                kernel.view(1, 1, -1),
                padding=padding
            ).squeeze(1)
        else:
            seasonal = seasonal_raw
        
        # 计算残差
        residual = detrended - seasonal
        
        return seasonal, residual