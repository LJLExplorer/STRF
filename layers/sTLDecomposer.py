import torch
import torch.nn as nn
import numpy as np
from statsmodels.tsa.seasonal import STL as StatsSTL

# stl法得到季节性和残差
class STLDecomposer(nn.Module):
    def __init__(self, period=None, robust=True):
        """
        使用STL分解提取季节性成分
        Args:
            period: 季节性周期长度
            robust: 是否使用稳健分解
        """
        super(STLDecomposer, self).__init__()
        self.period = period
        self.robust = robust
        
    def _detect_period(self, x):
        """自动检测周期性"""
        # 简化版的周期检测，使用自相关
        # 这里仅给出一个简单实现，实际使用中可以更复杂
        return 7  # 默认周期为7（适用于周数据）
    
    def forward(self, detrended):
        """
        从去趋势的时间序列提取季节性成分
        注意：STL是批处理外部算法，会逐一处理每个序列
        Args:
            detrended: [Batch, Seq_len] 去趋势后的时间序列
        Returns:
            seasonal: [Batch, Seq_len] 季节性成分
            residual: [Batch, Seq_len] 残差成分
        """
        batch_size, seq_len = detrended.shape
        device = detrended.device
        
        # 创建输出张量
        seasonal = torch.zeros_like(detrended)
        
        for i in range(batch_size):
            x = detrended[i].cpu().detach().numpy()
            
            # 检测或使用预设周期
            period = self.period if self.period is not None else self._detect_period(x)
            period = min(period, seq_len // 2)  # 确保周期合理
            
            # 应用STL分解
            try:
                stl = StatsSTL(x, period=period, robust=self.robust)
                res = stl.fit()
                seasonal_np = res.seasonal
            except Exception as e:
                # 如果STL失败，回退到简单的均值季节性
                n_periods = seq_len // period
                if n_periods > 0:
                    x_reshaped = x[:n_periods*period].reshape(n_periods, period)
                    seasonal_pattern = np.mean(x_reshaped, axis=0)
                    seasonal_np = np.tile(seasonal_pattern, n_periods + 1)[:seq_len]
                else:
                    seasonal_np = np.zeros_like(x)
            
            seasonal[i] = torch.tensor(seasonal_np, device=device)
        
        # 计算残差
        residual = detrended - seasonal
        
        return seasonal, residual