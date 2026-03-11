import torch
from torch import nn

from layers.ema import EMA
from layers.dema import DEMA

class MYDECOMP(nn.Module):
    """
    Series decomposition block - 简化稳健版
    """
    def __init__(self, ma_type, alpha, beta, seasonal_method='robust', period=None, n_harmonics=3):
        super(MYDECOMP, self).__init__()
        if ma_type == 'ema':
            self.ma = EMA(alpha)
        elif ma_type == 'dema':
            self.ma = DEMA(alpha, beta)

        # 季节性提取设置
        self.seasonal_method = seasonal_method
        self.period = period
        self.n_harmonics = n_harmonics

    def _robust_seasonal_extraction(self, res_component):
        """
        使用更稳健的方法提取季节性成分
        """
        batch_size, seq_len, n_channels = res_component.shape
        device = res_component.device
        
        # 检查输入是否包含NaN或Inf并处理
        if torch.isnan(res_component).any() or torch.isinf(res_component).any():
            res_component = torch.nan_to_num(res_component, nan=0.0, posinf=1e5, neginf=-1e5)
        
        # 对于大型数据集，使用简单的滑动平均来提取季节性成分
        # 此方法比FFT更稳定，特别是对于有噪声的大型数据集
        
        # 估计季节性成分 - 使用自适应的窗口大小
        if self.period is None:
            # 如果没有指定周期，使用序列长度的一小部分作为窗口大小
            window_size = max(3, min(seq_len // 10, 24))  # 根据数据调整
        else:
            window_size = max(3, min(self.period, seq_len // 4))
        
        # 确保窗口大小是奇数（对于中心化滑动平均）
        if window_size % 2 == 0:
            window_size += 1
            
        # 创建移动平均核
        padding = window_size // 2
        
        # 重塑以便于处理
        res_flat = res_component.permute(0, 2, 1).reshape(-1, seq_len)  # [Batch*Channel, Seq_len]
        
        # 使用卷积实现移动平均
        if device.type != 'mps':
            # 标准设备上使用卷积
            kernel = torch.ones(1, 1, window_size, device=device) / window_size
            res_flat_reshaped = res_flat.unsqueeze(1)  # [B*C, 1, seq_len]
            
            # 应用卷积（移动平均）
            try:
                # 使用卷积计算移动平均
                seasonal_flat = torch.nn.functional.conv1d(
                    res_flat_reshaped, kernel, padding=padding
                ).squeeze(1)
                
                # 处理填充引起的边界效应
                # 根据你的实际需求调整这部分
                if seq_len > window_size:
                    # 对边界使用反射填充
                    for i in range(padding):
                        seasonal_flat[:, i] = seasonal_flat[:, 2*padding - i]
                        seasonal_flat[:, -(i+1)] = seasonal_flat[:, -(2*padding-i)]
            except Exception as e:
                print(f"Moving average failed: {e}, using simple scaling")
                # 回退到简单的缩放
                seasonal_flat = res_flat * 0.5
        else:
            # MPS设备上手动实现移动平均
            seasonal_flat = torch.zeros_like(res_flat)
            
            try:
                # 手动实现滑动窗口
                for i in range(seq_len):
                    # 计算窗口的开始和结束
                    start = max(0, i - padding)
                    end = min(seq_len, i + padding + 1)
                    # 计算当前位置的移动平均
                    seasonal_flat[:, i] = torch.mean(res_flat[:, start:end], dim=1)
            except Exception as e:
                print(f"Moving average failed: {e}, using simple scaling")
                # 回退到简单的缩放
                seasonal_flat = res_flat * 0.5
        
        # 使用季节性成分减少噪声
        alpha = 0.7  # 季节性强度参数，可调整
        seasonal_flat = seasonal_flat * alpha
                
        # 重塑回原始维度
        seasonal = seasonal_flat.reshape(batch_size, n_channels, seq_len).permute(0, 2, 1)
        
        # 计算残差
        residual = res_component - seasonal
        
        # 确保没有NaN或Inf
        if torch.isnan(seasonal).any() or torch.isinf(seasonal).any():
            seasonal = torch.nan_to_num(seasonal, nan=0.0, posinf=1e5, neginf=-1e5)
            
        if torch.isnan(residual).any() or torch.isinf(residual).any():
            residual = torch.nan_to_num(residual, nan=0.0, posinf=1e5, neginf=-1e5)
            
        return seasonal, residual

    def forward(self, x):
        """
        将时间序列分解为趋势、季节性和残差分量
        Args:
            x: [Batch, Input seq, Channel]
        Returns:
            seasonal: [Batch, Input seq, Channel] 季节性分量
            trend: [Batch, Input seq, Channel] 趋势分量
            residual: [Batch, Input seq, Channel] 残差分量
        """
        # 输入检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
            
        # 步骤1: 原始的EMA/DEMA分解 - 提取趋势
        moving_average = self.ma(x)  # 趋势分量
        
        # 检查MA输出
        if torch.isnan(moving_average).any() or torch.isinf(moving_average).any():
            moving_average = torch.nan_to_num(moving_average, nan=0.0, posinf=1e5, neginf=-1e5)
            
        combined_res = x - moving_average  # 包含季节性和残差的组合
        
        # 步骤2: 使用稳健方法分解组合残差
        try:
            # 总是使用稳健的方法，避免FFT
            seasonal, residual = self._robust_seasonal_extraction(combined_res)
        except Exception as e:
            # 如果处理失败，使用简单的回退
            print(f"Decomposition failed: {e}, using basic decomposition")
            seasonal = combined_res * 0.5
            residual = combined_res * 0.5
        
        # 返回所有三个分量
        return seasonal, moving_average, residual