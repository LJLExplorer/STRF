import torch
from torch import nn

from layers.ema import EMA
from layers.dema import DEMA

class DECOMP(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, ma_type, alpha, beta):
        super(DECOMP, self).__init__()
        if ma_type == 'ema':
            self.ma = EMA(alpha)
        elif ma_type == 'dema':
            self.ma = DEMA(alpha, beta)

    def forward(self, x):
        # 移动平均得到的是趋势性
        moving_average = self.ma(x)
        # 在经典的时间序列分解中认为时间序列是由季节性+趋势性+残差得到的,stl分解也是得到趋势性+季节性+残差部分
        # 直接相减得到的并不是真的季节性 是 季节性+残差
        res = x - moving_average
        return res, moving_average