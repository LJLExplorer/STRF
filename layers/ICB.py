'''
Author: LJL . 
Date: 2025-04-17 18:40:02
LastEditors: LJL . 
LastEditTime: 2025-04-17 18:53:40
FilePath: /xPatch/layers/ICB.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import pytorch_lightning as pl
import torch
import torch.nn as nn


class ICB(pl.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x