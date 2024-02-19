# -*- coding: utf-8 -*-
"""
@Time    : 2021/12/30 15:40
@Author  : Lucius
@FileName: base_model.py
@Software: PyCharm
"""
import torch
from torch import nn
import torch.nn.functional as F


class HashLayer(nn.Module):
    def __init__(self, feature_in, feature_out, depth) -> None:
        super().__init__()
        if depth == 1:
            self.fc = nn.Linear(feature_in, feature_out)
        elif depth == 2:
            self.fc = nn.Sequential(
                nn.Linear(feature_in, 2 * feature_out),
                nn.Linear(2 * feature_out, feature_out),
            )

    def forward(self, x):
        x = self.fc(x)
        x = torch.tanh(x)
        return x


class HashEncoder(nn.Module):
    def __init__(self, feature_in, feature_out, depth) -> None:
        super().__init__()
        if depth == 1:
            self.fc = nn.Linear(feature_in, feature_out)
        elif depth == 2:
            self.fc = nn.Sequential(
                nn.Linear(feature_in, 2 * feature_out),
                nn.ReLU(),
                nn.Linear(2 * feature_out, feature_out),
            )

    def forward(self, x, no_pooling=False):
        x = self.fc(x)
        if not no_pooling:
            x = x.mean(-2)
        x = torch.tanh(x)
        return x


class AttenHashEncoder(nn.Module):
    def __init__(self, feature_in, feature_out, depth) -> None:
        super().__init__()
        if depth == 1:
            self.fc = nn.Linear(feature_in, feature_out)
        elif depth == 2:
            self.fc = nn.Sequential(
                nn.Linear(feature_in, 2 * feature_out),
                nn.ReLU(),
                nn.Linear(2 * feature_out, feature_out),
            )
        self.attention_layer = nn.Linear(feature_in, 1)

        # self.bn = nn.BatchNorm1d(feature_out)

    def forward(self, x, no_pooling=False, weight=False):
        out = self.fc(x)  # b x c x dim

        # test: bn
        # out = out.transpose(-1, -2)
        # out = self.bn(out)
        # out = out.transpose(-1, -2)

        att = self.attention_layer(x)
        att = F.softmax(att, dim=-2)  # b x c x 1
        if not no_pooling:
            out = out * att
            out = out.mean(-2)
            out = torch.tanh(out)
            return out

        out = torch.tanh(out)
        if weight:
            dim_c = out.shape[-2]
            return out, att * dim_c
        else:
            return out


class SqueezeOp(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x = x.squeeze()
        return x
