# encoding: utf-8
"""
@author: vinklibrary
@contact: shenxvdong1@gmail.com
@site: Todo

@version: 1.0
@license: None
@file: LogisticRegression.py
@time: 2019/7/11 17:21

这一行开始写关于本文件的说明与解释
"""
import torch
import torch.nn as nn
from utils.self_layers import one_hot_embedding

# 逻辑回归模型
class   LogisticRegression(nn.Module):
    def __init__(self, feature_sizes):
        super(LogisticRegression, self).__init__()
        self.feature_sizes = feature_sizes
        #各种Embedding+Linear 实现的逻辑回归
        feature_size = sum(feature_sizes)
        self.lr = nn.Linear(feature_size, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        #TOOD 将每个输出one-hot处理
        x = [one_hot_embedding(x_i, self.feature_sizes[i]).reshape(-1) for i,x_i in enumerate(x)]
        result = self.lr(x)
        return self.sm(result)

# FM 模型
class FM(nn.Module):
    def __init__(self):
        super(FM, self).__init__()
        pass

    def forward(self, *input):
        pass


