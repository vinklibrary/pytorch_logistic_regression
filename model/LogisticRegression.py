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

# 逻辑回归模型
class   LogisticRegression(nn.Module):
    def __init__(self, feature_sizes):
        super(LogisticRegression, self).__init__()
        #各种Embedding+Linear 实现的逻辑回归
        feature_size = sum(feature_sizes)
        self.lr = nn.Linear(feature_size, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        # one_hot = torch.zeros(batch_size, class_num).scatter_(1, label, 1)
        x =
        result = self.lr(x)
        return self.sm(result)


class FM(nn.Module):
    def __init__(self):
        super(FM, self).__init__()
        pass

    def forward(self, *input):
        pass


