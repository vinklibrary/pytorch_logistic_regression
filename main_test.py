# encoding: utf-8
"""
@author: vinklibrary
@contact: shenxvdong1@gmail.com
@site: Todo

@version: 1.0
@license: None
@file: main_test.py
@time: 2019/7/16 14:52

这一行开始写关于本文件的说明与解释
"""
import torch
from model import LogisticRegression

if __name__ == '__main__':
    x = torch.tensor([[2,6,8],[3,5,7]]).long()
    lr = LogisticRegression.LogisticRegression([10,10,10])
    lr.forward(x)