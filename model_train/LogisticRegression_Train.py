# encoding: utf-8
"""
@author: vinklibrary
@contact: shenxvdong1@gmail.com
@site: Todo

@version: 1.0
@license: None
@file: LogisticRegression_Train.py
@time: 2019/7/16 15:44

这一行开始写关于本文件的说明与解释
"""
from model.LogisticRegression import LogisticRegression
import torch
import torch.nn as nn

def LogisticRegression_Train(data_loader, num_rounds, optim, early_stop):
    logistic_model = LogisticRegression.LogisticRegression([10, 10, 10])

    if torch.cuda.is_available():
        logistic_model.cuda()

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-3, momentum=0.9)

    # 开始训练
    for epoch in range(num_rounds):
        if torch.cuda.is_available():
            x_data = Variable(x).cuda()
            y_data = Variable(y).cuda()
        else:
            x_data = Variable(x)
            y_data = Variable(y)

        out = logistic_model(x_data)
        loss = criterion(out, y_data)
        print_loss = loss.data.item()
        mask = out.ge(0.5).float()  # 以0.5为阈值进行分类
        correct = (mask == y_data).sum()  # 计算正确预测的样本个数
        acc = correct.item() / x_data.size(0)  # 计算精度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 每隔20轮打印一下当前的误差和精度
        if (epoch + 1) % 20 == 0:
            print('*' * 10)
            print('epoch {}'.format(epoch + 1))  # 训练轮数
            print('loss is {:.4f}'.format(print_loss))  # 误差
            print('acc is {:.4f}'.format(acc))  # 精度