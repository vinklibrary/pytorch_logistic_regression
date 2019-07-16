# encoding: utf-8
"""
@author: vinklibrary
@contact: shenxvdong1@gmail.com
@site: Todo

@version: 1.0
@license: None
@file: data_preprocess.py
@time: 2019/7/16 15:52

这一行开始写关于本文件的说明与解释
"""
import pandas as pd

class data_preprocess:
    # 需要读取数据，然后进行转换成新格式
    def __init__(self, raw_data_path):
        self.raw_data = pd.read_csv(raw_data_path)
        # 获得特征值对应的字典
        self.__columns_to_map__()
        pass

    def __columns_to_map__(self):
        pass

