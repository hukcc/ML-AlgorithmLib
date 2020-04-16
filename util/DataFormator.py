#coding = utf-8
import numpy as np

class DataFormator:
    def __init__(self):
        pass

    def linearDataFormator(data):
        # 用于处理感知机以及线性回归等模型的数据输入
        # 将输入向量的格式与W统一
        newData = []
        for sample in data:
            sample = np.append(sample,1)
            newData.append(sample)
        return newData
    
    def GDADataFormater(data):
        newData = []
        for sample in data:
            newData.append(sample)
        return newData