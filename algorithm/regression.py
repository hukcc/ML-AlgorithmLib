# coding=utf-8
import os
import sys
import numpy as np
import math
rootpath=str('/Users/hukcc/HYY_WorkSpace/ML_workspace')
syspath=sys.path
sys.path=[]
sys.path.append(rootpath)#将工程根目录加入到python搜索路径中

from open_ML_lib.util.DataLoader import TxtDataloader as DataLoader
from open_ML_lib.util.DataLoader import IrisDataLoader
from open_ML_lib.util.DataFormator import DataFormator

class LinearRegression:
    def __init__(self,inputNumber,inputData,inputLabel):
        self.name = 'Linear'
        self.func = None
        self.W = np.random.rand(1,inputNumber+1)  # 生成的是列向量 这边和公式保持一致
        self.lr = 0.001
        self.epoch = 10
        inputData = DataFormator.linearDataFormator(inputData)
        self.inputData = inputData
        self.inputLabel = inputLabel

    def fit(self):   # 输入应该是特征向量集合以及对应的label
        for epoch in range(self.epoch):
            for sampleIdx in range(len(self.inputData)):
                x = self.inputData[sampleIdx]
                y = self.inputLabel[sampleIdx]
                deltaW = self.getGradient(x,y)
                self.W = self.W + deltaW

    def getGradient(self,x,target):  #对loss求导之后  
        return self.lr*(target-np.dot(x,self.W.reshape(-1,1)) )*x

    def infer(self,x):   # 输入应该是特征向量
        return np.dot(x,self.W.reshape(-1,1))


class LocallyWeightRegression:
    def __init__(self,inputNumber,inputData,inputLabel):
        self.name = 'LWR'
        self.func = None
        self.W = np.random.rand(1,inputNumber+1)  # 生成的是列向量 这边和公式保持一致
        self.lr = 0.01
        self.epoch = 100
        inputData = DataFormator.linearDataFormator(inputData)
        self.inputData = inputData
        self.inputLabel = inputLabel

    def fitAndInfer(self, x0):   # 输入应该是特征向量集合以及对应的label
        for epoch in range(self.epoch):
            for sampleIdx in range(len(self.inputData)):
                x = self.inputData[sampleIdx]
                y = self.inputLabel[sampleIdx]
                deltaW = self.getGradient(x,y)
                weight = self.getWeight(x,x0)
                self.W = self.W + deltaW*weight # 做点乘 表示不同距离下对W的影响不同
        return np.dot(x0,self.W.reshape(-1,1))

    def getGradient(self,x,target):  #对loss求导之后  
        return self.lr*(target-np.dot(x,self.W.reshape(-1,1)) )*x
    
    def getWeight(self, x, x0):
        x = x[:-1] # 因为原本多增加了一维 所以这里不计算最后一个元素
        x0 = x0[:-1] # 因为原本多增加了一维 所以这里不计算最后一个元素
        weight = math.exp((x-x0)*(x-x0)/-2)
        return weight
    

class PolynomialRegression: # 这个类的写法还需要进一步改进 对于高纬的输入现在还没法算！目前只能highestPowerNum=2
    def __init__(self,highestPowerNum,inputData,inputLabel):
        # f = ax*x + b*x + c 定义最高次系数是W的第一个元素
        self.name = 'PolyRegression'
        self.func = None
        self.W = np.random.rand(1,highestPowerNum+1)  # 生成的是列向量 这边和公式保持一致
        self.lr = 0.0001
        self.epoch = 5
        inputData = DataFormator.linearDataFormator(inputData)
        self.inputData = inputData
        self.inputLabel = inputLabel

    def fit(self):   # 输入应该是特征向量集合以及对应的label
        for epoch in range(self.epoch):
            for sampleIdx in range(len(self.inputData)):
                x = self.inputData[sampleIdx][0]
                x = np.array([x*x, x, 1])
                y = self.inputLabel[sampleIdx]
                deltaW = self.getGradient(x,y)
                self.W = self.W + deltaW

    def getGradient(self,x,target):  #对loss求导之后  
        return self.lr*(target-self.getOutput(x))*x
    
    def getOutput(self,x):
        return np.dot(x.reshape(1,-1),self.W.reshape(-1,1)).squeeze()

    def infer(self,x):   # 输入应该是特征向量
        x = x[0]
        x = np.array([x*x, x, 1])
        return self.getOutput(x)

class LogisticRegression:
    pass



# 单元测试
if __name__ == "__main__":
    # LR

    # dataLoader = DataLoader(dataFile='data/workExpAndSalary.data')
    # # dataLoader = DataLoader(dataFile='data/housing.data') # 经测试 波士顿房价数据集无法用线性拟合
    # dataLoader.loadDataAsList(labelIdx=-1)
    # trainData = dataLoader.dataList[:int(len(dataLoader.dataList)*3/4)]
    # testData = dataLoader.dataList[int(len(dataLoader.dataList)*3/4 + 1):]
    # trainLabel = dataLoader.labelList[:int(len(dataLoader.dataList)*3/4)]
    # testLabel = dataLoader.labelList[int(len(dataLoader.dataList)*3/4 + 1):]
    # linearRegression = LinearRegression(inputNumber=len(trainData[0]),inputData=trainData,inputLabel=trainLabel)
    # linearRegression.fit()
    # testData = DataFormator.linearDataFormator(testData)
    # for testSampleIdx in range(len(testData)):
    #     predY = linearRegression.infer(testData[testSampleIdx])
    #     print ('predY: '+str(predY)+'----'+'targetY: '+str(testLabel[testSampleIdx]))

    #LWR

    # dataLoader = DataLoader(dataFile='data/workExpAndSalary.data')
    # # dataLoader = DataLoader(dataFile='data/housing.data') # 经测试 波士顿房价数据集无法用线性拟合
    # dataLoader.loadDataAsList(labelIdx=-1)
    # data = dataLoader.dataList
    # label = dataLoader.labelList
    # locallyWeightRegression = LocallyWeightRegression(inputNumber=len(data[0]),inputData=data,inputLabel=label)
    # data = DataFormator.linearDataFormator(data)
    # for testSampleIdx in range(len(data)):
    #     predY = locallyWeightRegression.fitAndInfer(x0 = data[testSampleIdx])
    #     print ('predY: '+str(predY)+'----'+'targetY: '+str(label[testSampleIdx]))

    # PolyR

    # dataLoader = DataLoader(dataFile='data/workExpAndSalary.data')
    # # dataLoader = DataLoader(dataFile='data/housing.data') # 经测试 波士顿房价数据集无法用线性拟合
    # dataLoader.loadDataAsList(labelIdx=-1)
    # trainData = dataLoader.dataList[:int(len(dataLoader.dataList)*3/4)]
    # testData = dataLoader.dataList[int(len(dataLoader.dataList)*3/4 + 1):]
    # trainLabel = dataLoader.labelList[:int(len(dataLoader.dataList)*3/4)]
    # testLabel = dataLoader.labelList[int(len(dataLoader.dataList)*3/4 + 1):]
    # polynomialRegression = PolynomialRegression(highestPowerNum=2,inputData=trainData,inputLabel=trainLabel)
    # polynomialRegression.fit()
    # testData = DataFormator.linearDataFormator(testData)
    # for testSampleIdx in range(len(testData)):
    #     predY = polynomialRegression.infer(testData[testSampleIdx])
    #     print ('predY: '+str(predY)+'----'+'targetY: '+str(testLabel[testSampleIdx]))

    