#coding = utf-8
import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
rootpath=str('/Users/hukcc/HYY_WorkSpace/ML_workspace')
syspath=sys.path
sys.path=[]
sys.path.append(rootpath)#将工程根目录加入到python搜索路径中
# 这里这样写主要是为了避免python的奇怪路径问题，也即用vscode时脚本很难认清自己所处的目录，给调试带来困难
from open_ML_lib.util.DataLoader import TxtDataloader as DataLoader
from open_ML_lib.util.DataLoader import IrisDataLoader
from open_ML_lib.util.DataLoader import SonarDataLoader
from open_ML_lib.util.DataFormator import DataFormator

class SoftmaxRegression:
    # 这里可以设置分类的类别 如果是两类 那么此时的softmax==logistic
    def __init__(self,featureNumber,kindNumber,inputData,inputLabel):
        self.name = 'softmax'
        self.func = None
        self.thetas = []
        for cnt in range(kindNumber):   # 这里去初始化每一个类别对应的theta -- 分几类就初始化几个
            self.thetas.append(np.random.rand(1,featureNumber+1))
        self.lr = 0.01
        self.epoch = 100
        inputData = DataFormator.linearDataFormator(inputData)
        self.inputData = inputData
        self.inputLabel = inputLabel

    def fit(self):
        for epoch in range(self.epoch):
            for sampleIdx in range(len(self.inputData)):
                # 遍历所有训练集样本 每一个样本都会更新所有的theta
                x = self.inputData[sampleIdx]
                y = self.inputLabel[sampleIdx]

                results = []
                newResults = []
                # 先正向推理计算出当前样本对应的概率值
                for theta in self.thetas:
                    results.append(math.exp(np.dot(x,theta.reshape(-1,1))))
                sum = np.sum(np.array(results))
                for result in results:
                    result = result/sum
                    newResults.append(result)
                results = newResults
                idx = 0
                newTheta = []
                # 根据更新公式（梯度下降法）去更新所有theta
                for theta in self.thetas:
                    theta = theta - self.lr*self.getGradient(x,results[idx],y,idx,theta)
                    newTheta.append(theta)
                    idx += 1
                self.thetas = newTheta
                

    def getGradient(self,x,y,target,idx,theta):  #对loss求导之后  
        if target == idx:
            return -x*(1-y)
        else:
            return -x*(0-y) 

    def infer(self,x):   # 输入应该是特征向量
        # 在推理的时候只需要计算当前样本的每一个概率值
        results = []
        newResults = []
        for theta in self.thetas:
            results.append(math.exp(np.dot(x,theta.reshape(-1,1))))
        sum = np.sum(np.array(results))
        for result in results:
            result = result/sum
            newResults.append(result)
        results = newResults
        return results

if __name__ == "__main__":
    # softmax

    ratio = 3/4
    # 这里直接用比例粗暴划分的原因是在做dataloader的时候又加入random.suffle，但是这样划分仍然会导致训练集和测试集分布不一致 这个问题可以在后面的作业中去进一步改进。
    dataLoader = SonarDataLoader(dataFile='data/sonar.all-data')
    dataLoader.loadDataAsList(labelIdx=-1) # 标记出label在数据中的位置 可以单独取出

    ratios = []
    accs = []

    for a in range(1,10):
        ratio = a/10

        trainData = dataLoader.dataList[:int(len(dataLoader.dataList)*ratio)]
        testData = dataLoader.dataList[int(len(dataLoader.dataList)*ratio + 1):]
        trainLabel = dataLoader.labelList[:int(len(dataLoader.dataList)*ratio)]
        testLabel = dataLoader.labelList[int(len(dataLoader.dataList)*ratio + 1):]
        softmaxRegression = SoftmaxRegression(featureNumber=len(trainData[0]),kindNumber=2,inputData=trainData,inputLabel=trainLabel)
        softmaxRegression.fit()
        testData = DataFormator.linearDataFormator(testData)    # 这里的formator实际上就是多增加一维特征 把偏置b加入进去
        acc = 0.0
        accCnt = 0
        for testSampleIdx in range(len(testData)):
            predY = softmaxRegression.infer(testData[testSampleIdx])
            predY = np.argmax(np.array(predY))
            if predY == testLabel[testSampleIdx]:
                accCnt += 1
            # print ('predY: '+str(predY)+'----'+'targetY: '+str(testLabel[testSampleIdx]))
        print('ratio: ' + str(ratio) + '------------------' +'acc: '+str(accCnt/len(testLabel)))
        ratios.append(ratio)
        accs.append(accCnt/len(testLabel))
    plt.plot(ratios,accs)
    plt.show()
