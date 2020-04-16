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

from open_ML_lib.util.DataLoader import TxtDataloader as DataLoader
from open_ML_lib.util.DataLoader import SonarDataLoader
from open_ML_lib.util.DataFormator import DataFormator

class GaussianDiscriminantAnalysisModel:
    # 这个类显得还是不够完善 以后想从这个类扩展出高斯混合模型估计比较困难
    def __init__(self,inputData,inputLabel):
        self.name = 'GDA'
        self.model = None
        # 初始化四个关键参数
        self.fi = None
        self.miu0 = None
        self.miu1 = None
        self.xigema = None

        self.inputData = inputData
        self.inputLabel = inputLabel
        self.n = len(inputLabel)

    def getModel(self):
        # GDA的思想其实很简单 这一点和他复杂的公式推导反差巨大
        # 获取模型的过程实际上就是根据不同的类别去求出该类别的分布
        self.fi = self.getFi()
        self.miu0 = self.getMiu0()
        self.miu1 = self.getMiu1()
        self.xigema = self.getXigema()

        self.model0 = self.gaussianModel0
        self.model1 = self.gaussianModel1

    def getFi(self):
        return np.mean(np.array(self.inputLabel))

    def getMiu0(self):
        temp = np.zeros(np.array(self.inputData[0]).shape)
        cnt = 0
        for sampleIdx in range(self.n):
            if self.inputLabel[sampleIdx] == 0:
                temp += self.inputData[sampleIdx]
                cnt += 1
        return temp/cnt

    def getMiu1(self):
        temp = np.zeros(np.array(self.inputData[0]).shape)
        cnt = 0
        for sampleIdx in range(self.n):
            if self.inputLabel[sampleIdx] == 1:
                temp += self.inputData[sampleIdx]
                cnt += 1
        return temp/cnt
    
    def getXigema(self):
        temp = np.zeros(np.array(self.inputData[0]).shape)
        temp = np.dot(temp.reshape(-1,1),temp.reshape(1,-1))
        for sampleIdx in range(self.n):
            if self.inputLabel[sampleIdx] == 0:
                temp += np.dot((np.array(self.inputData[sampleIdx]) - self.miu0).reshape(-1,1),(np.array(self.inputData[sampleIdx]) - self.miu0).reshape(1,-1))
            else:
                temp += np.dot((np.array(self.inputData[sampleIdx]) - self.miu1).reshape(-1,1),(np.array(self.inputData[sampleIdx]) - self.miu1).reshape(1,-1))
        return temp/self.n
    
    def gaussianModel0(self,x):
        return 1.0/(((2*math.pi)**(self.n/2.0))*(np.linalg.det(self.xigema)**0.5)) *\
             math.exp(-0.5*np.dot((x-self.miu0).reshape(1,-1),np.dot(np.linalg.inv(self.xigema),(x-self.miu0).reshape(-1,1))))
    
    def gaussianModel1(self,x):
        return 1.0/(((2*math.pi)**(self.n/2.0))*(np.linalg.det(self.xigema)**0.5)) *\
             math.exp(-0.5*np.dot((x-self.miu1).reshape(1,-1),np.dot(np.linalg.inv(self.xigema),(x-self.miu1).reshape(-1,1))))
    
    def infer(self,x):
        # 在推理的时候只需要比较这个模型属于哪个分布的概率更大就可以了
        p0 = self.model0(x)
        p1 = self.model1(x)
        return [p0,p1]

if __name__ == "__main__":
    ratio = 3/4

    dataLoader = SonarDataLoader(dataFile='data/sonar.all-data')
    dataLoader.loadDataAsList(labelIdx=-1)

    ratios = []
    accs = []

    for a in range(1,10):
        ratio = a/10
        trainData = dataLoader.dataList[:int(len(dataLoader.dataList)*0.9)]
        testData = dataLoader.dataList[int(len(dataLoader.dataList)*0.9 + 1):]
        trainLabel = dataLoader.labelList[:int(len(dataLoader.dataList)*0.9)]
        testLabel = dataLoader.labelList[int(len(dataLoader.dataList)*0.9 + 1):]
        gaussianDiscriminantAnalysisModel = GaussianDiscriminantAnalysisModel(inputData=trainData,inputLabel=trainLabel)
        gaussianDiscriminantAnalysisModel.getModel()

        acc = 0.0
        accCnt = 0
        for testSampleIdx in range(len(testData)):
            predY = gaussianDiscriminantAnalysisModel.infer(testData[testSampleIdx])
            predY = np.argmax(np.array(predY))
            if predY == testLabel[testSampleIdx]:
                accCnt += 1
            print ('predY: '+str(predY)+'----'+'targetY: '+str(testLabel[testSampleIdx]))
        print('acc: '+str(accCnt/len(testLabel)))
        ratios.append(ratio)
        accs.append(accCnt/len(testLabel))
    plt.plot(ratios,accs)
    plt.show()