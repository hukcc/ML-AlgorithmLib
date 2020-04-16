# coding=utf-8
import numpy as np
import math

def hardlim(num):
    if num>=0:
        return 1
    else:
        return 0
def sign(num):
    if num>=0:
        return 1
    else:
        return -1
def adjustDataFormat(data):
    newData = []
    for sample in data:
        sample = np.append(sample,1)
        newData.append(sample)
    return newData

class Perceptron():
    def __init__(self,inputNumber,activateFunc,inputData,inputLabel):
        # inputData = [ array[sample1],array[sample2] ]
        # inputLabel = [ label1,label2 ]
        self.W = np.random.rand(1,inputNumber+1)  # 生成的是列向量 这边和公式保持一致
        # 改变数据的格式 这边将W和b一起放到新的W向量中 另外np的转至实在是太麻烦了
        inputData = adjustDataFormat(inputData)
        self.inputData = inputData
        self.inputLabel = inputLabel
        # 选择激活函数 
        if activateFunc == 'hardlim':
            self.activateFunc = hardlim
        elif activateFunc == 'sign':
            self.activateFunc = sign

    def forward(self): 
        # 遍历数据集 并将分类错误的index保存在对象内 方便下面的adjust函数调用
        self.errorIdx = []
        for sampleIdx in range(len(self.inputData)):
            label = self.inputLabel[sampleIdx]
            sample = self.inputData[sampleIdx]

            result = np.dot(sample,self.W.reshape(-1,1))
            result = np.squeeze(result)
            result = result.tolist()
            result = self.activateFunc(result)
            if result != label:
                self.errorIdx.append(sampleIdx)
             

    def adjustW(self):
        # 把所有分错的样本利用更新公式更新W值
        for errorSampleIdx in self.errorIdx:
            label = self.inputLabel[errorSampleIdx]
            sample = self.inputData[errorSampleIdx]
            self.W = self.W + label*sample

# 单元测试
if __name__ == "__main__":
    testData = [np.array([1,1,1,1,1,1]).reshape(1,-1),np.array([2,2,2,2,2,2]).reshape(1,-1),np.array([10,-1,10,-1,-1,-1]).reshape(1,-1)]
    testLabel = [1,1,-1]
    test = Perceptron(inputNumber=6,activateFunc='sign',inputData=testData,inputLabel=testLabel)
    print(test.activateFunc(100))
    test.forward()
    test.adjustW()
    print('test')