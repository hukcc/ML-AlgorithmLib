#coding=utf-8
import numpy as np
import random

class BaseDataLoader:
    def __init__(self, dataDir=None, dataFile=None):
        self.dataDir = dataDir
        self.dataFile = dataFile
        self.dataList = None
        self.dataArray = None
        self.labelList = None

    def loadDataAsList(self):
        pass

    def loadDataAsArray(self):
        pass

class TxtDataloader(BaseDataLoader):
    def loadDataAsList(self, labelIdx):
        if self.dataFile != None:
            self.dataList = []
            self.labelList = []

            txtFile = open(self.dataFile, 'r', encoding='utf-8')
            lines = txtFile.readlines()
            txtFile.close()
            elements = None

            for line in lines:
                line = line.replace('\n','') # 先去除换行符
                line = line.replace('\t',',').replace('   ',',').replace('  ',',').replace(' ',',') # 解决各种杂七杂八的数据分隔符
                elements = line.split(',')
                # 清洗可能存在的错误分隔
                tempList = []
                for ele in elements:
                    if ele != '':
                        tempList.append(float(ele))
                elements = tempList
                label = elements.pop(labelIdx)
                self.labelList.append(label)
                self.dataList.append(elements)

    def loadDataAsArray(self, labelIdx):
        self.loadDataAsList(labelIdx)
        self.dataArray = np.array(self.dataList)

class ImgDirDataLoader(BaseDataLoader):
    pass

class IrisDataLoader(TxtDataloader):

    def loadDataAsList(self, labelIdx):
        if self.dataFile != None:
            self.dataList = []
            self.labelList = []

            txtFile = open(self.dataFile, 'r', encoding='utf-8')
            lines = txtFile.readlines()
            random.shuffle(lines)
            txtFile.close()
            elements = None

            for line in lines:
                line = line.replace('\n','') # 先去除换行符
                line = line.replace('\t',',').replace('   ',',').replace('  ',',').replace(' ',',') # 解决各种杂七杂八的数据分隔符
                elements = line.split(',')
                # 清洗可能存在的错误分隔
                tempList = []
                for ele in elements:
                    if ele != '':
                        ele = self.changeLabel2Number(ele)
                        tempList.append(float(ele))
                elements = tempList
                label = elements.pop(labelIdx)
                self.labelList.append(label)
                self.dataList.append(elements)

    def changeLabel2Number(self,label):
        if label == 'Iris-setosa':
            return 0
        elif label == 'Iris-versicolor':
            return 1
        elif label == 'Iris-virginica':
            return 2
        else:
            return label

class SonarDataLoader(IrisDataLoader):
    def changeLabel2Number(self,label):
        if label == 'R':
            return 0
        elif label == 'M':
            return 1
        else:
            return label
            
if __name__ == "__main__":
    txtDataloader = TxtDataloader(dataFile='data/housing.data')
    txtDataloader.loadDataAsList(labelIdx = -1)
    print('test')