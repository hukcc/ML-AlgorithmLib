#coding=utf-8
import matplotlib.pyplot as plt
import random

import os
import sys
import numpy as np
import math
rootpath=str('/Users/hukcc/HYY_WorkSpace/ML_workspace')
syspath=sys.path
sys.path=[]
sys.path.append(rootpath)#将工程根目录加入到python搜索路径中

from open_ML_lib.util.DataLoader import TxtDataloader as DataLoader
from open_ML_lib.util.DataLoader import SonarDataLoader
from open_ML_lib.util.DataLoader import IrisDataLoader
from open_ML_lib.util.DataFormator import DataFormator

color = ['#F0F8FF',
                          '#00FFFF',
                          '#7FFFD4',
                          '#F5F5DC',
                          '#FFE4C4',
                          '#000000',
                          '#FFEBCD',
                          '#0000FF',
                          '#8A2BE2',
                          '#A52A2A',
                          '#DEB887',
                          '#7FFF00',
                          '#DC143C',
                          '#00FFFF',
                          '#00008B',
                          '#B8860B',
                          '#A9A9A9',
                          '#006400',
                          '#8B008B',
                          '#556B2F',
                          '#9932CC',
                          '#8B0000',
                          '#8FBC8F',
                          '#9400D3',
                          '#FF1493',
                          '#00BFFF',
                          '#696969',
                          '#1E90FF',
                          '#B22222',
                          '#FFFAF0',
                          '#228B22',
                          '#FF00FF',
                          '#DCDCDC',
                          '#FFD700',
                          '#808080',
                          '#008000',
                          '#ADFF2F',
                          '#F0FFF0',
                          '#FF69B4',
                          '#CD5C5C',
                          '#4B0082',
                          '#FFFACD',
                          '#ADD8E6',
                          '#F08080',
                          '#E0FFFF',
                          '#FAFAD2',
                          '#90EE90',
                          '#D3D3D3',
                          '#FFB6C1',
                          '#FFA07A',
                          '#20B2AA',
                          '#87CEFA',
                          '#778899',
                          '#B0C4DE',
                          '#FFFFE0',
                          '#00FF00',
                          '#32CD32',
                          '#FAF0E6',
                          '#FF00FF',
                          '#800000',
                          '#66CDAA',
                          '#0000CD',
                          '#BA55D3',
                          '#9370DB',
                          '#3CB371',
                          '#7B68EE',
                          '#00FA9A',
                          '#48D1CC',
                          '#C71585',
                          '#191970',
                          '#F5FFFA',
                          '#FFE4E1',
                          '#FFE4B5',
                          '#DDA0DD',
                          '#B0E0E6',
                          '#800080',
                          '#FF0000',
                          '#BC8F8F',
                          '#4169E1',
                          '#8B4513',
                          '#FA8072',
                          '#FAA460',
                          '#2E8B57',
                          '#FFF5EE',
                          '#A0522D',
                          '#C0C0C0',
                          '#87CEEB',
                          '#6A5ACD',
                          '#708090',
                          '#FFFAFA',
                          '#00FF7F',
                          '#4682B4',
                          '#D2B48C',
                          '#008080',
                          '#D8BFD8',
                          '#FF6347',
                          '#40E0D0',
                          '#EE82EE',
                          '#F5DEB3',
                          '#FFFFFF',
                          '#F5F5F5',
                          '#FFFF00',
                          '#9ACD32']

class DataVisualize:
    def showPoints(inputData,inputLabel=None):
        if inputLabel == None:
            for sampleIdx in range(len(inputData)):
                drawFeatureX = inputData[sampleIdx][0]
                drawFeatureY = inputData[sampleIdx][1]
                plt.scatter(drawFeatureX, drawFeatureY, s=5, c='k', marker='.')
            plt.show()
        else:
            allLabel = []
            labelColors = []
            
            for label in inputLabel:
                if label not in allLabel:
                    allLabel.append(label)
            labelColors = random.sample(color,len(allLabel))
            for sampleIdx in range(len(inputData)):
                drawFeatureX = inputData[sampleIdx][0]
                drawFeatureY = inputData[sampleIdx][1]
                label = inputLabel[sampleIdx]
                plt.scatter(drawFeatureX, drawFeatureY, s=5, c=labelColors[allLabel.index(label)], marker='.')
            plt.show()

    def showLines():
        pass

if __name__ == "__main__":
    dataLoader = SonarDataLoader(dataFile='data/sonar.all-data')
    # dataLoader = IrisDataLoader(dataFile='data/iris.data')
    dataLoader.loadDataAsList(labelIdx=-1)
    # DataVisualize.showPoints(dataLoader.dataList)
    DataVisualize.showPoints(dataLoader.dataList, dataLoader.labelList)