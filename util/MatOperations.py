# coding=utf-8
import numpy as np

def MatTranspose(arr):
    if len(arr.shape) == 1:
        return arr.reshape(1,arr.shape[0])
    elif arr.shape[0] == None and arr.shape[1] != None:
        return arr.reshape(arr.shape[1],1)
    elif arr.shape[0] != None and arr.shape[1] != None:
        return arr.reshape(arr.shape[1],arr.shape[0])
    else:
        pass
    


if __name__ == "__main__":
    temp = np.array([1,2,34,1])
    print(temp.shape)
    temp = MatTranspose(temp)
    print(temp.shape)
