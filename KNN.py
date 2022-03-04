import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
import time
import matplotlib.pyplot as plt
from KNNHelper import datasetGen
from KNNHelper import doer

def KnnImplement():
    start = time.time()
    print(f'Program started')
    dataset = pd.read_csv('./iris.csv', names=['seplen', 'sepwid', 'petlen', 'petwid', 'label'])
    dic = {}
    ks = []
    testingAvg = []
    trainingAvg = []
    testingStd = []
    trainingStd = []
    print('yaha aya')
    for i in range(1, 21):
        curr = time.time()
        print(f'i = {i} started at {curr - start}')
        X_train, X_test, y_train, y_test = datasetGen(dataset, i)
        for k in range(1, 52, 2):
            now = time.time()
            print(f'k = {k} start at {now - start}')
            trData, teData = doer(X_train, X_test, y_train, y_test, k)
            if k in dic:
                dic[k][0].append(trData)
                dic[k][1].append(teData)
            else:
                dic[k] = [[trData], [teData]]

    for k in dic:
        ks.append(k)
        trainingAvg.append(np.average(dic[k][0]))
        trainingStd.append(np.std(dic[k][0]))
        testingAvg.append(np.average(dic[k][1]))
        testingStd.append(np.std(dic[k][1]))

    plot1 = plt.figure(1)
    plt.errorbar(x=ks, y=trainingAvg, yerr=trainingStd, marker='.')
    plt.xlabel('Value of k')
    plt.ylabel('(Accuracy over training data)')
    plt.title('Training Data')
    plt.ylim([0.88, 1.02])

    plot2 = plt.figure(2)
    plt.errorbar(x=ks, y=testingAvg, yerr=testingStd,marker = '.')
    plt.xlabel('Value of k')
    plt.ylabel('(Accuracy over testing data)')
    plt.title('testing Data')
    plt.ylim([0.88, 1.02])
    plt.show()
