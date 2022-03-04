import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def datasetGen(dataset,shuffler):
    dataset = sklearn.utils.shuffle(dataset, random_state=shuffler)
    X = dataset[['seplen', 'sepwid', 'petlen', 'petwid']]
    y = dataset['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=shuffler)

    # *****  normalization   ****:
    maxSeplen = X_train['seplen'].max()
    minSeplen = X_train['seplen'].min()

    maxSepwid = X_train['sepwid'].max()
    minSepwid = X_train['sepwid'].min()

    maxPetlen = X_train['petlen'].max()
    minPetlen = X_train['petlen'].min()

    maxPetwid = X_train['petwid'].max()
    minPetwid = X_train['petwid'].min()

    for i in range(0, X_train.shape[0]):
        X_train.iloc[i]['seplen'] = (X_train.iloc[i]['seplen'] - minSeplen) / (maxSeplen - minSeplen)
        X_train.iloc[i]['sepwid'] = (X_train.iloc[i]['sepwid'] - minSepwid) / (maxSepwid - minSepwid)
        X_train.iloc[i]['petlen'] = (X_train.iloc[i]['petlen'] - minPetlen) / (maxPetlen - minPetlen)
        X_train.iloc[i]['petwid'] = (X_train.iloc[i]['petwid'] - minPetwid) / (maxPetwid - minPetwid)

    for i in range(0, X_test.shape[0]):
        X_test.iloc[i]['seplen'] = (X_test.iloc[i]['seplen'] - minSeplen) / (maxSeplen - minSeplen)
        X_test.iloc[i]['sepwid'] = (X_test.iloc[i]['sepwid'] - minSepwid) / (maxSepwid - minSepwid)
        X_test.iloc[i]['petlen'] = (X_test.iloc[i]['petlen'] - minPetlen) / (maxPetlen - minPetlen)
        X_test.iloc[i]['petwid'] = (X_test.iloc[i]['petwid'] - minPetwid) / (maxPetwid - minPetwid)

    # ***** normalization done! *****
    return X_train, X_test, y_train, y_test

def doer(X_train, X_test, y_train, y_test,k):

    #knn start from here for training data
    Train_ans = []
    for i in range(0,X_train.shape[0]):
        curr_distance = []
        for j in range(0,X_train.shape[0]):
            curr_distance.append([distance.euclidean(X_train.iloc[i],X_train.iloc[j]),y_train.iloc[j]])
        temp = np.asarray(curr_distance)
        temp = temp[temp[:, 0].argsort()]
        names = ['Iris-virginica','Iris-versicolor','Iris-setosa']
        outputs = [0,0,0]
        for l in range(0,k):
            if(temp[l][1] == 'Iris-virginica'):
                outputs[0] += 1
            elif (temp[l][1] == 'Iris-versicolor'):
                outputs[1] += 1
            elif (temp[l][1] == 'Iris-setosa'):
                outputs[2] += 1
        max_value = max(outputs)
        max_index = outputs.index(max_value)
        if(names[max_index] == y_train.iloc[i]): Train_ans.append(1)
        else: Train_ans.append(0)

    #calculation for accuracy (Train Data)
    totalCorrect = 0;
    for i in Train_ans:
        if i == 1: totalCorrect+= 1
    accTrainData = totalCorrect / X_train.shape[0]
    # Training data done!

    # Test Data
    Test_data = []
    for i in range(0, X_test.shape[0]):
        curr_distance1 = []
        for j in range(0, X_train.shape[0]):
            curr_distance1.append([distance.euclidean(X_test.iloc[i], X_train.iloc[j]), y_train.iloc[j]])
        temp = np.asarray(curr_distance1)
        temp = temp[temp[:, 0].argsort()]
        names = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']
        outputs1 = [0, 0, 0]
        for l in range(0, k):
            if (temp[l][1] == 'Iris-virginica'):
                outputs1[0] += 1
            elif (temp[l][1] == 'Iris-versicolor'):
                outputs1[1] += 1
            elif (temp[l][1] == 'Iris-setosa'):
                outputs1[2] += 1
        max_value = max(outputs1)
        max_index = outputs1.index(max_value)
        if (names[max_index] == y_test.iloc[i]):
            Test_data.append(1)
        else:
            Test_data.append(0)

    # calculation for accuracy (Test Data)
    totalCorrect2 = 0;
    for i in Test_data:
        if i == 1: totalCorrect2 += 1
    accTrainData2 = totalCorrect2 / X_test.shape[0]

    return accTrainData,accTrainData2
