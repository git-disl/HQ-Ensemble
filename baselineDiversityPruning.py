#!/usr/bin/env python
# coding: utf-8

import os
import time
import timeit


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import Counter

from itertools import combinations


from pytorchUtility import *
import numpy as np
from operator import itemgetter
from EnsembleBench.groupMetrics import *
from EnsembleBench.teamSelection import *


# Dataset Configuration
predictionDir = './cifar10/prediction'
trainPredictionDir = './cifar10/train'
models = ['densenet-L190-k40', 'densenetbc-100-12', 'resnext8x64d', 'wrn-28-10-drop', 'vgg19_bn', 
          'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']
suffix = '.pt'

# baseline
maxModel = 0
thresholdAcc = 96.33


labelVectorsList = list()
predictionVectorsList = list()
tmpAccList = list()
for m in models:
    predictionPath = os.path.join(predictionDir, m+suffix)
    prediction = torch.load(predictionPath)
    predictionVectors = prediction['predictionVectors']
    predictionVectorsList.append(nn.functional.softmax(predictionVectors, dim=-1).cpu())
    labelVectors = prediction['labelVectors']
    labelVectorsList.append(labelVectors.cpu())
    tmpAccList.append(calAccuracy(predictionVectors, labelVectors))
    print(tmpAccList[-1])


minAcc = np.min(tmpAccList)
avgAcc = np.mean(tmpAccList)
maxAcc = np.max(tmpAccList)


trainLabelVectorsList = list()
trainPredictionVectorsList = list()
for m in models:
    trainPredictionPath = os.path.join(trainPredictionDir, m+suffix)
    trainPrediction = torch.load(trainPredictionPath)
    trainPredictionVectors = trainPrediction['predictionVectors']
    trainPredictionVectorsList.append(nn.functional.softmax(trainPredictionVectors, dim=-1).cpu())
    trainLabelVectors = trainPrediction['labelVectors']
    trainLabelVectorsList.append(labelVectors.cpu())


sampleID, sampleTarget, predictions, predVectors = calDisagreementSamplesNoGroundTruth(trainPredictionVectorsList, trainLabelVectorsList[0])


sampleID = np.array(sampleID)
sampleTarget = np.array(sampleTarget)
predictions = np.array(predictions)
predVectors = np.array([np.array([np.array(pp) for pp in p]) for p in predVectors])

nModels = len(predictions[0])
modelIdx = list(range(nModels))


# statistics for different metrics
np.random.seed(0)
crossValidationTimes = 3
nRandomSamples = 100

accuracyList = list()
#negAccuracyList = list()
kappaList = list()
binaryDisagreementList = list()
kwVarianceList = list()
GDList = list()

teamSizeList = list()
teamList = list()

startTime = timeit.default_timer()
for n in range(2, nModels+1):
    kappa_scores = []
    comb = combinations(modelIdx, n)
    best_comb = None
    best_kappa_score = 1.0
    best_accuracy = 0.0
    best_nSamples = len(predictions)
    accuracies = []
    for selectedModels in list(comb):
        teamSampleID, teamSampleTarget, teamPredictions, teamPredVectors =         filterModelsFixed(sampleID, sampleTarget, predictions, predVectors, selectedModels) 
        
        if len(teamPredictions) == 0:
            continue
        
        cur_kappa_scores = list()
        cur_binary_disagreements = list()
        cur_kw_variances = list()
        cur_GDs = list()
        
        for _ in range(crossValidationTimes):
            randomIdx = np.random.choice(np.arange(teamPredictions.shape[0]), nRandomSamples)
            cur_kappa_scores.append(group_kappa_score(teamPredictions[randomIdx]))
            cur_binary_disagreements.append(group_binary_disagreement(teamPredictions[randomIdx], teamSampleTarget[randomIdx]))
            cur_kw_variances.append(group_KW_variance(teamPredictions[randomIdx], teamSampleTarget[randomIdx]))
            cur_GDs.append(group_generalized_diversity(teamPredictions[randomIdx], teamSampleTarget[randomIdx]))
        
        kappaList.append(np.mean(cur_kappa_scores))
        binaryDisagreementList.append(np.mean(cur_binary_disagreements))
        kwVarianceList.append(np.mean(cur_kw_variances))
        GDList.append(np.mean(cur_GDs))
        
        tmpAccuracy = calAveragePredictionVectorAccuracy(predictionVectorsList, labelVectorsList[0], modelsList=selectedModels)[0].cpu().numpy()
        accuracyList.append(tmpAccuracy)
        teamSizeList.append(n)
        teamList.append(selectedModels)
endTime = timeit.default_timer()
print("Time: ", endTime-startTime)


accuracyList = np.array(accuracyList)
kappaList = np.array(kappaList)
binaryDisagreementList = np.array(binaryDisagreementList)
kwVarianceList = np.array(kwVarianceList)
GDList = np.array(GDList)

teamSizeList = np.array(teamSizeList)
teamList = np.array(teamList)

QData = {"Acc": accuracyList, 
         "CK": kappaList,
         "BD": binaryDisagreementList,
         "KW": kwVarianceList,
         "GD": GDList,
         "teamSizeList": teamSizeList,
         "teamList": teamList}
diversityMetricsList = ['CK', 'BD', 'KW', 'GD']


teamAccuracyDict = {}
for acc, t in zip(accuracyList, teamList):
    teamAccuracyDict["".join(map(str, t))] = acc


QMetrics = {}
QMetricsThreshold = {}
teamSelectedQAllDict = {}


for j, dm in enumerate(diversityMetricsList):
    if dm in ["CK", "QS", "FK"]:
        QMetricsThreshold[dm] = np.mean(QData[dm])
    elif dm in ["BD", "KW", "GD"]:
        QMetricsThreshold[dm] = np.mean(1.0-QData[dm])

print(QMetricsThreshold)

for i, t in enumerate(QData["teamList"]):
    teamName = "".join(map(str, t))
    for j, dm in enumerate(diversityMetricsList):
        QMetricsDM = QMetrics.get(dm, {})
        if dm in ["CK", "QS", "FK"]:
            QMetricsDM[teamName] = QData[dm][i]
        elif dm in ["BD", "KW", "GD"]:
            QMetricsDM[teamName] = 1.0 - QData[dm][i]
        QMetrics[dm] = QMetricsDM
        if QMetricsDM[teamName] < round(QMetricsThreshold[dm], 3):
            teamSelectedQAllSet = teamSelectedQAllDict.get(dm, set())
            teamSelectedQAllSet.add(teamName)
            teamSelectedQAllDict[dm] = teamSelectedQAllSet

for dm in diversityMetricsList:
    print(dm, getNTeamStatisticsTeamName(list(teamSelectedQAllDict[dm]), 
                                         teamAccuracyDict, minAcc, avgAcc, maxAcc, 
                                         targetModel=maxModel, thresholdAcc=thresholdAcc))
