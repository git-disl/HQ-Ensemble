#!/usr/bin/env python
# coding: utf-8

import os
import time

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


# obtain:
# team -> accuracy map
# model -> team
import timeit
teamAccuracyDict = dict()
modelTeamDict = dict()
teamNameDict = dict()
for n in range(2, len(models)+1):
    comb = combinations(list(range(len(models))), n)
    for selectedModels in list(comb):
        # accuracy
        tmpAccuracy = calAveragePredictionVectorAccuracy(predictionVectorsList, labelVectorsList[0], modelsList=selectedModels)[0].cpu().item()
        teamName = "".join(map(str, selectedModels))
        teamNameDict[teamName] = selectedModels
        teamAccuracyDict[teamName] = tmpAccuracy
        for m in teamName:
            if m in modelTeamDict:
                modelTeamDict[m].add(teamName)
            else:
                modelTeamDict[m] = set([teamName,])


# calculate the diversity measures for all configurations
np.random.seed(0)
nRandomSamples = 100
crossValidation = True
crossValidationTimes = 3

teamDiversityMetricMap = dict()
negAccuracyDict = dict()
diversityMetricsList = ['CK', 'BD','KW', 'GD']
startTime = timeit.default_timer()
for oneTargetModel in range(len(models)):
    sampleID, sampleTarget, predictions, predVectors = calDisagreementSamplesOneTargetNegative(trainPredictionVectorsList, trainLabelVectorsList[0], oneTargetModel)
    if len(predictions) == 0:
        print("negative sample not found")
        continue
    sampleID = np.array(sampleID)
    sampleTarget = np.array(sampleTarget)
    predictions = np.array(predictions)
    predVectors = np.array([np.array([np.array(pp) for pp in p]) for p in predVectors])
    for teamName in modelTeamDict[str(oneTargetModel)]:
        selectedModels = teamNameDict[teamName]
        teamSampleID, teamSampleTarget, teamPredictions, teamPredVectors =             filterModelsFixed(sampleID, sampleTarget, predictions, predVectors, selectedModels) 
        if crossValidation:
            tmpMetrics = list()
            for _ in range(crossValidationTimes):
                randomIdx = np.random.choice(np.arange(teamPredictions.shape[0]), nRandomSamples)        
                tmpMetrics.append(calAllDiversityMetrics(teamPredictions[randomIdx], teamSampleTarget[randomIdx], diversityMetricsList))
            tmpMetrics = np.mean(np.array(tmpMetrics), axis=0)
        else:
            tmpMetrics = np.array(calAllDiversityMetrics(teamPredictions, teamSampleTarget, diversityMetricsList))
        diversityMetricDict = {diversityMetricsList[i]:tmpMetrics[i].item()  for i in range(len(tmpMetrics))}
        targetDiversity = teamDiversityMetricMap.get(teamName, dict())
        targetDiversity[str(oneTargetModel)] = diversityMetricDict
        teamDiversityMetricMap[teamName] = targetDiversity
        
        tmpNegAccuracy = calAccuracy(torch.tensor(np.mean(np.transpose(teamPredVectors, (1, 0, 2)), axis=0)), torch.tensor(teamSampleTarget))[0].cpu().item()
        targetNegAccuracy = negAccuracyDict.get(teamName, dict())
        targetNegAccuracy[str(oneTargetModel)] = tmpNegAccuracy
        negAccuracyDict[teamName] = targetNegAccuracy

endTime = timeit.default_timer()
print("Time: ", endTime-startTime)


# calculate the targetTeamSizeDict
targetTeamSizeDict = dict()
for oneTargetModel in range(len(models)):
    for teamName in modelTeamDict[str(oneTargetModel)]:
        teamSize = len(teamName)
        teamSizeDict = targetTeamSizeDict.get(str(oneTargetModel), dict())
        fixedTeamDict = teamSizeDict.get(str(teamSize), dict())
        
        teamList = fixedTeamDict.get('TeamList', list())
        teamList.append(teamName)
        fixedTeamDict['TeamList'] = teamList
        
        # diversity measures
        diversityVector = np.expand_dims(np.array([teamDiversityMetricMap[teamName][str(oneTargetModel)][dm]
                                    for dm in diversityMetricsList]), axis=0)
        
        diversityMatrix = fixedTeamDict.get('DiversityMatrix', None)
        if diversityMatrix is None:
            diversityMatrix = diversityVector
        else:
            diversityMatrix = np.append(diversityMatrix, diversityVector, axis=0)
        fixedTeamDict['DiversityMatrix'] = diversityMatrix
        
        teamSizeDict[str(teamSize)] = fixedTeamDict
        targetTeamSizeDict[str(oneTargetModel)] = teamSizeDict 


for oneTargetModel in range(len(models)):
    for teamSize in range(2, len(models)):
        fixedTeamDict = targetTeamSizeDict[str(oneTargetModel)][str(teamSize)]
        teamList = fixedTeamDict['TeamList']
        accuracyList = [np.mean(negAccuracyDict[teamName].values()) for teamName in teamList]
        diversityMatrix = fixedTeamDict['DiversityMatrix']        
        scaledDiversityMeasures = list()
        for i in range(len(diversityMetricsList)):
            scaledDiversityMeasures.append(normalize01(diversityMatrix[:, i]))
        scaledDiversityMeasures = np.stack(scaledDiversityMeasures, axis=1)
        fixedTeamDict['ScaledDiversityMatrix'] = scaledDiversityMeasures
        targetTeamSizeDict[str(oneTargetModel)][str(teamSize)] = fixedTeamDict


# FQ diversity scores
teamList = set(teamAccuracyDict.keys()) - set(['0123456789'])
FQMetrics = dict()

for j, dm in enumerate(diversityMetricsList):
    FQMetricsDM = FQMetrics.get(dm, {})
    for teamName in teamList:
        if teamName in FQMetrics:
            continue
        tmpMetricList = list()
        teamModelIdx = map(int, [modelName for modelName in teamName])
        teamModelAcc = [tmpAccList[modelIdx][0].cpu().item() for modelIdx in teamModelIdx]
        teamModelWeights = np.argsort(teamModelAcc)
        tmpModelWeights = list()
        teamSize = len(teamName)
        if teamSize == len(tmpAccList):
            continue
        for (k, modelName) in enumerate(teamName):
            fixedTeamDict = targetTeamSizeDict[modelName][str(teamSize)]
            #print(modelName, teamSize)
            for i, tmpTeamName in enumerate(fixedTeamDict['TeamList']):
                if tmpTeamName == teamName:
                    tmpMetricList.append(fixedTeamDict['ScaledDiversityMatrix'][i, j])
                    tmpModelWeights.append(teamModelWeights[k])
        FQMetricsDM[teamName] = np.average(tmpMetricList, weights=tmpModelWeights)
    FQMetrics[dm] = FQMetricsDM


# team size list
teamList = set(teamAccuracyDict.keys()) - set(['0123456789'])
teamSizeDict = {} # teamSize -> teams map
for teamName in teamList:
    teamSize = len(teamName)
    sameTeamSizeSet = teamSizeDict.get(teamSize, set())
    sameTeamSizeSet.add(teamName)
    teamSizeDict[teamSize] = sameTeamSizeSet


# Mean-Threshold Based Pruning
FQMetricsThreshold = {}
teamSelectedFQMeanDict = {}

for j, dm in enumerate(diversityMetricsList):
    FQMetricsThreshold[dm] = np.mean(FQMetrics[dm].values())

print(FQMetricsThreshold)

for i, teamName in enumerate(teamList):
    for j, dm in enumerate(diversityMetricsList):
        if FQMetricsDM[teamName] < round(FQMetricsThreshold[dm], 3):
            teamSelectedFQMeanSet = teamSelectedFQMeanDict.get(dm, set())
            teamSelectedFQMeanSet.add(teamName)
            teamSelectedFQMeanDict[dm] = teamSelectedFQMeanSet
for dm in diversityMetricsList:
    print(dm, getNTeamStatisticsTeamName(list(teamSelectedFQMeanDict[dm]), 
                                     teamAccuracyDict, minAcc, avgAcc, maxAcc, 
                                     targetModel=maxModel, thresholdAcc=thresholdAcc))


# Hierarchical Pruning
targetTeamSize = 5
teamSelectedPrunedFQAllDict = {}
baseTeamsDict = {}

for dm in diversityMetricsList:
    toPrunePercentage = 0.1
    print(dm)
    teamSelectedPrunedFQSet = teamSelectedPrunedFQAllDict.get(dm, set())
    baseTeams = baseTeamsDict.get(dm, set())
    toPruneList = []
    FQAccTeamNamesDict = {}
    OutFQAccTeamNames = {}
    for teamSize in range(2, targetTeamSize+1):
        print("Team Size", teamSize, " Report")
        tmpFQAccTeamNames = FQAccTeamNamesDict.get(teamSize, [])
        tmpOutFQAccTeamNames = OutFQAccTeamNames.get(teamSize, [])
        for teamName in teamSizeDict[teamSize]:
            if not isTeamContainsAny(teamName, toPruneList):
                baseTeams.add(teamName)
                tmpFQAccTeamNames.append((FQMetrics[dm][teamName], teamAccuracyDict[teamName], teamName))
            else:
                tmpOutFQAccTeamNames.append((FQMetrics[dm][teamName], teamAccuracyDict[teamName], teamName))
        tmpFQAccTeamNames= sorted(tmpFQAccTeamNames, key=itemgetter(1))
        tmpFQAccTeamNames = sorted(tmpFQAccTeamNames, key=itemgetter(0), reverse=True)
        
        toPruneNum = int(len(tmpFQAccTeamNames) * toPrunePercentage)

        for tPIdx in range(min(toPruneNum, len(tmpFQAccTeamNames))):
            toPruneList.append(tmpFQAccTeamNames[tPIdx][2])
            tmpOutFQAccTeamNames.append(tmpFQAccTeamNames[tPIdx])
        tmpFQAccTeamNames = tmpFQAccTeamNames[tPIdx+1:]
        if len(tmpFQAccTeamNames) == 0:
            print("No in")
        else:
            tmpAcc = [fqATN[1] for fqATN in tmpFQAccTeamNames]
            teamSelectedPrunedFQSet.update([fqATN[2] for fqATN in tmpFQAccTeamNames])
            print("In Accuracy Range: ", min(tmpAcc), max(tmpAcc), getNTeamStatistics(tmpAcc, maxAcc, avgAcc, thresholdAcc))
        if len(tmpOutFQAccTeamNames) == 0:
            print("No out")
        else:
            tmpOutAcc = [fqOATN[1] for fqOATN in tmpOutFQAccTeamNames]
            print("Out Accuracy Range: ", min(tmpOutAcc), max(tmpOutAcc), getNTeamStatistics(tmpOutAcc, maxAcc, avgAcc, thresholdAcc))
        FQAccTeamNamesDict[teamSize] = tmpFQAccTeamNames
        OutFQAccTeamNames[teamSize] = tmpOutFQAccTeamNames

    teamSelectedPrunedFQAllDict[dm] = teamSelectedPrunedFQSet
    baseTeamsDict[dm] = baseTeams
