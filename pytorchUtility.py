import os
import time

import torch
import torch.nn as nn

import numpy as np

from collections import Counter

def calAccuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def calDisagreementSamplesNoGroundTruth(predictionVectorsList, target):
    """filter the disagreed samples without ground truth"""
    batchSize = target.size(0)
    predictionList = list()
    
    for pVL in predictionVectorsList:
        _, pred = pVL.max(dim=1)
        predictionList.append(pred)
    
    sampleID = list()
    sampleTarget = list()
    predictions = list()
    predVectors = list()
    
    for i in xrange(batchSize):
        pred = []
        predVect = []
        allAgreed = True
        previousPrediction = -1
        for j, p in enumerate(predictionList):
            pred.append(p[i].item())
            predVect.append(predictionVectorsList[j][i])
            if previousPrediction == -1:
                previousPrediction = p[i]
                continue
            if p[i] != previousPrediction:
                allAgreed = False
        if not allAgreed:
            sampleID.append(i)
            sampleTarget.append(target[i].item())
            predictions.append(pred)
            predVectors.append(predVect)
    return sampleID, sampleTarget, predictions, predVectors


def calDisagreementSamplesOneTargetNegative(predictionVectorsList, target, oneTargetIdx):
    """filter the disagreed samples"""
    batchSize = target.size(0)
    predictionList = list()
    
    for pVL in predictionVectorsList:
        _, pred = pVL.max(dim=1)
        predictionList.append(pred)
    
    sampleID = list()
    sampleTarget = list()
    predictions = list()
    predVectors = list()
    
    for i in xrange(batchSize):
        pred = []
        predVect = []
        for j, p in enumerate(predictionList):
            pred.append(p[i].item())
            predVect.append(predictionVectorsList[j][i])
        if predictionList[oneTargetIdx][i] != target[i]:
            sampleID.append(i)
            sampleTarget.append(target[i].item())
            predictions.append(pred)
            predVectors.append(predVect)
    return sampleID, sampleTarget, predictions, predVectors


def filterModelsFixed(sampleID, sampleTarget, predictions, predVectors, selectModels):
    filteredPredictions = predictions[:, selectModels]
    filteredPredVectors = predVectors[:, selectModels]
    return sampleID, sampleTarget, filteredPredictions, filteredPredVectors

