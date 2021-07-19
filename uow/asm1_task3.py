# -*- coding: utf-8 -*-

# Import relevant packages
import pandas as pd
import numpy as np

# Load dataset
lineList = pd.read_csv("wordsList", sep="\n", header=None).to_numpy()
wordList = np.array([np.array(document[0].split(',')) for document in lineList])
classList = pd.read_csv("classList", header=None).to_numpy().flatten()

# Create a vocabulary of words from the dataset
def createVocabList(dataSet):
    vocabSet = set([]) # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# Create a feature vector from an input
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1 # account for multiple occurrences of words
    return returnVec

# Train the Naive Bayesian Classifier from the train data matrix
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAdvert = np.sum(trainCategory) / float(numTrainDocs)
    p0Num = np.zeros(numWords) # as numerator
    p1Num = np.zeros(numWords) # as numerator
    p0Denom = 0 # as denominator
    p1Denom = 0 # as denominator
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return np.log(p0Vect), np.log(p1Vect), np.log(pAdvert) # Numerical Underflow

# Classify a single test case (Use logarithmic instead of probability)
def classifyNB0(vec2Classify, log0Vec, log1Vec, logAdvert):
    log1 = np.sum(np.multiply(log1Vec, vec2Classify)) + logAdvert
    log0 = np.sum(np.multiply(log0Vec, vec2Classify)) + np.log(1.0 - np.e**logAdvert)
    # return prediction with classification probability
    if log1 > log0:
        return (1, np.e**(log1-log0) / (np.e**(log1-log0) + 1))
    else:
        return (0, np.e**(log0-log1) / (np.e**(log0-log1) + 1))

# Initialize training and testing datasets (Stratified Sampling)
vocabList = createVocabList(wordList)
advertSet = wordList[(classList == 1)] # Advertisement email dataset
ordinarySet = wordList[(classList == 0)] # Original emails dataset

np.random.shuffle(advertSet)
np.random.shuffle(ordinarySet)

trainSet = np.concatenate((advertSet[:int(round(len(advertSet) * 66 / 72))], 
                          ordinarySet[:int(round(len(ordinarySet) * 66 / 72))]), axis=0)
trainLabels = [1 if i < int(round(len(advertSet) * 66 / 72)) else 0 for i in range(len(trainSet))]

testSet = np.concatenate((advertSet[int(round(len(advertSet) * 66 / 72)):], 
                         ordinarySet[int(round(len(ordinarySet) * 66 / 72)):]), axis=0)
testLabels = [1 if i < (len(advertSet) - int(round(len(advertSet) * 66 / 72))) else 0 for i in range(len(testSet))]

# Train the Naive Bayesian Classifier
trainMat = []
for email in trainSet:
    trainMat.append(bagOfWords2Vec(vocabList, email))

log0V, log1V, logAd = trainNB0(np.array(trainMat) + 0.01, np.array(trainLabels)) # Smoothing Zero Count

# Test the Bayesian Classifier
for idx, testEntry in enumerate(testSet):
    testDoc = np.array(bagOfWords2Vec(vocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB0(testDoc, log0V, log1V, logAd))
    print('Actual class: ', testLabels[idx])

