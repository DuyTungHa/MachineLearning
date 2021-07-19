# -*- coding: utf-8 -*-

# Import relevant packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from scipy import stats

# Load dataset
col_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConcl', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
magic04 = pd.read_csv("magic04.data", names=col_names, sep=',', header=None)

# Discretize the numerical features into 10 equally-sized bins
magic04_tr = magic04.copy()
est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
magic04_tr.iloc[:, 0:10] = est.fit_transform(magic04.iloc[:, 0:10])

class DecisionTreeIG:
    def __init__(self, ops=(10, )):
        # ops is a tree pre-pruning argument. 
        # If the size of split data set is smaller than ops[0], stop the split process. 
        self.ops = ops
        
    # calculate Shannon Entropy of a dataset
    def calcShannonEnt(self, dataSet):
        numEntries = len(dataSet) # number of tuples
        labelCounts = {}
        for idx, featVec in dataSet.iterrows():
            currentLabel = featVec[-1] # class label is last element in each tuple
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries
            shannonEnt -= prob * np.math.log(prob, 2)
        return shannonEnt
    
    # split dataset into two halves based in the splitting value 
    # delete the feature from the subdatasets after splitting
    def biSplitDataSet(self, dataSet, featIndex, splitVal):
        subDS0 = dataSet.loc[dataSet.iloc[:, featIndex] <= splitVal]
        subDS1 = dataSet.loc[dataSet.iloc[:, featIndex] > splitVal]
        return (subDS0.drop(subDS0.columns[featIndex], axis=1), subDS1.drop(subDS1.columns[featIndex], axis=1))
    
    # choose the best attribute for splitting dataset
    def chooseBestBiSplit(self, dataSet):
        tolN = self.ops[0]
        _, n = np.shape(dataSet)
        baseEntropy = self.calcShannonEnt(dataSet)
        bestInfoGain = 0.0; bestIndex = None; bestSplitValue = 0
        
        for featIndex in range(n - 1):
            for splitVal in set(dataSet.iloc[:, featIndex]):
                subDS0, subDS1 = self.biSplitDataSet(dataSet, featIndex, splitVal)
                
                if (len(subDS0) < tolN) or (len(subDS1) < tolN): # tree pre-pruning
                    continue
                
                probDS0 = len(subDS0) / float(len(dataSet))
                newEntropy = probDS0 * self.calcShannonEnt(subDS0) + (1 - probDS0) * self.calcShannonEnt(subDS1)
                
                infoGain = baseEntropy - newEntropy
                if (infoGain > bestInfoGain):
                    bestInfoGain = infoGain
                    bestIndex = featIndex
                    bestSplitValue = splitVal
        
        return bestIndex, bestSplitValue
    
    # train the decision tree model
    def fit(self, dataSet):
        self.classLabels = list(set(dataSet.iloc[:, -1]))
        self.root = self.generateDecisionTree(dataSet)
    
    def generateDecisionTree(self, dataSet):
        tolN = self.ops[0]
        node = [0.0 for label in self.classLabels] # default a node to a leaf
        # if tuples in dataset are of the same class, return a leaf node 
        if len(set(dataSet.iloc[:, -1])) == 1:
            node[self.classLabels.index(dataSet.iloc[:, -1].values[0])] = 1.0
            return node
        
        # if no attributes left or dataset is too small, return a leaf node 
        if dataSet.shape[1] == 1 or dataSet.shape[0] < (tolN * 2):
            for idx, label in enumerate(self.classLabels):
                node[idx] = len(dataSet.loc[dataSet.iloc[:, -1] == label]) / len(dataSet)
            return node
        
        bestIndex, bestSplitValue = self.chooseBestBiSplit(dataSet)
        
        # if cannot find a feature to split, return a leaf node
        if bestIndex == None:
            for idx, label in enumerate(self.classLabels):
                node[idx] = len(dataSet.loc[dataSet.iloc[:, -1] == label]) / len(dataSet)
            return node
        
        subDS0, subDS1 = self.biSplitDataSet(dataSet, bestIndex, bestSplitValue)
        
        # node now become a subtree 
        # attach a new child node returned by applying generateDecisionTree on the subset recursively
        node = {"splitting_feature": bestIndex, "splitting_threshold": bestSplitValue, 
                "left": self.generateDecisionTree(subDS0), "right": self.generateDecisionTree(subDS1)}
        
        return node
    
    def classify(self, node, data):
        if not isinstance(node, dict):
            return self.classLabels[node.index(max(node))]
        elif data[node["splitting_feature"]] <= node["splitting_threshold"]:
            return self.classify(node["left"], data)
        else:
            return self.classify(node["right"], data)
        
    def predict(self, dataSet):
        return [self.classify(self.root, data) for idx, data in dataSet.iterrows()]

class DecisionTreeGR:
    def __init__(self, ops=(10, )):
        # ops is a tree pre-pruning argument. 
        # If the size of split data set is smaller than ops[0], stop the split process. 
        self.ops = ops
    
    def calcSplitInfo(self, dataSet, subsets):
        splitInfo = 0.0
        for subset in subsets:
            splitInfo -= (len(subset) / len(dataSet)) * np.math.log(len(subset) / len(dataSet), 2)
        return splitInfo
        
    # calculate Shannon Entropy of a dataset
    def calcShannonEnt(self, dataSet):
        numEntries = len(dataSet) # number of tuples
        labelCounts = {}
        for idx, featVec in dataSet.iterrows():
            currentLabel = featVec[-1] # class label is last element in each tuple
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries
            shannonEnt -= prob * np.math.log(prob, 2)
        return shannonEnt
    
    # split dataset into two halves based in the splitting value 
    # delete the feature from the subdatasets after splitting
    def biSplitDataSet(self, dataSet, featIndex, splitVal):
        subDS0 = dataSet.loc[dataSet.iloc[:, featIndex] <= splitVal]
        subDS1 = dataSet.loc[dataSet.iloc[:, featIndex] > splitVal]
        return (subDS0.drop(subDS0.columns[featIndex], axis=1), subDS1.drop(subDS1.columns[featIndex], axis=1))
    
    # choose the best attribute for splitting dataset
    def chooseBestBiSplit(self, dataSet):
        tolN = self.ops[0]
        _, n = np.shape(dataSet)
        baseEntropy = self.calcShannonEnt(dataSet)
        bestGainRatio = 0.0; bestIndex = None; bestSplitValue = 0
        
        for featIndex in range(n - 1):
            for splitVal in set(dataSet.iloc[:, featIndex]):
                subDS0, subDS1 = self.biSplitDataSet(dataSet, featIndex, splitVal)
                
                if (len(subDS0) < tolN) or (len(subDS1) < tolN): # tree pre-pruning
                    continue
                
                probDS0 = len(subDS0) / float(len(dataSet))
                newEntropy = probDS0 * self.calcShannonEnt(subDS0) + (1 - probDS0) * self.calcShannonEnt(subDS1)
                
                gainRatio = (baseEntropy - newEntropy) / self.calcSplitInfo(dataSet, [subDS0, subDS1])
                if (gainRatio > bestGainRatio):
                    bestGainRatio = gainRatio
                    bestIndex = featIndex
                    bestSplitValue = splitVal
        
        return bestIndex, bestSplitValue
    
    # train the decision tree model
    def fit(self, dataSet):
        self.classLabels = list(set(dataSet.iloc[:, -1]))
        self.root = self.generateDecisionTree(dataSet)
    
    def generateDecisionTree(self, dataSet):
        tolN = self.ops[0]
        node = [0.0 for label in self.classLabels] # default a node to a leaf
        # if tuples in dataset are of the same class, return a leaf node 
        if len(set(dataSet.iloc[:, -1])) == 1:
            node[self.classLabels.index(dataSet.iloc[:, -1].values[0])] = 1.0
            return node
        
        # if no attributes left or dataset is too small, return a leaf node 
        if dataSet.shape[1] == 1 or dataSet.shape[0] < (tolN * 2):
            for idx, label in enumerate(self.classLabels):
                node[idx] = len(dataSet.loc[dataSet.iloc[:, -1] == label]) / len(dataSet)
            return node
        
        bestIndex, bestSplitValue = self.chooseBestBiSplit(dataSet)
        
        # if cannot find a feature to split, return a leaf node
        if bestIndex == None:
            for idx, label in enumerate(self.classLabels):
                node[idx] = len(dataSet.loc[dataSet.iloc[:, -1] == label]) / len(dataSet)
            return node
        
        subDS0, subDS1 = self.biSplitDataSet(dataSet, bestIndex, bestSplitValue)
        
        # node now become a subtree 
        # attach a new child node returned by applying generateDecisionTree on the subset recursively
        node = {"splitting_feature": bestIndex, "splitting_threshold": bestSplitValue, 
                "left": self.generateDecisionTree(subDS0), "right": self.generateDecisionTree(subDS1)}
        
        return node
    
    def classify(self, node, data):
        if not isinstance(node, dict):
            return self.classLabels[node.index(max(node))]
        elif data[node["splitting_feature"]] <= node["splitting_threshold"]:
            return self.classify(node["left"], data)
        else:
            return self.classify(node["right"], data)
        
    def predict(self, dataSet):
        return [self.classify(self.root, data) for idx, data in dataSet.iterrows()]

class DecisionTreeVA:
    def __init__(self, ops=(10, )):
        # ops is a tree pre-pruning argument. 
        # If the size of split data set is smaller than ops[0], stop the split process. 
        self.ops = ops
        
    # calculate Variance of a dataset (binary classification)
    def calcVariance(self, dataSet):
        numEntries = len(dataSet) # number of tuples
        labelCounts = {}
        for idx, featVec in dataSet.iterrows():
            currentLabel = featVec[-1] # class label is last element in each tuple
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        prob = float(labelCounts[list(labelCounts.keys())[0]]) / numEntries
        return prob * (1 - prob)
    
    # split dataset into two halves based in the splitting value 
    # delete the feature from the subdatasets after splitting
    def biSplitDataSet(self, dataSet, featIndex, splitVal):
        subDS0 = dataSet.loc[dataSet.iloc[:, featIndex] <= splitVal]
        subDS1 = dataSet.loc[dataSet.iloc[:, featIndex] > splitVal]
        return (subDS0.drop(subDS0.columns[featIndex], axis=1), subDS1.drop(subDS1.columns[featIndex], axis=1))
    
    # choose the best attribute for splitting dataset
    def chooseBestBiSplit(self, dataSet):
        tolN = self.ops[0]
        _, n = np.shape(dataSet)
        baseVariance = self.calcVariance(dataSet)
        bestDeltaVariance = 0.0; bestIndex = None; bestSplitValue = 0
        
        for featIndex in range(n - 1):
            for splitVal in set(dataSet.iloc[:, featIndex]):
                subDS0, subDS1 = self.biSplitDataSet(dataSet, featIndex, splitVal)
                
                if (len(subDS0) < tolN) or (len(subDS1) < tolN): # tree pre-pruning
                    continue
                
                probDS0 = len(subDS0) / float(len(dataSet))
                newVariance = probDS0 * self.calcVariance(subDS0) + (1 - probDS0) * self.calcVariance(subDS1)
                
                deltaVariance = baseVariance - newVariance
                if (deltaVariance > bestDeltaVariance):
                    bestDeltaVariance = deltaVariance
                    bestIndex = featIndex
                    bestSplitValue = splitVal
        
        return bestIndex, bestSplitValue
    
    # train the decision tree model
    def fit(self, dataSet):
        self.classLabels = list(set(dataSet.iloc[:, -1]))
        self.root = self.generateDecisionTree(dataSet)
    
    def generateDecisionTree(self, dataSet):
        tolN = self.ops[0]
        node = [0.0 for label in self.classLabels] # default a node to a leaf
        # if tuples in dataset are of the same class, return a leaf node 
        if len(set(dataSet.iloc[:, -1])) == 1:
            node[self.classLabels.index(dataSet.iloc[:, -1].values[0])] = 1.0
            return node
        
        # if no attributes left or dataset is too small, return a leaf node 
        if dataSet.shape[1] == 1 or dataSet.shape[0] < (tolN * 2):
            for idx, label in enumerate(self.classLabels):
                node[idx] = len(dataSet.loc[dataSet.iloc[:, -1] == label]) / len(dataSet)
            return node
        
        bestIndex, bestSplitValue = self.chooseBestBiSplit(dataSet)
        
        # if cannot find a feature to split, return a leaf node
        if bestIndex == None:
            for idx, label in enumerate(self.classLabels):
                node[idx] = len(dataSet.loc[dataSet.iloc[:, -1] == label]) / len(dataSet)
            return node
        
        subDS0, subDS1 = self.biSplitDataSet(dataSet, bestIndex, bestSplitValue)
        
        # node now become a subtree 
        # attach a new child node returned by applying generateDecisionTree on the subset recursively
        node = {"splitting_feature": bestIndex, "splitting_threshold": bestSplitValue, 
                "left": self.generateDecisionTree(subDS0), "right": self.generateDecisionTree(subDS1)}
        
        return node
    
    def classify(self, node, data):
        if not isinstance(node, dict):
            return self.classLabels[node.index(max(node))]
        elif data[node["splitting_feature"]] <= node["splitting_threshold"]:
            return self.classify(node["left"], data)
        else:
            return self.classify(node["right"], data)
        
    def predict(self, dataSet):
        return [self.classify(self.root, data) for idx, data in dataSet.iterrows()]

class DecisionTreeGI:
    def __init__(self, ops=(10, )):
        # ops is a tree pre-pruning argument. 
        # If the size of split data set is smaller than ops[0], stop the split process. 
        self.ops = ops
        
    # calculate Gini Index of a dataset
    def calcGiniIndex(self, dataSet):
        numEntries = len(dataSet) # number of tuples
        labelCounts = {}
        for idx, featVec in dataSet.iterrows():
            currentLabel = featVec[-1] # class label is last element in each tuple
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        giniIdx = 1.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries
            giniIdx -= prob ** 2
        return giniIdx
    
    # split dataset into two halves based in the splitting value
    # delete the feature from the subdatasets after splitting
    def biSplitDataSet(self, dataSet, featIndex, splitVal):
        subDS0 = dataSet.loc[dataSet.iloc[:, featIndex] <= splitVal]
        subDS1 = dataSet.loc[dataSet.iloc[:, featIndex] > splitVal]
        return (subDS0.drop(subDS0.columns[featIndex], axis=1), subDS1.drop(subDS1.columns[featIndex], axis=1))
    
    # choose the best attribute for splitting dataset
    def chooseBestBiSplit(self, dataSet):
        tolN = self.ops[0]
        _, n = np.shape(dataSet)
        baseGiniIdx = self.calcGiniIndex(dataSet)
        bestdeltaGini = 0.0; bestIndex = None; bestSplitValue = 0
        
        for featIndex in range(n - 1):
            for splitVal in set(dataSet.iloc[:, featIndex]):
                subDS0, subDS1 = self.biSplitDataSet(dataSet, featIndex, splitVal)
                
                if (len(subDS0) < tolN) or (len(subDS1) < tolN): # tree pre-pruning
                    continue
                
                probDS0 = len(subDS0) / float(len(dataSet))
                newGiniIdx = probDS0 * self.calcGiniIndex(subDS0) + (1 - probDS0) * self.calcGiniIndex(subDS1)
                
                deltaGini = baseGiniIdx - newGiniIdx
                if (deltaGini > bestdeltaGini):
                    bestdeltaGini = deltaGini
                    bestIndex = featIndex
                    bestSplitValue = splitVal
        
        return bestIndex, bestSplitValue
    
    # train the decision tree model
    def fit(self, dataSet):
        self.classLabels = list(set(dataSet.iloc[:, -1]))
        self.root = self.generateDecisionTree(dataSet)
    
    def generateDecisionTree(self, dataSet):
        tolN = self.ops[0]
        node = [0.0 for label in self.classLabels] # default a node to a leaf
        # if tuples in dataset are of the same class, return a leaf node 
        if len(set(dataSet.iloc[:, -1])) == 1:
            node[self.classLabels.index(dataSet.iloc[:, -1].values[0])] = 1.0
            return node
        
        # if no attributes left or dataset is too small, return a leaf node 
        if dataSet.shape[1] == 1 or dataSet.shape[0] < (tolN * 2):
            for idx, label in enumerate(self.classLabels):
                node[idx] = len(dataSet.loc[dataSet.iloc[:, -1] == label]) / len(dataSet)
            return node
        
        bestIndex, bestSplitValue = self.chooseBestBiSplit(dataSet)
        
        # if cannot find a feature to split, return a leaf node
        if bestIndex == None:
            for idx, label in enumerate(self.classLabels):
                node[idx] = len(dataSet.loc[dataSet.iloc[:, -1] == label]) / len(dataSet)
            return node
        
        subDS0, subDS1 = self.biSplitDataSet(dataSet, bestIndex, bestSplitValue)
        
        # node now become a subtree 
        # attach a new child node returned by applying generateDecisionTree on the subset recursively
        node = {"splitting_feature": bestIndex, "splitting_threshold": bestSplitValue, 
                "left": self.generateDecisionTree(subDS0), "right": self.generateDecisionTree(subDS1)}
        
        return node
    
    def classify(self, node, data):
        if not isinstance(node, dict):
            return self.classLabels[node.index(max(node))]
        elif data[node["splitting_feature"]] <= node["splitting_threshold"]:
            return self.classify(node["left"], data)
        else:
            return self.classify(node["right"], data)
        
    def predict(self, dataSet):
        return [self.classify(self.root, data) for idx, data in dataSet.iterrows()]
    
    def getMean(self, node):
        if isinstance(node["left"], dict):
            node["left"] = self.getMean(node["left"])
        if isinstance(node["right"], dict):
            node["right"] = self.getMean(node["right"])
        return [sum(x) / 2.0 for x in zip(node["left"], node["right"])]
    
    def prune(self, node, dataSet):
        if len(dataSet) == 0:
            return self.getMean(node) # if no test data collapse the tree
        lSet, rSet = self.biSplitDataSet(dataSet, node["splitting_feature"], node["splitting_threshold"])
        if isinstance(node["left"], dict):
            node["left"] = self.prune(node["left"], lSet)
        if isinstance(node["right"], dict):
            node["right"] = self.prune(node["right"], rSet)
        # if they are both leaves, see if we can merge them
        if not isinstance(node["left"], dict) and not isinstance(node["right"], dict):
            leftPred = self.classLabels[node["left"].index(max(node["left"]))]
            rightPred = self.classLabels[node["right"].index(max(node["right"]))]
            nodeMean = self.getMean(node)
            mergePred = self.classLabels[nodeMean.index(max(nodeMean))]
            errNoMerge = sum(lSet.iloc[:, -1] != leftPred) + sum(rSet.iloc[:, -1] != rightPred)
            errMerge = sum(dataSet.iloc[:, -1] != mergePred)
            if errMerge < errNoMerge:
                return nodeMean
        return node
    
    def postPruning(self, testData):
        self.root = self.prune(self.root, testData)

class DecisionTreeMStar:
    def __init__(self, ops=(10, )):
        self.modelIG = DecisionTreeIG(ops)
        self.modelGR = DecisionTreeGR(ops)
        self.modelGA = DecisionTreeVA(ops)
    
    def fit(self, dataSet):
        self.modelIG.fit(dataSet)
        self.modelGR.fit(dataSet)
        self.modelGA.fit(dataSet)
    
    def predict(self, dataSet):
        predictionIG = self.modelIG.predict(dataSet)
        predictionGR = self.modelGR.predict(dataSet)
        predictionGA = self.modelGA.predict(dataSet)
        return [label if (label == predictionGR[idx]) or (label == predictionGA[idx]) else predictionGR[idx]
                for idx, label in enumerate(predictionIG)]

# 10-fold cross-validation to evaluate M* and Mgi

# Initialize models
modelGI = DecisionTreeGI(ops=(100, ))
modelMS = DecisionTreeMStar(ops=(100, ))

# Randomly partition the data into 10 mutually exclusive subsets, each approximately equal size
dataShuffled = magic04_tr.sample(frac = 1)
subsetSize = int(len(dataShuffled) / 10)
dataSubsets = [dataShuffled.iloc[i * subsetSize: (i + 1) * subsetSize, :] if i < 9 else dataShuffled.iloc[i * subsetSize: , :]
               for i in range(10)]

# Compute error rate of two models
errGI = np.zeros(10); errMS = np.zeros(10)
for i in range(10):
    valSet = dataSubsets[i]
    trainSet = pd.concat(dataSubsets[:i] + dataSubsets[i + 1:])
    labels = valSet.iloc[:, -1].values
    modelGI.fit(trainSet)
    modelMS.fit(trainSet)
    predictionGI = modelGI.predict(valSet)
    predictionMS = modelMS.predict(valSet)
    errGI[i] = sum([0 if predict == labels[idx] else 1 for idx, predict in enumerate(predictionGI)]) / len(valSet)
    errMS[i] = sum([0 if predict == labels[idx] else 1 for idx, predict in enumerate(predictionMS)]) / len(valSet)

# Compute the p-value of the error rate difference between two models
meanErrGI = np.mean(errGI)
meanErrMS = np.mean(errMS)
print('The mean error rate difference between Mgi and M* is: ', meanErrGI - meanErrMS)
varGIMS = sum([(errGI[i] - errMS[i] - (meanErrGI - meanErrMS))**2 for i in range(10)]) * 0.1
tScore = (meanErrGI - meanErrMS) / ((varGIMS / 10)**0.5)
pval = stats.t.sf(np.abs(tScore), 9) * 2
print('p-value: ', pval)

# Whether to reject the null hypothesis based on significance level (0.05)
if pval > 0.05:
    print('Any difference between M* and Mgi is by chance')
else:
    print('Statistically significant difference between M* and Mgi')

# Use stratified sampling to split dataset into training, pruning, and evaluation sets
gamma_set = magic04_tr.loc[magic04_tr.iloc[:, -1] == 'g'].sample(frac = 1)
hadron_set = magic04_tr.loc[magic04_tr.iloc[:, -1] == 'h'].sample(frac = 1)
strat_train_set = pd.concat([gamma_set[0: int(len(gamma_set) / 3)], hadron_set[0: int(len(hadron_set) / 3)]]).sample(frac = 1)
strat_val_set = pd.concat([gamma_set[int(len(gamma_set) / 3): int(len(gamma_set) * 2 / 3)], 
                           hadron_set[int(len(hadron_set) / 3) : int(len(hadron_set) * 2 / 3)]]).sample(frac = 1)
strat_test_set = pd.concat([gamma_set[int(len(gamma_set) * 2 / 3):], 
                           hadron_set[int(len(hadron_set) * 2 / 3):]]).sample(frac = 1)

# Calculate true positive and false positive rates
def calcTPFP(labels, predictions):
    TP = 0
    FP = 0
    for idx, label in enumerate(labels):
        if label == 'g' and predictions[idx] == 'g':
            TP += 1
        if label == 'h' and predictions[idx] == 'g':
            FP += 1
    return (TP / len(labels), FP / len(labels))

modelGI = DecisionTreeGI(ops=(50, )) # Initialize model
modelGI.fit(strat_train_set) # Train model
pred = modelGI.predict(strat_test_set) # Make predictions for the unpruned Mgi
modelGI.postPruning(strat_val_set) # Post Pruning
predPrune = modelGI.predict(strat_test_set) # Make predictions for the pruned Mgi
TP, FP = calcTPFP(strat_test_set.iloc[:, -1], pred) # TP, FP for the unpruned Mgi
TPPrune, FPPrune = calcTPFP(strat_test_set.iloc[:, -1], predPrune) # TP, FP for the pruned Mgi

# Report result
print('True positive and False Positive rates for the unpruned Mgi are: ', TP, FP)
print('True Positive and False Positive rates for the pruned Mgi are: ', TPPrune, FPPrune)

