### Randall Fowler
### EEC 266: Final Project

### To use, specify threshold, kmi, and beta values.
### Datasets will be read from the Data folder.
### Selected features will be stored in the SelectedFeatures folder.

import os
from sys import stdout
import time
import scipy
import numpy as np
from math import floor
from typing import Tuple, List
from numpy.matlib import repmat
from scipy.special import digamma
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KDTree, NearestNeighbors

threshold = 40          # Number of selected features
kmi = 5                 # k for kNN in MI est.

# Function that takes in dataset and labels and creates list of new datasets and labels with binary classes.
def ClassBinarization(X: np.ndarray, Y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if(min(Y) == 1):
        Y = Y - 1
    numSamples, numFeatures = X.shape
    numClasses = np.max(Y) + 1

    labelSpaces = []
    featureSpaces = []

    for i in range(numClasses):
        trueMatrix = (Y == i).flatten()
        falseClasses = Y[np.invert(trueMatrix)]
        numTrueSamples = np.sum(trueMatrix)
        numFalseSamples = numSamples - numTrueSamples
        trueFeatures = X[trueMatrix]

        featureSpace = []
        originalClasses = []
        if(numTrueSamples < numFalseSamples):
            featureSpace = np.zeros((2*numFalseSamples, numFeatures))
            originalClasses = np.zeros((2*numFalseSamples, 1), dtype=int)
            
            mult = floor((numFalseSamples - numTrueSamples) / numTrueSamples) + 1
            bound = numTrueSamples*mult

            featureSpace[:bound,:] = repmat(trueFeatures, mult, 1)
            originalClasses[:numFalseSamples] = i
            featureSpace[bound:numFalseSamples,:] = trueFeatures[:numFalseSamples - bound,:]
            featureSpace[numFalseSamples:,:] = X[np.invert(trueMatrix)]
            originalClasses[numFalseSamples:] = falseClasses

        elif(numTrueSamples > numFalseSamples):
            featureSpace = np.zeros((2*numTrueSamples, numFeatures))
            originalClasses = np.zeros((2*numTrueSamples, 1), dtype=int)
            falseFeatures = X[np.invert(trueMatrix)]
            
            mult = floor((numTrueSamples - numFalseSamples) / numFalseSamples) + 1
            bound = numFalseSamples*mult

            featureSpace[:numTrueSamples,:] = trueFeatures
            originalClasses[:numTrueSamples] = i
            featureSpace[numTrueSamples:numTrueSamples + bound,:] = repmat(falseFeatures, mult, 1)
            featureSpace[numTrueSamples + bound:,:] = falseFeatures[:numTrueSamples - bound,:]
            originalClasses[numTrueSamples:numTrueSamples + bound] = repmat(falseClasses, mult, 1)
            originalClasses[numTrueSamples + bound:] = falseClasses[:numTrueSamples - bound]
    
        else:          
            featureSpace = X
            originalClasses = Y
        
        featureSpaces.append(featureSpace)
        labels = np.zeros(featureSpace.shape[0], dtype=int)
        labels[:int(featureSpace.shape[0]/2)] = 1
        labelSpaces.append([labels,originalClasses.squeeze()])

    return featureSpaces, labelSpaces

# I(X;Y,Z) with X continuous, Y discrete and Z continuous
def jmi_c_dc(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, n_neighbors: int) -> float:
    n_samples = X.shape[0]
    X = X.reshape((-1,1))
    Z = Z.reshape((-1,1))

    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    nn = NearestNeighbors()

    for label in np.unique(Y):  #for each label
        mask = Y == label       #get the data for that label
        count = np.sum(mask)    #get number of points
        if count > 1:
            k = min(n_neighbors, count - 1) #Need to have enough data points for k
            nn.set_params(n_neighbors=k)    #set k
            nn.fit(np.hstack((X[mask], Z[mask])))                 #fit X,Z data with Y = label
            r = nn.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count
    
    mask = label_counts > 1
    n_samples = np.sum(mask)
    k_all = k_all[mask]
    X = X[mask]
    Y = Y[mask] #remove small groups
    Z = Z[mask]
    radius = radius[mask]

    kd = KDTree(X)
    nx = kd.query_radius(X, radius, count_only=True, return_distance=False)

    nyz = np.zeros((len(Z),))
    for label in np.unique(Y):
        mask = Y == label
        kd = KDTree(Z[mask])
        nyz[mask] = kd.query_radius(Z[mask], radius[mask], count_only=True, return_distance=False)

    mi = (digamma(n_samples) + np.mean(digamma(k_all)) - np.mean(digamma(nx)) - np.mean(digamma(nyz)))
    return max(0, mi)

#featureSpace shape = (3000,670)
def CSMI(featureSpace: np.ndarray, labelSpace: List[np.ndarray], threshold: int = 40, kmi: int = 5) -> np.array:
    N, numFeatures = featureSpace.shape
    selectedFeatures = np.zeros(threshold, dtype=int) - 1
    J = np.zeros(numFeatures)

    t = 0
    fj = -1
    lastFeature = None
    print("Features collected at time [sec]: [", end="")
    tic = time.perf_counter()
    J = mutual_info_regression(featureSpace, labelSpace[0], n_neighbors= kmi)
    while(t < threshold):
        if(t > 0):  #I(fj;Pi|fk) = I(fj;Pi,fk) - I(fk;fj)
            #J += I(fk;Pi,fj) - I(fk;C,fj) + I(fj;Pi,fk) - 2*mutual_info_regression(featureSpace, featureSpace[fj], n_neighbors= kmi)
            for fk in range(numFeatures):
                J[fk] += jmi_c_dc(featureSpace[:,fk], labelSpace[0], featureSpace[:,fj], kmi)
                J[fk] -= jmi_c_dc(featureSpace[:,fk], labelSpace[1], featureSpace[:,fj], kmi)
                J[fk] += jmi_c_dc(featureSpace[:,fj], labelSpace[0], featureSpace[:,fk], kmi)

            J -= 2*mutual_info_regression(featureSpace, featureSpace[:,fj], n_neighbors= kmi)
            J[selectedFeatures[selectedFeatures != -1]] = -np.inf

        fj = np.argmax(J)
        selectedFeatures[t] = fj
        t += 1
        print(f"{time.perf_counter() - tic: .2f}",end=",")
        stdout.flush()
    print("]")
    return selectedFeatures



dirPath = os.path.join(os.getcwd(),"Data")
datasetNames = [filename[:-6] for filename in os.listdir(dirPath) if filename.endswith("X.npy")]
datasetNames.sort()
datasetNames = [datasetNames[2]]
print(datasetNames)

featurePath = os.path.join(os.getcwd(),"SelectedFeatures")
for dataset in datasetNames:
    try:
        X = np.load(os.path.join(dirPath, dataset+"_X.npy"))
        Y = np.load(os.path.join(dirPath, dataset+"_Y.npy"))
        print(f"Feature Space size: {X.shape}")
        print(f"Class Space size: {Y.shape}")

        featureSpaces, labelSpaces = ClassBinarization(X,Y)
        
        selectedFeatures = []
        start = time.perf_counter()
        for i in range(len(featureSpaces)):
            selectedFeatures.append( CSMI(featureSpaces[i], labelSpaces[i], kmi=kmi, threshold=threshold) )
            print(f"\tFeature space {i} completed at {time.perf_counter() - start}")
            print(f"\t---Selected Features: {selectedFeatures[i]}")
            stdout.flush()
        print()
        selectedFeatures = np.array(selectedFeatures)
        np.save(os.path.join(featurePath, dataset + "_CSMIFeatures"), selectedFeatures)
    except Exception as err:
        print()
        print("***Failed on:"+dataset+"***")
        print(err)
        print()
        stdout.flush()
        pass

