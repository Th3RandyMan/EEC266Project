### Randall Fowler
### EEC 266: Final Project

### To use, specify threshold, kmi, and beta values.
### Datasets will be read from the Data folder.
### Selected features will be stored in the SelectedFeatures folder.

import os
import time
import numpy as np
from sys import stdout
from numpy.matlib import repmat
from scipy.special import digamma
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KDTree, NearestNeighbors

threshold = 40          # Number of selected features
kmi = 5                 # k for kNN in MI est.
beta = 1/10             # Hyperparam for MIFS

ITFS_Methods = ["MIM", "MIFS", "JMI", "mRMR", "CIFE", "MRI", "DCSF"]

# I(X;Y,Z) with X continuous, Y discrete, and Z continuous
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

# I(X;Y,Z) with X discete, Y continuous, and Z continuous
def jmi_d_cc(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, n_neighbors: int) -> float:
    n_samples = X.shape[0]
    Y = Y.reshape((-1,1))
    Z = Z.reshape((-1,1))

    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    nn = NearestNeighbors()

    for label in np.unique(X):  #for each label
        mask = X == label       #get the data for that label
        count = np.sum(mask)    #get number of points
        if count > 1:
            k = min(n_neighbors, count - 1) #Need to have enough data points for k
            nn.set_params(n_neighbors=k)    #set k
            nn.fit(np.hstack((Y[mask], Z[mask])))
            r = nn.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count
    
    mask = label_counts > 1
    n_samples = np.sum(mask)
    k_all = k_all[mask]
    Y = Y[mask] #remove small groups
    Z = Z[mask]
    label_counts = label_counts[mask]
    radius = radius[mask]

    YZ = np.hstack((Y, Z))
    kd = KDTree(YZ)
    nyz = kd.query_radius(YZ, radius, count_only=True, return_distance=False)

    mi = (digamma(n_samples) + np.mean(digamma(k_all)) - np.mean(digamma(label_counts)) - np.mean(digamma(nyz)))
    return max(0, mi)


dirPath = os.path.join(os.getcwd(),"Data")
datasetNames = [filename[:-6] for filename in os.listdir(dirPath) if filename.endswith("X.npy")]
print(datasetNames)

featurePath = os.path.join(os.getcwd(),"SelectedFeatures")
for dataset in datasetNames:
    try:
        X = np.load(os.path.join(dirPath, dataset+"_X.npy"))
        Y = np.load(os.path.join(dirPath, dataset+"_Y.npy")).flatten()
        print(f"Feature Space size: {X.shape}")
        print(f"Class Space size: {Y.shape}")
        print()

        n_ITFS = len(ITFS_Methods)
        N, numFeatures = X.shape
        selectedFeatures = np.zeros((n_ITFS,threshold), dtype=int) - 1
        J = np.zeros((n_ITFS,numFeatures))

        t = 0
        fj = np.zeros(n_ITFS, dtype=int) - 1
        print("Features collected at time [sec]: [", end="")
        tic = time.perf_counter()
        I_fk_C = mutual_info_regression(X, Y, n_neighbors=kmi)
        J[:,:] = I_fk_C
        while(t < threshold):
            if(t > 0):
                J[1,:] -= beta*mutual_info_regression(X, X[:,fj[1]], n_neighbors=kmi)  #done
                J[3,:] -= mutual_info_regression(X, X[:,fj[3]], n_neighbors=kmi)/(t+1)
                J[4,:] -= mutual_info_regression(X, X[:,fj[4]], n_neighbors=kmi) + I_fk_C
                J[5,:] -= 2*mutual_info_regression(X, X[:,fj[5]], n_neighbors=kmi)
                J[6,:] -= (3*mutual_info_regression(X, X[:,fj[6]], n_neighbors=kmi) + 2*I_fk_C)/(t+1)
                
                for fk in range(numFeatures):
                    J[2,fk] += jmi_d_cc(Y, X[:,fk], X[:,fj[2]], kmi)
                    J[4,fk] += jmi_c_dc(X[:,fk], Y, X[:,fj[4]], kmi)  #I(fk;fj|C) = I(fk;fj,C) - I(fk;C)
                    J[5,fk] += jmi_c_dc(X[:,fk], Y, X[:,fj[5]], kmi) + jmi_c_dc(X[:,fj[5]], Y, X[:,fk], kmi)#ICI(C;fj,fk) = I(fj;C|fk) + I(fk;C|fj) = I(fj;C,fk) + I(fk;C,fj) - 2*I(fj;fk)
                    J[6,fk] += 2*jmi_c_dc(X[:,fk], Y, X[:,fj[6]], kmi)/(t+1)

                for i in range(n_ITFS):
                    J[i, selectedFeatures[i, selectedFeatures[i,:] != -1]] = -np.inf
            
            fj = np.argmax(J,axis=1)
            selectedFeatures[:,t] = fj
            t += 1
            print(f"{time.perf_counter() - tic: .2f}",end=",")
            stdout.flush()
        print("]")
        print(selectedFeatures)
        np.save(os.path.join(featurePath, dataset + "_ITFSFeatures"), selectedFeatures)
        print()
    except Exception as err:
        print()
        print("***Failed on:"+dataset+"***")
        print(err)
        print()
        stdout.flush()
        pass