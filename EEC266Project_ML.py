### Randall Fowler
### EEC 266: Final Project

### To use, set the number of averages in avg variable and run.
### This code will also use the first available GPU for XGBoost.
### Selected features will be read from SelectedFeatures folder.
### Model results will be stored in Plot folder.

import os
import time
import numpy as np
import xgboost as xgb
from sys import stdout
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

avg = 10
ITFS_Methods = ["MIM", "MIFS", "JMI", "mRMR", "CIFE", "MRI", "DCSF", "CSMI"]

# function for taking in dataset X with labels Y and evaluating accuracy and goodness-of-fit for XGBoost
# GoF is a bool to decide whether to measure goodness-of-fit
# nomralize is a bool to decide whether to normalize the gof
# avg is the number of times to train and evaluate the model
def XGBoostModelRun(X: np.ndarray, Y: np.ndarray, GoF: bool = False, normalize: bool = False, avg: int = 1) -> [float, [float]]:#Dict[str, Dict[str, List[float]]]]:
    gof_avg = [0,0]  #default value for unused
    acc_avg = 0      #average of the classification accuracy
    
    for a in range(avg):    # run avg times to average the results
        model = xgb.XGBClassifier(tree_method = "hist", device = "cuda")
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

        if(GoF):    #if measuring goodness-of-fit
            evalset = [(X_train, Y_train), (X_test, Y_test)]
            model.fit(X_train, Y_train, eval_set=evalset, verbose=False) #logloss doesn't work, but mlogloss does

            results = model.evals_result()  
            if 'mlogloss' in results['validation_0']:
                gof = [np.trapz(results['validation_0']['mlogloss']), np.trapz(results['validation_1']['mlogloss'])]    #get area under curve
            elif 'logloss' in results['validation_0']:
                gof = [np.trapz(results['validation_0']['logloss']), np.trapz(results['validation_1']['logloss'])]

            if(normalize):  # if we want to normalize with the number of labels
                norm = np.max(Y) + 1
                gof = [val/norm for val in gof]
            gof_avg[0] += gof[0]    #add to average 
            gof_avg[1] += gof[1]
        else:
            model.fit(X_train, Y_train, verbose=False)  #fit without gof

        Y_pred = np.rint(model.predict(X_test)) #round all elements in array
        acc_avg += accuracy_score(Y_test, Y_pred.astype(int))   #convert elements to integers and get accuracy

    if(GoF):    #if measuring goodness-of-fit
        gof_avg[0] /= avg   #divide to get average
        gof_avg[1] /= avg
    return acc_avg/avg, gof_avg

# function for taking in dataset X with labels Y and evaluating accuracy and goodness-of-fit for XGBoost
# csmi_f is a ndarray with size (n_labels, n_features) for indexing X
# GoF is a bool to decide whether to measure goodness-of-fit
# nomralize is a bool to decide whether to normalize the gof
# avg is the number of times to train and evaluate the model
def CSMI_XGBoostModelsRun(X: np.ndarray, Y: np.ndarray, csmi_f: np.ndarray, GoF: bool = False, normalize: bool = False, avg: int = 1) -> [float, [float]]:
    n_labels = csmi_f.shape[0]
    acc_avg = 0
    gof_avg = [0,0]

    for a in range(avg):    # run avg times for averaging measurements
        model_list = [xgb.XGBClassifier(tree_method = "hist", device = "cuda") for i in range(n_labels)]                 # list of models to train and evaluate
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)   # split data and training sets
        Y_pred = np.zeros((n_labels,len(Y_test)), dtype=int)                        # array for predictions from each model
        votes = np.zeros((n_labels,1),dtype=int)                                        # votes for each label (reused later)
        Y_vote = np.zeros((len(Y_test),1),dtype=int)                                    # 

        for l in range(n_labels):
            features = csmi_f[l,:]
            X_train1 = X_train[:,features]
            X_test1 = X_test[:,features]

            if(GoF):
                evalset = [(X_train1, Y_train), (X_test1, Y_test)]
                model_list[l].fit(X_train1, Y_train, eval_set=evalset, verbose=False)

                results = model_list[l].evals_result()
                if 'mlogloss' in results['validation_0']:
                    gof = [np.trapz(results['validation_0']['mlogloss']), np.trapz(results['validation_1']['mlogloss'])]    #get area under curve
                elif 'logloss' in results['validation_0']:
                    gof = [np.trapz(results['validation_0']['logloss']), np.trapz(results['validation_1']['logloss'])]
                    
                if(normalize):
                    norm = np.max(Y) + 1
                    gof = [val/norm for val in gof]
                gof_avg[0] += gof[0]
                gof_avg[1] += gof[1]
            else:
                model_list[l].fit(X_train1, Y_train, verbose=False)
            
            Y_pred[l,:] = np.rint(model_list[l].predict(X_test1)).astype(int)
    
        for sample in range(len(Y_test)):   # go through all samples
            votes[:] = 0
            for l in range(n_labels):   # go through all labels
                vote = Y_pred[l,sample] # get the vote the model made
                if(vote > n_labels):
                    pass        # this should be removed
                else:
                    votes[vote] += 10
                    if(l == vote):
                        votes[vote] += 5    #weight of model predicting its own label
            Y_vote[sample] = np.argmax(votes)   # the label with the most votes will be the prediction
        acc_avg += accuracy_score(Y_test, Y_vote)

    if(GoF):
        gof_avg[0] /= n_labels*avg
        gof_avg[1] /= n_labels*avg
    return acc_avg/avg, gof_avg


dirPath = os.path.join(os.getcwd(),"Data")
featurePath = os.path.join(os.getcwd(),"SelectedFeatures")
plotPath = os.path.join(os.getcwd(),"Plot")
CSMI_Features = [filename[:-4] for filename in os.listdir(featurePath) if filename.endswith("CSMIFeatures.npy")]
ITFS_Features = [filename[:-4] for filename in os.listdir(featurePath) if filename.endswith("ITFSFeatures.npy")]
CSMI_Features.sort()
ITFS_Features.sort()

availableFeatureSets = [CSMI_Features[i].split("_")[0] for i in range(len(CSMI_Features))]
print(availableFeatureSets)
for setName in availableFeatureSets:
    print()
    print(setName)
    X = np.load(os.path.join(dirPath, setName+"_X.npy"))
    Y = np.load(os.path.join(dirPath, setName+"_Y.npy")) - 1
    n_labels = np.max(Y) + 1
    n_methods = len(ITFS_Methods)

    csmi_f = np.load(os.path.join(featurePath, setName + "_CSMIFeatures.npy"))
    itfs_f = np.load(os.path.join(featurePath, setName + "_ITFSFeatures.npy"))

    classify_acc = np.zeros((n_methods,40)) #accuracy for each method with varying number of features
    GoF = np.zeros(n_methods)   #maybe add 2nd dimension for validation or training curve

    tic = time.perf_counter()
    for m in range(itfs_f.shape[0]):
        print("ITFS Method: "+ITFS_Methods[m])
        for n_features in range(40):
            acc, result = XGBoostModelRun(X[:,itfs_f[m,:n_features+1]], Y, GoF=(n_features==39), avg=avg)
            classify_acc[m,n_features] = acc
            if(n_features==39):
                GoF[m] = result[0]
        print(f"\tCompleted at time: {time.perf_counter() - tic:.2f} seconds")
        stdout.flush()
    
    print("ITFS Method: "+ITFS_Methods[-1])
    for n_features in range(40):
        stdout.flush()
        print(f"\tTraining with {n_features+1} features: ",end="")
        acc, result = CSMI_XGBoostModelsRun(X, Y, csmi_f[:,:n_features+1], GoF=(n_features==39), avg=avg)
        classify_acc[n_methods-1,n_features] = acc
        if(n_features==39):
            GoF[n_methods-1] = result[0]
        print(f"{time.perf_counter() - tic:.2f} seconds")
        
    print(f"\tCompleted at time: {time.perf_counter() - tic:.2f} seconds")
    np.save(os.path.join(plotPath, setName+"_acc.npy"),classify_acc)
    np.save(os.path.join(plotPath, setName+"_gof.npy"),GoF)