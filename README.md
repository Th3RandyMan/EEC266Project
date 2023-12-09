# Information Theory Final Project
Within this repository is tools to reproduce figures 10 and 11 from "A General Framework for Class Label Specific Mutual Information Feature Selection Method."

## Overview
This project focuses on selecting features optimal features from datasets to reduce complexity in processing as well as maximizing on accuracy. Multiple files were created to reduce the time required to run each algorithm. Class-label Specific Mutual Information (CSMI) and other information theory-based feature selection (ITFS) methods are compared.
## Requirements
* numpy
* matplotlib
* scipy
* sklearn
* xgboost

## Usuage
To get started, open the `EEC266Project_CSMI.py` and `EEC266Project_ITFSMethods.py` to adjust the `threshold` for number of features to extract and the `kmi`, the k value in kNN estimations for mutual information. By default, a threshold of 40 features is used and k is 5. A folder named Data will be expected to hold numpy files for the datasets. Selected features will be stored in a folder named SelectedFeatures. Both folders should be within the same directory as the python scripts.

***Warning: Feature selection for CSMI may take multiple days.**

Running the python scripts for selected features for all ITFS methods can be done with an exported log with the following command lines:
```
python EEC266Project_CSMI.py > csmi_log.txt
```
and
```
python EEC266Project_ITFSMethods.py > itfs_log.txt
```
With the selected features stored as numpy files, the `EEC266Project_ML.py` script will run training and evaluation for the XGBoost ML model for each of the ITFS methods. Enter into the python script to adjust the number of average by setting the `avg` variable. Results will be stored in a folder called Plot.

***Warning: Training and evaluation for all models may take at least a day to complete.**

Similar to before, the python script to train and evaluate models using the selected features from all ITFS methods can be ran using:
```
python EEC266Project_ML.py > plot_log.txt
```

Once the models have been evaluated, results can then be plotted using `EEC266Project_ML.ipynb`. Saving figures will be optional, but the notebook will display the figures that were needed to be reproduced. It is currently setup to display 1 and 10 averaging results, but can be adjusted for more or different values.

## Results
The results of 1 and 10 averages for all ITFS methods are saved in the Figures folder. Records of the runs are within the plot_log test files. A 100 average session was not completed as the amount of time required exceeded the due date of the project.

Reproduced figures are not ideal, but ORL does show similar results to the accuracy figure. Read more about the project in `EEC266ProjectPaper.pdf`.

## Acknowledgements
This project was completed for EEC 266 at UC Davis, Fall 2023, by Randall Fowler. 

## Helpful links
* https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/feature_selection/_mutual_info.py
* https://github.com/jannisteunissen/mutual_information/tree/main
* https://github.com/robince/gcmi/blob/master/python/gcmi.py
* https://machinelearningmastery.com/tune-xgboost-performance-with-learning-curves/
* https://jundongl.github.io/scikit-feature/datasets.html


## References
[1] D. K. Rakesh and P. K. Jana, “A General Framework for Class Label
Specific Mutual Information Feature Selection Method,” IEEE Trans.
Info. Theory, vol. 68, no. 12, Dec. 2022.

[2] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach.
Learn. Res., vol. 12, pp. 2825–2830, Nov. 2011.

[3] P. Zhoa and L. Lai, “Analysis of KNN Information Estimators for Smooth
Distributions,” IEEE Trans. Info. Theory, vol. 66, no. 6, Jun. 2020.

[4] O. C. Mesner and C. R. Shalizi, “Conditional Mutual Information
Estimation for Mixed, Discrete and Continuous Data,” IEEE Trans. Info.
Theory, vol. 67, no. 1, Jan. 2021.

[5] J. Wang, J. Wei, Z. Yang, and S. Wang, “Feature Selection by Maximizing
Independent Classification Information,” IEEE Trans. Info. Theory, vol.
29, no. 4, Apr. 2017.