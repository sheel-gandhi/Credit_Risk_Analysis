# Credit_Risk_Analysis

Application of Machine Learning Model to evaluate Credit Card Risk

## Overview

A credit card dataset from LendingClub, a peer-to-peer lending services company will be analyised to classify the risky loans from the good loans. Data will be preprocessed by removing any unwanted/ redundand columns, removing rows with NaNs, then statistical reasonsing will be applied, and then we can apply Machine Learning by training data, fitting, and predicting. Different techniques will be applied to train and evaluate models with unbalanced classes by using imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. 

We will oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, we will use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, to compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier will be applied to predict credit risk. Once done, we will evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Results

### Random Over Sampler

In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced.
The results for Credit Card data under Random Over Sampler is as follows:
Balanced Accuracy Score: 
![RandomOverSampler](https://user-images.githubusercontent.com/108366412/198415167-cf3afacb-ff49-47b7-a8c6-c3e624be9b96.png)

### SMOTE
![SMOTE](https://user-images.githubusercontent.com/108366412/198415201-e70a8413-b670-412b-b4e9-a8e333482c8e.png)

### Cluster Centroids
![ClusterCentroids](https://user-images.githubusercontent.com/108366412/198415229-e4f56a7c-a477-41bf-af47-6c088ced46c0.png)

### SMOTEENN
![SMOTEENN](https://user-images.githubusercontent.com/108366412/198415249-fa53ae2a-ed9e-409e-a726-bc225d8c51ee.png)

### Balanced Random Forest Classifier
![BalancedRandomForestClassifier](https://user-images.githubusercontent.com/108366412/198415274-53f49d67-5dab-42ad-8597-bd26f41fe00a.png)

### Easy Ensemble Classifier
![EasyEnsembleClassifier](https://user-images.githubusercontent.com/108366412/198415317-dda15141-c7d0-474e-9bc5-11ed3a747e4e.png)


## Summary

### Recommendation
