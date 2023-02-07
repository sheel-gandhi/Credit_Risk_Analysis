# Credit_Risk_Analysis

Application of Machine Learning Model to evaluate Credit Card Risk

## Purpose

* **Explain how a machine learning algorithm is used in data analytics.**
* **Create training and test groups from a given data set.**
* **Implement the logistic regression, decision tree, random forest, and support vector machine algorithms.**
* **Interpret the results of the logistic regression, decision tree, random forest, and support vector machine algorithms.**
* **Compare the advantages and disadvantages of each supervised learning algorithm.**
* **Determine which supervised learning algorithm is best used for a given data set or scenario.**
* **Use ensemble and resampling techniques to improve model performance.**

## Overview

A credit card dataset from LendingClub, a peer-to-peer lending services company will be analyised to classify the risky loans from the good loans. Data will be preprocessed by removing any unwanted/ redundand columns, removing rows with NaNs, then statistical reasonsing will be applied, and then we can apply Machine Learning by training data, fitting, and predicting. Different techniques will be applied to train and evaluate models with unbalanced classes by using imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. 

We will oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, we will use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, to compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier will be applied to predict credit risk. Once done, we will evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Results

### Random Over Sampler
In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced.
![RandomOverSampler](https://user-images.githubusercontent.com/108366412/198415167-cf3afacb-ff49-47b7-a8c6-c3e624be9b96.png)### SMOTE

The results for Credit Card data under Random Over Sampler are as follows:
* **Balanced Accuracy Score is at 0.6257 which means that 62.6% of the time the minority class is balanced by oversampling**
* **Precision: The precision for high risk and low risk applicants is 0.01 and 1.00 respectively. This means that 100% of the predicted low risk applicants are actually low risk, whereas only 1% of the predicted high risk applicants actually fall under high risk.**
* **Recall: The recall rate for high risk and low risk applicants is 0.68 and 0.57 respectively. This means that 57% of the high risk applicants are classified as high risk whereas only 68% of the low risk applicants are classified as low risk.**

### SMOTE
In SMOTE, new instances are interpolated. That is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.
![SMOTE](https://user-images.githubusercontent.com/108366412/198415201-e70a8413-b670-412b-b4e9-a8e333482c8e.png)

The results for Credit Card data under SMOTE are as follows:
* **Balanced Accuracy Score is at 0.6277 which means that 62.8% of the time the minority class is balanced by oversampling**
* **Precision: The precision for high risk and low risk applicants is 0.01 and 1.00 respectively. This means that 100% of the predicted low risk applicants are actually low risk, whereas only 1% of the predicted high risk applicants actually fall under high risk.**
* **Recall: The recall rate for high risk and low risk applicants is 0.62 and 0.53 respectively. This means that 62% of the high risk applicants are classified as high risk whereas only 63% of the low risk applicants are classified as low risk.**

### Cluster Centroids
In Cluster centroid undersampling, algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.
![ClusterCentroids](https://user-images.githubusercontent.com/108366412/198415229-e4f56a7c-a477-41bf-af47-6c088ced46c0.png)

The results for Credit Card data under Cluster Centroids are as follows:
* **Balanced Accuracy Score is at 0.5293 which means that 52.93% of the time the minority class is balanced by oversampling**
* **Precision: The precision for high risk and low risk applicants is 0.01 and 1.00 respectively. This means that 100% of the predicted low risk applicants are actually low risk, whereas only 1% of the predicted high risk applicants actually fall under high risk.**
* **Recall: The recall rate for high risk and low risk applicants is 0.45 and 0.61 respectively. This means that 45% of the high risk applicants are classified as high risk whereas only 61% of the low risk applicants are classified as low risk.**

### SMOTEENN
SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. SMOTEENN is a two-step process: Oversample the minority class with SMOTE.
Clean the resulting data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.
![SMOTEENN](https://user-images.githubusercontent.com/108366412/198415249-fa53ae2a-ed9e-409e-a726-bc225d8c51ee.png)

The results for Credit Card data under SMOTEENN are as follows:
* **Balanced Accuracy Score is at 0.6548 which means that 65.48% of the time the minority class is balanced by oversampling**
* **Precision: The precision for high risk and low risk applicants is 0.01 and 1.00 respectively. This means that 100% of the predicted low risk applicants are actually low risk, whereas only 1% of the predicted high risk applicants actually fall under high risk.**
* **Recall: The recall rate for high risk and low risk applicants is 0.61 and 0.70 respectively. This means that 61% of the high risk applicants are classified as high risk whereas only 70% of the low risk applicants are classified as low risk.**

### Balanced Random Forest Classifier
The Random Forests Classifier is composed of several small decision trees created from random sampling. By using the Balanced Random Forests, we oversample from the minority class to balance the classes.
![BalancedRandomForestClassifier](https://user-images.githubusercontent.com/108366412/198415274-53f49d67-5dab-42ad-8597-bd26f41fe00a.png)

The results for Credit Card data under Balanced Random Forest Classifier are as follows:
* **Balanced Accuracy Score is at 0.8160 which means that 81.60% of the time the minority class is balanced by oversampling**
* **Precision: The precision for high risk and low risk applicants is 0.04 and 1.00 respectively. This means that 100% of the predicted low risk applicants are actually low risk, whereas only 4% of the predicted high risk applicants actually fall under high risk.**
* **Recall: The recall rate for high risk and low risk applicants is 0.72 and 0.91 respectively. This means that 72% of the high risk applicants are classified as high risk whereas only 91% of the low risk applicants are classified as low risk.**

### Easy Ensemble Classifier
In AdaBoost, a model is trained then evaluated. After evaluating the errors of the first model, another model is trained. This time, however, the model gives extra weight to the errors from the previous model. The purpose of this weighting is to minimize similar errors in subsequent models. Then, the errors from the second model are given extra weight for the third model. This process is repeated until the error rate is minimized.

![EasyEnsembleClassifier](https://user-images.githubusercontent.com/108366412/198415317-dda15141-c7d0-474e-9bc5-11ed3a747e4e.png)

The results for Credit Card data under Easy Ensemble Classifier are as follows:
* **Balanced Accuracy Score is at 0.9253 which means that 92.53% of the time the minority class is balanced by oversampling**
* **Precision: The precision for high risk and low risk applicants is 0.07 and 1.00 respectively. This means that 100% of the predicted low risk applicants are actually low risk, whereas only 7% of the predicted high risk applicants actually fall under high risk.**
* **Recall: The recall rate for high risk and low risk applicants is 0.91 and 0.94 respectively. This means that 91% of the high risk applicants are classified as high risk whereas only 94% of the low risk applicants are classified as low risk.**

## Summary

With resampling we tried to address the imbalance in the data where one cluster is having too low data points to have a precise prediction. Through our study, we tried different models and through Easy Ensemble Classifier, we got the best results. We are able to get a Balanced Accuracy Score of 92.53% which means 92% of the data is balanced under the model. The Precision rate and Recall rate for high risk candidates is at 7% and 91% which is maximum among all the models. Similarly, low risk candidates Precision rate and Recall rate is also at 100% and 94% respectively. Model. thus is able to predict both high and low risk candidates with high precision.

Considering the high accuracy results from the Easy Ensemble Classifier model, we would not recommend running any other model. We should further look into the results from the model and work with other tools beyond Machine Learning. 
