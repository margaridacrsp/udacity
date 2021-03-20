# Predict Customers Churn Project

The objective of this project was to predict the churn of customers based on demographic, services and account data.

Some exploratory data analysis was done to understand which variables could have an impact on the churn event. After exploring the data, the columns of interest were transformed into numerical or binary columns to be used in the modeling phase.

The dataset was divided into training and testing, in the proportion 67% and 33%, respectively, and 5 classification models were trained and the hyper-parameters where adjusted using Grid Search.

Finally, the model that showed the best results for predicting churn was the Logistic Regression, with an AUC of 0.717. This model obtained the highest scores in all 5 metrics chosen.

To view the full article go to https://margarida-crespo.medium.com/why-customers-say-goodbye-b341bed96502 .



This project was developed under the Udacity Data Science Nanodegree program.


## Overview

This project is divided in four steps:
0. Dataset analysis
1. Exploratory Analysis
2. Feature engineering
3. Modeling


## Files

Main folder:
- 0-Dataset.ipynb : jupyter notebook to analyze the dataset
- 1-ExploratoryAnalysis.ipynb : jupyter notebook to explore the data and investigate relationships between variables
- 2-FeatureEngineering.ipynb : jupyter notebook to transform and adapt the data to fit the model
- 3-Modeling.ipynb : jupyter notebook to train and test the dataset with the different classifiers
- ParameteresAnalysis.ipynb : jupyter notebook used to understand the impact of some parameters in the different models

data folder:
- TelcoCustomerChurn.csv : CSV file containing the original data obtained here: https://www.kaggle.com/blastchar/telco-customer-churn
- TransformedTelcoCustomerChurn.csv : CSV file containing the transformed data to use the the 3-Modeling.ipynb file

images folder: 
- several images created in the jupyter notebook used in the articles


## Run

Each jupyter notebook can be run without dependencies.

