# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 10:55:11 2018

@author: Sharan
"""

#Classification of cancer dignosis
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset 
dataset = pd.read_csv('./cancer_short.csv')
X = dataset.iloc[:, 1:10]
Y = dataset.iloc[:, 10]

#Accuracy using RandomForest
#from sklearn import model_selection
#from sklearn.ensemble import RandomForestClassifier
#seed = 0
#num_trees = 50
#max_features = 5
#kfold = model_selection.KFold(n_splits=10, random_state=seed)
#model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
#results = model_selection.cross_val_score(model, X, Y, cv=kfold)
#print(results.mean())

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

#Fitting Random Forest Classification Algorithm
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X, Y)
#98.6 Acuracy


#Fitting the Logistic Regression Algorithm to the Training Set
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X, Y)
#95.8 Acuracy

#Fitting K-NN Algorithm
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#classifier.fit(X_train, Y_train)
#95.1 Acuracy

#Fitting SVM
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state = 0)
#classifier.fit(X_train, Y_train) 
#97.2 Acuracy

#Fitting K-SVM
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier.fit(X_train, Y_train)
#96.5 Acuracy

#Fitting Naive_Bayes
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(X_train, Y_train)
#91.6 Acuracy

#Fitting Decision Tree Algorithm
#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#classifier.fit(X_train, Y_train)
#95.8 Acuracy

#predicting the Test set results
#Y_pred = classifier.predict(X_test)

#Creating the confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(Y_test, Y_pred)
#print(cm)

#Export model
from sklearn.externals import joblib
joblib.dump(classifier, 'model.pkl')

# load the model from disk
#loaded_model = joblib.load(filename)
#result = loaded_model.score(X_test, Y_test)
#print(result)