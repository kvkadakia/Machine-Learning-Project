#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 00:06:39 2018

@author: karan
"""

# Load libraries
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt, pylab


warnings.simplefilter("ignore")

#read dataset
dataset = pandas.read_csv('responses.csv')

print('Data cleaning: Filling up the na values with the column median \n')

#fill up with median
dataset = dataset.fillna(dataset.median(axis=0))



############# Applying one hot encoding on Categorical values ################

print('Data cleaning: Applying one hot encoding in order to deal with Categorical values \n')
dataset = pandas.get_dummies(dataset, columns=['Punctuality','Alcohol','Smoking', 'Lying', 'Internet usage', 'Gender','Left - right handed','Education','Only child','Village - town','House - block of flats'])


############# Selecting the best features to predict Empathy #################

print('Selecting the best features to predict Empathy using RFE \n')
#drop empathy column which can't be used for prediction
X = dataset.drop(['Empathy'], axis=1)
Y = dataset[['Empathy']]        

#performs RFE which selects the best 5 features
model = LogisticRegression()
rfe = RFE(model, 141)
fit = rfe.fit(X, Y)

#Append the index of the top 5 elements by checking the columns ranked 1st
count = 0
val_index = []
for x in np.nditer(fit.ranking_):
    if(x==1):
        val_index.append(count)
    count+=1
    
#Print the column names    
pos = 0
top_feat = []
for i,k in enumerate(X):
    if(i==val_index[pos]):
        top_feat.append(k)
        if(pos == len(val_index)-1):
            break
        pos+=1

print('Assigning the top features to X \n')    
#Put the top features in X and train the model
X = dataset[top_feat]
Y = dataset[['Empathy']]

############# Splitting the data into train test and validation################

X_train, X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.25,random_state=26)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)  

############# Using Cross validation to find out which model works best########

print('Applying Cross Validation, we get the following scores for each models:')
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
models.append(('Ensemble', VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')))

seed = 7
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_val, Y_val, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

print('\n')

################### SVC works best by above evaluation thus, using SVC #########    

model = SVC()
model = model.fit(X_train, Y_train)
predictions=model.predict(X_test)
print('Model Accuracy using SVC for Predicting Empathy:')
print(model.score(X_test, Y_test))