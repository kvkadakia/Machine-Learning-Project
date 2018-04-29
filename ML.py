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
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.simplefilter("ignore")
#read dataset
dataset = pandas.read_csv('responses.csv')
#fill up with median
dataset = dataset.fillna(dataset.median(axis=0))

#drop columns which dont have numeric values
X = dataset.drop(['Empathy','Punctuality','Alcohol','Smoking', 'Lying', 'Internet usage', 'Gender','Left - right handed','Education','Only child','Village - town','House - block of flats'], axis=1)
Y = dataset[['Empathy']]

#performs RFE which selects the best 5 features
model = LogisticRegression()
rfe = RFE(model, 41)
fit = rfe.fit(X, Y)
print("Num Features: {}".format(fit.n_features_))
print("Selected Features: {}".format(fit.support_))
print("Feature Ranking: {}".format(fit.ranking_))

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
    
print(top_feat)

#Put the top features in X and train the model
X = dataset[top_feat]
Y = dataset[['Empathy']]
X_train, X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.25,random_state=26)

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
model = model.fit(X_train, Y_train)
predictions=model.predict(X_test)
print(model.score(X_test, Y_test))


###################Cross validation##########################
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

seed = 7
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()