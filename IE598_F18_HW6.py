#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 15:11:53 2018

@author: stille
"""
#read Iris dataset
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

tree = DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=0)

test_score=[]
for num in range(1,11):
    #print("random_state={}".format(num))
    #split the train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=num)
    
    #standardization
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    tree.fit(X_train, y_train)
    
    #print("    Test score:{}".format(tree.score(X_test,y_test)))
    print(tree.score(X_test,y_test))
    test_score.append(tree.score(X_test,y_test))

print("mean of test score={}".format(np.mean(test_score)))
print("std of test score={}".format(np.std(test_score)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
cv_scores = cross_val_score(tree,X_train,y_train,cv=10)
print(cv_scores)
print("means:{}".format(np.mean(cv_scores)))
print("std:{}".format(np.std(cv_scores)))
print("out of sample acc:{}".format(tree.score(X_test,y_test)))

print("My name is {Wenyu Ni}")
print("My NetID is: {wenyuni2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")