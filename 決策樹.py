#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
活動25

@author: linweicheng
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn import tree

train_data = pd.read_csv('train.csv')
train_data = train_data.drop(['Id'], axis = 1)
x_data = train_data.drop(['SalePrice'], axis = 1)
y_data = train_data['SalePrice']
x_data = x_data.dropna(axis = 1 , how = 'any')

encoder = LabelEncoder()
x_encoded = pd.DataFrame(x_data , columns = x_data.columns).apply(lambda col:encoder.fit_transform(col))
kbest = SelectKBest(f_regression , k = 40) 
x_new = kbest.fit_transform(x_encoded , y_data)

x_train, x_test, y_train, y_test = train_test_split(x_new, y_data, train_size=0.7, test_size=0.3, random_state=0)

model_dtree = tree.DecisionTreeRegressor(max_depth = 6, random_state=0)
model_dtree.fit(x_train, y_train)

model_dtree.score(x_test, y_test)
score = model_dtree.score(x_test, y_test)

print('\nScore : ', score)
print('Accuracy : ' + str(score*100) + '%')