# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 21:43:15 2019

@author: arpit
"""

# prqctice for simple regression
"""
X is chirps per sec
Y is temperature in farenheit
our aim is to predict chirps using regression

"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read data
data_chirp = pd.read_excel("C:\\Users\\arpit\\Downloads\\slr02.xls")
chirp = data_chirp.iloc[:,:-1].values
temp = data_chirp.iloc[:,1].values

#split data
from sklearn.model_selection import train_test_split
train_chirp, test_chirp, train_temp, test_temp  = train_test_split(chirp,temp, test_size = 0.2, random_state = 0)

#use simple regression model to train model
from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(train_chirp, train_temp)

#predict the values of training
pred_2 = regressor1.predict(test_chirp)

#plot the graph for test data set
plt.scatter(test_chirp,test_temp, color='red')
plt.plot(train_chirp, regressor1.predict(train_chirp), color = 'Blue')
plt.title("chirp vs temp")
plt.xlabel("chirps")
plt.ylabel("temp")
