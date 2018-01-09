# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:25:51 2017

@author: zheyiyi
"""


import sys
import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv

def open_matrix(filename):
    list_data = []
    with open(filename) as csvfile:
        data = csv.reader(csvfile, delimiter = ",")
        for row in data:
            list_data.append(row)
        return np.array(list_data, dtype=float)


#random give a and b
def bastys(x,y):
    new_a = random.randrange(1,10)
    new_b = random.randrange(1,10)  
    a = 11
    b = 11
    n = len(y)
    #print(x.shape, y.shape)
    #print(a,b,new_a,new_b)
    while abs(new_a - a) > 0.000001 and abs(new_b - b) > 0.000001:
        a = new_a
        b = new_b
        #print(a)
        t = np.dot(x.T,x)
        #print(t.shape)
        #print(np.shape(np.identity(len(t[0]))))
        s_n_1 = np.add(a * np.identity(len(t[0])) ,b * t)
        s_n = np.linalg.inv(s_n_1)  ## see hoe to -1
        m_n = b * np.dot(np.dot(s_n, x.T), y)
        
        evige_value = np.linalg.eigvals(s_n_1)
       # print(evige_value)
        lamb_da = evige_value - a #不确定能不能这么减
       # print(lamb_da)
        #print(evige_value)
       # print(a)
        #print(evige_value)
        #print(a)
        #print(lamb_da)
       # print(len(evige_value))
        r = 0
        for lamb in lamb_da:
            r += lamb / (a + lamb)
        
        new_a = r / np.dot(m_n.T,m_n)[0][0]
        #print(new_a)
        sum = 0
        for i in range(n):
            sum += math.pow((y[i] - np.dot(m_n.T, x[i])),2)
        new_b = (n - r)/ sum

    return new_a,new_b    
 
def w_map(x,y):
    a,b = bastys(x,y) 
    lambd = a / b
    t = np.dot(x.T,x)
    s_n = np.linalg.inv(np.add(a * np.identity(len(t[0])) ,b * t)) 
    w_max =  b * np.dot(np.dot(s_n, x.T), y)
    
    return w_max, lambd, a, b
    
def mean_squre(x,y,w):
    sum = 0
    n = len(y)
    for i in range(n):
        a = np.dot(x[i].T,w)
        sum += math.pow((a - y[i]),2)
    return sum / n

# training set mse as a function of the regularization parameter
# train_100_10 and trainr_100_10 and test

train_100_10 = open_matrix("train-100-10.csv")
trainr_100_10 = open_matrix("trainR-100-10.csv")
w,lambd_100_10, a_100_10, b_100_10 = w_map(train_100_10, trainr_100_10)

test_100_10 = open_matrix("test-100-10.csv")
testr_100_10 = open_matrix("testR-100-10.csv")
w_100_10 = mean_squre(test_100_10, testr_100_10, w)

print(w_100_10,lambd_100_10,a_100_10, b_100_10)

train_100_100 = open_matrix("train-100-100.csv")
trainr_100_100 = open_matrix("trainR-100-100.csv")
w,lambd_100_100, a_100_100, b_100_100 = w_map(train_100_100, trainr_100_100)

test_100_100 = open_matrix("test-100-100.csv")
testr_100_100 = open_matrix("testR-100-100.csv")
w_100_100 = mean_squre(test_100_100, testr_100_100, w)

print(w_100_100,lambd_100_100, a_100_100, b_100_100)


train_1000_100 = open_matrix("train-1000-100.csv")
trainr_1000_100 = open_matrix("trainR-1000-100.csv")
w, lambd_1000_100, a_1000_100, b_1000_100 = w_map(train_1000_100, trainr_1000_100)

test_1000_100 = open_matrix("test-1000-100.csv")
testr_1000_100 = open_matrix("testR-1000-100.csv")
w_1000_100 = mean_squre(test_1000_100, testr_1000_100, w)

print(w_1000_100,lambd_1000_100, a_1000_100, b_1000_100 )

train_crime= open_matrix("train-crime.csv")
trainr_crime = open_matrix("trainR-crime.csv")
w,lambd_crime,a_crime, b_crime   = w_map(train_crime, trainr_crime) 

test_crime = open_matrix("test-crime.csv")
testr_crime = open_matrix("testR-crime.csv")
w_crime = mean_squre(test_crime, testr_crime, w)
print(w_crime,lambd_crime,a_crime, b_crime)


train_wine= open_matrix("train-wine.csv")
trainr_wine = open_matrix("trainR-wine.csv")
w, lambd_wine, a_wine, b_wine = w_map(train_wine, trainr_wine) 

test_wine = open_matrix("test-wine.csv")
testr_wine = open_matrix("testR-wine.csv")
w_wine = mean_squre(test_wine, testr_wine, w)
print(w_wine,lambd_wine, a_wine, b_wine)

