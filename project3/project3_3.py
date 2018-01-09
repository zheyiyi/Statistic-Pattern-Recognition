# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:53:23 2017

@author: zheyiyi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 00:46:56 2017

@author: zheyiyi
"""

import time

import csv

import numpy as np
from numpy import linalg as LA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# open file and store data into train list (2/3 total) and test list(1/ 3 total)

def open_file(filename, filename1):
    list_data = []
    with open(filename) as csvfile:
        data = csv.reader(csvfile, delimiter = "," )
        for row in data:
            row = ['1'] + row
            list_data.append(row)
    list_data1 = []
    with open(filename1) as csvfile1:
        data1 = csv.reader(csvfile1, delimiter = "," )
        for row in data1:
            list_data1.append(row)
        new_list_data = []    
        for i in range(len(list_data)):
            new_list_data.append(list_data[i] + list_data1[i])
              
        n = len(new_list_data) 
        test =  n // 3   
        test_list = new_list_data[:test]   
        train_list = new_list_data[test:]
        
        return train_list, test_list
        
"""

def open_file(filename, filename1):
    list_data = []
    with open(filename) as csvfile:
        data = csv.reader(csvfile, delimiter = "," )
        for row in data:
            row = ['1'] + row
            list_data.append(row)
    list_data1 = []
    with open(filename1) as csvfile1:
        data1 = csv.reader(csvfile1, delimiter = "," )
        for row in data1:
            list_data1.append(row)
        new_list_data = []    
        for i in range(len(list_data)):
            new_list_data.append(list_data[i] + list_data1[i])
              
        return new_list_data     
"""

# select list as a function of increasing raining set size everytime 50  




def train_select(train_list):
    train_incre = []           
    train_incre.append(train_list)    
    train_s = []    
    for list in train_incre:
        temp_s = []
        temp_l = []
        for i in list:
            
            temp_s.append(i[:-1])
            temp_l.append([i[-1]])
    
        train_s.append([temp_s,temp_l])
       
      
    return train_s



  
def sigmoid(z):
    s = 1 / (1 + np.exp(-1 * z)) 
    return s

def update_gradient(train_set, train_label, eita): 
    time_value = []
    w_value = []
    t = 0
 
    train_set = np.array(train_set, dtype = float)
    train_label = np.array(train_label, dtype = float)
    Wo = np.zeros((len(train_set[0]),1)) 
    w_value.append(Wo)
    time_value.append(0)
    
    t0 = time.time()
    a = [np.dot(Wo.T, train_set[i]) for i in range(len(train_set))]
    a = np.array(a)
    y = sigmoid(a)
    value = np.dot(train_set.T, (y - train_label)) + 0.1 * Wo
    Wn = Wo - eita * value
    count = 1
    t1 = time.time()
    t += t1 - t0
    w_value.append(Wn)
    time_value.append(t)
   
    while LA.norm(Wo) == 0 or ((LA.norm(Wn- Wo) / LA.norm(Wo)) >= 0.001 and count < 6000):
        t0 = time.time()
        for i in range(10):
           Wo = Wn
           a = [np.dot(Wo.T, train_set[i]) for i in range(len(train_set))]
           a = np.array(a)
           y = sigmoid(a)        
           value = np.dot(train_set.T, (y - train_label)) + 0.1 * Wo
           Wn = Wo - eita * value   
           count += 1
           
        t1 = time.time()
        t += t1 - t0
        w_value.append(Wn)
        time_value.append(t)
     
    return w_value, time_value
   


  
def predict_bayesian(test_sample, wn):
    
    test_data, test_label = test_sample[0]
    test_data = np.array(test_data, dtype = float)
    wn = np.array(wn, dtype = float)
    a = [np.dot(wn.T, test_data[i]) for i in range(len(test_data))]
    a = np.array(a)
   
    predict_value = np.where(a > 0, 1, 0)
    count = 0
    n = len(predict_value)
    
    for i in range(n):
        if predict_value[i][0] == int(test_label[i][0]):
            count += 1
    
    accuracy = count / n
    return 1 - accuracy




def bey_prediction_g(filename1, filename2):
    train_data, test_data = open_file(filename1, filename2)
    train = train_select(train_data)
    train_s = train[0][0]
    train_l = train[0][1]
   
    eitas = [0.005, 0.001, 0.0007, 0.0005, 0.0001]
    result3 = []
    for eita in eitas:
        result = []
        for i in range(3):
            w_set, time_set = update_gradient(train_s, train_l,eita)
            result.append(time_set)
        new_time = np.array(result)
        mean_time = np.mean(new_time, axis = 0) 
        test = train_select(test_data)
        result2 = []
        for w in w_set:
           error = predict_bayesian(test, w)
           result2.append(error)
        result3.append((mean_time, result2))
    return result3
    
result3 = bey_prediction_g("A.csv", "labels-A.csv")

plt.figure()

plt.plot(result3[0][0],  result3[0][1], 'r-', label = "error as a function of run time for eita = 0.005")
plt.plot(result3[1][0],  result3[1][1], 'b-', label = "error as a function of run time for eita = 0.001")
plt.plot(result3[2][0],  result3[2][1], 'k-', label = "error as a function of run time for eita = 0.0007")
plt.plot(result3[3][0],  result3[3][1], 'g-', label = "error as a function of run time for eita = 0.0005")
plt.plot(result3[4][0],  result3[4][1], 'm-', label = "error as a function of run time for eita = 0.0001")

plt.xlim(0,15 + 1)
plt.ylim((0, 0.7))
plt.legend()
plt.title("performance of the gradient method for A.csv")
plt.ylabel("error rate")
plt.xlabel("time")
plt.savefig('graph2.png')