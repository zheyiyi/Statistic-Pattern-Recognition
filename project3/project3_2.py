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

def update_gradient(train_set, train_label): 
    time_value = []
    w_value = []
    t = 0
    eita = 0.001
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
   
    
def update_bayesian(train_set, train_label):
    time_value = []
    w_value = []
    t = 0
    train_set = np.array(train_set, dtype = float)
    train_label = np.array(train_label, dtype = float)
### shaop 问题  wo 应该是什么shape y 是什么shape 有问题。。。    
    Wo = np.zeros((len(train_set[0]),1))
    w_value.append(Wo)
    time_value.append(0)
    
    t0 = time.time()     
    a = [np.dot(Wo.T, train_set[i]) for i in range(len(train_set))]
    a = np.array(a)
    y = sigmoid(a)
    
    d = np.array([i * (1 - i) for i in y])
    r = np.diag(d.T[0])
    
    #a = np.linalg.inv((0.1 * np.identity(len(train_set[0]))  + np.dot(np.dot(train_set.T, r), train_set)))
   
    value = np.dot(np.linalg.inv((0.1 * np.identity(len(train_set[0]))  + np.dot(np.dot(train_set.T, r), train_set))),(np.dot(train_set.T, (y - train_label)) + 0.1 * Wo))
   
   
    Wn = Wo - value
    count = 1
    t1 = time.time()
    t += t1 - t0
    w_value.append(Wn)
    time_value.append(t)
    
    while  LA.norm(Wo) == 0 or (LA.norm(Wn- Wo) / LA.norm(Wo) > 0.001 and count < 100):
        t0 = time.time()
        Wo = Wn
        a = [np.dot(Wo.T, train_set[i]) for i in range(len(train_set))]
        a = np.array(a)
        y = sigmoid(a)        
        d = np.array([i * (1 - i) for i in y])
        r = np.diag(d.T[0])        
        value = np.dot(np.linalg.inv((0.1 * np.identity(len(train_set[0]))  + np.dot(np.dot(train_set.T, r), train_set))),(np.dot(train_set.T, (y - train_label)) + 0.1 * Wo))
        Wn = Wo - value
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


def bey_prediction(filename1, filename2):
    train_data, test_data = open_file(filename1, filename2)
    train = train_select(train_data)
    train_s = train[0][0]
    train_l = train[0][1]
    result = []

    for i in range(3):
        w_set, time_set = update_bayesian(train_s, train_l)
        result.append(time_set)
    new_time = np.array(result)
    mean_time = np.mean(new_time, axis = 0) 
    test = train_select(test_data)
    result2 = []
    for w in w_set:
       error = predict_bayesian(test, w)
       result2.append(error)
    
    return mean_time, result2  


def bey_prediction_g(filename1, filename2):
    train_data, test_data = open_file(filename1, filename2)
    train = train_select(train_data)
    train_s = train[0][0]
    train_l = train[0][1]
    result = []

    for i in range(3):
        w_set, time_set = update_gradient(train_s, train_l)
        result.append(time_set)
    new_time = np.array(result)
    mean_time = np.mean(new_time, axis = 0) 
    test = train_select(test_data)
    result2 = []
    for w in w_set:
       error = predict_bayesian(test, w)
       result2.append(error)
    
    return mean_time, result2 


time_A, result_A = bey_prediction("A.csv", "labels-A.csv")
time_A_g, result_A_g = bey_prediction_g("A.csv", "labels-A.csv")

plt.figure()
plt.plot(time_A,  result_A, 'r-', label = "error as a function of run time for newton method for A.csv")
n = time_A[-1]
plt.xlim(0,n + 0.01)
plt.ylim((0, 0.7))
plt.legend()
plt.title("performance of the newton method ")
plt.ylabel("error rate")
plt.xlabel("time")
plt.savefig('graph1.png')


plt.figure()
plt.plot(time_A_g,  result_A_g, 'r-', label = "error as a function of run time for A.csv")
n = time_A_g[-1]
plt.xlim(0,n + 1)
plt.ylim((0, 0.7))
plt.legend()
plt.title("performance of the gradient method ")
plt.ylabel("error rate")
plt.xlabel("time")
plt.savefig('graph2.png')

time_u_g, result_u_g = bey_prediction_g("usps.csv", "labels-usps.csv")
time_u, result_u = bey_prediction("usps.csv", "labels-usps.csv")


plt.figure()
plt.plot(time_u,  result_u, 'r-', label = "error as a function of run time for newton method for usps.csv")
n = time_u[-1]
plt.xlim(0,n + 0.01)
plt.ylim((0, 0.7))
plt.legend()
plt.title("performance of the newton method ")
plt.ylabel("error rate")
plt.xlabel("time")
plt.savefig('graph3.png')


plt.figure()
plt.plot(time_u_g,  result_u_g, 'r-', label = "error as a function of run time for usps.csv")
n = time_u_g[-1]
plt.xlim(0,0.14)
plt.ylim((0, 0.7))
plt.legend()
plt.title("performance of the gradient method ")
plt.ylabel("error rate")
plt.xlabel("time")
plt.savefig('graph4.png')
"""
train_data, test_data = open_file("A.csv", "labels-A.csv")
train = train_select(train_data)
train_s = train[0][0]
train_l = train[0][1]
result = []
    
for i in range(3):
   w_set, time_set = update_gradient(train_s, train_l)
   result.append(time_set)

new_time = np.array(result)
mean_time = np.mean(new_time, axis = 0) 
test = train_select(test_data)
result2 = []
for w in w_set:
        error = predict_bayesian(test, w)
        result2.append(error)
        
 """       

"""

#print(len(train_data))
#print(len(test_data))
#print(mean_A)
#print(std_A)
#3print(data_length_A)

plt.figure()
plt.plot(data_length_A ,  mean_A, 'r-', label = "accuracy as a function of data size of bayesian model")
plt.errorbar(data_length_A , mean_A, yerr = std_A, linestyle = "None")
#plt.plot(data,  mean_amazon,'k-', label = "smooth accuracy as a function of data size of amazon")
#plt.errorbar(data, mean_amazon, yerr = std_amazon, linestyle = "None")
n = data_length_A[-1]
plt.xlim(0,n + 100)
plt.ylim((0.4, 1))
plt.legend()
plt.title("learning Rate Curve")
plt.ylabel("Accuracy")
plt.xlabel("data size")
plt.savefig('graph1.png')
"""