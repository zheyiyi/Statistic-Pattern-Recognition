# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:03:38 2017

@author: zheyiyi

"""

import math
import csv
import random
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
        test_list = random.sample(new_list_data, test)
        train_list = []
        for i in new_list_data:
            if i not in test_list:
                train_list.append(i)
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
    data_length = []
    for i in range(10,len(train_list) + 1, 50):
        data_length.append(i)
        train_incre.append(random.sample(train_list,i))
    
    train_s = []
    
    for list in train_incre:
        temp_s = []
        temp_l = []
        for i in list:
            
            temp_s.append(i[:-1])
            temp_l.append([i[-1]])
        
        train_s.append([temp_s, temp_l])
      
    return train_s, data_length


def train_select_1(train_list):
    train_incre = []           
    train_incre.append(train_list)    
    train_s = []    
    for list in train_incre:
        temp_s = []
        temp_l = []
        for i in list:
            
            temp_s.append(i[:-1])
            temp_l.append([i[-1]])
        
        train_s.append([temp_s, temp_l])
      
    return train_s



  
def sigmoid(z):
    s = 1 / (1 + np.exp(-1 * z)) 
    return s
      
def update_bayesian(train_set, train_label):

    train_set = np.array(train_set, dtype = float)
    train_label = np.array(train_label, dtype = float)
### shaop 问题  wo 应该是什么shape y 是什么shape 有问题。。。    
    Wo = np.zeros((len(train_set[0]),1))     
    a = [np.dot(Wo.T, train_set[i]) for i in range(len(train_set))]
    a = np.array(a)
    y = sigmoid(a)
    
    d = np.array([i * (1 - i) for i in y])
    r = np.diag(d.T[0])
    
    
    ## 不确定 np.identity 是否正确
    #print(np.shape(y), np.shape(train_set))  
   
    a = np.linalg.inv((0.1 * np.identity(len(train_set[0]))  + np.dot(np.dot(train_set.T, r), train_set)))
   
    #value = (np.dot(train_set.T, (y - train_label)) + 0.1 * Wo) /(0.1 * np.identity(len(train_set[0]))  + np.dot(np.dot(train_set.T, r), train_set))
    value = np.dot(np.linalg.inv((0.1 * np.identity(len(train_set[0]))  + np.dot(np.dot(train_set.T, r), train_set))),(np.dot(train_set.T, (y - train_label)) + 0.1 * Wo))
   
   
    Wn = Wo - value
    count = 1
    # 有问题
    while  LA.norm(Wo) == 0 or (LA.norm(Wn- Wo) / LA.norm(Wo) > 0.001 or count < 100):
        Wo = Wn
        a = [np.dot(Wo.T, train_set[i]) for i in range(len(train_set))]
        a = np.array(a)
        y = sigmoid(a)        
        d = np.array([i * (1 - i) for i in y])
        r = np.diag(d.T[0])        
        value = np.dot(np.linalg.inv((0.1 * np.identity(len(train_set[0]))  + np.dot(np.dot(train_set.T, r), train_set))),(np.dot(train_set.T, (y - train_label)) + 0.1 * Wo))
        Wn = Wo - value
        count += 1
    
    return Wn

  
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
    return 1- accuracy

    
    

def open_file_g(filename, filename1):
    list_data = []
    with open(filename) as csvfile:
        data = csv.reader(csvfile, delimiter = "," )
        for row in data:
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
        test_list = random.sample(new_list_data, test)
        train_list = []
        for i in new_list_data:
            if i not in test_list:
                train_list.append(i)
        return train_list, test_list
 
def train_select_g(train_list):
    train_incre = []
    data_length = []
    for i in range(10,len(train_list) + 1, 50):
        data_length.append(i)
        train_incre.append(random.sample(train_list,i))
    
    train_s = []
    
    for list in train_incre:
        temp_s = []
        temp_l = []
        for i in list:
            
            temp_s.append(i[:-1])
            temp_l.append([i[-1]])
        
        train_s.append([temp_s, temp_l])
      
    return train_s, data_length
       
def train_select_g_1(train_list):
    train_incre = []           
    train_incre.append(train_list)    
    train_s = []    
    for list in train_incre:
        temp_s = []
        temp_l = []
        for i in list:
            
            temp_s.append(i[:-1])
            temp_l.append([i[-1]])
        
        train_s.append([temp_s, temp_l])
      
    return train_s
    
def genertation_w(train_s, train_label):
    train_s = np.array(train_s, dtype = float)
    train_label = np.array(train_label, dtype = float)
    n = len(train_label)
    n1 = np.sum(train_label)
    
    n2 = n - n1
    pai = n1 / n
   
    u1 = np.dot(train_s.T, train_label) / n1
    u2 = np.dot(train_s.T, 1 - train_label) / n2
    
    x_u1 = np.subtract(train_s, u1.T)
    s1n = np.dot(x_u1.T, x_u1)
    
    x_u2 = np.subtract(train_s,u2.T)
    s2n = np.dot(x_u2.T, x_u2)
    
    s = (s1n + s2n) / n    
    s = s + np.identity(len(s)) * (10 ** (-9))
    s_i = np.linalg.inv(s)
    
    w = np.dot(s_i,(u1 - u2))
    w0 = - np.dot(np.dot(u1.T, s_i), u1) / 2 + np.dot(np.dot(u2.T, s_i), u2) / 2 + np.log(pai) - np.log(1 - pai)
    return w,w0

def predict_generation(test_sample, wn, w0):
    
    test_data, test_label = test_sample[0]
    test_data = np.array(test_data, dtype = float)
    wn = np.array(wn, dtype = float)
  
    a = [np.dot(wn.T, test_data[i]) + w0 for i in range(len(test_data))]
    a = np.array(a)
   
    predict_value = np.where(a > 0, 1, 0)
    count = 0
    n = len(predict_value)
    
    for i in range(n):
        if predict_value[i][0] == int(test_label[i][0]):
            count += 1
    
    accuracy = count / n
    return 1 - accuracy

def generation_prediction(filename1, filename2):
    train_x, test_x= open_file_g(filename1, filename2)
    result = []
    for i in range(30):
        train_s, data_length = train_select_g(train_x)
        test_sample = train_select_g_1(test_x)
        sub_result = []
        for train_data, train_label in train_s:
            w, w0 = genertation(train_data, train_label)
            error = predict_generation(test_sample,w,w0)
            sub_result.append(error)
        result.append(sub_result)
    #print(result)
    ac = np.array(result)
    mean = np.mean(ac, axis=0)
    standard = np.std(ac, axis=0)
    return mean, standard, data_length
    



def bey_prediction(filename1, filename2):
    train_x, test_x= open_file(filename1, filename2)
    result = []
    for i in range(30):
        train_s, data_length = train_select(train_x)
        test_sample = train_select_1(test_x)
        sub_result = []
        for train_data, train_label in train_s:
            new_n = update_bayesian(train_data, train_label)
            accuracy = predict_bayesian(test_sample, new_n)
            sub_result.append(accuracy)
        result.append(sub_result)
    #print(result)
    ac = np.array(result)
    mean = np.mean(ac, axis=0)
    standard = np.std(ac, axis=0)
    return mean, standard, data_length


mean_A, std_A, data_length_A = bey_prediction("A.csv", "labels-A.csv")
mean_A_g, std_A_g, data_length_A_g = generation_prediction("A.csv", "labels-A.csv")




plt.figure()
plt.plot(data_length_A ,  mean_A, 'r-', label = "error as a function of data size of bayesian model")
plt.errorbar(data_length_A , mean_A, yerr = std_A, linestyle = "None")
plt.plot(data_length_A_g,  mean_A_g,'k-', label = "error as a function of data size of generation model")
plt.errorbar(data_length_A_g , mean_A_g, yerr = std_A_g, linestyle = "None")
n = data_length_A_g[-1]
plt.xlim(0,n + 100)
plt.ylim((0, 0.7))
plt.legend()
plt.title("A sample")
plt.ylabel("error rate")
plt.xlabel("data size")
plt.savefig('graph1.png')




mean_B, std_B, data_length_B = bey_prediction("B.csv", "labels-B.csv")
mean_B_g, std_B_g, data_length_B_g = generation_prediction("B.csv", "labels-B.csv")


plt.figure()
plt.plot(data_length_B ,  mean_B, 'r-', label = "error as a function of data size of bayesian model")
plt.errorbar(data_length_B , mean_B, yerr = std_B, linestyle = "None")
plt.plot(data_length_B_g,  mean_B_g,'k-', label = "error as a function of data size of generation model")
plt.errorbar(data_length_B_g , mean_B_g, yerr = std_B_g, linestyle = "None")
n = data_length_B[-1]
plt.xlim(0,n + 5)
plt.ylim((0, 0.5))
plt.legend()
plt.title("B sample")
plt.ylabel("error rate")
plt.xlabel("data size")
plt.savefig('graph2.png')


mean_u, std_u, data_length_u = bey_prediction("usps.csv", "labels-usps.csv")
mean_u_g, std_u_g, data_length_u_g = generation_prediction("usps.csv", "labels-usps.csv")


plt.figure()
plt.plot(data_length_u ,  mean_u, 'r-', label = "error as a function of data size of bayesian model")
plt.errorbar(data_length_u , mean_u, yerr = std_u, linestyle = "None")
plt.plot(data_length_u_g,  mean_u_g,'k-', label = "error as a function of data size of generation model")
plt.errorbar(data_length_u_g , mean_u_g, yerr = std_u_g, linestyle = "None")
n = data_length_u[-1]
plt.xlim(0,n + 100)
plt.ylim((0, 0.7))
plt.legend()
plt.title("usps sample")
plt.ylabel("error rate")
plt.xlabel("data size")
plt.savefig('graph3.png')



